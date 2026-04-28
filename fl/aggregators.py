import collections
import math
from abc import ABC, abstractmethod

import torch

from fl.expert_filter_state import predict_expert_state, wolf_scalar_update


def parse_expert_ref_from_key(key):
    parts = key.split(".")
    if "blocks" not in parts or "experts" not in parts:
        return None

    blocks_idx = parts.index("blocks")
    experts_idx = parts.index("experts")
    if blocks_idx + 1 >= len(parts) or experts_idx + 1 >= len(parts):
        return None
    if not parts[blocks_idx + 1].isdigit() or not parts[experts_idx + 1].isdigit():
        return None

    return parts[blocks_idx + 1], int(parts[experts_idx + 1])


def stable_sigmoid(value):
    value = float(value)
    if math.isnan(value):
        return 0.5
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def format_scientific_list(values):
    return [f"{float(value):.12e}" for value in values]


class Aggregator(ABC):
    # 聚合器统一接口。后续新增聚合方法时，只需要新增实现类并在 build_aggregator 中注册。
    @abstractmethod
    def aggregate(self, client_updates, client_weights, global_model=None, **kwargs):
        pass


class FedAvgAggregator(Aggregator):
    # 标准 FedAvg：
    # 对完整 state_dict 做按客户端样本数加权平均，权重 w_i = n_i / sum_j n_j。
    def aggregate(self, client_updates, client_weights, global_model=None, **kwargs):
        if len(client_updates) == 0:
            raise ValueError("FedAvg requires at least one client update")
        if len(client_updates) != len(client_weights):
            raise ValueError("client_updates and client_weights must have the same length")

        total_weight = sum(client_weights)
        if total_weight <= 0:
            raise ValueError("FedAvg requires positive total client weight")

        aggregated_state = collections.OrderedDict()
        for key in client_updates[0].keys():
            first_value = client_updates[0][key].detach().cpu()
            if torch.is_floating_point(first_value):
                aggregated_state[key] = torch.zeros_like(first_value)
                for update, weight in zip(client_updates, client_weights):
                    aggregated_state[key] += update[key].detach().cpu() * (weight / total_weight)
            else:
                # 非浮点 buffer 通常不能加权平均，沿用第一个客户端的值。
                aggregated_state[key] = first_value.clone()

        return aggregated_state


class ExpertFedAvgAggregator(Aggregator):
    # FL + MoE 专家级 FedAvg：
    # - 普通共享层仍按客户端训练样本数 n_i 做标准 FedAvg；
    # - blocks.{layer}.ffn.experts.{expert_id}.* 参数按该层该专家实际处理的 token 数 n_{i,l,e} 加权。
    def __init__(self):
        pass

    def aggregate(self, client_updates, client_weights, global_model=None, **kwargs):
        if len(client_updates) == 0:
            raise ValueError("ExpertFedAvg requires at least one client update")
        if len(client_updates) != len(client_weights):
            raise ValueError("client_updates and client_weights must have the same length")

        expert_weights = kwargs.get("expert_weights")
        if expert_weights is None:
            raise ValueError("ExpertFedAvg requires expert_weights for expert-level aggregation")
        if len(expert_weights) != len(client_updates):
            raise ValueError("expert_weights and client_updates must have the same length")

        global_state = global_model.state_dict() if global_model is not None else None
        aggregated_state = collections.OrderedDict()
        total_client_weight = sum(client_weights)
        if total_client_weight <= 0:
            raise ValueError("ExpertFedAvg requires positive total client weight")

        for key in client_updates[0].keys():
            first_value = client_updates[0][key].detach().cpu()
            if not torch.is_floating_point(first_value):
                aggregated_state[key] = first_value.clone()
                continue

            expert_ref = self._parse_expert_ref(key)
            if expert_ref is None:
                weights = client_weights
            else:
                layer_id, expert_id = expert_ref
                weights = [
                    self._get_expert_weight(client_usage, layer_id, expert_id)
                    for client_usage in expert_weights
                ]

            total_weight = sum(weights)
            if total_weight <= 0:
                # 某一轮没有客户端使用该专家时，不用随机客户端覆盖它，保留服务端旧参数更稳。
                if global_state is not None:
                    aggregated_state[key] = global_state[key].detach().cpu().clone()
                else:
                    aggregated_state[key] = first_value.clone()
                continue

            aggregated_state[key] = torch.zeros_like(first_value)
            for update, weight in zip(client_updates, weights):
                aggregated_state[key] += update[key].detach().cpu() * (weight / total_weight)

        return aggregated_state

    def _parse_expert_ref(self, key):
        return parse_expert_ref_from_key(key)

    def _get_expert_weight(self, client_usage, layer_id, expert_id):
        if isinstance(client_usage, dict):
            if layer_id is None:
                usage = client_usage.get("expert_activations")
            else:
                layer_stats = client_usage.get("expert_stats_by_layer", {}).get(str(layer_id), {})
                usage = layer_stats.get("expert_activations")
                if usage is None:
                    usage = client_usage.get("expert_activations_by_layer", {}).get(str(layer_id))

            if usage is None:
                return 0.0
            if expert_id >= len(usage):
                raise ValueError(f"Missing expert weight for expert id {expert_id}")
            return float(usage[expert_id])

        if expert_id >= len(client_usage):
            raise ValueError(f"Missing expert weight for expert id {expert_id}")
        return float(client_usage[expert_id])


class FedWoLFAggregator(Aggregator):
    # FedWoLF 聚合器：
    # - shared / router / classifier 等非 expert 参数仍按客户端样本数做 FedAvg；
    # - expert 参数按客户端上传的 expert_fisher_score_by_layer 做 Fisher-score 加权平均。
    # - agg_method=fedwolf 时额外吸收 Fisher log evidence 更新 WoLF-IMQ filter state。
    # - agg_method=fedwolf 时使用 gamma=sigmoid(mu) 在 old global expert 和 Theta_bar 之间插值。
    def __init__(self, args=None, use_wolf_filter=True, use_gamma=True):
        self.eps = float(getattr(args, "fedwolf_eps", 1e-8))
        self.process_noise_q = float(getattr(args, "fedwolf_process_noise_q", 0.01))
        self.sigma_e2 = float(getattr(args, "fedwolf_sigma_e2", 1.0))
        self.imq_c = float(getattr(args, "fedwolf_imq_c", 1.0))
        self.num_experts = getattr(args, "num_experts", None)
        self.use_wolf_filter = use_wolf_filter
        self.use_gamma = use_gamma
        if self.use_wolf_filter and self.imq_c <= 0:
            raise ValueError("fedwolf_imq_c must be positive")
        self.expert_filter_mu = {}
        self.expert_filter_P = {}
        self.last_filter_summary = {}
        self.last_gamma_summary = {}

    def aggregate(self, client_updates, client_weights, global_model=None, **kwargs):
        if len(client_updates) == 0:
            raise ValueError("FedWoLF requires at least one client update")
        if len(client_updates) != len(client_weights):
            raise ValueError("client_updates and client_weights must have the same length")

        client_stats = kwargs.get("client_stats")
        if client_stats is None:
            client_stats = kwargs.get("expert_weights")
        if client_stats is None:
            raise ValueError("FedWoLF Fisher-only requires client_stats or expert_weights")
        if len(client_stats) != len(client_updates):
            raise ValueError("client_stats/expert_weights and client_updates must have the same length")

        global_state = global_model.state_dict() if global_model is not None else None
        if self.use_gamma and global_state is None:
            raise ValueError("FedWoLF gamma interpolation requires global_model for old expert parameters")
        total_client_weight = sum(client_weights)
        if total_client_weight <= 0:
            raise ValueError("FedWoLF requires positive total client weight")

        if self.use_wolf_filter:
            self._update_filter_state(client_updates[0].keys(), client_stats)
        else:
            self.last_filter_summary = {}
        self.last_gamma_summary = {}

        aggregated_state = collections.OrderedDict()
        for key in client_updates[0].keys():
            first_value = client_updates[0][key].detach().cpu()
            if not torch.is_floating_point(first_value):
                aggregated_state[key] = first_value.clone()
                continue

            expert_ref = parse_expert_ref_from_key(key)
            if expert_ref is None:
                weights = client_weights
                total_weight = total_client_weight
                denominator = total_weight
            else:
                layer_id, expert_id = expert_ref
                weights = [
                    self._get_fisher_score(client_stat, layer_id, expert_id)
                    for client_stat in client_stats
                ]
                total_weight = sum(weights)
                denominator = total_weight + self.eps

            if total_weight <= 0:
                # 该 expert 本轮没有有效 Fisher evidence 时，保留上一轮 global expert 参数。
                if global_state is not None:
                    aggregated_state[key] = global_state[key].detach().cpu().clone()
                else:
                    aggregated_state[key] = first_value.clone()
                continue

            theta_bar = torch.zeros_like(first_value)
            for update, weight in zip(client_updates, weights):
                theta_bar += update[key].detach().cpu() * (weight / denominator)

            if expert_ref is None or not self.use_gamma:
                aggregated_state[key] = theta_bar
                continue

            layer_id, expert_id = expert_ref
            gamma = self._get_gamma(layer_id, expert_id)
            theta_old = global_state[key].detach().cpu()
            aggregated_state[key] = theta_old * (1.0 - gamma) + theta_bar * gamma
            self._record_gamma(layer_id, expert_id, gamma, total_weight)

        if self.use_gamma:
            self._merge_gamma_summary()

        return aggregated_state

    def _update_filter_state(self, state_keys, client_stats):
        expert_refs = sorted(
            {expert_ref for key in state_keys if (expert_ref := parse_expert_ref_from_key(key)) is not None},
            key=lambda item: (int(item[0]), item[1]),
        )
        if not expert_refs:
            self.last_filter_summary = {}
            return

        layer_sizes = {}
        for layer_id, expert_id in expert_refs:
            layer_sizes[layer_id] = max(layer_sizes.get(layer_id, 0), expert_id + 1)
        if self.num_experts is not None:
            for layer_id in layer_sizes:
                layer_sizes[layer_id] = max(layer_sizes[layer_id], int(self.num_experts))

        round_filter_stats = {
            layer_id: self._new_filter_stat_buffers(size)
            for layer_id, size in layer_sizes.items()
        }

        for layer_id, expert_id in expert_refs:
            self._ensure_filter_state_for_layer(layer_id, layer_sizes[layer_id])
            mu_current, p_current = predict_expert_state(
                self.expert_filter_mu[layer_id][expert_id].item(),
                self.expert_filter_P[layer_id][expert_id].item(),
                self.process_noise_q,
            )

            for client_stat in client_stats:
                z = self._get_fisher_log_score(client_stat, layer_id, expert_id)
                if z is None:
                    round_filter_stats[layer_id]["skipped_observations"][expert_id] += 1
                    continue
                s = self._get_fisher_score(client_stat, layer_id, expert_id)
                mu_current, p_current, update_info = wolf_scalar_update(
                    mu=mu_current,
                    p=p_current,
                    observation=z,
                    score=s,
                    sigma_e2=self.sigma_e2,
                    eps=self.eps,
                    imq_c=self.imq_c,
                )
                self._add_filter_update_info(
                    stat_buffers=round_filter_stats[layer_id],
                    expert_id=expert_id,
                    update_info=update_info,
                    score=s,
                    observation=z,
                )

            self.expert_filter_mu[layer_id][expert_id] = mu_current
            self.expert_filter_P[layer_id][expert_id] = p_current

        self.last_filter_summary = self._build_filter_summary(round_filter_stats)

    def _ensure_filter_state_for_layer(self, layer_id, num_experts):
        layer_id = str(layer_id)
        if layer_id not in self.expert_filter_mu:
            self.expert_filter_mu[layer_id] = torch.zeros(num_experts, dtype=torch.float32)
            self.expert_filter_P[layer_id] = torch.ones(num_experts, dtype=torch.float32)
            return

        current_size = self.expert_filter_mu[layer_id].numel()
        if current_size >= num_experts:
            return

        mu_padding = torch.zeros(num_experts - current_size, dtype=torch.float32)
        p_padding = torch.ones(num_experts - current_size, dtype=torch.float32)
        self.expert_filter_mu[layer_id] = torch.cat([self.expert_filter_mu[layer_id], mu_padding])
        self.expert_filter_P[layer_id] = torch.cat([self.expert_filter_P[layer_id], p_padding])

    def _new_filter_stat_buffers(self, num_experts):
        return {
            "count": torch.zeros(num_experts, dtype=torch.float32),
            "score_sum": torch.zeros(num_experts, dtype=torch.float32),
            "max_score": torch.zeros(num_experts, dtype=torch.float32),
            "observation_sum": torch.zeros(num_experts, dtype=torch.float32),
            "max_observation": torch.zeros(num_experts, dtype=torch.float32),
            "R_sum": torch.zeros(num_experts, dtype=torch.float32),
            "min_R": torch.full((num_experts,), float("inf"), dtype=torch.float32),
            "max_R": torch.zeros(num_experts, dtype=torch.float32),
            "weight_sum": torch.zeros(num_experts, dtype=torch.float32),
            "min_weight": torch.full((num_experts,), float("inf"), dtype=torch.float32),
            "max_weight": torch.zeros(num_experts, dtype=torch.float32),
            "abs_residual_sum": torch.zeros(num_experts, dtype=torch.float32),
            "kalman_gain_sum": torch.zeros(num_experts, dtype=torch.float32),
            "max_kalman_gain": torch.zeros(num_experts, dtype=torch.float32),
            "skipped_observations": torch.zeros(num_experts, dtype=torch.float32),
        }

    def _add_filter_update_info(self, stat_buffers, expert_id, update_info, score, observation):
        score = max(float(score), 0.0)
        observation = float(observation)
        observation_noise = float(update_info["R"])
        weight = float(update_info["weight"])
        residual = float(update_info["residual"])
        kalman_gain = float(update_info["kalman_gain"])

        stat_buffers["count"][expert_id] += 1
        stat_buffers["score_sum"][expert_id] += score
        stat_buffers["max_score"][expert_id] = max(float(stat_buffers["max_score"][expert_id]), score)
        stat_buffers["observation_sum"][expert_id] += observation
        stat_buffers["max_observation"][expert_id] = max(float(stat_buffers["max_observation"][expert_id]), observation)
        stat_buffers["R_sum"][expert_id] += observation_noise
        stat_buffers["min_R"][expert_id] = min(float(stat_buffers["min_R"][expert_id]), observation_noise)
        stat_buffers["max_R"][expert_id] = max(float(stat_buffers["max_R"][expert_id]), observation_noise)
        stat_buffers["weight_sum"][expert_id] += weight
        stat_buffers["min_weight"][expert_id] = min(float(stat_buffers["min_weight"][expert_id]), weight)
        stat_buffers["max_weight"][expert_id] = max(float(stat_buffers["max_weight"][expert_id]), weight)
        stat_buffers["abs_residual_sum"][expert_id] += abs(residual)
        stat_buffers["kalman_gain_sum"][expert_id] += kalman_gain
        stat_buffers["max_kalman_gain"][expert_id] = max(float(stat_buffers["max_kalman_gain"][expert_id]), kalman_gain)

    def _safe_mean(self, total, count):
        return torch.where(count > 0, total / count.clamp_min(1.0), torch.zeros_like(total))

    def _build_filter_summary(self, round_filter_stats):
        summary = {}
        for layer_id in sorted(self.expert_filter_mu, key=lambda item: int(item)):
            stats = round_filter_stats.get(layer_id)
            if stats is None:
                stats = self._new_filter_stat_buffers(self.expert_filter_mu[layer_id].numel())

            count = stats["count"]
            mean_s = self._safe_mean(stats["score_sum"], count)
            mean_z = self._safe_mean(stats["observation_sum"], count)
            mean_R = self._safe_mean(stats["R_sum"], count)
            mean_weight = self._safe_mean(stats["weight_sum"], count)
            mean_abs_residual = self._safe_mean(stats["abs_residual_sum"], count)
            mean_kalman_gain = self._safe_mean(stats["kalman_gain_sum"], count)
            min_R = torch.where(
                count > 0,
                stats["min_R"],
                torch.zeros_like(stats["min_R"]),
            )
            min_weight = torch.where(
                count > 0,
                stats["min_weight"],
                torch.zeros_like(stats["min_weight"]),
            )
            summary[layer_id] = {
                "total_fisher_weight": format_scientific_list(stats["score_sum"].tolist()),
                "mean_s": format_scientific_list(mean_s.tolist()),
                "max_s": format_scientific_list(stats["max_score"].tolist()),
                "mean_z": format_scientific_list(mean_z.tolist()),
                "max_z": format_scientific_list(stats["max_observation"].tolist()),
                "mean_R": format_scientific_list(mean_R.tolist()),
                "min_R": format_scientific_list(min_R.tolist()),
                "max_R": format_scientific_list(stats["max_R"].tolist()),
                "mean_kalman_gain": format_scientific_list(mean_kalman_gain.tolist()),
                "max_kalman_gain": format_scientific_list(stats["max_kalman_gain"].tolist()),
                "mean_abs_residual": format_scientific_list(mean_abs_residual.tolist()),
                "mean_imq_weight": format_scientific_list(mean_weight.tolist()),
                "mean_weight": format_scientific_list(mean_weight.tolist()),
                "min_weight": format_scientific_list(min_weight.tolist()),
                "max_weight": format_scientific_list(stats["max_weight"].tolist()),
                "mu": format_scientific_list(self.expert_filter_mu[layer_id].tolist()),
                "P": format_scientific_list(self.expert_filter_P[layer_id].tolist()),
                "skipped_observations": [int(v) for v in stats["skipped_observations"].tolist()],
            }
        return summary

    def _get_gamma(self, layer_id, expert_id):
        layer_id = str(layer_id)
        if layer_id not in self.expert_filter_mu or expert_id >= self.expert_filter_mu[layer_id].numel():
            return 0.5
        return stable_sigmoid(self.expert_filter_mu[layer_id][expert_id].item())

    def _record_gamma(self, layer_id, expert_id, gamma, total_fisher_weight):
        layer_id = str(layer_id)
        layer_summary = self.last_gamma_summary.setdefault(
            layer_id,
            {"gamma": []},
        )
        while len(layer_summary["gamma"]) <= expert_id:
            layer_summary["gamma"].append("0.000000000000e+00")
        layer_summary["gamma"][expert_id] = f"{float(gamma):.12e}"

    def _merge_gamma_summary(self):
        for layer_id, gamma_summary in self.last_gamma_summary.items():
            non_gamma_summary = {
                key: value
                for key, value in gamma_summary.items()
                if key != "gamma"
            }
            if non_gamma_summary:
                self.last_filter_summary.setdefault(layer_id, {}).update(non_gamma_summary)
        for layer_id, layer_summary in self.last_filter_summary.items():
            if layer_id in self.expert_filter_mu:
                layer_summary["gamma"] = format_scientific_list(
                    stable_sigmoid(value)
                    for value in self.expert_filter_mu[layer_id].tolist()
                )

    def _get_fisher_score(self, client_stats, layer_id, expert_id):
        score = self._get_layer_expert_value(
            client_stats=client_stats,
            field_name="expert_fisher_score_by_layer",
            layer_id=layer_id,
            expert_id=expert_id,
        )
        if score is None:
            return 0.0
        return max(score, 0.0)

    def _get_fisher_log_score(self, client_stats, layer_id, expert_id):
        return self._get_layer_expert_value(
            client_stats=client_stats,
            field_name="expert_fisher_log_score_by_layer",
            layer_id=layer_id,
            expert_id=expert_id,
        )

    def _get_layer_expert_value(self, client_stats, field_name, layer_id, expert_id):
        if not isinstance(client_stats, dict):
            return None

        value_by_layer = client_stats.get(field_name, {})
        layer_values = value_by_layer.get(str(layer_id))
        if layer_values is None:
            return None

        if torch.is_tensor(layer_values):
            flat_values = layer_values.detach().cpu().flatten()
            if expert_id >= flat_values.numel():
                return None
            value = flat_values[expert_id].item()
        else:
            if expert_id >= len(layer_values):
                return None
            value = layer_values[expert_id]

        value = float(value)
        if not math.isfinite(value):
            return None
        return value


def build_aggregator(args):
    if args.agg_method == "fedwolf_fisher_only":
        return FedWoLFAggregator(args, use_wolf_filter=False, use_gamma=False)
    if args.agg_method == "fedwolf":
        return FedWoLFAggregator(args, use_wolf_filter=True, use_gamma=True)
    if args.agg_method == "expert_fedavg":
        return ExpertFedAvgAggregator()
    if args.agg_method == "fedavg":
        return FedAvgAggregator()
    raise ValueError(f"Unknown aggregation method: {args.agg_method}")
