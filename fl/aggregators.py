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


def resolve_aggregation_device(args):
    raw_device = getattr(args, "aggregation_device", "cpu")
    if raw_device is None:
        raw_device = "cpu"
    raw_device = str(raw_device).strip().lower()

    if raw_device in {"", "none", "null"}:
        raw_device = "cpu"

    if raw_device == "cpu":
        return torch.device("cpu")

    if raw_device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("aggregation_device='cuda' but CUDA is not available.")
        # 多 GPU 环境下优先跟随训练 device，避免聚合跑到错误的默认卡上。
        train_device = str(getattr(args, "device", "cuda")).strip().lower()
        if train_device.startswith("cuda"):
            return torch.device(train_device)
        return torch.device("cuda")

    if raw_device.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError(f"aggregation_device={raw_device!r} but CUDA is not available.")
        return torch.device(raw_device)

    raise ValueError(
        "aggregation_device must be 'cpu', 'cuda', or 'cuda:<index>', "
        f"got {raw_device!r}."
    )


class Aggregator(ABC):
    # 聚合器统一接口。后续新增聚合方法时，只需要新增实现类并在 build_aggregator 中注册。
    def __init__(self, args=None):
        self.aggregation_device = resolve_aggregation_device(args)

    def _to_agg_device(self, tensor):
        tensor = tensor.detach()
        if tensor.device == self.aggregation_device:
            return tensor
        if self.aggregation_device.type == "cuda":
            return tensor.to(self.aggregation_device, non_blocking=True)
        return tensor.to(self.aggregation_device)

    def _to_output_device(self, tensor):
        return tensor.detach().cpu()

    @abstractmethod
    def aggregate(self, client_updates, client_weights, global_model=None, **kwargs):
        pass


class FedAvgAggregator(Aggregator):
    # 标准 FedAvg：
    # 对完整 state_dict 做按客户端样本数加权平均，权重 w_i = n_i / sum_j n_j。
    def __init__(self, args=None):
        super().__init__(args)

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
            first_value = client_updates[0][key].detach()
            if torch.is_floating_point(first_value):
                first_on_device = self._to_agg_device(first_value)
                aggregated_tensor = torch.zeros_like(first_on_device)
                for update, weight in zip(client_updates, client_weights):
                    update_tensor = self._to_agg_device(update[key])
                    aggregated_tensor += update_tensor * (weight / total_weight)
                aggregated_state[key] = self._to_output_device(aggregated_tensor)
            else:
                # 非浮点 buffer 通常不能加权平均，沿用第一个客户端的值。
                aggregated_state[key] = first_value.cpu().clone()

        return aggregated_state


class ExpertFedAvgAggregator(Aggregator):
    # FL + MoE 专家级 FedAvg：
    # - 普通共享层仍按客户端训练样本数 n_i 做标准 FedAvg；
    # - blocks.{layer}.ffn.experts.{expert_id}.* 参数按该层该专家实际处理的 token 数 n_{i,l,e} 加权。
    def __init__(self, args=None):
        super().__init__(args)

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
            first_value = client_updates[0][key].detach()
            if not torch.is_floating_point(first_value):
                aggregated_state[key] = first_value.cpu().clone()
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
                    aggregated_state[key] = first_value.cpu().clone()
                continue

            first_on_device = self._to_agg_device(first_value)
            aggregated_tensor = torch.zeros_like(first_on_device)
            for update, weight in zip(client_updates, weights):
                update_tensor = self._to_agg_device(update[key])
                aggregated_tensor += update_tensor * (weight / total_weight)
            aggregated_state[key] = self._to_output_device(aggregated_tensor)

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
    # - fedwolf_fisher_only 的 expert 参数按 raw Fisher score 做旧版加权平均；
    # - fedwolf 的 expert 参数按 sqrt(relative Fisher) 聚合，filter R 使用 evidence active tokens。
    # - agg_method=fedwolf 时额外吸收 Fisher log evidence 更新 WoLF-IMQ filter state。
    # - agg_method=fedwolf 时使用 learned/fixed gamma 在 old global expert 和 Theta_bar 之间插值。
    def __init__(self, args=None, use_wolf_filter=True, use_gamma=True):
        super().__init__(args)
        self.eps = float(getattr(args, "fedwolf_eps", 1e-8))
        self.process_noise_q = float(getattr(args, "fedwolf_process_noise_q", 0.01))
        self.sigma_e2 = float(getattr(args, "fedwolf_sigma_e2", 1.0))
        self.imq_c = float(getattr(args, "fedwolf_imq_c", 1.0))
        self.gamma_temperature = float(getattr(args, "fedwolf_gamma_temperature", 1.0))
        self.gamma_mode = str(getattr(args, "fedwolf_gamma_mode", "learned")).strip().lower()
        self.fixed_gamma = None
        try:
            self.gamma_min = float(getattr(args, "fedwolf_gamma_min", 0.0))
            self.gamma_max = float(getattr(args, "fedwolf_gamma_max", 1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("fedwolf_gamma_min and fedwolf_gamma_max must be convertible to float") from exc
        self.num_experts = getattr(args, "num_experts", None)
        self.use_wolf_filter = use_wolf_filter
        self.use_gamma = use_gamma
        self.use_relative_sqrt_fisher_aggregation = bool(use_wolf_filter and use_gamma)
        if self.use_wolf_filter and self.imq_c <= 0:
            raise ValueError("fedwolf_imq_c must be positive")
        if self.gamma_temperature <= 0:
            raise ValueError("fedwolf_gamma_temperature must be positive")
        if not math.isfinite(self.gamma_min) or not math.isfinite(self.gamma_max):
            raise ValueError("fedwolf_gamma_min and fedwolf_gamma_max must be finite")
        if self.gamma_min < 0.0 or self.gamma_min > 1.0:
            raise ValueError("fedwolf_gamma_min must be in [0, 1]")
        if self.gamma_max < 0.0 or self.gamma_max > 1.0:
            raise ValueError("fedwolf_gamma_max must be in [0, 1]")
        if self.gamma_min > self.gamma_max:
            raise ValueError("fedwolf_gamma_min must be <= fedwolf_gamma_max")
        if self.gamma_mode not in {"learned", "fixed"}:
            raise ValueError("fedwolf_gamma_mode must be either 'learned' or 'fixed'")
        if self.use_gamma and self.gamma_mode == "fixed":
            raw_fixed_gamma = getattr(args, "fedwolf_fixed_gamma", None)
            if raw_fixed_gamma is None:
                raise ValueError("fedwolf_fixed_gamma must be set when fedwolf_gamma_mode is 'fixed'")
            try:
                self.fixed_gamma = float(raw_fixed_gamma)
            except (TypeError, ValueError) as exc:
                raise ValueError("fedwolf_fixed_gamma must be convertible to float") from exc
            if self.fixed_gamma < 0.0 or self.fixed_gamma > 1.0:
                raise ValueError("fedwolf_fixed_gamma must be in [0, 1]")
        self.expert_filter_mu = {}
        self.expert_filter_P = {}
        self.last_filter_summary = {}
        self.last_gamma_summary = {}
        self.last_aggregation_weight_summary = {}

    def aggregate(self, client_updates, client_weights, global_model=None, **kwargs):
        self.last_aggregation_weight_summary = {}

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
        recorded_expert_weight_refs = set()

        aggregated_state = collections.OrderedDict()
        for key in client_updates[0].keys():
            first_value = client_updates[0][key].detach()
            if not torch.is_floating_point(first_value):
                aggregated_state[key] = first_value.cpu().clone()
                continue

            expert_ref = parse_expert_ref_from_key(key)
            if expert_ref is None:
                weights = client_weights
                total_weight = total_client_weight
                denominator = total_weight
            else:
                layer_id, expert_id = expert_ref
                raw_scores = [
                    self._get_fisher_score(client_stat, layer_id, expert_id)
                    for client_stat in client_stats
                ]
                if self.use_relative_sqrt_fisher_aggregation:
                    mean_s = self._mean_positive(raw_scores)
                    weights = [
                        self._sqrt_relative_weight(score, mean_s)
                        for score in raw_scores
                    ]
                else:
                    # fedwolf_fisher_only 保持旧消融语义：expert 直接按 raw Fisher score 聚合。
                    weights = raw_scores
                total_weight = sum(weights)
                denominator = total_weight + self.eps
                weight_ref = (str(layer_id), int(expert_id))
                if weight_ref not in recorded_expert_weight_refs:
                    self._record_expert_aggregation_weight_summary(
                        layer_id=layer_id,
                        expert_id=expert_id,
                        weights=weights,
                    )
                    recorded_expert_weight_refs.add(weight_ref)

            if total_weight <= 0:
                # 该 expert 本轮没有有效 Fisher evidence 时，保留上一轮 global expert 参数。
                if global_state is not None:
                    aggregated_state[key] = global_state[key].detach().cpu().clone()
                else:
                    aggregated_state[key] = first_value.cpu().clone()
                continue

            first_on_device = self._to_agg_device(first_value)
            theta_bar = torch.zeros_like(first_on_device)
            for update, weight in zip(client_updates, weights):
                update_tensor = self._to_agg_device(update[key])
                theta_bar += update_tensor * (weight / denominator)

            if expert_ref is None or not self.use_gamma:
                aggregated_state[key] = self._to_output_device(theta_bar)
                continue

            layer_id, expert_id = expert_ref
            gamma = self._get_gamma(layer_id, expert_id)
            theta_old = self._to_agg_device(global_state[key])
            aggregated_tensor = theta_old * (1.0 - gamma) + theta_bar * gamma
            aggregated_state[key] = self._to_output_device(aggregated_tensor)
            self._record_gamma(layer_id, expert_id, gamma, total_weight)

        if self.use_gamma:
            self._merge_gamma_summary()
        self._merge_aggregation_weight_summary()

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
            raw_scores = [
                self._get_fisher_score(client_stat, layer_id, expert_id)
                for client_stat in client_stats
            ]
            mean_s = self._mean_positive(raw_scores)
            active_tokens = [
                self._get_evidence_active_tokens(client_stat, layer_id, expert_id)
                for client_stat in client_stats
            ]
            mean_n_active = self._mean_positive(active_tokens)
            mu_current, p_current = predict_expert_state(
                self.expert_filter_mu[layer_id][expert_id].item(),
                self.expert_filter_P[layer_id][expert_id].item(),
                self.process_noise_q,
            )

            for client_stat, s in zip(client_stats, raw_scores):
                z = self._get_fisher_log_score(client_stat, layer_id, expert_id)
                if z is None:
                    round_filter_stats[layer_id]["skipped_observations"][expert_id] += 1
                    continue
                s_agg = self._sqrt_relative_weight(s, mean_s)
                n_active = self._get_evidence_active_tokens(client_stat, layer_id, expert_id)
                if n_active is None or mean_n_active <= 0.0:
                    n_rel = 1.0
                    n_reliability = 1.0
                else:
                    n_rel = max(float(n_active), 0.0) / (mean_n_active + self.eps)
                    n_reliability = math.sqrt(max(n_rel, 0.0) + self.eps)
                mu_current, p_current, update_info = wolf_scalar_update(
                    mu=mu_current,
                    p=p_current,
                    observation=z,
                    score=s,
                    sigma_e2=self.sigma_e2,
                    eps=self.eps,
                    imq_c=self.imq_c,
                    observation_reliability=n_reliability,
                )
                self._add_filter_update_info(
                    stat_buffers=round_filter_stats[layer_id],
                    expert_id=expert_id,
                    update_info=update_info,
                    score=s,
                    observation=z,
                    n_active=n_active,
                    n_rel=n_rel,
                    n_reliability=n_reliability,
                    s_agg=s_agg,
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
            "n_active_sum": torch.zeros(num_experts, dtype=torch.float32),
            "max_n_active": torch.zeros(num_experts, dtype=torch.float32),
            "n_rel_sum": torch.zeros(num_experts, dtype=torch.float32),
            "max_n_rel": torch.zeros(num_experts, dtype=torch.float32),
            "n_reliability_sum": torch.zeros(num_experts, dtype=torch.float32),
            "min_n_reliability": torch.full((num_experts,), float("inf"), dtype=torch.float32),
            "max_n_reliability": torch.zeros(num_experts, dtype=torch.float32),
            "noise_score_sum": torch.zeros(num_experts, dtype=torch.float32),
            "s_agg_sum": torch.zeros(num_experts, dtype=torch.float32),
            "max_s_agg": torch.zeros(num_experts, dtype=torch.float32),
        }

    def _safe_nonnegative_float(self, value, default=0.0):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(value):
            return default
        return max(value, 0.0)

    def _add_filter_update_info(
        self,
        stat_buffers,
        expert_id,
        update_info,
        score,
        observation,
        n_active=None,
        n_rel=None,
        n_reliability=None,
        s_agg=None,
    ):
        score = self._safe_nonnegative_float(score)
        observation = self._safe_nonnegative_float(observation)
        observation_noise = self._safe_nonnegative_float(update_info["R"])
        weight = self._safe_nonnegative_float(update_info["weight"])
        residual = float(update_info["residual"])
        if not math.isfinite(residual):
            residual = 0.0
        kalman_gain = self._safe_nonnegative_float(update_info["kalman_gain"])
        n_active = self._safe_nonnegative_float(n_active)
        n_rel = self._safe_nonnegative_float(n_rel, default=1.0)
        n_reliability = self._safe_nonnegative_float(n_reliability, default=1.0)
        noise_score = self._safe_nonnegative_float(update_info.get("noise_score", n_reliability))
        s_agg = self._safe_nonnegative_float(s_agg)

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
        stat_buffers["n_active_sum"][expert_id] += n_active
        stat_buffers["max_n_active"][expert_id] = max(float(stat_buffers["max_n_active"][expert_id]), n_active)
        stat_buffers["n_rel_sum"][expert_id] += n_rel
        stat_buffers["max_n_rel"][expert_id] = max(float(stat_buffers["max_n_rel"][expert_id]), n_rel)
        stat_buffers["n_reliability_sum"][expert_id] += n_reliability
        stat_buffers["min_n_reliability"][expert_id] = min(
            float(stat_buffers["min_n_reliability"][expert_id]),
            n_reliability,
        )
        stat_buffers["max_n_reliability"][expert_id] = max(
            float(stat_buffers["max_n_reliability"][expert_id]),
            n_reliability,
        )
        stat_buffers["noise_score_sum"][expert_id] += noise_score
        stat_buffers["s_agg_sum"][expert_id] += s_agg
        stat_buffers["max_s_agg"][expert_id] = max(float(stat_buffers["max_s_agg"][expert_id]), s_agg)

    def _safe_mean(self, total, count):
        return torch.where(count > 0, total / count.clamp_min(1.0), torch.zeros_like(total))

    def _mean_positive(self, values):
        positives = []
        for value in values:
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0.0:
                positives.append(value)
        if not positives:
            return 0.0
        return sum(positives) / len(positives)

    def _sqrt_relative_weight(self, raw_score, positive_mean):
        try:
            raw_score = float(raw_score)
            positive_mean = float(positive_mean)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(raw_score) or not math.isfinite(positive_mean):
            return 0.0
        if raw_score <= 0.0 or positive_mean <= 0.0:
            return 0.0
        relative_score = raw_score / (positive_mean + self.eps)
        return math.sqrt(max(relative_score, 0.0) + self.eps)

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
            mean_n_active = self._safe_mean(stats["n_active_sum"], count)
            mean_n_rel = self._safe_mean(stats["n_rel_sum"], count)
            mean_n_reliability = self._safe_mean(stats["n_reliability_sum"], count)
            mean_noise_score = self._safe_mean(stats["noise_score_sum"], count)
            mean_s_agg = self._safe_mean(stats["s_agg_sum"], count)
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
            min_n_reliability = torch.where(
                count > 0,
                stats["min_n_reliability"],
                torch.zeros_like(stats["min_n_reliability"]),
            )
            summary[layer_id] = {
                "total_fisher_weight": format_scientific_list(stats["score_sum"].tolist()),
                "mean_s": format_scientific_list(mean_s.tolist()),
                "max_s": format_scientific_list(stats["max_score"].tolist()),
                "mean_s_agg": format_scientific_list(mean_s_agg.tolist()),
                "max_s_agg": format_scientific_list(stats["max_s_agg"].tolist()),
                "total_s_agg_weight": format_scientific_list(stats["s_agg_sum"].tolist()),
                "mean_z": format_scientific_list(mean_z.tolist()),
                "max_z": format_scientific_list(stats["max_observation"].tolist()),
                "mean_n_active": format_scientific_list(mean_n_active.tolist()),
                "max_n_active": format_scientific_list(stats["max_n_active"].tolist()),
                "mean_n_rel": format_scientific_list(mean_n_rel.tolist()),
                "max_n_rel": format_scientific_list(stats["max_n_rel"].tolist()),
                "mean_n_reliability": format_scientific_list(mean_n_reliability.tolist()),
                "min_n_reliability": format_scientific_list(min_n_reliability.tolist()),
                "max_n_reliability": format_scientific_list(stats["max_n_reliability"].tolist()),
                "mean_noise_score": format_scientific_list(mean_noise_score.tolist()),
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

    def _compute_gamma_from_mu(self, mu):
        if self.gamma_mode == "fixed":
            return self.fixed_gamma
        base_gamma = stable_sigmoid(float(mu) / self.gamma_temperature)
        return self.gamma_min + (self.gamma_max - self.gamma_min) * base_gamma

    def _get_gamma(self, layer_id, expert_id):
        layer_id = str(layer_id)
        if layer_id not in self.expert_filter_mu or expert_id >= self.expert_filter_mu[layer_id].numel():
            return self._compute_gamma_from_mu(0.0)
        return self._compute_gamma_from_mu(self.expert_filter_mu[layer_id][expert_id].item())

    def _record_gamma(self, layer_id, expert_id, gamma, total_fisher_weight):
        layer_id = str(layer_id)
        layer_summary = self.last_gamma_summary.setdefault(
            layer_id,
            {"gamma": []},
        )
        while len(layer_summary["gamma"]) <= expert_id:
            layer_summary["gamma"].append("0.000000000000e+00")
        layer_summary["gamma"][expert_id] = f"{float(gamma):.12e}"

    def _get_aggregation_weight_mode(self):
        if self.use_relative_sqrt_fisher_aggregation:
            return "sqrt_relative_fisher"
        return "raw_fisher"

    def _ensure_summary_slot(self, values, expert_id, default):
        while len(values) <= expert_id:
            values.append(default)

    def _record_expert_aggregation_weight_summary(self, layer_id, expert_id, weights):
        layer_id = str(layer_id)
        clean_weights = []
        for weight in weights:
            try:
                weight = float(weight)
            except (TypeError, ValueError):
                continue
            if math.isfinite(weight):
                clean_weights.append(max(weight, 0.0))

        total_weight = sum(clean_weights)
        count = len(clean_weights)
        mean_weight = total_weight / count if count > 0 else 0.0
        min_weight = min(clean_weights) if clean_weights else 0.0
        max_weight = max(clean_weights) if clean_weights else 0.0
        positive_count = sum(1 for weight in clean_weights if weight > 0.0)

        layer_summary = self.last_aggregation_weight_summary.setdefault(
            layer_id,
            {
                "aggregation_weight_mode": self._get_aggregation_weight_mode(),
                "total_s_agg_weight": [],
                "mean_s_agg": [],
                "min_s_agg": [],
                "max_s_agg": [],
                "positive_s_agg_clients": [],
            },
        )
        layer_summary["aggregation_weight_mode"] = self._get_aggregation_weight_mode()
        for key in ["total_s_agg_weight", "mean_s_agg", "min_s_agg", "max_s_agg"]:
            self._ensure_summary_slot(layer_summary[key], expert_id, "0.000000000000e+00")
        self._ensure_summary_slot(layer_summary["positive_s_agg_clients"], expert_id, 0)

        layer_summary["total_s_agg_weight"][expert_id] = f"{total_weight:.12e}"
        layer_summary["mean_s_agg"][expert_id] = f"{mean_weight:.12e}"
        layer_summary["min_s_agg"][expert_id] = f"{min_weight:.12e}"
        layer_summary["max_s_agg"][expert_id] = f"{max_weight:.12e}"
        layer_summary["positive_s_agg_clients"][expert_id] = int(positive_count)

    def _merge_aggregation_weight_summary(self):
        for layer_id, aggregation_summary in self.last_aggregation_weight_summary.items():
            self.last_filter_summary.setdefault(layer_id, {}).update(aggregation_summary)

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
                    self._compute_gamma_from_mu(value)
                    for value in self.expert_filter_mu[layer_id].tolist()
                )
                layer_summary["gamma_temperature"] = f"{self.gamma_temperature:.12e}"
                layer_summary["gamma_mode"] = self.gamma_mode
                layer_summary["fixed_gamma"] = (
                    "None" if self.fixed_gamma is None else f"{float(self.fixed_gamma):.12e}"
                )
                layer_summary["gamma_min"] = f"{self.gamma_min:.12e}"
                layer_summary["gamma_max"] = f"{self.gamma_max:.12e}"

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

    def _get_evidence_active_tokens(self, client_stats, layer_id, expert_id):
        if not isinstance(client_stats, dict):
            return None

        value = self._get_layer_expert_value(
            client_stats=client_stats,
            field_name="evidence_expert_activations_by_layer",
            layer_id=layer_id,
            expert_id=expert_id,
        )
        if value is not None:
            return max(value, 0.0)

        stats_by_layer = client_stats.get("evidence_expert_stats_by_layer", {})
        if not isinstance(stats_by_layer, dict):
            return None

        layer_stats = self._get_value_by_id(stats_by_layer, layer_id)
        if layer_stats is None:
            layer_stats = {}
        if not isinstance(layer_stats, dict):
            return None

        value = self._get_expert_value_from_sequence(
            layer_stats.get("expert_activations"),
            expert_id,
        )
        if value is None:
            return None
        return max(value, 0.0)

    def _get_layer_expert_value(self, client_stats, field_name, layer_id, expert_id):
        if not isinstance(client_stats, dict):
            return None

        value_by_layer = client_stats.get(field_name, {})
        if not isinstance(value_by_layer, dict):
            return None

        layer_values = self._get_value_by_id(value_by_layer, layer_id)
        if layer_values is None:
            return None

        return self._get_expert_value_from_sequence(layer_values, expert_id)

    def _get_value_by_id(self, mapping, key):
        if not isinstance(mapping, dict):
            return None

        candidates = [key, str(key)]
        try:
            candidates.append(int(key))
        except (TypeError, ValueError):
            pass

        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate in mapping:
                return mapping[candidate]
        return None

    def _get_expert_value_from_sequence(self, values, expert_id):
        if values is None:
            return None
        try:
            expert_id = int(expert_id)
        except (TypeError, ValueError):
            return None
        if expert_id < 0:
            return None

        if isinstance(values, dict):
            value = self._get_value_by_id(values, expert_id)
            if value is None:
                return None
        elif torch.is_tensor(values):
            flat_values = values.detach().cpu().flatten()
            if expert_id >= flat_values.numel():
                return None
            value = flat_values[expert_id].item()
        else:
            try:
                if expert_id >= len(values):
                    return None
                value = values[expert_id]
            except (TypeError, KeyError, IndexError):
                return None
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
        return ExpertFedAvgAggregator(args)
    if args.agg_method == "fedavg":
        return FedAvgAggregator(args)
    raise ValueError(f"Unknown aggregation method: {args.agg_method}")
