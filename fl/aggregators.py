import collections
import math
from abc import ABC, abstractmethod

import torch

from fl.expert_filter_state import (
    batch_update_filter_state,
    compute_filter_precision,
    compute_imq_weight,
    compute_standardized_residual,
    compute_support_noise,
    predict_expert_state,
    safe_float,
)


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

    def _parse_expert_ref_from_key(self, key):
        expert_ref = parse_expert_ref_from_key(key)
        if expert_ref is None:
            return None
        layer_id, expert_id = expert_ref
        return int(layer_id), int(expert_id)

    def _collect_expert_param_keys_by_ref(self, state_keys):
        expert_param_keys_by_ref = {}
        for key in state_keys:
            expert_ref = self._parse_expert_ref_from_key(key)
            if expert_ref is None:
                continue
            expert_param_keys_by_ref.setdefault(expert_ref, []).append(key)
        return expert_param_keys_by_ref

    def _aggregate_fedavg_param(self, key, first_value, client_updates, client_weights, total_weight):
        first_on_device = self._to_agg_device(first_value)
        aggregated_tensor = torch.zeros_like(first_on_device)
        for update, weight in zip(client_updates, client_weights):
            aggregated_tensor += self._to_agg_device(update[key]) * (weight / total_weight)
        return self._to_output_device(aggregated_tensor)

    def _aggregate_fisher_only_expert_param(
        self,
        key,
        first_value,
        expert_ref,
        client_updates,
        client_stats,
        global_state,
    ):
        layer_id, expert_id = expert_ref
        weights = [
            self._nonnegative_or_zero(self._get_fisher_score(client_stat, layer_id, expert_id))
            for client_stat in client_stats
        ]
        total_weight = sum(weights)
        if total_weight <= 0.0:
            if global_state is not None and key in global_state:
                return global_state[key].detach().cpu().clone()
            return first_value.cpu().clone()

        first_on_device = self._to_agg_device(first_value)
        aggregated_tensor = torch.zeros_like(first_on_device)
        denominator = total_weight + self.eps
        for update, weight in zip(client_updates, weights):
            if weight <= 0.0 or key not in update:
                continue
            aggregated_tensor += self._to_agg_device(update[key]) * (weight / denominator)
        return self._to_output_device(aggregated_tensor)

    def _aggregate_precision_fusion_expert_param(
        self,
        key,
        first_value,
        expert_ref,
        client_updates,
        global_state,
        precision_cache,
    ):
        if global_state is None or key not in global_state:
            return first_value.cpu().clone()

        cache = precision_cache.get(expert_ref)
        if cache is None or sum(cache["lambda_clients"]) <= 0.0:
            return global_state[key].detach().cpu().clone()

        theta_old = self._to_agg_device(global_state[key])
        lambda0 = max(float(cache["lambda0"]), 0.0)
        if self.use_old_prior:
            numerator = theta_old * lambda0
            denominator = lambda0
        else:
            # No-prior mode intentionally removes the explicit old-global-expert
            # term from the final fusion. This keeps the update pace closer to
            # ExpertFedAvg. The previous global expert is still implicitly
            # preserved because clients start local training from the previous
            # global model, and explicit retention only happens as a fallback
            # when no valid client update exists. Weak evidence does not trigger
            # an extra old-prior retention term here; that is a deliberate
            # design choice, not a bug.
            numerator = torch.zeros_like(theta_old)
            denominator = 0.0
        valid_client_count = 0

        for update, lambda_client in zip(client_updates, cache["lambda_clients"]):
            lambda_client = max(float(lambda_client), 0.0)
            if lambda_client <= 0.0 or key not in update:
                continue
            numerator += self._to_agg_device(update[key]) * lambda_client
            denominator += lambda_client
            valid_client_count += 1

        if valid_client_count == 0 or denominator <= 0.0:
            return global_state[key].detach().cpu().clone()

        return self._to_output_device(numerator / (denominator + self.eps))

    def _prepare_fedwolf_precision_cache(
        self,
        expert_param_keys_by_ref,
        client_updates,
        client_stats,
        global_state,
    ):
        precision_cache = {}
        self.last_filter_summary = {}

        layer_sizes = {}
        for layer_id, expert_id in expert_param_keys_by_ref:
            layer_key = str(layer_id)
            layer_sizes[layer_key] = max(layer_sizes.get(layer_key, 0), expert_id + 1)
        if self.num_experts is not None:
            for layer_key in layer_sizes:
                layer_sizes[layer_key] = max(layer_sizes[layer_key], int(self.num_experts))

        for (layer_id, expert_id), param_keys in sorted(expert_param_keys_by_ref.items()):
            layer_key = str(layer_id)
            self._ensure_filter_state_for_layer(layer_key, layer_sizes[layer_key])
            mu_pred, p_pred = predict_expert_state(
                self.expert_filter_mu[layer_key][expert_id].item(),
                self.expert_filter_P[layer_key][expert_id].item(),
                self.process_noise_q,
                eps=self.eps,
            )
            lambda0 = 1.0 / (p_pred + self.eps)
            has_global_params = all(key in global_state for key in param_keys)

            raw_scores = []
            observations = []
            active_tokens = []
            valid = []
            for update, client_stat in zip(client_updates, client_stats):
                s_value = self._get_fisher_score(client_stat, layer_id, expert_id)
                z_value = self._get_fisher_log_score(client_stat, layer_id, expert_id)
                n_value = self._get_evidence_active_tokens(client_stat, layer_id, expert_id)
                has_client_params = self._client_has_all_expert_params(update, param_keys)
                is_valid = (
                    has_global_params
                    and has_client_params
                    and s_value is not None
                    and z_value is not None
                    and s_value >= 0.0
                    and math.isfinite(float(z_value))
                )
                raw_scores.append(s_value)
                observations.append(z_value)
                active_tokens.append(n_value)
                valid.append(bool(is_valid))

            positive_ns = [
                float(n_value)
                for n_value, is_valid in zip(active_tokens, valid)
                if is_valid and n_value is not None and math.isfinite(float(n_value)) and float(n_value) > 0.0
            ]
            mean_positive_n = sum(positive_ns) / len(positive_ns) if positive_ns else 0.0

            R_values = []
            n_rel_values = []
            support_values = []
            nu_values = []
            rho_values = []
            lambda_filters = []
            # Diagnostic only: n_rel/support_values explain the support-derived
            # observation noise R. Do not multiply them into lambda_raw again,
            # otherwise activation support would be counted twice.
            for is_valid, z_value, n_value in zip(valid, observations, active_tokens):
                if not is_valid:
                    R_values.append(0.0)
                    n_rel_values.append(0.0)
                    support_values.append(0.0)
                    nu_values.append(0.0)
                    rho_values.append(0.0)
                    lambda_filters.append(0.0)
                    continue

                R, n_rel, support_reliability = compute_support_noise(
                    n_active=n_value,
                    mean_positive_n=mean_positive_n,
                    sigma_e2=self.sigma_e2,
                    eps=self.eps,
                )
                nu = compute_standardized_residual(
                    observation=z_value,
                    mu_pred=mu_pred,
                    p_pred=p_pred,
                    R=R,
                    eps=self.eps,
                )
                rho = compute_imq_weight(
                    standardized_residual=nu,
                    imq_c=self.imq_c,
                    eps=self.eps,
                )
                lambda_filter = compute_filter_precision(rho=rho, R=R, eps=self.eps)
                R_values.append(R)
                n_rel_values.append(n_rel)
                support_values.append(support_reliability)
                nu_values.append(nu)
                rho_values.append(rho)
                lambda_filters.append(lambda_filter)

            valid_observations = [
                z_value for z_value, is_valid in zip(observations, valid) if is_valid
            ]
            valid_lambda_filters = [
                lambda_filter for lambda_filter, is_valid in zip(lambda_filters, valid) if is_valid
            ]
            mu_new, p_new = batch_update_filter_state(
                mu_pred=mu_pred,
                p_pred=p_pred,
                observations=valid_observations,
                lambda_filters=valid_lambda_filters,
                eps=self.eps,
            )
            self.expert_filter_mu[layer_key][expert_id] = mu_new
            self.expert_filter_P[layer_key][expert_id] = p_new

            positive_scores = [
                float(score)
                for score, is_valid in zip(raw_scores, valid)
                if is_valid and score is not None and math.isfinite(float(score)) and float(score) > 0.0
            ]
            mean_positive_s = sum(positive_scores) / len(positive_scores) if positive_scores else 0.0
            fisher_salience = [
                self._sqrt_relative_weight(score, mean_positive_s) if is_valid and score is not None else 0.0
                for score, is_valid in zip(raw_scores, valid)
            ]
            w_pre = [
                salience * lambda_filter if is_valid else 0.0
                for salience, lambda_filter, is_valid in zip(fisher_salience, lambda_filters, valid)
            ]
            update_consistency = self._compute_leave_one_out_update_consistency(
                param_keys=param_keys,
                client_updates=client_updates,
                global_state=global_state,
                valid=valid,
                w_pre=w_pre,
            )
            lambda_raw = [
                salience * lambda_filter * consistency if is_valid else 0.0
                for salience, lambda_filter, consistency, is_valid in zip(
                    fisher_salience,
                    lambda_filters,
                    update_consistency,
                    valid,
                )
            ]
            positive_lambda_raw = [
                value for value, is_valid in zip(lambda_raw, valid) if is_valid and value > 0.0
            ]
            mean_lambda_raw = (
                sum(positive_lambda_raw) / len(positive_lambda_raw)
                if positive_lambda_raw else 0.0
            )
            if mean_lambda_raw <= 0.0:
                lambda_clients = [0.0 for _ in lambda_raw]
            else:
                lambda_clients = [
                    self._clip_lambda(value / (mean_lambda_raw + self.eps)) if is_valid and value > 0.0 else 0.0
                    for value, is_valid in zip(lambda_raw, valid)
                ]

            sum_lambda_clients = sum(lambda_clients)
            if self.use_old_prior:
                old_prior_fraction = lambda0 / (lambda0 + sum_lambda_clients + self.eps)
            else:
                old_prior_fraction = 0.0

            precision_cache[(layer_id, expert_id)] = {
                "lambda0": float(lambda0),
                "use_old_prior": bool(self.use_old_prior),
                "old_prior_fraction": float(old_prior_fraction),
                "lambda_clients": lambda_clients,
                "lambda_filter": lambda_filters,
                "lambda_raw": lambda_raw,
                "fisher_salience": fisher_salience,
                "update_consistency": update_consistency,
                "R": R_values,
                "rho": rho_values,
                "nu": nu_values,
                "valid": valid,
                "mu_pred": float(mu_pred),
                "P_pred": float(p_pred),
                "mu_new": float(mu_new),
                "P_new": float(p_new),
                "mean_positive_s": float(mean_positive_s),
                "mean_positive_n": float(mean_positive_n),
                "n_rel_values": n_rel_values,
                "support_values": support_values,
                "mean_n_rel": self._mean_or_zero(
                    self._finite_values(n_rel_values, mask=valid)
                ),
                "min_n_rel": self._min_or_zero(
                    self._finite_values(n_rel_values, mask=valid)
                ),
                "max_n_rel": self._max_or_zero(
                    self._finite_values(n_rel_values, mask=valid)
                ),
                "mean_support_reliability": self._mean_or_zero(
                    self._finite_values(support_values, mask=valid)
                ),
                "min_support_reliability": self._min_or_zero(
                    self._finite_values(support_values, mask=valid)
                ),
                "max_support_reliability": self._max_or_zero(
                    self._finite_values(support_values, mask=valid)
                ),
            }

        return precision_cache

    def _client_has_all_expert_params(self, client_update, param_keys):
        return all(key in client_update for key in param_keys)

    def _clip_lambda(self, value):
        value = safe_float(value, default=0.0)
        if value <= 0.0:
            return 0.0
        return min(max(value, self.lambda_min), self.lambda_max)

    def _get_delta_tensor(self, update, global_state, key):
        return self._to_agg_device(update[key]) - self._to_agg_device(global_state[key])

    def _compute_leave_one_out_update_consistency(
        self,
        param_keys,
        client_updates,
        global_state,
        valid,
        w_pre,
    ):
        valid_indices = [idx for idx, is_valid in enumerate(valid) if is_valid]
        if len(valid_indices) < 2:
            return [1.0 if is_valid else 0.0 for is_valid in valid]

        weighted_sum_delta = {
            key: torch.zeros_like(self._to_agg_device(global_state[key]))
            for key in param_keys
        }
        sum_w_pre = 0.0
        for idx in valid_indices:
            weight = max(safe_float(w_pre[idx], default=0.0), 0.0)
            if weight <= 0.0:
                continue
            sum_w_pre += weight
            for key in param_keys:
                weighted_sum_delta[key] += weight * self._get_delta_tensor(
                    client_updates[idx],
                    global_state,
                    key,
                )

        consistency = []
        for idx, is_valid in enumerate(valid):
            if not is_valid:
                consistency.append(0.0)
                continue

            weight = max(safe_float(w_pre[idx], default=0.0), 0.0)
            denom_minus = sum_w_pre - weight
            if denom_minus <= self.eps:
                consistency.append(1.0)
                continue

            dot = 0.0
            norm_delta_sq = 0.0
            norm_center_sq = 0.0
            for key in param_keys:
                delta = self._get_delta_tensor(client_updates[idx], global_state, key)
                center = (weighted_sum_delta[key] - weight * delta) / (denom_minus + self.eps)
                dot += float(torch.sum(delta * center).detach().cpu().item())
                norm_delta_sq += float(torch.sum(delta * delta).detach().cpu().item())
                norm_center_sq += float(torch.sum(center * center).detach().cpu().item())

            norm_delta = math.sqrt(max(norm_delta_sq, 0.0))
            norm_center = math.sqrt(max(norm_center_sq, 0.0))
            if norm_delta <= self.eps or norm_center <= self.eps:
                consistency.append(1.0)
                continue

            cosine = dot / (norm_delta * norm_center + self.eps)
            cosine = min(max(cosine, -1.0), 1.0)
            consistency.append(
                self.consistency_min
                + (1.0 - self.consistency_min) * max(0.0, cosine)
            )

        return consistency

    def _finite_values(self, values, mask=None, positive_only=False):
        finite_values = []
        for idx, value in enumerate(values or []):
            if mask is not None and not mask[idx]:
                continue
            value = safe_float(value, default=None)
            if value is None:
                continue
            if positive_only and value <= 0.0:
                continue
            finite_values.append(value)
        return finite_values

    def _mean_or_zero(self, values):
        return sum(values) / len(values) if values else 0.0

    def _min_or_zero(self, values):
        return min(values) if values else 0.0

    def _max_or_zero(self, values):
        return max(values) if values else 0.0

    def _build_filter_summary_from_precision_cache(self, precision_cache):
        summary = {
            "num_experts": 0,
            "num_valid_experts": 0,
            "aggregation_weight_mode": self._get_aggregation_weight_mode(),
            "use_old_prior": bool(self.use_old_prior),
            "lambda0": 0.0,
            "mean_old_prior_fraction": 0.0,
            "min_old_prior_fraction": 0.0,
            "max_old_prior_fraction": 0.0,
            "mean_lambda_filter": 0.0,
            "min_lambda_filter": 0.0,
            "max_lambda_filter": 0.0,
            "mean_lambda_raw": 0.0,
            "min_lambda_raw": 0.0,
            "max_lambda_raw": 0.0,
            "mean_lambda_final": 0.0,
            "min_lambda_final": 0.0,
            "max_lambda_final": 0.0,
            "mean_R": 0.0,
            "min_R": 0.0,
            "max_R": 0.0,
            "mean_n_rel": 0.0,
            "min_n_rel": 0.0,
            "max_n_rel": 0.0,
            "mean_support_reliability": 0.0,
            "min_support_reliability": 0.0,
            "max_support_reliability": 0.0,
            "mean_rho": 0.0,
            "min_rho": 0.0,
            "max_rho": 0.0,
            "mean_std_residual": 0.0,
            "mean_abs_standardized_residual": 0.0,
            "mean_fisher_salience": 0.0,
            "mean_update_consistency": 0.0,
            "mean_mu": 0.0,
            "mean_P": 0.0,
            "skipped_observations": 0,
        }

        if not precision_cache:
            return summary

        lambda0_values = []
        lambda_filter_values = []
        lambda_raw_values = []
        lambda_final_values = []
        old_prior_fraction_values = []
        R_values = []
        rho_values = []
        std_residual_values = []
        abs_std_residual_values = []
        n_rel_values = []
        support_reliability_values = []
        fisher_salience_values = []
        update_consistency_values = []
        mu_values = []
        P_values = []
        skipped_observations = 0

        for cache in precision_cache.values():
            summary["num_experts"] += 1
            valid = [bool(flag) for flag in cache.get("valid", [])]
            if any(valid):
                summary["num_valid_experts"] += 1

            lambda0 = safe_float(cache.get("lambda0"), default=None)
            if lambda0 is not None:
                lambda0_values.append(lambda0)

            lambda_filter_values.extend(self._finite_values(cache.get("lambda_filter"), mask=valid))
            lambda_raw_values.extend(self._finite_values(cache.get("lambda_raw"), mask=valid))
            lambda_final_values.extend(self._finite_values(cache.get("lambda_clients"), mask=valid))
            old_prior_fraction = safe_float(cache.get("old_prior_fraction"), default=None)
            if old_prior_fraction is not None:
                old_prior_fraction_values.append(old_prior_fraction)
            R_values.extend(self._finite_values(cache.get("R"), mask=valid))
            n_rel_values.extend(self._finite_values(cache.get("n_rel_values"), mask=valid))
            support_reliability_values.extend(
                self._finite_values(cache.get("support_values"), mask=valid)
            )
            rho_values.extend(self._finite_values(cache.get("rho"), mask=valid))
            std_residual_values.extend(self._finite_values(cache.get("nu"), mask=valid))
            abs_std_residual_values.extend(
                abs(value) for value in self._finite_values(cache.get("nu"), mask=valid)
            )
            fisher_salience_values.extend(
                self._finite_values(cache.get("fisher_salience"), mask=valid)
            )
            update_consistency_values.extend(
                self._finite_values(cache.get("update_consistency"), mask=valid)
            )

            mu_new = safe_float(cache.get("mu_new"), default=None)
            if mu_new is not None:
                mu_values.append(mu_new)
            p_new = safe_float(cache.get("P_new"), default=None)
            if p_new is not None:
                P_values.append(p_new)

            skipped_observations += int(len(valid) - sum(valid))

        summary["lambda0"] = self._mean_or_zero(lambda0_values)
        summary["use_old_prior"] = bool(self.use_old_prior)
        summary["mean_old_prior_fraction"] = self._mean_or_zero(old_prior_fraction_values)
        summary["min_old_prior_fraction"] = self._min_or_zero(old_prior_fraction_values)
        summary["max_old_prior_fraction"] = self._max_or_zero(old_prior_fraction_values)
        summary["mean_lambda_filter"] = self._mean_or_zero(lambda_filter_values)
        summary["min_lambda_filter"] = self._min_or_zero(lambda_filter_values)
        summary["max_lambda_filter"] = self._max_or_zero(lambda_filter_values)
        summary["mean_lambda_raw"] = self._mean_or_zero(lambda_raw_values)
        summary["min_lambda_raw"] = self._min_or_zero(lambda_raw_values)
        summary["max_lambda_raw"] = self._max_or_zero(lambda_raw_values)
        summary["mean_lambda_final"] = self._mean_or_zero(lambda_final_values)
        summary["min_lambda_final"] = self._min_or_zero(lambda_final_values)
        summary["max_lambda_final"] = self._max_or_zero(lambda_final_values)
        summary["mean_R"] = self._mean_or_zero(R_values)
        summary["min_R"] = self._min_or_zero(R_values)
        summary["max_R"] = self._max_or_zero(R_values)
        summary["mean_n_rel"] = self._mean_or_zero(n_rel_values)
        summary["min_n_rel"] = self._min_or_zero(n_rel_values)
        summary["max_n_rel"] = self._max_or_zero(n_rel_values)
        summary["mean_support_reliability"] = self._mean_or_zero(support_reliability_values)
        summary["min_support_reliability"] = self._min_or_zero(support_reliability_values)
        summary["max_support_reliability"] = self._max_or_zero(support_reliability_values)
        summary["mean_rho"] = self._mean_or_zero(rho_values)
        summary["min_rho"] = self._min_or_zero(rho_values)
        summary["max_rho"] = self._max_or_zero(rho_values)
        summary["mean_std_residual"] = self._mean_or_zero(std_residual_values)
        summary["mean_abs_standardized_residual"] = self._mean_or_zero(abs_std_residual_values)
        summary["mean_fisher_salience"] = self._mean_or_zero(fisher_salience_values)
        summary["mean_update_consistency"] = self._mean_or_zero(update_consistency_values)
        summary["mean_mu"] = self._mean_or_zero(mu_values)
        summary["mean_P"] = self._mean_or_zero(P_values)
        summary["skipped_observations"] = skipped_observations
        return summary


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


class FedWoLFAggregator(FedAvgAggregator):
    # FedWoLF 聚合器：
    # - shared / router / classifier 等非 expert 参数仍按客户端样本数做 FedAvg；
    # - fedwolf_fisher_only 的 expert 参数按 raw Fisher score 做 Fisher-only baseline；
    # - fedwolf 的 expert 参数使用 client-expert precision fusion：
    #   Fisher salience × filter reliability × leave-one-out update consistency，
    #   再 normalize + clip 后用于 expert 聚合。
    # - old global expert prior 是可选项，由 fedwolf_use_old_prior 控制：
    #   true 时显式加入 lambda0 prior，false 时不显式加入旧 prior；
    #   false 时旧 expert 仍通过客户端本地训练初始化被隐式保留，
    #   且在没有有效客户端更新时仍 fallback 保留旧 global expert。
    #
    # 注意：
    # - fedwolf 不再使用旧的插值路径。
    # - Fisher salience 只是最终 client-expert precision 的一个因子。
    # - filter 输出 lambda_filter。
    def __init__(self, args=None, use_wolf_filter=True):
        super().__init__(args)
        self.eps = float(getattr(args, "fedwolf_eps", 1e-8))
        self.process_noise_q = float(getattr(args, "fedwolf_process_noise_q", 0.01))
        self.sigma_e2 = float(getattr(args, "fedwolf_sigma_e2", 1.0))
        self.imq_c = float(getattr(args, "fedwolf_imq_c", 1.0))
        self.consistency_min = float(getattr(args, "fedwolf_consistency_min", 0.05))
        self.lambda_min = float(getattr(args, "fedwolf_lambda_min", 0.05))
        self.lambda_max = float(getattr(args, "fedwolf_lambda_max", 5.0))
        self.use_old_prior = bool(getattr(args, "fedwolf_use_old_prior", False))
        self.num_experts = getattr(args, "num_experts", None)
        self.use_wolf_filter = use_wolf_filter
        if self.imq_c <= 0:
            raise ValueError("fedwolf_imq_c must be positive")
        if self.sigma_e2 <= 0:
            raise ValueError("fedwolf_sigma_e2 must be positive")
        if self.process_noise_q < 0:
            raise ValueError("fedwolf_process_noise_q must be non-negative")
        if self.consistency_min < 0.0 or self.consistency_min >= 1.0:
            raise ValueError("fedwolf_consistency_min must satisfy 0 <= value < 1")
        if self.lambda_min <= 0.0:
            raise ValueError("fedwolf_lambda_min must be positive")
        if self.lambda_max < self.lambda_min:
            raise ValueError("fedwolf_lambda_max must be >= fedwolf_lambda_min")
        self.expert_filter_mu = {}
        self.expert_filter_P = {}
        self.last_filter_summary = {}

    def aggregate(self, client_updates, client_weights, global_model=None, **kwargs):
        if len(client_updates) == 0:
            raise ValueError("FedWoLF requires at least one client update")
        if len(client_updates) != len(client_weights):
            raise ValueError("client_updates and client_weights must have the same length")

        client_stats = kwargs.get("client_stats")
        if client_stats is None:
            client_stats = kwargs.get("expert_weights")
        if client_stats is None:
            raise ValueError("FedWoLF requires client_stats or expert_weights")
        if len(client_stats) != len(client_updates):
            raise ValueError("client_stats/expert_weights and client_updates must have the same length")

        global_state = global_model.state_dict() if global_model is not None else None
        total_client_weight = sum(client_weights)
        if total_client_weight <= 0:
            raise ValueError("FedWoLF requires positive total client weight")

        expert_state_keys = global_state.keys() if global_state is not None else client_updates[0].keys()
        expert_param_keys_by_ref = self._collect_expert_param_keys_by_ref(expert_state_keys)
        precision_cache = {}
        if self.use_wolf_filter:
            if global_state is None:
                raise ValueError("FedWoLF precision fusion requires global_model for expert deltas and fallback")
            precision_cache = self._prepare_fedwolf_precision_cache(
                expert_param_keys_by_ref=expert_param_keys_by_ref,
                client_updates=client_updates,
                client_stats=client_stats,
                global_state=global_state,
            )
            self.last_filter_summary = self._build_filter_summary_from_precision_cache(precision_cache)
        else:
            self.last_filter_summary = {}

        aggregated_state = collections.OrderedDict()
        for key in client_updates[0].keys():
            first_value = client_updates[0][key].detach()
            if not torch.is_floating_point(first_value):
                aggregated_state[key] = first_value.cpu().clone()
                continue

            expert_ref = self._parse_expert_ref_from_key(key)
            if expert_ref is None:
                aggregated_state[key] = self._aggregate_fedavg_param(
                    key=key,
                    first_value=first_value,
                    client_updates=client_updates,
                    client_weights=client_weights,
                    total_weight=total_client_weight,
                )
                continue

            if not self.use_wolf_filter:
                aggregated_state[key] = self._aggregate_fisher_only_expert_param(
                    key=key,
                    first_value=first_value,
                    expert_ref=expert_ref,
                    client_updates=client_updates,
                    client_stats=client_stats,
                    global_state=global_state,
                )
                continue

            aggregated_state[key] = self._aggregate_precision_fusion_expert_param(
                key=key,
                first_value=first_value,
                expert_ref=expert_ref,
                client_updates=client_updates,
                global_state=global_state,
                precision_cache=precision_cache,
            )
        return aggregated_state

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

    def _safe_nonnegative_float(self, value, default=0.0):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(value):
            return default
        return max(value, 0.0)

    def _nonnegative_or_zero(self, value):
        return self._safe_nonnegative_float(value)

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

    def _get_aggregation_weight_mode(self):
        return "precision_fusion" if self.use_wolf_filter else "raw_fisher"

    def _get_fisher_score(self, client_stats, layer_id, expert_id):
        score = self._get_layer_expert_value(
            client_stats=client_stats,
            field_name="expert_fisher_score_by_layer",
            layer_id=layer_id,
            expert_id=expert_id,
        )
        if score is None:
            return None
        return max(score, 0.0)

    def _get_fisher_log_score(self, client_stats, layer_id, expert_id):
        log_score = self._get_layer_expert_value(
            client_stats=client_stats,
            field_name="expert_fisher_log_score_by_layer",
            layer_id=layer_id,
            expert_id=expert_id,
        )
        if log_score is not None:
            return log_score

        score = self._get_fisher_score(client_stats, layer_id, expert_id)
        if score is None:
            return None
        return math.log1p(max(score, 0.0))

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
        return FedWoLFAggregator(args, use_wolf_filter=False)
    if args.agg_method == "fedwolf":
        return FedWoLFAggregator(args, use_wolf_filter=True)
    if args.agg_method == "expert_fedavg":
        return ExpertFedAvgAggregator(args)
    if args.agg_method == "fedavg":
        return FedAvgAggregator(args)
    raise ValueError(f"Unknown aggregation method: {args.agg_method}")
