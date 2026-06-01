"""State holder for the upcoming History-WoLF expert filter."""

from __future__ import annotations

import math
import statistics
from typing import Any

import torch


_QUADRANT_NAMES = (
    "current_good_history_good",
    "current_good_history_bad",
    "current_bad_history_good",
    "current_bad_history_bad",
)
_QUADRANT_VALUE_KEYS = (
    "q",
    "mu_eff",
    "direction",
    "magnitude",
    "usage_conf",
    "filter_raw",
    "final_raw",
    "mu_old",
    "mu_new",
    "p_pred",
    "p_new",
    "fisher_multiplier",
)


class HistoryWolfExpertFilter:
    """Keep History-WoLF filter configuration and checkpointable state."""

    def __init__(self, args: object, eps: float = 1e-8) -> None:
        self.eps = float(eps)

        self.min_usage = int(getattr(args, "history_wolf_min_usage", 1))
        self.c = float(getattr(args, "history_wolf_c", 2.0))
        self.process_noise = float(getattr(args, "history_wolf_process_noise", 0.01))
        self.obs_noise = float(getattr(args, "history_wolf_obs_noise", 0.05))
        self.quadrant_log = bool(getattr(args, "history_wolf_quadrant_log", True))
        self.quadrant_threshold = self._clip(
            float(getattr(args, "history_wolf_quadrant_threshold", 0.5)),
            0.0,
            1.0,
        )

        self.mu_init = 0.5
        self.p_init = 0.25
        self.p_min = 0.01
        self.p_max = 0.25
        self.direction_weight = 0.75
        self.magnitude_weight = 0.25
        self.filter_current_weight = 0.55
        self.filter_history_weight = 0.35
        self.filter_joint_weight = 0.10
        self.fisher_clip_min = 0.5
        self.fisher_clip_max = 2.0

        self.history_mu: dict[tuple[int, str, int], float] = {}
        self.history_p: dict[tuple[int, str, int], float] = {}
        self.last_summary: dict[str, Any] = {}

    def compute_weights(
        self,
        client_updates: list[dict[str, Any]],
        client_stats: list[dict[str, Any]],
        global_state: dict[str, Any],
        expert_keys_by_ref: dict[tuple[str, int], list[str]],
        use_fisher: bool = False,
    ) -> dict[tuple[str, int], list[float]]:
        """Return raw per-client expert weights for each layer/expert reference.

        weights_by_ref[(layer_id, expert_id)] = [w_client_0, w_client_1, ...]
        """

        # 基础输入校验：filter 只产出 raw weights，不在这里做归一化。
        num_clients = len(client_updates)
        if num_clients == 0:
            self._update_summary(
                num_experts=len(expert_keys_by_ref or {}),
                num_clients=0,
                use_fisher=use_fisher,
            )
            return {}
        if client_stats is None or len(client_stats) != num_clients:
            raise ValueError("client_stats must be set and match client_updates length.")
        if global_state is None:
            raise ValueError("global_state is required for History-WoLF expert filtering.")
        if not expert_keys_by_ref:
            self._update_summary(
                num_experts=0,
                num_clients=num_clients,
                use_fisher=use_fisher,
            )
            return {}

        weights_by_ref: dict[tuple[str, int], list[float]] = {}
        # summary 只统计真正参与贡献的 client-expert，避免无效样本污染均值。
        summary_values: dict[str, list[float]] = {
            "usage_conf": [],
            "q": [],
            "direction": [],
            "magnitude": [],
            "mu_old": [],
            "mu_eff": [],
            "mu_new": [],
            "p_pred": [],
            "p_new": [],
            "omega": [],
            "filter_raw": [],
            "final_raw": [],
            "fisher_multiplier": [],
        }
        quadrant_values = (
            {
                quadrant: {key: [] for key in _QUADRANT_VALUE_KEYS}
                for quadrant in _QUADRANT_NAMES
            }
            if self.quadrant_log
            else None
        )
        valid_contrib_count = 0
        skipped_zero_delta_count = 0
        fisher_all_zero_expert_count = 0

        with torch.no_grad():
            # 每个 expert_ref 独立计算一组 client raw weights。
            for expert_ref, param_keys in expert_keys_by_ref.items():
                layer_id, expert_id = expert_ref
                weights = [0.0 for _ in range(num_clients)]
                if not param_keys:
                    weights_by_ref[expert_ref] = weights
                    continue

                # usage 只用于判断观测是否可信，并不直接进入 q_i 质量分。
                usages = [
                    self._get_expert_usage(stats, layer_id, expert_id)
                    for stats in client_stats
                ]
                valid_indices = [
                    client_index
                    for client_index, usage in enumerate(usages)
                    # min_usage=1 means clients with at least one activation can contribute.
                    if usage >= self.min_usage and usage > 0
                ]
                if not valid_indices:
                    weights_by_ref[expert_ref] = weights
                    continue

                # 用相对 usage 得到置信度 u_i；高 usage 不超过 1，低 usage 会降低最终 raw weight。
                valid_usages = [usages[client_index] for client_index in valid_indices]
                median_usage = statistics.median(valid_usages)
                usage_conf_by_client = {
                    client_index: self._clip(
                        math.sqrt(usages[client_index] / (median_usage + self.eps)),
                        0.0,
                        1.0,
                    )
                    for client_index in valid_indices
                }

                # 逐参数流式计算 delta norm，不拼接大向量，降低额外内存占用。
                delta_by_client: dict[int, dict[str, Any]] = {}
                delta_norm_by_client: dict[int, float] = {}
                total_delta_by_key: dict[str, Any] = {}

                for client_index in valid_indices:
                    client_delta_by_key: dict[str, Any] = {}
                    norm_sq = 0.0
                    for key in param_keys:
                        delta = self._get_delta(
                            client_updates[client_index],
                            global_state,
                            key,
                        )
                        if delta is None:
                            continue

                        client_delta_by_key[key] = delta
                        norm_sq += float((delta * delta).sum().item())
                        if key in total_delta_by_key:
                            total_delta_by_key[key] = total_delta_by_key[key] + delta
                        else:
                            total_delta_by_key[key] = delta.clone()

                    delta_by_client[client_index] = client_delta_by_key
                    delta_norm_by_client[client_index] = math.sqrt(max(norm_sq, 0.0))

                # near-zero delta 表示该 client-expert 基本没有实际更新：权重为 0，且不更新历史。
                contributing_indices = []
                for client_index in valid_indices:
                    delta_norm = delta_norm_by_client[client_index]
                    if delta_norm <= self.eps or not self._is_finite(delta_norm):
                        weights[client_index] = 0.0
                        skipped_zero_delta_count += 1
                        continue
                    contributing_indices.append(client_index)

                if not contributing_indices:
                    weights_by_ref[expert_ref] = weights
                    continue

                # leave-one-out 方向分只使用真正有更新的贡献者，避免 no-op 稀释参考方向。
                total_delta_by_key = {}
                for client_index in contributing_indices:
                    for key, delta in delta_by_client[client_index].items():
                        if key in total_delta_by_key:
                            total_delta_by_key[key] = total_delta_by_key[key] + delta
                        else:
                            total_delta_by_key[key] = delta.clone()

                # magnitude health 只惩罚异常大的更新，参考尺度来自有效 delta 的中位数。
                valid_norms = [
                    delta_norm_by_client[client_index]
                    for client_index in contributing_indices
                    if self._is_finite(delta_norm_by_client[client_index])
                ]
                median_norm = statistics.median(valid_norms) if valid_norms else 0.0
                # fisher_history_wolf 只在 filter raw weight 之后乘 sqrt multiplier，不覆盖滤波器判断。
                fisher_scores = {
                    client_index: self._get_fisher_score(
                        client_stats[client_index],
                        layer_id,
                        expert_id,
                    )
                    for client_index in contributing_indices
                }
                fisher_multipliers = self._get_fisher_multipliers(
                    fisher_scores=fisher_scores,
                    valid_indices=contributing_indices,
                    use_fisher=use_fisher,
                )
                if use_fisher and all(fisher_scores[index] <= 0 for index in contributing_indices):
                    fisher_all_zero_expert_count += 1

                for client_index in contributing_indices:
                    usage_conf = usage_conf_by_client[client_index]
                    delta_norm = delta_norm_by_client[client_index]
                    direction = self._compute_direction_score(
                        client_delta_by_key=delta_by_client[client_index],
                        total_delta_by_key=total_delta_by_key,
                        delta_norm=delta_norm,
                        num_valid_clients=len(contributing_indices),
                    )
                    magnitude = self._compute_magnitude_health(
                        delta_norm=delta_norm,
                        median_norm=median_norm,
                    )
                    # 当前质量 q_i 由方向一致性和幅度健康度组合；usage 不参与 q_i。
                    q_value = self._clip(
                        self.direction_weight * direction
                        + self.magnitude_weight * magnitude,
                        0.0,
                        1.0,
                    )

                    # mu/P 是每个 client-layer-expert 的跨轮可靠性状态。
                    history_key = (client_index, str(layer_id), int(expert_id))
                    mu_old = self._clip(
                        self._as_finite_float(
                            self.history_mu.get(history_key, self.mu_init),
                            self.mu_init,
                        ),
                        0.0,
                        1.0,
                    )
                    p_old = self._clip(
                        self._as_finite_float(
                            self.history_p.get(history_key, self.p_init),
                            self.p_init,
                        ),
                        self.p_min,
                        self.p_max,
                    )
                    # 先使用预测态 mu_pred/p_pred 计算本轮权重，再用观测 q_i 更新历史。
                    mu_pred = mu_old
                    p_pred = self._clip(
                        p_old + max(self.process_noise, 0.0),
                        self.p_min,
                        self.p_max,
                    )
                    # usage_conf 越高，观测噪声越小；WoLF omega 会软抑制大残差。
                    obs_noise = max(self.obs_noise, self.eps) / (usage_conf + self.eps)
                    residual = (q_value - mu_pred) / math.sqrt(
                        p_pred + obs_noise + self.eps
                    )
                    wolf_c = max(abs(self.c), self.eps)
                    omega = (
                        1.0 + (residual * residual) / (wolf_c * wolf_c)
                    ) ** -0.5
                    omega2 = omega * omega
                    gain = (omega2 * p_pred) / (
                        omega2 * p_pred + obs_noise + self.eps
                    )
                    mu_new = self._clip(
                        mu_pred + gain * (q_value - mu_pred),
                        0.0,
                        1.0,
                    )
                    p_new = self._clip(
                        (1.0 - gain) * p_pred,
                        self.p_min,
                        self.p_max,
                    )

                    self.history_mu[history_key] = mu_new
                    self.history_p[history_key] = p_new

                    # P 越小代表历史越可信；P 大时 mu_eff 回到 0.5 的中性可靠性。
                    history_conf = 1.0 - self._clip(p_pred / self.p_max, 0.0, 1.0)
                    mu_eff = history_conf * mu_pred + (1.0 - history_conf) * 0.5
                    filter_raw = usage_conf * (
                        self.filter_current_weight * q_value
                        + self.filter_history_weight * mu_eff
                        + self.filter_joint_weight * q_value * mu_eff
                    )
                    if not self._is_finite(filter_raw) or filter_raw < 0:
                        filter_raw = 0.0

                    # filter-only 时 multiplier 为 1；fisher 组合模式只做温和缩放。
                    fisher_multiplier = fisher_multipliers[client_index]
                    final_raw = filter_raw * fisher_multiplier
                    if not self._is_finite(final_raw) or final_raw < 0:
                        final_raw = 0.0

                    weights[client_index] = float(final_raw)
                    valid_contrib_count += 1
                    summary_values["usage_conf"].append(usage_conf)
                    summary_values["q"].append(q_value)
                    summary_values["direction"].append(direction)
                    summary_values["magnitude"].append(magnitude)
                    summary_values["mu_old"].append(mu_old)
                    summary_values["mu_eff"].append(mu_eff)
                    summary_values["mu_new"].append(mu_new)
                    summary_values["p_pred"].append(p_pred)
                    summary_values["p_new"].append(p_new)
                    summary_values["omega"].append(omega)
                    summary_values["filter_raw"].append(filter_raw)
                    summary_values["final_raw"].append(final_raw)
                    if use_fisher:
                        summary_values["fisher_multiplier"].append(fisher_multiplier)
                    if quadrant_values is not None:
                        current_good = q_value >= self.quadrant_threshold
                        history_good = mu_eff >= self.quadrant_threshold
                        if current_good and history_good:
                            quadrant = "current_good_history_good"
                        elif current_good:
                            quadrant = "current_good_history_bad"
                        elif history_good:
                            quadrant = "current_bad_history_good"
                        else:
                            quadrant = "current_bad_history_bad"

                        values = quadrant_values[quadrant]
                        values["q"].append(float(q_value))
                        values["mu_eff"].append(float(mu_eff))
                        values["direction"].append(float(direction))
                        values["magnitude"].append(float(magnitude))
                        values["usage_conf"].append(float(usage_conf))
                        values["filter_raw"].append(float(filter_raw))
                        values["final_raw"].append(float(final_raw))
                        values["mu_old"].append(float(mu_old))
                        values["mu_new"].append(float(mu_new))
                        values["p_pred"].append(float(p_pred))
                        values["p_new"].append(float(p_new))
                        if use_fisher:
                            values["fisher_multiplier"].append(float(fisher_multiplier))

                weights_by_ref[expert_ref] = weights

        weight_dispersion_summary = self._summarize_weight_dispersion(weights_by_ref)
        self._update_summary(
            num_experts=len(expert_keys_by_ref),
            num_clients=num_clients,
            use_fisher=use_fisher,
            valid_contrib_count=valid_contrib_count,
            summary_values=summary_values,
            fisher_all_zero_expert_count=fisher_all_zero_expert_count,
            skipped_zero_delta_count=skipped_zero_delta_count,
            quadrant_values=quadrant_values,
            weight_dispersion_summary=weight_dispersion_summary,
        )
        return weights_by_ref

    def state_dict(self) -> dict[str, dict[tuple[int, str, int], float]]:
        return {
            "history_mu": self.history_mu,
            "history_p": self.history_p,
        }

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        if not state:
            return

        self.history_mu = dict(state.get("history_mu") or {})
        self.history_p = dict(state.get("history_p") or {})

    def _get_expert_usage(self, stats: dict[str, Any], layer_id: str, expert_id: int) -> float:
        # 兼容 client_stats 的新旧结构，缺失或非法值都按 0 usage 处理。
        value = None
        if isinstance(stats, dict):
            layer_stats_by_layer = stats.get("expert_stats_by_layer") or {}
            layer_stats = layer_stats_by_layer.get(str(layer_id), {})
            if isinstance(layer_stats, dict):
                value = layer_stats.get("expert_activations")
            if value is None:
                activations_by_layer = stats.get("expert_activations_by_layer") or {}
                value = activations_by_layer.get(str(layer_id))
            if value is None:
                value = stats.get("expert_activations")

        return max(self._get_indexed_value(value, expert_id), 0.0)

    def _get_fisher_score(self, stats: dict[str, Any], layer_id: str, expert_id: int) -> float:
        value = None
        if isinstance(stats, dict):
            score_by_layer = stats.get("expert_fisher_score_by_layer") or {}
            value = score_by_layer.get(str(layer_id))

        return max(self._get_indexed_value(value, expert_id), 0.0)

    def _get_indexed_value(self, values: Any, expert_id: int) -> float:
        if values is None:
            return 0.0

        try:
            if torch.is_tensor(values):
                flat_values = values.detach().cpu().flatten()
                if expert_id >= flat_values.numel():
                    return 0.0
                return self._as_finite_float(flat_values[expert_id].item(), 0.0)
            if isinstance(values, (list, tuple)):
                if expert_id >= len(values):
                    return 0.0
                return self._as_finite_float(values[expert_id], 0.0)
        except (TypeError, ValueError, RuntimeError):
            return 0.0

        return 0.0

    def _get_delta(
        self,
        client_update: dict[str, Any],
        global_state: dict[str, Any],
        key: str,
    ) -> Any | None:
        # 只比较双方都存在的浮点参数；buffer/整数状态直接跳过。
        if key not in client_update or key not in global_state:
            return None

        local_value = client_update[key]
        global_value = global_state[key]
        if not torch.is_tensor(local_value) or not torch.is_tensor(global_value):
            return None

        local_tensor = local_value.detach().cpu()
        global_tensor = global_value.detach().cpu()
        if not torch.is_floating_point(local_tensor) or not torch.is_floating_point(global_tensor):
            return None

        try:
            return local_tensor.float() - global_tensor.float()
        except RuntimeError:
            return None

    def _compute_direction_score(
        self,
        client_delta_by_key: dict[str, Any],
        total_delta_by_key: dict[str, Any],
        delta_norm: float,
        num_valid_clients: int,
    ) -> float:
        # 单个贡献者或参考方向过小时，没有可靠方向信号，返回中性分。
        if num_valid_clients < 2 or delta_norm <= self.eps:
            return 0.5

        dot = 0.0
        ref_norm_sq = 0.0
        for key, delta in client_delta_by_key.items():
            ref_delta = total_delta_by_key[key] - delta
            dot += float((delta * ref_delta).sum().item())
            ref_norm_sq += float((ref_delta * ref_delta).sum().item())

        ref_norm = math.sqrt(max(ref_norm_sq, 0.0))
        if ref_norm <= self.eps:
            return 0.5

        cosine = dot / (delta_norm * ref_norm + self.eps)
        return 0.5 * (1.0 + self._clip(cosine, -1.0, 1.0))

    def _compute_magnitude_health(self, delta_norm: float, median_norm: float) -> float:
        if median_norm <= self.eps or delta_norm <= self.eps:
            return 1.0

        ratio = delta_norm / (median_norm + self.eps)
        if ratio <= 0 or not self._is_finite(ratio):
            return 1.0

        penalty = max(0.0, math.log(ratio))
        return 1.0 / (1.0 + penalty * penalty)

    def _get_fisher_multipliers(
        self,
        fisher_scores: dict[int, float],
        valid_indices: list[int],
        use_fisher: bool,
    ) -> dict[int, float]:
        if not use_fisher:
            return {client_index: 1.0 for client_index in valid_indices}

        positive_log_scores = [
            math.log1p(score)
            for score in fisher_scores.values()
            if score > 0 and self._is_finite(score)
        ]
        if not positive_log_scores:
            return {client_index: 1.0 for client_index in valid_indices}

        median_log_score = statistics.median(positive_log_scores)
        if median_log_score <= self.eps:
            return {client_index: 1.0 for client_index in valid_indices}

        multipliers = {}
        for client_index in valid_indices:
            log_score = math.log1p(max(fisher_scores[client_index], 0.0))
            score_ratio = self._clip(
                log_score / (median_log_score + self.eps),
                self.fisher_clip_min,
                self.fisher_clip_max,
            )
            multipliers[client_index] = math.sqrt(score_ratio)
        return multipliers

    def _update_summary(
        self,
        num_experts: int,
        num_clients: int,
        use_fisher: bool,
        valid_contrib_count: int = 0,
        summary_values: dict[str, list[float]] | None = None,
        fisher_all_zero_expert_count: int = 0,
        skipped_zero_delta_count: int = 0,
        quadrant_values: dict[str, dict[str, list[float]]] | None = None,
        weight_dispersion_summary: dict[str, float | int] | None = None,
    ) -> None:
        # summary 只保存 Python 标量，便于 server 日志打印和 checkpoint 外使用。
        summary_values = summary_values or {}
        self.last_summary = {
            "num_experts": int(num_experts),
            "num_clients": int(num_clients),
            "use_fisher": bool(use_fisher),
            "valid_contrib_count": int(valid_contrib_count),
            "skipped_zero_delta_count": int(skipped_zero_delta_count),
            "mean_usage_conf": self._safe_mean(summary_values.get("usage_conf", [])),
            "mean_q": self._safe_mean(summary_values.get("q", [])),
            "mean_direction": self._safe_mean(summary_values.get("direction", [])),
            "mean_magnitude": self._safe_mean(summary_values.get("magnitude", [])),
            "mean_mu_old": self._safe_mean(summary_values.get("mu_old", [])),
            "mean_mu_eff": self._safe_mean(summary_values.get("mu_eff", [])),
            "mean_mu_new": self._safe_mean(summary_values.get("mu_new", [])),
            "mean_p_pred": self._safe_mean(summary_values.get("p_pred", [])),
            "mean_p_new": self._safe_mean(summary_values.get("p_new", [])),
            "mean_omega": self._safe_mean(summary_values.get("omega", [])),
            "mean_filter_raw": self._safe_mean(summary_values.get("filter_raw", [])),
            "mean_final_raw": self._safe_mean(summary_values.get("final_raw", [])),
        }
        self.last_summary.update(
            weight_dispersion_summary or self._summarize_weight_dispersion({})
        )
        if use_fisher:
            self.last_summary.update(
                {
                    "mean_fisher_multiplier": self._safe_mean(
                        summary_values.get("fisher_multiplier", [])
                    ),
                    "fisher_all_zero_expert_count": int(fisher_all_zero_expert_count),
                }
            )
        if self.quadrant_log:
            quadrants = {}
            for quadrant in _QUADRANT_NAMES:
                values = (quadrant_values or {}).get(quadrant, {})
                quadrant_summary = {
                    "count": int(len(values.get("q", []))),
                    "mean_q": self._safe_mean(values.get("q", [])),
                    "mean_mu_eff": self._safe_mean(values.get("mu_eff", [])),
                    "mean_direction": self._safe_mean(values.get("direction", [])),
                    "mean_magnitude": self._safe_mean(values.get("magnitude", [])),
                    "mean_usage_conf": self._safe_mean(values.get("usage_conf", [])),
                    "mean_filter_raw": self._safe_mean(values.get("filter_raw", [])),
                    "mean_final_raw": self._safe_mean(values.get("final_raw", [])),
                    "mean_mu_old": self._safe_mean(values.get("mu_old", [])),
                    "mean_mu_new": self._safe_mean(values.get("mu_new", [])),
                    "mean_p_pred": self._safe_mean(values.get("p_pred", [])),
                    "mean_p_new": self._safe_mean(values.get("p_new", [])),
                }
                if use_fisher:
                    quadrant_summary["mean_fisher_multiplier"] = self._safe_mean(
                        values.get("fisher_multiplier", [])
                    )
                quadrants[quadrant] = quadrant_summary
            self.last_summary["quadrants"] = quadrants

    def _summarize_weight_dispersion(
        self,
        weights_by_ref: dict[tuple[str, int], list[float]],
    ) -> dict[str, float | int]:
        # 每个 expert_ref 只用正 raw weight 诊断；全零 ref 单独计数，不混入均值。
        nonzero_client_counts = []
        top1_shares = []
        ess_ratios = []
        raw_weight_cvs = []
        all_zero_ref_count = 0

        for weights in weights_by_ref.values():
            positive_weights = [
                float(weight)
                for weight in weights
                if self._is_finite(weight) and float(weight) > 0
            ]
            weight_sum = sum(positive_weights)
            if weight_sum <= 0:
                all_zero_ref_count += 1
                continue

            probabilities = [weight / weight_sum for weight in positive_weights]
            nonzero_client_count = len(positive_weights)
            ess = 1.0 / sum(probability * probability for probability in probabilities)
            mean_raw_weight = weight_sum / nonzero_client_count

            nonzero_client_counts.append(float(nonzero_client_count))
            top1_shares.append(float(max(probabilities)))
            ess_ratios.append(float(ess / nonzero_client_count))
            raw_weight_cvs.append(
                float(statistics.pstdev(positive_weights) / mean_raw_weight)
            )

        return {
            "weight_ref_count": int(len(weights_by_ref)),
            "weight_all_zero_ref_count": int(all_zero_ref_count),
            "mean_nonzero_clients_per_ref": self._safe_mean(nonzero_client_counts),
            "mean_top1_share": self._safe_mean(top1_shares),
            "mean_ess_ratio": self._safe_mean(ess_ratios),
            "mean_raw_weight_cv": self._safe_mean(raw_weight_cvs),
        }

    def _safe_mean(self, values: list[float]) -> float:
        finite_values = [float(value) for value in values if self._is_finite(value)]
        if not finite_values:
            return 0.0
        return float(sum(finite_values) / len(finite_values))

    def _as_finite_float(self, value: Any, default: float) -> float:
        try:
            float_value = float(value)
        except (TypeError, ValueError):
            return default
        if not self._is_finite(float_value):
            return default
        return float_value

    def _clip(self, value: float, lower: float, upper: float) -> float:
        if not self._is_finite(value):
            return lower
        return min(max(float(value), lower), upper)

    def _is_finite(self, value: Any) -> bool:
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False
