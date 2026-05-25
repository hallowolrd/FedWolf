import collections
import math
from abc import ABC, abstractmethod

import torch

FEDWOLF_FUSION_MODE_ROBUST_UPDATE_FUSION = "robust_update_fusion"
FEDWOLF_UPDATE_FUSION_VARIANT_UNIFORM_UPDATE = "uniform_update"
FEDWOLF_UPDATE_FUSION_VARIANT_FISHER_ONLY = "fisher_only"
FEDWOLF_UPDATE_FUSION_VARIANT_ROBUST_ONLY = "robust_only"
FEDWOLF_UPDATE_FUSION_VARIANT_FISHER_WOLF = "fisher_wolf"
FEDWOLF_UPDATE_FUSION_VARIANT_FISHER_HISTORY_WOLF = "fisher_history_wolf"
FEDWOLF_UPDATE_FUSION_VARIANTS = (
    FEDWOLF_UPDATE_FUSION_VARIANT_UNIFORM_UPDATE,
    FEDWOLF_UPDATE_FUSION_VARIANT_FISHER_ONLY,
    FEDWOLF_UPDATE_FUSION_VARIANT_ROBUST_ONLY,
    FEDWOLF_UPDATE_FUSION_VARIANT_FISHER_WOLF,
    FEDWOLF_UPDATE_FUSION_VARIANT_FISHER_HISTORY_WOLF,
)
FEDWOLF_FISHER_PRECISION_GRANULARITY_BLOCK = "block"
FEDWOLF_FISHER_PRECISION_GRANULARITY_EXPERT = "expert"
FEDWOLF_FISHER_PRECISION_GRANULARITIES = (
    FEDWOLF_FISHER_PRECISION_GRANULARITY_BLOCK,
    FEDWOLF_FISHER_PRECISION_GRANULARITY_EXPERT,
)
FEDWOLF_FISHER_WEIGHT_SOURCE_PRECISION = "precision"
FEDWOLF_FISHER_WEIGHT_SOURCE_LEGACY_SCORE = "legacy_score"
FEDWOLF_FISHER_WEIGHT_SOURCES = (
    FEDWOLF_FISHER_WEIGHT_SOURCE_PRECISION,
    FEDWOLF_FISHER_WEIGHT_SOURCE_LEGACY_SCORE,
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


def parse_expert_block_ref_from_key(key):
    expert_ref = parse_expert_ref_from_key(key)
    if expert_ref is None:
        return None

    parts = key.split(".")
    experts_idx = parts.index("experts")
    if experts_idx + 2 >= len(parts):
        return None

    layer_id, expert_id = expert_ref
    block_name = ".".join(parts[experts_idx + 2:])
    if not block_name:
        return None
    return int(layer_id), int(expert_id), block_name


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


def _lookup_mapping_by_id(mapping, key):
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


def _as_finite_positive_scalar(value):
    if value is None:
        return None, "missing"
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None, "nonfinite"
        try:
            value = value.detach().cpu().item()
        except (RuntimeError, TypeError, ValueError):
            return None, "nonfinite"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None, "nonfinite"
    if not math.isfinite(value):
        return None, "nonfinite"
    if value <= 0.0:
        return None, "nonpositive"
    return value, None


def _get_client_block_fisher_precision(client_stat, layer_id, expert_id, block_name):
    if not isinstance(client_stat, dict):
        return None, "missing"

    precision_by_layer = client_stat.get("expert_block_fisher_precision_by_layer")
    if not isinstance(precision_by_layer, dict):
        return None, "missing"

    layer_dict = _lookup_mapping_by_id(precision_by_layer, layer_id)
    if not isinstance(layer_dict, dict):
        return None, "missing"

    expert_dict = _lookup_mapping_by_id(layer_dict, expert_id)
    if not isinstance(expert_dict, dict):
        return None, "missing"

    if block_name not in expert_dict:
        return None, "missing"

    return _as_finite_positive_scalar(expert_dict.get(block_name))


def _get_client_expert_fisher_precision(client_stat, layer_id, expert_id):
    if not isinstance(client_stat, dict):
        return None, "missing"

    precision_by_layer = client_stat.get("expert_fisher_precision_by_layer")
    if not isinstance(precision_by_layer, dict):
        return None, "missing"

    layer_dict = _lookup_mapping_by_id(precision_by_layer, layer_id)
    if not isinstance(layer_dict, dict):
        return None, "missing"

    precision = _lookup_mapping_by_id(layer_dict, expert_id)
    return _as_finite_positive_scalar(precision)


def _get_client_fisher_precision(
    client_stat,
    layer_id,
    expert_id,
    block_name,
    precision_granularity,
):
    if precision_granularity == FEDWOLF_FISHER_PRECISION_GRANULARITY_BLOCK:
        return _get_client_block_fisher_precision(
            client_stat,
            layer_id,
            expert_id,
            block_name,
        )
    if precision_granularity == FEDWOLF_FISHER_PRECISION_GRANULARITY_EXPERT:
        return _get_client_expert_fisher_precision(
            client_stat,
            layer_id,
            expert_id,
        )
    raise ValueError(
        "fedwolf_fisher_precision_granularity must be one of "
        f"{list(FEDWOLF_FISHER_PRECISION_GRANULARITIES)}, "
        f"got {precision_granularity!r}."
    )


def _get_client_block_fisher_score(client_stat, layer_id, expert_id, block_name):
    if not isinstance(client_stat, dict):
        return None, "missing"

    score_by_layer = client_stat.get("expert_block_fisher_score_by_layer")
    if not isinstance(score_by_layer, dict):
        return None, "missing"

    layer_dict = _lookup_mapping_by_id(score_by_layer, layer_id)
    if not isinstance(layer_dict, dict):
        return None, "missing"

    expert_dict = _lookup_mapping_by_id(layer_dict, expert_id)
    if not isinstance(expert_dict, dict):
        return None, "missing"

    if block_name not in expert_dict:
        return None, "missing"

    return _as_finite_positive_scalar(expert_dict.get(block_name))


def _get_client_expert_fisher_score(client_stat, layer_id, expert_id):
    if not isinstance(client_stat, dict):
        return None, "missing"

    score_by_layer = client_stat.get("expert_fisher_score_by_layer")
    if isinstance(score_by_layer, dict):
        layer_dict = _lookup_mapping_by_id(score_by_layer, layer_id)
        if isinstance(layer_dict, dict):
            score = _lookup_mapping_by_id(layer_dict, expert_id)
            score_value, score_error = _as_finite_positive_scalar(score)
            if score_error is None:
                return score_value, None
            if score_error != "missing":
                return None, score_error

    block_score_by_layer = client_stat.get("expert_block_fisher_score_by_layer")
    if not isinstance(block_score_by_layer, dict):
        return None, "missing"

    layer_dict = _lookup_mapping_by_id(block_score_by_layer, layer_id)
    if not isinstance(layer_dict, dict):
        return None, "missing"

    expert_dict = _lookup_mapping_by_id(layer_dict, expert_id)
    if not isinstance(expert_dict, dict) or not expert_dict:
        return None, "missing"

    score_sum = 0.0
    valid_count = 0
    saw_nonfinite = False
    saw_nonpositive = False
    for block_score in expert_dict.values():
        score_value, score_error = _as_finite_positive_scalar(block_score)
        if score_error is None:
            score_sum += score_value
            valid_count += 1
        elif score_error == "nonfinite":
            saw_nonfinite = True
        elif score_error == "nonpositive":
            saw_nonpositive = True

    if valid_count > 0:
        return score_sum, None
    if saw_nonfinite:
        return None, "nonfinite"
    if saw_nonpositive:
        return None, "nonpositive"
    return None, "missing"


def _get_client_fisher_weight(
    client_stat,
    layer_id,
    expert_id,
    block_name,
    precision_granularity,
    weight_source,
):
    if weight_source == FEDWOLF_FISHER_WEIGHT_SOURCE_PRECISION:
        return _get_client_fisher_precision(
            client_stat,
            layer_id,
            expert_id,
            block_name,
            precision_granularity,
        )
    if weight_source == FEDWOLF_FISHER_WEIGHT_SOURCE_LEGACY_SCORE:
        if precision_granularity == FEDWOLF_FISHER_PRECISION_GRANULARITY_BLOCK:
            return _get_client_block_fisher_score(
                client_stat,
                layer_id,
                expert_id,
                block_name,
            )
        if precision_granularity == FEDWOLF_FISHER_PRECISION_GRANULARITY_EXPERT:
            return _get_client_expert_fisher_score(
                client_stat,
                layer_id,
                expert_id,
            )
    raise ValueError(
        "fedwolf_fisher_weight_source must be one of "
        f"{list(FEDWOLF_FISHER_WEIGHT_SOURCES)}, got {weight_source!r}; "
        "fedwolf_fisher_precision_granularity must be one of "
        f"{list(FEDWOLF_FISHER_PRECISION_GRANULARITIES)}, "
        f"got {precision_granularity!r}."
    )


def _get_client_id(client_stat, fallback_idx):
    if isinstance(client_stat, dict) and "client_id" in client_stat:
        client_id = client_stat.get("client_id")
    else:
        client_id = fallback_idx

    if torch.is_tensor(client_id):
        if client_id.numel() != 1:
            return fallback_idx
        try:
            client_id = client_id.detach().cpu().item()
        except (RuntimeError, TypeError, ValueError):
            return fallback_idx

    if isinstance(client_id, float) and not math.isfinite(client_id):
        return fallback_idx

    try:
        hash(client_id)
    except TypeError:
        return fallback_idx
    return client_id


def _get_client_expert_usage_count(client_stat, layer_id, expert_id):
    def read_usage_value(usage):
        if usage is None:
            return None
        try:
            expert_index = int(expert_id)
        except (TypeError, ValueError):
            return None

        if isinstance(usage, dict):
            value = _lookup_mapping_by_id(usage, expert_id)
        elif torch.is_tensor(usage):
            try:
                flat_usage = usage.detach().cpu().reshape(-1)
            except (RuntimeError, TypeError, ValueError):
                return None
            if expert_index < 0 or expert_index >= flat_usage.numel():
                return None
            value = flat_usage[expert_index].item()
        else:
            try:
                value = usage[expert_index]
            except (IndexError, KeyError, TypeError):
                return None

        if torch.is_tensor(value):
            if value.numel() != 1:
                return None
            try:
                value = value.detach().cpu().item()
            except (RuntimeError, TypeError, ValueError):
                return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value) or value < 0.0:
            return None
        return value

    if not isinstance(client_stat, dict):
        return 0.0

    stats_by_layer = client_stat.get("expert_stats_by_layer")
    if isinstance(stats_by_layer, dict):
        layer_stats = _lookup_mapping_by_id(stats_by_layer, layer_id)
        if isinstance(layer_stats, dict):
            usage_value = read_usage_value(layer_stats.get("expert_activations"))
            if usage_value is not None:
                return float(usage_value)

    activations_by_layer = client_stat.get("expert_activations_by_layer")
    if isinstance(activations_by_layer, dict):
        layer_usage = _lookup_mapping_by_id(activations_by_layer, layer_id)
        usage_value = read_usage_value(layer_usage)
        if usage_value is not None:
            return float(usage_value)

    usage_value = read_usage_value(client_stat.get("expert_activations"))
    if usage_value is not None:
        return float(usage_value)
    return 0.0


def _get_client_expert_legacy_fisher_score(client_stat, layer_id, expert_id):
    score, score_error = _get_client_expert_fisher_score(
        client_stat,
        layer_id,
        expert_id,
    )
    if score_error is None:
        return float(score)
    return None


def _safe_median(values, eps):
    try:
        fallback = float(eps)
    except (TypeError, ValueError):
        fallback = 0.0
    if not math.isfinite(fallback):
        fallback = 0.0

    finite_values = []

    def append_value(value):
        if torch.is_tensor(value):
            try:
                flat_value = value.detach().float().cpu().reshape(-1)
            except (RuntimeError, TypeError, ValueError):
                return
            finite_mask = torch.isfinite(flat_value)
            if bool(finite_mask.any().detach().cpu().item()):
                finite_values.extend(
                    float(item.item()) for item in flat_value[finite_mask]
                )
            return
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        if math.isfinite(value):
            finite_values.append(value)

    if torch.is_tensor(values):
        append_value(values)
    else:
        try:
            iterator = iter(values)
        except TypeError:
            iterator = (values,)
        for value in iterator:
            append_value(value)

    if not finite_values:
        return fallback

    finite_values.sort()
    mid = len(finite_values) // 2
    if len(finite_values) % 2 == 1:
        return float(finite_values[mid])
    return float((finite_values[mid - 1] + finite_values[mid]) * 0.5)


def _summarize_float_values(values, prefix):
    finite_values = []

    def append_value(value):
        if torch.is_tensor(value):
            try:
                flat_value = value.detach().float().cpu().reshape(-1)
            except (RuntimeError, TypeError, ValueError):
                return
            finite_mask = torch.isfinite(flat_value)
            if bool(finite_mask.any().detach().cpu().item()):
                finite_values.extend(
                    float(item.item()) for item in flat_value[finite_mask]
                )
            return
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        if math.isfinite(value):
            finite_values.append(value)

    if torch.is_tensor(values):
        append_value(values)
    else:
        try:
            iterator = iter(values)
        except TypeError:
            iterator = (values,)
        for value in iterator:
            append_value(value)

    if not finite_values:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
        }

    mean_value = sum(finite_values) / float(len(finite_values))
    variance = sum((value - mean_value) ** 2 for value in finite_values) / float(
        len(finite_values)
    )
    return {
        f"{prefix}_mean": float(mean_value),
        f"{prefix}_std": float(math.sqrt(max(variance, 0.0))),
        f"{prefix}_min": float(min(finite_values)),
        f"{prefix}_max": float(max(finite_values)),
    }


def _compute_fisher_history_expert_weights(
    client_stats,
    layer_id,
    expert_id,
    history_state,
    history_enabled,
    history_eta,
    history_init,
    eps,
):
    if client_stats is None:
        client_stats = []
    elif isinstance(client_stats, dict):
        client_stats = [client_stats]
    else:
        try:
            client_stats = list(client_stats)
        except TypeError:
            client_stats = []

    diagnostics = {
        "usage_conf_values": [],
        "fisher_factor_values": [],
        "observation_values": [],
        "history_factor_values": [],
        "weight_values": [],
        "missing_score_count": 0,
        "nonfinite_or_nonpositive_score_count": 0,
        "zero_usage_count": 0,
        "history_good_current_good_count": 0,
        "history_bad_current_good_count": 0,
        "history_good_current_bad_count": 0,
        "history_bad_current_bad_count": 0,
    }
    pending_history_updates = {}
    num_clients = len(client_stats)
    if num_clients <= 0:
        return [], diagnostics, pending_history_updates

    if not isinstance(history_state, dict):
        history_state = {}
    history_enabled = bool(history_enabled)
    try:
        history_eta = float(history_eta)
    except (TypeError, ValueError):
        history_eta = 0.0
    if not math.isfinite(history_eta):
        history_eta = 0.0
    try:
        history_init = float(history_init)
    except (TypeError, ValueError):
        history_init = 0.5
    if not math.isfinite(history_init):
        history_init = 0.5
    history_init = min(1.0, max(0.0, history_init))
    try:
        eps = float(eps)
    except (TypeError, ValueError):
        eps = 0.0
    if not math.isfinite(eps) or eps < 0.0:
        eps = 0.0

    records = []
    positive_scores = []
    for fallback_idx, client_stat in enumerate(client_stats):
        client_id = _get_client_id(client_stat, fallback_idx)
        usage_count = _get_client_expert_usage_count(
            client_stat,
            layer_id,
            expert_id,
        )
        if usage_count <= 0.0:
            diagnostics["zero_usage_count"] += 1
        sqrt_usage = math.sqrt(max(usage_count, 0.0))

        score = _get_client_expert_legacy_fisher_score(
            client_stat,
            layer_id,
            expert_id,
        )
        if score is not None:
            positive_scores.append(score)
        else:
            _, score_error = _get_client_expert_fisher_score(
                client_stat,
                layer_id,
                expert_id,
            )
            if score_error == "missing":
                diagnostics["missing_score_count"] += 1
            else:
                diagnostics["nonfinite_or_nonpositive_score_count"] += 1

        records.append(
            {
                "client_id": client_id,
                "sqrt_usage": sqrt_usage,
                "score": score,
            }
        )

    positive_sqrt_usages = [
        record["sqrt_usage"]
        for record in records
        if record["sqrt_usage"] > eps
    ]
    if positive_sqrt_usages:
        median_sqrt_usage = _safe_median(positive_sqrt_usages, eps)
    else:
        median_sqrt_usage = 0.0
    if positive_scores:
        median_score = _safe_median(positive_scores, eps)
    else:
        median_score = eps

    unnormalized_weights = []
    for record in records:
        if median_sqrt_usage <= eps:
            usage_conf = 0.0
        else:
            usage_conf = record["sqrt_usage"] / (median_sqrt_usage + eps)
            usage_conf = min(1.0, max(0.0, usage_conf))

        if positive_scores and record["score"] is not None:
            z_value = record["score"] / (median_score + eps)
            if math.isfinite(z_value) and z_value >= 0.0:
                fisher_factor = z_value / (1.0 + z_value)
            else:
                fisher_factor = 0.5
        else:
            fisher_factor = 0.5

        observation = 0.5 + usage_conf * (fisher_factor - 0.5)
        observation = min(1.0, max(0.0, observation))

        history_key = (record["client_id"], layer_id, expert_id)
        if history_enabled:
            h_prev = history_state.get(history_key, history_init)
            if torch.is_tensor(h_prev):
                if h_prev.numel() == 1:
                    try:
                        h_prev = h_prev.detach().cpu().item()
                    except (RuntimeError, TypeError, ValueError):
                        h_prev = history_init
                else:
                    h_prev = history_init
            try:
                h_prev = float(h_prev)
            except (TypeError, ValueError):
                h_prev = history_init
            if not math.isfinite(h_prev):
                h_prev = history_init
            h_prev = min(1.0, max(0.0, h_prev))
            history_factor = 0.75 + 0.5 * h_prev
        else:
            h_prev = history_init
            history_factor = 1.0

        unnormalized_weight = usage_conf * observation * history_factor
        unnormalized_weights.append(float(unnormalized_weight))

        if history_enabled:
            h_new = h_prev + history_eta * usage_conf * (observation - h_prev)
            if not math.isfinite(h_new):
                h_new = h_prev
            pending_history_updates[history_key] = min(1.0, max(0.0, h_new))

        history_good = h_prev > 0.5
        current_good = observation > 0.5
        if history_good and current_good:
            diagnostics["history_good_current_good_count"] += 1
        elif (not history_good) and current_good:
            diagnostics["history_bad_current_good_count"] += 1
        elif history_good and (not current_good):
            diagnostics["history_good_current_bad_count"] += 1
        else:
            diagnostics["history_bad_current_bad_count"] += 1

        diagnostics["usage_conf_values"].append(float(usage_conf))
        diagnostics["fisher_factor_values"].append(float(fisher_factor))
        diagnostics["observation_values"].append(float(observation))
        diagnostics["history_factor_values"].append(float(history_factor))

    weight_sum = sum(unnormalized_weights)
    if math.isfinite(weight_sum) and weight_sum > eps:
        weights = [float(weight / weight_sum) for weight in unnormalized_weights]
    else:
        weights = [0.0 for _ in range(num_clients)]
    diagnostics["weight_values"] = weights

    return weights, diagnostics, pending_history_updates


def _tensor_is_all_finite(tensor):
    try:
        return bool(torch.isfinite(tensor).all().detach().cpu().item())
    except (RuntimeError, TypeError, ValueError):
        return False


def _tensor_values_as_floats(tensor):
    flat = tensor.detach().float().cpu().reshape(-1)
    return [float(value.item()) for value in flat]


def _compute_robust_only_center(deltas, eps, irls_steps):
    if len(deltas) == 1:
        return deltas[0], {
            "residual_values": [0.0],
            "rho2_values": [1.0],
            "weight_sum": 1.0,
            "effective_num_clients": 1.0,
            "nonfinite_rho2_count": 0,
        }

    stacked = torch.stack(deltas, dim=0)
    center = torch.median(stacked, dim=0).values
    residuals = None
    rho2 = None
    weight_sum = None
    nonfinite_rho2_count = 0

    num_steps = max(int(irls_steps), 0)
    num_weight_evals = max(num_steps, 1)
    for step_idx in range(num_weight_evals):
        residuals = torch.stack(
            [
                torch.sqrt(torch.mean((delta - center) * (delta - center)) + eps)
                for delta in deltas
            ],
            dim=0,
        )
        scale = torch.median(residuals) + eps
        u = residuals / scale
        rho2 = 1.0 / (1.0 + u * u)
        finite_rho2 = torch.isfinite(rho2)
        if not bool(finite_rho2.all().detach().cpu().item()):
            nonfinite_rho2_count += int((~finite_rho2).sum().detach().cpu().item())
            rho2 = torch.where(finite_rho2, rho2, torch.zeros_like(rho2))

        weight_sum = torch.sum(rho2)
        weight_sum_value = float(weight_sum.detach().cpu().item())
        if weight_sum_value <= eps:
            break

        if step_idx < num_steps:
            weighted_center = torch.zeros_like(center)
            for delta, weight in zip(deltas, rho2):
                weighted_center += weight.to(device=center.device, dtype=center.dtype) * delta
            center = weighted_center / (weight_sum + eps)

    if rho2 is None or weight_sum is None:
        return center, {
            "residual_values": [],
            "rho2_values": [],
            "weight_sum": 0.0,
            "effective_num_clients": 0.0,
            "nonfinite_rho2_count": int(nonfinite_rho2_count),
        }

    weight_sum_value = float(weight_sum.detach().cpu().item())
    sum_weight_sq = torch.sum(rho2 * rho2)
    sum_weight_sq_value = float(sum_weight_sq.detach().cpu().item())
    effective_num_clients = (
        (weight_sum_value * weight_sum_value) / (sum_weight_sq_value + eps)
        if sum_weight_sq_value > 0.0
        else 0.0
    )
    return center, {
        "residual_values": _tensor_values_as_floats(residuals),
        "rho2_values": _tensor_values_as_floats(rho2),
        "weight_sum": weight_sum_value,
        "effective_num_clients": float(effective_num_clients),
        "nonfinite_rho2_count": int(nonfinite_rho2_count),
    }


def _compute_fisher_wolf_center(deltas, precisions, eps, irls_steps):
    if len(deltas) != len(precisions):
        raise ValueError("fisher_wolf deltas and precisions must have the same length")
    if len(deltas) == 1:
        precision = float(precisions[0])
        return deltas[0], {
            "residual_values": [0.0],
            "whitened_residual_values": [0.0],
            "rho2_values": [1.0],
            "weight_values": [precision],
            "weight_sum": precision,
            "effective_num_clients": 1.0,
            "nonfinite_rho2_count": 0,
            "nonfinite_weight_count": 0,
        }

    stacked = torch.stack(deltas, dim=0)
    center = torch.median(stacked, dim=0).values
    precision_tensor = torch.as_tensor(
        precisions,
        device=deltas[0].device,
        dtype=deltas[0].dtype,
    )
    residuals = None
    whitened_residuals = None
    rho2 = None
    weights = None
    weight_sum = None
    nonfinite_rho2_count = 0
    nonfinite_weight_count = 0

    num_steps = max(int(irls_steps), 1)
    for _ in range(num_steps):
        residuals = torch.stack(
            [
                torch.sqrt(torch.mean((delta - center) * (delta - center)) + eps)
                for delta in deltas
            ],
            dim=0,
        )
        whitened_residuals = torch.sqrt(precision_tensor + eps) * residuals
        scale = torch.median(whitened_residuals) + eps
        u = whitened_residuals / scale
        rho2 = 1.0 / (1.0 + u * u)
        finite_rho2 = torch.isfinite(rho2)
        if not bool(finite_rho2.all().detach().cpu().item()):
            nonfinite_rho2_count += int((~finite_rho2).sum().detach().cpu().item())
            rho2 = torch.where(finite_rho2, rho2, torch.zeros_like(rho2))

        weights = precision_tensor * rho2
        finite_weights = torch.isfinite(weights)
        if not bool(finite_weights.all().detach().cpu().item()):
            nonfinite_weight_count += int((~finite_weights).sum().detach().cpu().item())
            weights = torch.where(finite_weights, weights, torch.zeros_like(weights))

        weight_sum = torch.sum(weights)
        weight_sum_value = float(weight_sum.detach().cpu().item())
        if weight_sum_value <= eps:
            break

        weighted_center = torch.zeros_like(center)
        for delta, weight in zip(deltas, weights):
            weighted_center += weight.to(device=center.device, dtype=center.dtype) * delta
        center = weighted_center / (weight_sum + eps)

    if weights is None or weight_sum is None:
        return center, {
            "residual_values": [],
            "whitened_residual_values": [],
            "rho2_values": [],
            "weight_values": [],
            "weight_sum": 0.0,
            "effective_num_clients": 0.0,
            "nonfinite_rho2_count": int(nonfinite_rho2_count),
            "nonfinite_weight_count": int(nonfinite_weight_count),
        }

    weight_sum_value = float(weight_sum.detach().cpu().item())
    sum_weight_sq = torch.sum(weights * weights)
    sum_weight_sq_value = float(sum_weight_sq.detach().cpu().item())
    effective_num_clients = (
        (weight_sum_value * weight_sum_value) / (sum_weight_sq_value + eps)
        if sum_weight_sq_value > 0.0
        else 0.0
    )
    return center, {
        "residual_values": _tensor_values_as_floats(residuals),
        "whitened_residual_values": _tensor_values_as_floats(whitened_residuals),
        "rho2_values": _tensor_values_as_floats(rho2),
        "weight_values": _tensor_values_as_floats(weights),
        "weight_sum": weight_sum_value,
        "effective_num_clients": float(effective_num_clients),
        "nonfinite_rho2_count": int(nonfinite_rho2_count),
        "nonfinite_weight_count": int(nonfinite_weight_count),
    }


def aggregate_experts_robust_update_fusion(
    *,
    args=None,
    global_model=None,
    global_state_dict=None,
    client_updates=None,
    client_weights=None,
    client_stats=None,
    aggregation_device=None,
    round_idx=None,
    history_state=None,
):
    """Run the new FedWoLF expert update-fusion path.

    `uniform_update` updates expert parameters as
    theta_old + mean_m(theta_m - theta_old). `fisher_only` uses Fisher
    precision A at the configured granularity to weight expert updates.
    `robust_only` ignores Fisher and uses residual-based rho2 weights over
    expert update tensors. `fisher_wolf` uses Fisher precision A at the same
    configured granularity with robust rho2, with final weights approximately
    A * rho2. When fedwolf_fisher_precision_granularity='block', A is read
    from `expert_block_fisher_precision_by_layer`; when it is 'expert', A is
    read from `expert_fisher_precision_by_layer`.
    """

    if history_state is None:
        history_state = {}

    variant = str(
        getattr(
            args,
            "fedwolf_update_fusion_variant",
            FEDWOLF_UPDATE_FUSION_VARIANT_UNIFORM_UPDATE,
        )
    ).strip().lower()

    if variant not in {
        FEDWOLF_UPDATE_FUSION_VARIANT_UNIFORM_UPDATE,
        FEDWOLF_UPDATE_FUSION_VARIANT_FISHER_ONLY,
        FEDWOLF_UPDATE_FUSION_VARIANT_ROBUST_ONLY,
        FEDWOLF_UPDATE_FUSION_VARIANT_FISHER_WOLF,
        FEDWOLF_UPDATE_FUSION_VARIANT_FISHER_HISTORY_WOLF,
    }:
        raise ValueError(
            f"Unknown fedwolf_update_fusion_variant: {variant}. "
            f"Expected one of {list(FEDWOLF_UPDATE_FUSION_VARIANTS)}."
        )
    if not client_updates:
        raise ValueError("robust_update_fusion requires at least one client update")
    if client_weights is None or len(client_updates) != len(client_weights):
        raise ValueError("client_updates and client_weights must have the same length")

    precision_granularity = FEDWOLF_FISHER_PRECISION_GRANULARITY_BLOCK
    fisher_weight_source = FEDWOLF_FISHER_WEIGHT_SOURCE_PRECISION
    if variant in {
        FEDWOLF_UPDATE_FUSION_VARIANT_FISHER_ONLY,
        FEDWOLF_UPDATE_FUSION_VARIANT_FISHER_WOLF,
    }:
        precision_granularity = str(
            getattr(
                args,
                "fedwolf_fisher_precision_granularity",
                FEDWOLF_FISHER_PRECISION_GRANULARITY_BLOCK,
            )
        ).strip().lower()
        if precision_granularity not in FEDWOLF_FISHER_PRECISION_GRANULARITIES:
            raise ValueError(
                "fedwolf_fisher_precision_granularity must be one of "
                f"{list(FEDWOLF_FISHER_PRECISION_GRANULARITIES)}, "
                f"got {precision_granularity!r}."
            )
        if variant == FEDWOLF_UPDATE_FUSION_VARIANT_FISHER_ONLY:
            fisher_weight_source = str(
                getattr(
                    args,
                    "fedwolf_fisher_weight_source",
                    FEDWOLF_FISHER_WEIGHT_SOURCE_PRECISION,
                )
            ).strip().lower()
            if fisher_weight_source not in FEDWOLF_FISHER_WEIGHT_SOURCES:
                raise ValueError(
                    "fedwolf_fisher_weight_source must be one of "
                    f"{list(FEDWOLF_FISHER_WEIGHT_SOURCES)}, "
                    f"got {fisher_weight_source!r}."
                )
        if client_stats is None or len(client_stats) != len(client_updates):
            raise RuntimeError(
                f"{variant} requires client_stats with one entry per client update."
            )
        if (
            variant == FEDWOLF_UPDATE_FUSION_VARIANT_FISHER_ONLY
            and fisher_weight_source == FEDWOLF_FISHER_WEIGHT_SOURCE_LEGACY_SCORE
        ):
            score_fields = (
                "expert_block_fisher_score_by_layer",
                "expert_fisher_score_by_layer",
            )
            has_score_field = any(
                isinstance(client_stat, dict)
                and any(score_field in client_stat for score_field in score_fields)
                for client_stat in client_stats
            )
            if not has_score_field:
                raise RuntimeError(
                    "fisher_only with fedwolf_fisher_weight_source="
                    + repr(FEDWOLF_FISHER_WEIGHT_SOURCE_LEGACY_SCORE)
                    + " requires expert_block_fisher_score_by_layer or "
                    "expert_fisher_score_by_layer."
                )
        else:
            precision_field = (
                "expert_block_fisher_precision_by_layer"
                if precision_granularity == FEDWOLF_FISHER_PRECISION_GRANULARITY_BLOCK
                else "expert_fisher_precision_by_layer"
            )
            has_precision_field = any(
                isinstance(client_stat, dict)
                and precision_field in client_stat
                for client_stat in client_stats
            )
            if not has_precision_field:
                raise RuntimeError(
                    f"{variant} with fedwolf_fisher_precision_granularity="
                    f"{precision_granularity!r} requires client_stats[{precision_field!r}], "
                    "but it was not found."
                )

    if variant == FEDWOLF_UPDATE_FUSION_VARIANT_FISHER_HISTORY_WOLF:
        if client_stats is None or len(client_stats) != len(client_updates):
            raise RuntimeError(
                f"{variant} requires client_stats with one entry per client update."
            )

    if global_state_dict is None:
        if global_model is None:
            raise ValueError("robust_update_fusion requires global_model or global_state_dict")
        global_state_dict = global_model.state_dict()

    total_client_weight = sum(client_weights)
    if total_client_weight <= 0:
        raise ValueError("robust_update_fusion requires positive total client weight")

    if aggregation_device is None:
        aggregation_device = resolve_aggregation_device(args)

    eps = float(getattr(args, "fedwolf_eps", 1e-8))
    eps = max(eps, 0.0)
    update_fusion_eps = float(getattr(args, "fedwolf_update_fusion_eps", 1e-12))
    update_fusion_eps = max(update_fusion_eps, 0.0)
    irls_steps = int(getattr(args, "fedwolf_irls_steps", 2))
    if irls_steps < 0:
        raise ValueError("fedwolf_irls_steps must be non-negative")
    irls_steps = max(1, irls_steps)
    history_enabled = bool(getattr(args, "fedwolf_history_filter_enabled", False))
    history_eta = float(getattr(args, "fedwolf_history_eta", 0.1))
    history_init = float(getattr(args, "fedwolf_history_init", 0.5))
    old_expert_state = {
        key: global_state_dict[key].detach().clone()
        for key in client_updates[0].keys()
        if parse_expert_ref_from_key(key) is not None and key in global_state_dict
    }
    aggregated_state = collections.OrderedDict()
    updated_expert_params = 0
    skipped_expert_params = 0
    clients_per_param = []
    delta_norms = []
    fisher_only_valid_client_contribs = 0
    fisher_only_missing_precision_count = 0
    fisher_only_nonfinite_precision_count = 0
    fisher_only_nonpositive_precision_count = 0
    fisher_only_A_values = []
    fisher_only_weight_sums = []
    robust_only_valid_client_contribs = 0
    robust_only_missing_param_count = 0
    robust_only_shape_mismatch_count = 0
    robust_only_nonfinite_param_count = 0
    robust_only_nonfinite_rho2_count = 0
    robust_only_residual_values = []
    robust_only_rho2_values = []
    robust_only_weight_sums = []
    robust_only_effective_num_clients = []
    fisher_wolf_valid_client_contribs = 0
    fisher_wolf_missing_param_count = 0
    fisher_wolf_shape_mismatch_count = 0
    fisher_wolf_nonfinite_param_count = 0
    fisher_wolf_missing_precision_count = 0
    fisher_wolf_nonfinite_precision_count = 0
    fisher_wolf_nonpositive_precision_count = 0
    fisher_wolf_nonfinite_rho2_count = 0
    fisher_wolf_nonfinite_weight_count = 0
    fisher_wolf_A_values = []
    fisher_wolf_residual_values = []
    fisher_wolf_whitened_residual_values = []
    fisher_wolf_rho2_values = []
    fisher_wolf_W_values = []
    fisher_wolf_weight_sums = []
    fisher_wolf_effective_num_clients = []
    fisher_history_weight_cache = {}
    fisher_history_pending_updates = {}
    fisher_history_valid_client_contribs = 0
    fisher_history_usage_conf_values = []
    fisher_history_fisher_factor_values = []
    fisher_history_observation_values = []
    fisher_history_factor_values = []
    fisher_history_weight_values = []
    fisher_history_effective_num_clients = []
    fisher_history_missing_score_count = 0
    fisher_history_nonfinite_or_nonpositive_score_count = 0
    fisher_history_zero_usage_count = 0
    fisher_history_good_current_good_count = 0
    fisher_history_bad_current_good_count = 0
    fisher_history_good_current_bad_count = 0
    fisher_history_bad_current_bad_count = 0

    with torch.no_grad():
        for key in client_updates[0].keys():
            first_value = client_updates[0][key].detach()
            if not torch.is_floating_point(first_value):
                aggregated_state[key] = first_value.cpu().clone()
                continue

            expert_ref = parse_expert_ref_from_key(key)
            if expert_ref is None:
                first_on_device = first_value.to(aggregation_device)
                aggregated_tensor = torch.zeros_like(first_on_device)
                for update, weight in zip(client_updates, client_weights):
                    aggregated_tensor += update[key].detach().to(aggregation_device) * (weight / total_client_weight)
                aggregated_state[key] = aggregated_tensor.detach().cpu()
                continue

            if key not in global_state_dict:
                raise ValueError(
                    "robust_update_fusion requires the old global expert "
                    f"parameter for key {key!r}."
                )

            theta_old = old_expert_state[key].detach().to(aggregation_device)
            acc_delta = torch.zeros_like(theta_old)
            valid_client_count = 0
            weight_sum = None
            expert_block_ref = None
            robust_deltas = []
            fisher_wolf_deltas = []
            fisher_wolf_precisions = []
            fisher_history_weights = None
            if variant == FEDWOLF_UPDATE_FUSION_VARIANT_FISHER_HISTORY_WOLF:
                layer_id, expert_id = expert_ref
                fisher_history_ref = (int(layer_id), int(expert_id))
                if fisher_history_ref not in fisher_history_weight_cache:
                    weights, fisher_history_diag, pending_updates = (
                        _compute_fisher_history_expert_weights(
                            client_stats=client_stats,
                            layer_id=fisher_history_ref[0],
                            expert_id=fisher_history_ref[1],
                            history_state=history_state,
                            history_enabled=history_enabled,
                            history_eta=history_eta,
                            history_init=history_init,
                            eps=eps,
                        )
                    )
                    fisher_history_weight_cache[fisher_history_ref] = weights
                    fisher_history_pending_updates.update(pending_updates)
                    fisher_history_usage_conf_values.extend(
                        fisher_history_diag.get("usage_conf_values", [])
                    )
                    fisher_history_fisher_factor_values.extend(
                        fisher_history_diag.get("fisher_factor_values", [])
                    )
                    fisher_history_observation_values.extend(
                        fisher_history_diag.get("observation_values", [])
                    )
                    fisher_history_factor_values.extend(
                        fisher_history_diag.get("history_factor_values", [])
                    )
                    fisher_history_weight_values.extend(
                        fisher_history_diag.get("weight_values", [])
                    )
                    weight_square_sum = sum(float(weight) ** 2 for weight in weights)
                    if math.isfinite(weight_square_sum) and weight_square_sum > 0.0:
                        fisher_history_effective_num_clients.append(
                            1.0 / weight_square_sum
                        )
                    fisher_history_missing_score_count += int(
                        fisher_history_diag.get("missing_score_count", 0)
                    )
                    fisher_history_nonfinite_or_nonpositive_score_count += int(
                        fisher_history_diag.get(
                            "nonfinite_or_nonpositive_score_count",
                            0,
                        )
                    )
                    fisher_history_zero_usage_count += int(
                        fisher_history_diag.get("zero_usage_count", 0)
                    )
                    fisher_history_good_current_good_count += int(
                        fisher_history_diag.get("history_good_current_good_count", 0)
                    )
                    fisher_history_bad_current_good_count += int(
                        fisher_history_diag.get("history_bad_current_good_count", 0)
                    )
                    fisher_history_good_current_bad_count += int(
                        fisher_history_diag.get("history_good_current_bad_count", 0)
                    )
                    fisher_history_bad_current_bad_count += int(
                        fisher_history_diag.get("history_bad_current_bad_count", 0)
                    )
                fisher_history_weights = fisher_history_weight_cache[fisher_history_ref]
            if variant in {
                FEDWOLF_UPDATE_FUSION_VARIANT_FISHER_ONLY,
                FEDWOLF_UPDATE_FUSION_VARIANT_FISHER_WOLF,
            }:
                expert_block_ref = parse_expert_block_ref_from_key(key)
                if expert_block_ref is None:
                    raise ValueError(
                        f"{variant} could not map expert parameter key to a Fisher "
                        f"precision block: {key!r}."
                    )
                if variant == FEDWOLF_UPDATE_FUSION_VARIANT_FISHER_ONLY:
                    weight_sum = torch.zeros((), device=theta_old.device, dtype=theta_old.dtype)

            for client_idx, update in enumerate(client_updates):
                if key not in update:
                    if variant == FEDWOLF_UPDATE_FUSION_VARIANT_ROBUST_ONLY:
                        robust_only_missing_param_count += 1
                    elif variant == FEDWOLF_UPDATE_FUSION_VARIANT_FISHER_WOLF:
                        fisher_wolf_missing_param_count += 1
                    continue
                client_param = update[key].detach()
                if not torch.is_floating_point(client_param):
                    if variant == FEDWOLF_UPDATE_FUSION_VARIANT_ROBUST_ONLY:
                        robust_only_nonfinite_param_count += 1
                    elif variant == FEDWOLF_UPDATE_FUSION_VARIANT_FISHER_WOLF:
                        fisher_wolf_nonfinite_param_count += 1
                    continue
                if variant == FEDWOLF_UPDATE_FUSION_VARIANT_ROBUST_ONLY:
                    if client_param.shape != theta_old.shape:
                        robust_only_shape_mismatch_count += 1
                        continue
                    if not _tensor_is_all_finite(client_param):
                        robust_only_nonfinite_param_count += 1
                        continue
                    client_param = client_param.to(device=theta_old.device, dtype=theta_old.dtype)
                    delta = client_param - theta_old
                    robust_deltas.append(delta)
                    valid_client_count += 1
                    robust_only_valid_client_contribs += 1
                    delta_norms.append(
                        float(torch.linalg.vector_norm(delta.detach().float()).detach().cpu().item())
                    )
                    continue
                if variant == FEDWOLF_UPDATE_FUSION_VARIANT_FISHER_WOLF:
                    if client_param.shape != theta_old.shape:
                        fisher_wolf_shape_mismatch_count += 1
                        continue
                    if not _tensor_is_all_finite(client_param):
                        fisher_wolf_nonfinite_param_count += 1
                        continue
                    layer_id, expert_id, block_name = expert_block_ref
                    precision, precision_error = _get_client_fisher_precision(
                        client_stats[client_idx],
                        layer_id,
                        expert_id,
                        block_name,
                        precision_granularity,
                    )
                    if precision_error == "missing":
                        fisher_wolf_missing_precision_count += 1
                        continue
                    if precision_error == "nonfinite":
                        fisher_wolf_nonfinite_precision_count += 1
                        continue
                    if precision_error == "nonpositive":
                        fisher_wolf_nonpositive_precision_count += 1
                        continue
                    client_param = client_param.to(device=theta_old.device, dtype=theta_old.dtype)
                    delta = client_param - theta_old
                    fisher_wolf_deltas.append(delta)
                    fisher_wolf_precisions.append(float(precision))
                    valid_client_count += 1
                    fisher_wolf_valid_client_contribs += 1
                    fisher_wolf_A_values.append(float(precision))
                    delta_norms.append(
                        float(torch.linalg.vector_norm(delta.detach().float()).detach().cpu().item())
                    )
                    continue
                if variant == FEDWOLF_UPDATE_FUSION_VARIANT_FISHER_HISTORY_WOLF:
                    if client_param.shape != theta_old.shape:
                        continue
                    if not _tensor_is_all_finite(client_param):
                        continue
                    if fisher_history_weights is None or client_idx >= len(fisher_history_weights):
                        continue
                    client_weight = float(fisher_history_weights[client_idx])
                    if not math.isfinite(client_weight) or client_weight < 0.0:
                        continue
                    client_param = client_param.to(device=theta_old.device, dtype=theta_old.dtype)
                    delta = client_param - theta_old
                    weight = torch.as_tensor(
                        client_weight,
                        device=theta_old.device,
                        dtype=theta_old.dtype,
                    )
                    acc_delta += weight * delta
                    if weight_sum is None:
                        weight_sum = torch.zeros((), device=theta_old.device, dtype=theta_old.dtype)
                    weight_sum += weight
                    valid_client_count += 1
                    fisher_history_valid_client_contribs += 1
                    delta_norms.append(
                        float(torch.linalg.vector_norm(delta.detach().float()).detach().cpu().item())
                    )
                    continue

                client_param = client_param.to(device=theta_old.device, dtype=theta_old.dtype)
                delta = client_param - theta_old

                if variant == FEDWOLF_UPDATE_FUSION_VARIANT_UNIFORM_UPDATE:
                    acc_delta += delta
                    valid_client_count += 1
                    delta_norms.append(
                        float(torch.linalg.vector_norm(delta.detach().float()).detach().cpu().item())
                    )
                    continue

                layer_id, expert_id, block_name = expert_block_ref
                fisher_weight, weight_error = _get_client_fisher_weight(
                    client_stats[client_idx],
                    layer_id,
                    expert_id,
                    block_name,
                    precision_granularity,
                    fisher_weight_source,
                )
                if weight_error == "missing":
                    fisher_only_missing_precision_count += 1
                    continue
                if weight_error == "nonfinite":
                    fisher_only_nonfinite_precision_count += 1
                    continue
                if weight_error == "nonpositive":
                    fisher_only_nonpositive_precision_count += 1
                    continue

                weight = torch.as_tensor(fisher_weight, device=theta_old.device, dtype=theta_old.dtype)
                acc_delta += weight * delta
                weight_sum += weight
                valid_client_count += 1
                fisher_only_valid_client_contribs += 1
                fisher_only_A_values.append(float(fisher_weight))
                delta_norms.append(
                    float(torch.linalg.vector_norm(delta.detach().float()).detach().cpu().item())
                )

            if valid_client_count <= 0:
                aggregated_state[key] = theta_old.detach().cpu().clone()
                skipped_expert_params += 1
                continue

            if variant == FEDWOLF_UPDATE_FUSION_VARIANT_UNIFORM_UPDATE:
                fused_delta = acc_delta / float(valid_client_count)
                clients_per_param.append(float(valid_client_count))
            elif variant == FEDWOLF_UPDATE_FUSION_VARIANT_FISHER_ONLY:
                weight_sum_value = float(weight_sum.detach().cpu().item())
                fisher_only_weight_sums.append(weight_sum_value)
                if weight_sum_value <= eps:
                    aggregated_state[key] = theta_old.detach().cpu().clone()
                    skipped_expert_params += 1
                    continue
                fused_delta = acc_delta / (weight_sum + eps)
                clients_per_param.append(float(valid_client_count))
            elif variant == FEDWOLF_UPDATE_FUSION_VARIANT_ROBUST_ONLY:
                fused_delta, robust_diag = _compute_robust_only_center(
                    robust_deltas,
                    update_fusion_eps,
                    irls_steps,
                )
                weight_sum_value = float(robust_diag["weight_sum"])
                robust_only_weight_sums.append(weight_sum_value)
                robust_only_effective_num_clients.append(
                    float(robust_diag["effective_num_clients"])
                )
                robust_only_nonfinite_rho2_count += int(robust_diag["nonfinite_rho2_count"])
                robust_only_residual_values.extend(robust_diag["residual_values"])
                robust_only_rho2_values.extend(robust_diag["rho2_values"])
                if weight_sum_value <= update_fusion_eps:
                    aggregated_state[key] = theta_old.detach().cpu().clone()
                    skipped_expert_params += 1
                    continue
                clients_per_param.append(float(valid_client_count))
            elif variant == FEDWOLF_UPDATE_FUSION_VARIANT_FISHER_HISTORY_WOLF:
                weight_sum_value = float(weight_sum.detach().cpu().item()) if weight_sum is not None else 0.0
                if weight_sum_value <= eps:
                    aggregated_state[key] = theta_old.detach().cpu().clone()
                    skipped_expert_params += 1
                    continue
                fused_delta = acc_delta / (weight_sum + eps)
                clients_per_param.append(float(valid_client_count))
            else:
                fused_delta, fisher_wolf_diag = _compute_fisher_wolf_center(
                    fisher_wolf_deltas,
                    fisher_wolf_precisions,
                    update_fusion_eps,
                    irls_steps,
                )
                weight_sum_value = float(fisher_wolf_diag["weight_sum"])
                fisher_wolf_weight_sums.append(weight_sum_value)
                fisher_wolf_effective_num_clients.append(
                    float(fisher_wolf_diag["effective_num_clients"])
                )
                fisher_wolf_nonfinite_rho2_count += int(
                    fisher_wolf_diag["nonfinite_rho2_count"]
                )
                fisher_wolf_nonfinite_weight_count += int(
                    fisher_wolf_diag["nonfinite_weight_count"]
                )
                fisher_wolf_residual_values.extend(fisher_wolf_diag["residual_values"])
                fisher_wolf_whitened_residual_values.extend(
                    fisher_wolf_diag["whitened_residual_values"]
                )
                fisher_wolf_rho2_values.extend(fisher_wolf_diag["rho2_values"])
                fisher_wolf_W_values.extend(fisher_wolf_diag["weight_values"])
                if valid_client_count > 1 and weight_sum_value <= update_fusion_eps:
                    aggregated_state[key] = theta_old.detach().cpu().clone()
                    skipped_expert_params += 1
                    continue
                clients_per_param.append(float(valid_client_count))

            theta_new = theta_old + fused_delta
            aggregated_state[key] = theta_new.detach().cpu()
            updated_expert_params += 1

    if variant == FEDWOLF_UPDATE_FUSION_VARIANT_FISHER_HISTORY_WOLF and history_enabled:
        history_state.update(fisher_history_pending_updates)

    diagnostics = {
        "fedwolf_fusion_mode": FEDWOLF_FUSION_MODE_ROBUST_UPDATE_FUSION,
        "fedwolf_update_fusion_variant": variant,
        "fedwolf_fisher_precision_granularity": precision_granularity,
        "fedwolf_fisher_weight_source": fisher_weight_source,
        "robust_update_fusion_updated_expert_params": int(updated_expert_params),
        "robust_update_fusion_skipped_expert_params": int(skipped_expert_params),
        "robust_update_fusion_mean_clients_per_param": (
            sum(clients_per_param) / len(clients_per_param) if clients_per_param else 0.0
        ),
        "robust_update_fusion_delta_norm_mean": (
            sum(delta_norms) / len(delta_norms) if delta_norms else 0.0
        ),
        "robust_update_fusion_delta_norm_max": max(delta_norms) if delta_norms else 0.0,
    }

    if variant == FEDWOLF_UPDATE_FUSION_VARIANT_FISHER_ONLY:
        invalid_precision_count = (
            fisher_only_missing_precision_count
            + fisher_only_nonfinite_precision_count
            + fisher_only_nonpositive_precision_count
        )
        total_precision_count = fisher_only_valid_client_contribs + invalid_precision_count
        A_mean = sum(fisher_only_A_values) / len(fisher_only_A_values) if fisher_only_A_values else 0.0
        A_var = (
            sum((value - A_mean) ** 2 for value in fisher_only_A_values) / len(fisher_only_A_values)
            if fisher_only_A_values
            else 0.0
        )
        # Historical A/precision fields are kept for compatibility. With
        # legacy_score they describe the actual score weight, not precision A.
        diagnostics.update(
            {
                "fisher_only_updated_expert_params": int(updated_expert_params),
                "fisher_only_skipped_expert_params": int(skipped_expert_params),
                "fisher_only_valid_client_contribs": int(fisher_only_valid_client_contribs),
                "fisher_only_weight_source": fisher_weight_source,
                "fisher_only_missing_weight_count": int(fisher_only_missing_precision_count),
                "fisher_only_nonfinite_weight_count": int(fisher_only_nonfinite_precision_count),
                "fisher_only_nonpositive_weight_count": int(fisher_only_nonpositive_precision_count),
                "fisher_only_missing_precision_count": int(fisher_only_missing_precision_count),
                "fisher_only_nonfinite_precision_count": int(fisher_only_nonfinite_precision_count),
                "fisher_only_nonpositive_precision_count": int(fisher_only_nonpositive_precision_count),
                "fisher_only_weight_mean": float(A_mean),
                "fisher_only_weight_std": float(math.sqrt(max(A_var, 0.0))),
                "fisher_only_weight_min": min(fisher_only_A_values) if fisher_only_A_values else 0.0,
                "fisher_only_weight_max": max(fisher_only_A_values) if fisher_only_A_values else 0.0,
                "fisher_only_weight_zero_or_invalid_fraction": (
                    invalid_precision_count / float(total_precision_count)
                    if total_precision_count > 0
                    else 0.0
                ),
                "fisher_only_A_mean": float(A_mean),
                "fisher_only_A_std": float(math.sqrt(max(A_var, 0.0))),
                "fisher_only_A_min": min(fisher_only_A_values) if fisher_only_A_values else 0.0,
                "fisher_only_A_max": max(fisher_only_A_values) if fisher_only_A_values else 0.0,
                "fisher_only_A_zero_or_invalid_fraction": (
                    invalid_precision_count / float(total_precision_count)
                    if total_precision_count > 0
                    else 0.0
                ),
                "fisher_only_weight_sum_mean": (
                    sum(fisher_only_weight_sums) / len(fisher_only_weight_sums)
                    if fisher_only_weight_sums
                    else 0.0
                ),
                "fisher_only_weight_sum_min": min(fisher_only_weight_sums) if fisher_only_weight_sums else 0.0,
                "fisher_only_weight_sum_max": max(fisher_only_weight_sums) if fisher_only_weight_sums else 0.0,
                "fisher_only_delta_norm_mean": diagnostics["robust_update_fusion_delta_norm_mean"],
                "fisher_only_delta_norm_max": diagnostics["robust_update_fusion_delta_norm_max"],
            }
        )
    elif variant == FEDWOLF_UPDATE_FUSION_VARIANT_ROBUST_ONLY:
        residual_mean = (
            sum(robust_only_residual_values) / len(robust_only_residual_values)
            if robust_only_residual_values
            else 0.0
        )
        residual_sorted = sorted(robust_only_residual_values)
        residual_median = (
            residual_sorted[len(residual_sorted) // 2] if residual_sorted else 0.0
        )
        rho2_mean = (
            sum(robust_only_rho2_values) / len(robust_only_rho2_values)
            if robust_only_rho2_values
            else 0.0
        )
        diagnostics.update(
            {
                "robust_only_updated_expert_params": int(updated_expert_params),
                "robust_only_skipped_expert_params": int(skipped_expert_params),
                "robust_only_valid_client_contribs": int(robust_only_valid_client_contribs),
                "robust_only_missing_param_count": int(robust_only_missing_param_count),
                "robust_only_shape_mismatch_count": int(robust_only_shape_mismatch_count),
                "robust_only_nonfinite_param_count": int(robust_only_nonfinite_param_count),
                "robust_only_nonfinite_rho2_count": int(robust_only_nonfinite_rho2_count),
                "robust_only_irls_steps": int(irls_steps),
                "robust_only_residual_mean": float(residual_mean),
                "robust_only_residual_median": float(residual_median),
                "robust_only_residual_max": (
                    max(robust_only_residual_values) if robust_only_residual_values else 0.0
                ),
                "robust_only_rho2_mean": float(rho2_mean),
                "robust_only_rho2_min": min(robust_only_rho2_values) if robust_only_rho2_values else 0.0,
                "robust_only_rho2_max": max(robust_only_rho2_values) if robust_only_rho2_values else 0.0,
                "robust_only_rho2_low_fraction": (
                    sum(1 for value in robust_only_rho2_values if value < 0.5)
                    / float(len(robust_only_rho2_values))
                    if robust_only_rho2_values
                    else 0.0
                ),
                "robust_only_weight_sum_mean": (
                    sum(robust_only_weight_sums) / len(robust_only_weight_sums)
                    if robust_only_weight_sums
                    else 0.0
                ),
                "robust_only_weight_sum_min": min(robust_only_weight_sums) if robust_only_weight_sums else 0.0,
                "robust_only_weight_sum_max": max(robust_only_weight_sums) if robust_only_weight_sums else 0.0,
                "robust_only_effective_num_clients_mean": (
                    sum(robust_only_effective_num_clients)
                    / len(robust_only_effective_num_clients)
                    if robust_only_effective_num_clients
                    else 0.0
                ),
                "robust_only_effective_num_clients_min": (
                    min(robust_only_effective_num_clients)
                    if robust_only_effective_num_clients
                    else 0.0
                ),
                "robust_only_delta_norm_mean": diagnostics["robust_update_fusion_delta_norm_mean"],
                "robust_only_delta_norm_max": diagnostics["robust_update_fusion_delta_norm_max"],
            }
        )
    elif variant == FEDWOLF_UPDATE_FUSION_VARIANT_FISHER_WOLF:
        precision_error_count = (
            fisher_wolf_missing_precision_count
            + fisher_wolf_nonfinite_precision_count
            + fisher_wolf_nonpositive_precision_count
        )
        A_mean = sum(fisher_wolf_A_values) / len(fisher_wolf_A_values) if fisher_wolf_A_values else 0.0
        A_var = (
            sum((value - A_mean) ** 2 for value in fisher_wolf_A_values) / len(fisher_wolf_A_values)
            if fisher_wolf_A_values
            else 0.0
        )
        residual_sorted = sorted(fisher_wolf_residual_values)
        whitened_sorted = sorted(fisher_wolf_whitened_residual_values)
        rho2_mean = (
            sum(fisher_wolf_rho2_values) / len(fisher_wolf_rho2_values)
            if fisher_wolf_rho2_values
            else 0.0
        )
        W_mean = sum(fisher_wolf_W_values) / len(fisher_wolf_W_values) if fisher_wolf_W_values else 0.0
        W_var = (
            sum((value - W_mean) ** 2 for value in fisher_wolf_W_values) / len(fisher_wolf_W_values)
            if fisher_wolf_W_values
            else 0.0
        )
        diagnostics.update(
            {
                "fisher_wolf_updated_expert_params": int(updated_expert_params),
                "fisher_wolf_skipped_expert_params": int(skipped_expert_params),
                "fisher_wolf_valid_client_contribs": int(fisher_wolf_valid_client_contribs),
                "fisher_wolf_missing_param_count": int(fisher_wolf_missing_param_count),
                "fisher_wolf_shape_mismatch_count": int(fisher_wolf_shape_mismatch_count),
                "fisher_wolf_nonfinite_param_count": int(fisher_wolf_nonfinite_param_count),
                "fisher_wolf_missing_precision_count": int(fisher_wolf_missing_precision_count),
                "fisher_wolf_nonfinite_precision_count": int(fisher_wolf_nonfinite_precision_count),
                "fisher_wolf_nonpositive_precision_count": int(fisher_wolf_nonpositive_precision_count),
                "fisher_wolf_precision_error_count": int(precision_error_count),
                "fisher_wolf_nonfinite_rho2_count": int(fisher_wolf_nonfinite_rho2_count),
                "fisher_wolf_nonfinite_weight_count": int(fisher_wolf_nonfinite_weight_count),
                "fisher_wolf_irls_steps": int(max(irls_steps, 1)),
                "fisher_wolf_A_mean": float(A_mean),
                "fisher_wolf_A_std": float(math.sqrt(max(A_var, 0.0))),
                "fisher_wolf_A_min": min(fisher_wolf_A_values) if fisher_wolf_A_values else 0.0,
                "fisher_wolf_A_max": max(fisher_wolf_A_values) if fisher_wolf_A_values else 0.0,
                "fisher_wolf_residual_mean": (
                    sum(fisher_wolf_residual_values) / len(fisher_wolf_residual_values)
                    if fisher_wolf_residual_values
                    else 0.0
                ),
                "fisher_wolf_residual_median": (
                    residual_sorted[len(residual_sorted) // 2] if residual_sorted else 0.0
                ),
                "fisher_wolf_residual_max": (
                    max(fisher_wolf_residual_values) if fisher_wolf_residual_values else 0.0
                ),
                "fisher_wolf_whitened_residual_mean": (
                    sum(fisher_wolf_whitened_residual_values) / len(fisher_wolf_whitened_residual_values)
                    if fisher_wolf_whitened_residual_values
                    else 0.0
                ),
                "fisher_wolf_whitened_residual_median": (
                    whitened_sorted[len(whitened_sorted) // 2] if whitened_sorted else 0.0
                ),
                "fisher_wolf_whitened_residual_max": (
                    max(fisher_wolf_whitened_residual_values)
                    if fisher_wolf_whitened_residual_values
                    else 0.0
                ),
                "fisher_wolf_rho2_mean": float(rho2_mean),
                "fisher_wolf_rho2_min": min(fisher_wolf_rho2_values) if fisher_wolf_rho2_values else 0.0,
                "fisher_wolf_rho2_max": max(fisher_wolf_rho2_values) if fisher_wolf_rho2_values else 0.0,
                "fisher_wolf_rho2_low_fraction": (
                    sum(1 for value in fisher_wolf_rho2_values if value < 0.5)
                    / float(len(fisher_wolf_rho2_values))
                    if fisher_wolf_rho2_values
                    else 0.0
                ),
                "fisher_wolf_W_mean": float(W_mean),
                "fisher_wolf_W_std": float(math.sqrt(max(W_var, 0.0))),
                "fisher_wolf_W_min": min(fisher_wolf_W_values) if fisher_wolf_W_values else 0.0,
                "fisher_wolf_W_max": max(fisher_wolf_W_values) if fisher_wolf_W_values else 0.0,
                "fisher_wolf_weight_sum_mean": (
                    sum(fisher_wolf_weight_sums) / len(fisher_wolf_weight_sums)
                    if fisher_wolf_weight_sums
                    else 0.0
                ),
                "fisher_wolf_weight_sum_min": min(fisher_wolf_weight_sums) if fisher_wolf_weight_sums else 0.0,
                "fisher_wolf_weight_sum_max": max(fisher_wolf_weight_sums) if fisher_wolf_weight_sums else 0.0,
                "fisher_wolf_effective_num_clients_mean": (
                    sum(fisher_wolf_effective_num_clients)
                    / len(fisher_wolf_effective_num_clients)
                    if fisher_wolf_effective_num_clients
                    else 0.0
                ),
                "fisher_wolf_effective_num_clients_min": (
                    min(fisher_wolf_effective_num_clients)
                    if fisher_wolf_effective_num_clients
                    else 0.0
                ),
                "fisher_wolf_delta_norm_mean": diagnostics["robust_update_fusion_delta_norm_mean"],
                "fisher_wolf_delta_norm_max": diagnostics["robust_update_fusion_delta_norm_max"],
            }
        )
    elif variant == FEDWOLF_UPDATE_FUSION_VARIANT_FISHER_HISTORY_WOLF:
        history_values = []
        if isinstance(history_state, dict):
            for history_value in history_state.values():
                if torch.is_tensor(history_value):
                    if history_value.numel() != 1:
                        continue
                    try:
                        history_value = history_value.detach().cpu().item()
                    except (RuntimeError, TypeError, ValueError):
                        continue
                try:
                    history_value = float(history_value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(history_value):
                    history_values.append(min(1.0, max(0.0, history_value)))
        history_state_size = len(history_state) if isinstance(history_state, dict) else 0
        diagnostics.update(
            {
                "fisher_history_enabled": bool(history_enabled),
                "fisher_history_eta": float(history_eta),
                "fisher_history_init": float(history_init),
                "fisher_history_state_size": int(history_state_size),
                "fisher_history_updated_expert_params": int(updated_expert_params),
                "fisher_history_skipped_expert_params": int(skipped_expert_params),
                "fisher_history_valid_client_contribs": int(fisher_history_valid_client_contribs),
                "fisher_history_missing_score_count": int(fisher_history_missing_score_count),
                "fisher_history_nonfinite_or_nonpositive_score_count": int(
                    fisher_history_nonfinite_or_nonpositive_score_count
                ),
                "fisher_history_zero_usage_count": int(fisher_history_zero_usage_count),
                "fisher_history_good_current_good_count": int(
                    fisher_history_good_current_good_count
                ),
                "fisher_history_bad_current_good_count": int(
                    fisher_history_bad_current_good_count
                ),
                "fisher_history_good_current_bad_count": int(
                    fisher_history_good_current_bad_count
                ),
                "fisher_history_bad_current_bad_count": int(
                    fisher_history_bad_current_bad_count
                ),
            }
        )
        diagnostics.update(
            _summarize_float_values(
                fisher_history_usage_conf_values,
                "fisher_history_usage_conf",
            )
        )
        diagnostics.update(
            _summarize_float_values(
                fisher_history_fisher_factor_values,
                "fisher_history_fisher_factor",
            )
        )
        diagnostics.update(
            _summarize_float_values(
                fisher_history_observation_values,
                "fisher_history_observation",
            )
        )
        diagnostics.update(
            _summarize_float_values(
                fisher_history_factor_values,
                "fisher_history_factor",
            )
        )
        diagnostics.update(
            _summarize_float_values(
                fisher_history_weight_values,
                "fisher_history_weight",
            )
        )
        diagnostics.update(
            _summarize_float_values(
                history_values,
                "fisher_history_state",
            )
        )
        effective_num_clients_summary = _summarize_float_values(
            fisher_history_effective_num_clients,
            "fisher_history_effective_num_clients",
        )
        diagnostics.update(
            {
                "fisher_history_effective_num_clients_mean": effective_num_clients_summary[
                    "fisher_history_effective_num_clients_mean"
                ],
                "fisher_history_effective_num_clients_min": effective_num_clients_summary[
                    "fisher_history_effective_num_clients_min"
                ],
                "fisher_history_effective_num_clients_max": effective_num_clients_summary[
                    "fisher_history_effective_num_clients_max"
                ],
            }
        )
    return aggregated_state, diagnostics


class FedWoLFRobustUpdateFusionAggregator(FedAvgAggregator):
    def __init__(self, args=None):
        super().__init__(args)
        self.args = args
        self.last_robust_update_summary = {}
        # Persistent per-(client_id, layer_id, expert_id) history reliability state
        # used by the fisher_history_wolf variant. Values are scalar h in [0, 1].
        self.fisher_history_state = {}

    def aggregate(self, client_updates, client_weights, global_model=None, **kwargs):
        client_stats = kwargs.get("client_stats")
        if client_stats is None:
            client_stats = kwargs.get("expert_weights")
        aggregated_state, diagnostics = aggregate_experts_robust_update_fusion(
            args=self.args,
            global_model=global_model,
            client_updates=client_updates,
            client_weights=client_weights,
            client_stats=client_stats,
            aggregation_device=self.aggregation_device,
            history_state=self.fisher_history_state,
        )
        self.last_robust_update_summary = diagnostics
        return aggregated_state


def build_aggregator(args):
    if args.agg_method == "fedwolf_fisher_only":
        raise ValueError(
            "agg_method='fedwolf_fisher_only' has been removed with the legacy "
            "FedWoLF filter path. Use agg_method='fedwolf', "
            "fedwolf_fusion_mode='robust_update_fusion', and "
            "fedwolf_update_fusion_variant='fisher_only' instead."
        )
    if args.agg_method == "fedwolf":
        fusion_mode = str(
            getattr(
                args,
                "fedwolf_fusion_mode",
                FEDWOLF_FUSION_MODE_ROBUST_UPDATE_FUSION,
            )
        ).strip().lower()
        if fusion_mode != FEDWOLF_FUSION_MODE_ROBUST_UPDATE_FUSION:
            raise ValueError(
                f"fedwolf_fusion_mode={fusion_mode!r} is no longer enabled for "
                "agg_method='fedwolf'. Current supported mode is "
                "'robust_update_fusion'. Use fedwolf_update_fusion_variant in "
                "{uniform_update, fisher_only, robust_only, fisher_wolf, "
                "fisher_history_wolf}."
            )
        return FedWoLFRobustUpdateFusionAggregator(args)
    if args.agg_method == "expert_fedavg":
        return ExpertFedAvgAggregator(args)
    if args.agg_method == "fedavg":
        return FedAvgAggregator(args)
    raise ValueError(f"Unknown aggregation method: {args.agg_method}")
