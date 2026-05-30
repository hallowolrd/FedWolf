import collections
import math
from abc import ABC, abstractmethod

import torch

from fl.history_wolf_filter import HistoryWolfExpertFilter


# 从模型参数名中解析 expert 所在的 block 层号和 expert 编号。
# 典型 key 形如：blocks.0.ffn.experts.3.xxx
# 返回值：
# - 如果是 expert 参数：返回 (layer_id, expert_id)
# - 如果不是 expert 参数：返回 None
def parse_expert_ref_from_key(key):
    parts = key.split(".")
    if "blocks" not in parts or "experts" not in parts:
        return None

    # 找到参数名中 blocks 和 experts 的位置。
    blocks_idx = parts.index("blocks")
    experts_idx = parts.index("experts")

    # 确保 blocks 后面有 layer id，experts 后面有 expert id。
    if blocks_idx + 1 >= len(parts) or experts_idx + 1 >= len(parts):
        return None

    # layer id 和 expert id 必须是数字，否则说明不是标准 expert 参数名。
    if not parts[blocks_idx + 1].isdigit() or not parts[experts_idx + 1].isdigit():
        return None

    # layer_id 保持字符串形式，方便后面匹配统计字典中的 key；
    # expert_id 转成 int，方便作为列表/张量下标。
    return parts[blocks_idx + 1], int(parts[experts_idx + 1])


class Aggregator(ABC):
    # 聚合器统一接口。后续新增聚合方法时，只需要新增实现类并在 build_aggregator 中注册。
    @abstractmethod
    def aggregate(self, client_updates, client_weights, global_model=None, **kwargs):
        pass


_SPLIT_NON_EXPERT_AGG_ALIASES = {
    "equal_avg": "equal_avg",
    "direct_avg": "equal_avg",
    "fedavg": "sample_weighted_avg",
    "sample_weighted": "sample_weighted_avg",
    "sample_weighted_avg": "sample_weighted_avg",
}
_SPLIT_EXPERT_AGG_ALIASES = {
    "equal_avg": "equal_avg",
    "direct_avg": "equal_avg",
    "fedavg": "sample_weighted_avg",
    "sample_weighted": "sample_weighted_avg",
    "sample_weighted_avg": "sample_weighted_avg",
    "expert_usage": "expert_usage",
    "token_usage": "expert_usage",
    "expert_fedavg": "expert_usage",
    "fisher": "fisher_raw_score",
    "fisher_raw_score": "fisher_raw_score",
    "fedwolf_fisher_only": "fisher_raw_score",
    "history_wolf_filter": "history_wolf_filter",
    "fedwolf_history_wolf": "history_wolf_filter",
    "fisher_history_wolf": "fisher_history_wolf",
    "fedwolf_fisher_history_wolf": "fisher_history_wolf",
}
_SPLIT_LEGACY_AGG_METHOD_MAP = {
    "fedavg": ("sample_weighted_avg", "sample_weighted_avg"),
    "equal_avg": ("equal_avg", "equal_avg"),
    "expert_fedavg": ("sample_weighted_avg", "expert_usage"),
    "expert_equal_avg": ("sample_weighted_avg", "equal_avg"),
    "fedwolf_fisher_only": ("sample_weighted_avg", "fisher_raw_score"),
    "fedwolf_history_wolf": ("sample_weighted_avg", "history_wolf_filter"),
    "fedwolf_fisher_history_wolf": ("sample_weighted_avg", "fisher_history_wolf"),
}
HISTORY_WOLF_EXPERT_AGG_METHODS = {
    "history_wolf_filter",
    "fisher_history_wolf",
}


def _normalize_split_agg_choice(value, aliases, field_name):
    normalized = str(value).strip().lower()
    if normalized in aliases:
        return aliases[normalized]

    valid_values = sorted(set(aliases))
    raise ValueError(f"{field_name} must be one of {valid_values}, got {value!r}")


def _get_split_agg_methods(args):
    non_expert_method = getattr(args, "non_expert_agg_method", None)
    expert_method = getattr(args, "expert_agg_method", None)

    if non_expert_method is not None or expert_method is not None:
        if non_expert_method is None or expert_method is None:
            raise ValueError(
                "non_expert_agg_method and expert_agg_method must be set together"
            )

        return (
            _normalize_split_agg_choice(
                non_expert_method,
                _SPLIT_NON_EXPERT_AGG_ALIASES,
                "non_expert_agg_method",
            ),
            _normalize_split_agg_choice(
                expert_method,
                _SPLIT_EXPERT_AGG_ALIASES,
                "expert_agg_method",
            ),
        )

    agg_method = str(getattr(args, "agg_method", "")).strip().lower()
    if agg_method in _SPLIT_LEGACY_AGG_METHOD_MAP:
        return _SPLIT_LEGACY_AGG_METHOD_MAP[agg_method]

    valid_values = sorted(_SPLIT_LEGACY_AGG_METHOD_MAP)
    raise ValueError(
        "Missing split aggregation config. Set non_expert_agg_method and "
        f"expert_agg_method, or use legacy agg_method in {valid_values}."
    )


class SplitAggregator(Aggregator):
    # 按参数类型拆分聚合策略：非 expert 参数和 expert 参数可分别选择权重来源。
    def __init__(self, args=None):
        self.non_expert_agg_method, self.expert_agg_method = _get_split_agg_methods(args)
        self.eps = float(getattr(args, "fedwolf_eps", 1e-8))
        # History-WoLF 需要跨轮保存 mu/P 历史，因此聚合器持有同一个 filter 实例。
        self.history_wolf_filter = HistoryWolfExpertFilter(args, eps=self.eps)
        # 每轮聚合开始前清空；本轮内按 expert_ref 复用，避免同一 expert 多个参数重复更新历史。
        self._history_wolf_weights_cache = None

    def aggregate(self, client_updates, client_weights, global_model=None, **kwargs):
        self._validate_inputs(client_updates, client_weights)
        self._history_wolf_weights_cache = None

        client_stats = kwargs.get("client_stats")
        if client_stats is None:
            client_stats = kwargs.get("expert_weights")

        is_history_wolf_method = self.expert_agg_method in HISTORY_WOLF_EXPERT_AGG_METHODS
        global_state = None

        # History-WoLF 要比较 client update 与当前 global state 的 expert delta。
        if is_history_wolf_method:
            global_state = global_model.state_dict() if global_model is not None else None
            if global_state is None:
                raise ValueError(
                    "history_wolf_filter requires global_state for expert delta computation."
                )
            if client_stats is None:
                raise ValueError("history_wolf_filter requires client_stats.")

        if self._expert_method_needs_stats():
            if client_stats is None:
                raise ValueError(
                    f"expert_agg_method={self.expert_agg_method!r} requires client expert stats"
                )
            if len(client_stats) != len(client_updates):
                raise ValueError("client expert stats and client_updates must have the same length")

        if not is_history_wolf_method:
            global_state = global_model.state_dict() if global_model is not None else None

        if is_history_wolf_method:
            # 按 layer/expert 先分组参数 key，再一次性计算本轮 expert 权重。
            expert_keys_by_ref = self._collect_expert_keys_by_ref(client_updates[0].keys())
            self._history_wolf_weights_cache = self.history_wolf_filter.compute_weights(
                client_updates=client_updates,
                client_stats=client_stats,
                global_state=global_state,
                expert_keys_by_ref=expert_keys_by_ref,
                use_fisher=(self.expert_agg_method == "fisher_history_wolf"),
            )

        aggregated_state = collections.OrderedDict()

        for key in client_updates[0].keys():
            first_value = client_updates[0][key].detach().cpu()

            if not torch.is_floating_point(first_value):
                aggregated_state[key] = first_value.clone()
                continue

            expert_ref = parse_expert_ref_from_key(key)
            if expert_ref is None:
                weights, add_eps = self._get_non_expert_weights(client_weights)
            else:
                weights, add_eps = self._get_expert_weights(
                    client_weights=client_weights,
                    client_stats=client_stats,
                    expert_ref=expert_ref,
                )

            total_weight = sum(weights)
            if total_weight <= 0:
                if global_state is not None:
                    aggregated_state[key] = global_state[key].detach().cpu().clone()
                else:
                    aggregated_state[key] = first_value.clone()
                continue

            denominator = total_weight + (self.eps if add_eps else 0.0)
            aggregated_state[key] = torch.zeros_like(first_value)
            for update, weight in zip(client_updates, weights):
                aggregated_state[key] += update[key].detach().cpu() * (weight / denominator)

        return aggregated_state

    @property
    def last_history_wolf_summary(self):
        if hasattr(self, "history_wolf_filter"):
            return getattr(self.history_wolf_filter, "last_summary", None)
        return None

    def state_dict(self):
        # checkpoint 只保存 filter 的历史状态，不保存本轮日志 summary/cache。
        state = {}
        if hasattr(self, "history_wolf_filter"):
            state["history_wolf_filter"] = self.history_wolf_filter.state_dict()
        return state

    def load_state_dict(self, state):
        if not state:
            return
        if hasattr(self, "history_wolf_filter"):
            self.history_wolf_filter.load_state_dict(
                state.get("history_wolf_filter", {})
            )

    def _collect_expert_keys_by_ref(self, keys):
        # 只负责把参数名分桶到 (layer_id, expert_id)，不计算任何权重。
        expert_keys_by_ref = collections.defaultdict(list)
        for key in keys:
            expert_ref = parse_expert_ref_from_key(key)
            if expert_ref is not None:
                expert_keys_by_ref[expert_ref].append(key)
        return dict(expert_keys_by_ref)

    def _validate_inputs(self, client_updates, client_weights):
        if len(client_updates) == 0:
            raise ValueError("SplitAggregator requires at least one client update")
        if len(client_updates) != len(client_weights):
            raise ValueError("client_updates and client_weights must have the same length")

        if (
            self.non_expert_agg_method == "sample_weighted_avg"
            or self.expert_agg_method == "sample_weighted_avg"
        ):
            total_client_weight = sum(float(weight) for weight in client_weights)
            if total_client_weight <= 0:
                raise ValueError("sample-weighted aggregation requires positive client weights")

    def _expert_method_needs_stats(self):
        return self.expert_agg_method in (
            {"expert_usage", "fisher_raw_score"} | HISTORY_WOLF_EXPERT_AGG_METHODS
        )

    def _get_non_expert_weights(self, client_weights):
        if self.non_expert_agg_method == "equal_avg":
            return [1.0 for _ in client_weights], False
        if self.non_expert_agg_method == "sample_weighted_avg":
            return [float(weight) for weight in client_weights], False

        raise ValueError(f"Unknown non_expert_agg_method: {self.non_expert_agg_method}")

    def _get_expert_weights(self, client_weights, client_stats, expert_ref):
        if self.expert_agg_method == "equal_avg":
            return [1.0 for _ in client_weights], False
        if self.expert_agg_method == "sample_weighted_avg":
            return [float(weight) for weight in client_weights], False

        if self.expert_agg_method in HISTORY_WOLF_EXPERT_AGG_METHODS:
            # History-WoLF 权重必须来自本轮预计算 cache，禁止退化成 uniform/usage/Fisher。
            if self._history_wolf_weights_cache is None:
                raise RuntimeError(
                    "History-WoLF weights cache is empty. compute_weights must be called "
                    "before per-key aggregation."
                )
            weights = self._history_wolf_weights_cache.get(
                expert_ref,
                [0.0 for _ in client_weights],
            )
            return weights, False

        layer_id, expert_id = expert_ref
        if self.expert_agg_method == "expert_usage":
            return [
                self._get_expert_usage(client_stat, layer_id, expert_id)
                for client_stat in client_stats
            ], False
        if self.expert_agg_method == "fisher_raw_score":
            return [
                self._get_fisher_score(client_stat, layer_id, expert_id)
                for client_stat in client_stats
            ], True

        raise ValueError(f"Unknown expert_agg_method: {self.expert_agg_method}")

    def _get_expert_usage(self, client_stats, layer_id, expert_id):
        if isinstance(client_stats, dict):
            layer_stats = client_stats.get("expert_stats_by_layer", {}).get(str(layer_id), {})
            usage = layer_stats.get("expert_activations")

            if usage is None:
                usage = client_stats.get("expert_activations_by_layer", {}).get(str(layer_id))
            if usage is None:
                usage = client_stats.get("expert_activations")
        else:
            usage = client_stats

        if usage is None:
            return 0.0
        if expert_id >= len(usage):
            raise ValueError(f"Missing expert usage for expert id {expert_id}")

        return float(usage[expert_id])

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
    # 根据 non_expert_agg_method 和 expert_agg_method 构造拆分聚合器。
    # 旧 agg_method 会在配置加载阶段映射到这两个字段；这里也保留兜底兼容。
    return SplitAggregator(args)
