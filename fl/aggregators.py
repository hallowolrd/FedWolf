import collections
import math
from abc import ABC, abstractmethod

import torch

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


class ExpertEqualAvgAggregator(Aggregator):
    # Shared/router/classifier 按客户端样本数 FedAvg；expert 参数按客户端数等权平均。
    def aggregate(self, client_updates, client_weights, global_model=None, **kwargs):
        if len(client_updates) == 0:
            raise ValueError("ExpertEqualAvg requires at least one client update")
        if len(client_updates) != len(client_weights):
            raise ValueError("client_updates and client_weights must have the same length")

        total_client_weight = sum(client_weights)
        if total_client_weight <= 0:
            raise ValueError("ExpertEqualAvg requires positive total client weight")

        aggregated_state = collections.OrderedDict()
        expert_weight = 1.0 / len(client_updates)
        for key in client_updates[0].keys():
            first_value = client_updates[0][key].detach().cpu()
            if not torch.is_floating_point(first_value):
                aggregated_state[key] = first_value.clone()
                continue

            if parse_expert_ref_from_key(key) is None:
                aggregated_state[key] = torch.zeros_like(first_value)
                for update, weight in zip(client_updates, client_weights):
                    aggregated_state[key] += update[key].detach().cpu() * (weight / total_client_weight)
            else:
                aggregated_state[key] = torch.zeros_like(first_value)
                for update in client_updates:
                    aggregated_state[key] += update[key].detach().cpu() * expert_weight

        return aggregated_state


class FisherOnlyAggregator(Aggregator):
    # Fisher-only expert aggregation:
    # - shared / router / classifier 等非 expert 参数仍按客户端样本数做 FedAvg；
    # - expert 参数按客户端上传的 expert_fisher_score_by_layer 做 Fisher-score 加权平均。
    def __init__(self, args=None):
        self.eps = float(getattr(args, "fedwolf_eps", 1e-8))

    def aggregate(self, client_updates, client_weights, global_model=None, **kwargs):
        if len(client_updates) == 0:
            raise ValueError("FisherOnly requires at least one client update")
        if len(client_updates) != len(client_weights):
            raise ValueError("client_updates and client_weights must have the same length")

        client_stats = kwargs.get("client_stats")
        if client_stats is None:
            client_stats = kwargs.get("expert_weights")
        if client_stats is None:
            raise ValueError("FisherOnly requires client_stats or expert_weights")
        if len(client_stats) != len(client_updates):
            raise ValueError("client_stats/expert_weights and client_updates must have the same length")

        global_state = global_model.state_dict() if global_model is not None else None
        total_client_weight = sum(client_weights)
        if total_client_weight <= 0:
            raise ValueError("FisherOnly requires positive total client weight")

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
                if global_state is not None:
                    aggregated_state[key] = global_state[key].detach().cpu().clone()
                else:
                    aggregated_state[key] = first_value.clone()
                continue

            aggregated_state[key] = torch.zeros_like(first_value)
            for update, weight in zip(client_updates, weights):
                aggregated_state[key] += update[key].detach().cpu() * (weight / denominator)

        return aggregated_state

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
    if args.agg_method == "fedwolf_fisher_only":
        return FisherOnlyAggregator(args)
    if args.agg_method == "expert_equal_avg":
        return ExpertEqualAvgAggregator()
    if args.agg_method == "expert_fedavg":
        return ExpertFedAvgAggregator()
    if args.agg_method == "fedavg":
        return FedAvgAggregator()
    raise ValueError(f"Unknown aggregation method: {args.agg_method}")
