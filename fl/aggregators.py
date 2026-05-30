import collections
import math
from abc import ABC, abstractmethod

import torch


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


class FedAvgAggregator(Aggregator):
    # 标准 FedAvg：
    # 对完整 state_dict 做按客户端样本数加权平均，权重 w_i = n_i / sum_j n_j。
    def aggregate(self, client_updates, client_weights, global_model=None, **kwargs):
        # 至少需要一个客户端更新，否则没有可聚合内容。
        if len(client_updates) == 0:
            raise ValueError("FedAvg requires at least one client update")

        # 每个客户端更新必须对应一个客户端权重。
        if len(client_updates) != len(client_weights):
            raise ValueError("client_updates and client_weights must have the same length")

        # FedAvg 中 client_weights 通常是每个客户端的数据量。
        total_weight = sum(client_weights)
        if total_weight <= 0:
            raise ValueError("FedAvg requires positive total client weight")

        aggregated_state = collections.OrderedDict()

        # 遍历模型 state_dict 中的每一个参数 / buffer。
        for key in client_updates[0].keys():
            # 先取第一个客户端对应 key 的值，用于判断类型和创建同形状张量。
            first_value = client_updates[0][key].detach().cpu()

            if torch.is_floating_point(first_value):
                # 浮点参数可以做加权平均。
                aggregated_state[key] = torch.zeros_like(first_value)
                for update, weight in zip(client_updates, client_weights):
                    # 按样本数权重归一化后累加。
                    aggregated_state[key] += update[key].detach().cpu() * (weight / total_weight)
            else:
                # 非浮点 buffer 通常不能加权平均，沿用第一个客户端的值。
                aggregated_state[key] = first_value.clone()

        return aggregated_state


class EqualAvgAggregator(Aggregator):
    # 对完整 state_dict 做客户端等权平均，每个客户端权重都是 1 / client_num。
    def aggregate(self, client_updates, client_weights, global_model=None, **kwargs):
        # 至少需要一个客户端更新。
        if len(client_updates) == 0:
            raise ValueError("EqualAvg requires at least one client update")

        # 虽然等权平均不使用 client_weights 的数值，但仍要求长度一致，避免调用端传参错误。
        if len(client_updates) != len(client_weights):
            raise ValueError("client_updates and client_weights must have the same length")

        aggregated_state = collections.OrderedDict()

        # 所有客户端等权。
        client_weight = 1.0 / len(client_updates)

        for key in client_updates[0].keys():
            first_value = client_updates[0][key].detach().cpu()

            # 非浮点参数 / buffer 不参与平均，直接取第一个客户端的值。
            if not torch.is_floating_point(first_value):
                aggregated_state[key] = first_value.clone()
                continue

            # 对浮点参数做简单算术平均。
            aggregated_state[key] = torch.zeros_like(first_value)
            for update in client_updates:
                aggregated_state[key] += update[key].detach().cpu() * client_weight

        return aggregated_state


class ExpertFedAvgAggregator(Aggregator):
    # FL + MoE 专家级 FedAvg：
    # - 普通共享层仍按客户端训练样本数 n_i 做标准 FedAvg；
    # - blocks.{layer}.ffn.experts.{expert_id}.* 参数按该层该专家实际处理的 token 数 n_{i,l,e} 加权。
    def __init__(self):
        pass

    def aggregate(self, client_updates, client_weights, global_model=None, **kwargs):
        # 至少需要一个客户端更新。
        if len(client_updates) == 0:
            raise ValueError("ExpertFedAvg requires at least one client update")

        # 每个客户端更新必须对应一个客户端样本权重。
        if len(client_updates) != len(client_weights):
            raise ValueError("client_updates and client_weights must have the same length")

        # expert_weights 记录每个客户端中每个 expert 的使用量。
        # 对 expert 参数聚合时，不再用客户端样本数，而是用该 expert 的 usage。
        expert_weights = kwargs.get("expert_weights")
        if expert_weights is None:
            raise ValueError("ExpertFedAvg requires expert_weights for expert-level aggregation")
        if len(expert_weights) != len(client_updates):
            raise ValueError("expert_weights and client_updates must have the same length")

        # 如果传入了 global_model，则当某个 expert 本轮无人使用时，可以保留服务端旧参数。
        global_state = global_model.state_dict() if global_model is not None else None

        aggregated_state = collections.OrderedDict()

        # 非 expert 参数仍然使用标准 FedAvg，所以需要客户端总样本权重。
        total_client_weight = sum(client_weights)
        if total_client_weight <= 0:
            raise ValueError("ExpertFedAvg requires positive total client weight")

        for key in client_updates[0].keys():
            first_value = client_updates[0][key].detach().cpu()

            # 非浮点 buffer 不做平均，直接沿用第一个客户端。
            if not torch.is_floating_point(first_value):
                aggregated_state[key] = first_value.clone()
                continue

            # 判断当前参数是不是 MoE expert 参数。
            expert_ref = self._parse_expert_ref(key)
            if expert_ref is None:
                # 非 expert 参数：按客户端样本数做 FedAvg。
                weights = client_weights
            else:
                # expert 参数：按该 layer、该 expert 在各客户端中的 usage 加权。
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

            # 对当前参数做加权平均。
            aggregated_state[key] = torch.zeros_like(first_value)
            for update, weight in zip(client_updates, weights):
                aggregated_state[key] += update[key].detach().cpu() * (weight / total_weight)

        return aggregated_state

    def _parse_expert_ref(self, key):
        # 复用全局解析函数，判断参数 key 是否属于某个 expert。
        return parse_expert_ref_from_key(key)

    def _get_expert_weight(self, client_usage, layer_id, expert_id):
        # client_usage 可能是 dict，也可能是简单列表 / 张量。
        # dict 格式通常来自客户端记录的 expert usage 统计。
        if isinstance(client_usage, dict):
            if layer_id is None:
                # 兼容没有按 layer 统计的旧格式。
                usage = client_usage.get("expert_activations")
            else:
                # 优先读取 expert_stats_by_layer 中当前 layer 的 expert 激活次数。
                layer_stats = client_usage.get("expert_stats_by_layer", {}).get(str(layer_id), {})
                usage = layer_stats.get("expert_activations")

                # 如果新格式没有，就回退到 expert_activations_by_layer。
                if usage is None:
                    usage = client_usage.get("expert_activations_by_layer", {}).get(str(layer_id))

            # 如果找不到该 expert 的 usage，则认为这个客户端对该 expert 没有贡献。
            if usage is None:
                return 0.0

            # usage 长度不够说明统计信息缺失，直接报错暴露问题。
            if expert_id >= len(usage):
                raise ValueError(f"Missing expert weight for expert id {expert_id}")

            return float(usage[expert_id])

        # 非 dict 格式：假设 client_usage 本身就是 expert usage 列表。
        if expert_id >= len(client_usage):
            raise ValueError(f"Missing expert weight for expert id {expert_id}")
        return float(client_usage[expert_id])


class ExpertEqualAvgAggregator(Aggregator):
    # Shared/router/classifier 按客户端样本数 FedAvg；expert 参数按客户端数等权平均。
    def aggregate(self, client_updates, client_weights, global_model=None, **kwargs):
        # 至少需要一个客户端更新。
        if len(client_updates) == 0:
            raise ValueError("ExpertEqualAvg requires at least one client update")

        # 每个客户端更新必须对应一个客户端样本权重。
        if len(client_updates) != len(client_weights):
            raise ValueError("client_updates and client_weights must have the same length")

        # 非 expert 参数仍然按客户端样本数做 FedAvg。
        total_client_weight = sum(client_weights)
        if total_client_weight <= 0:
            raise ValueError("ExpertEqualAvg requires positive total client weight")

        aggregated_state = collections.OrderedDict()

        # expert 参数不看 usage，也不看样本数，直接对参与客户端等权。
        expert_weight = 1.0 / len(client_updates)

        for key in client_updates[0].keys():
            first_value = client_updates[0][key].detach().cpu()

            # 非浮点 buffer 不参与平均。
            if not torch.is_floating_point(first_value):
                aggregated_state[key] = first_value.clone()
                continue

            if parse_expert_ref_from_key(key) is None:
                # 非 expert 参数：标准 FedAvg。
                aggregated_state[key] = torch.zeros_like(first_value)
                for update, weight in zip(client_updates, client_weights):
                    aggregated_state[key] += update[key].detach().cpu() * (weight / total_client_weight)
            else:
                # expert 参数：客户端等权平均。
                aggregated_state[key] = torch.zeros_like(first_value)
                for update in client_updates:
                    aggregated_state[key] += update[key].detach().cpu() * expert_weight

        return aggregated_state


class FisherOnlyAggregator(Aggregator):
    # Fisher-only expert aggregation:
    # - shared / router / classifier 等非 expert 参数仍按客户端样本数做 FedAvg；
    # - expert 参数按客户端上传的 expert_fisher_score_by_layer 做 Fisher-score 加权平均。
    def __init__(self, args=None):
        # eps 用于 Fisher 权重归一化分母，避免极小数值导致数值不稳定。
        self.eps = float(getattr(args, "fedwolf_eps", 1e-8))

    def aggregate(self, client_updates, client_weights, global_model=None, **kwargs):
        # 至少需要一个客户端更新。
        if len(client_updates) == 0:
            raise ValueError("FisherOnly requires at least one client update")

        # 每个客户端更新必须对应一个客户端样本权重。
        if len(client_updates) != len(client_weights):
            raise ValueError("client_updates and client_weights must have the same length")

        # FisherOnly 需要客户端统计信息：
        # - 优先读取 client_stats；
        # - 如果没有，则兼容性地回退到 expert_weights。
        client_stats = kwargs.get("client_stats")
        if client_stats is None:
            client_stats = kwargs.get("expert_weights")
        if client_stats is None:
            raise ValueError("FisherOnly requires client_stats or expert_weights")
        if len(client_stats) != len(client_updates):
            raise ValueError("client_stats/expert_weights and client_updates must have the same length")

        # 如果某个 expert 的 Fisher 权重全为 0，则可以保留 global_model 中的旧参数。
        global_state = global_model.state_dict() if global_model is not None else None

        # 非 expert 参数仍然使用客户端样本数做 FedAvg。
        total_client_weight = sum(client_weights)
        if total_client_weight <= 0:
            raise ValueError("FisherOnly requires positive total client weight")

        aggregated_state = collections.OrderedDict()

        for key in client_updates[0].keys():
            first_value = client_updates[0][key].detach().cpu()

            # 非浮点 buffer 不做加权平均。
            if not torch.is_floating_point(first_value):
                aggregated_state[key] = first_value.clone()
                continue

            # 判断当前参数是否属于某个 expert。
            expert_ref = parse_expert_ref_from_key(key)
            if expert_ref is None:
                # 非 expert 参数：使用标准 FedAvg 权重。
                weights = client_weights
                total_weight = total_client_weight
                denominator = total_weight
            else:
                # expert 参数：使用 Fisher score 作为聚合权重。
                layer_id, expert_id = expert_ref
                weights = [
                    self._get_fisher_score(client_stat, layer_id, expert_id)
                    for client_stat in client_stats
                ]

                # Fisher score 总和作为归一化分母。
                total_weight = sum(weights)

                # 加 eps 是为了防止分母过小造成数值问题。
                denominator = total_weight + self.eps

            if total_weight <= 0:
                # 当前参数没有有效贡献时，优先保留服务端旧参数。
                if global_state is not None:
                    aggregated_state[key] = global_state[key].detach().cpu().clone()
                else:
                    aggregated_state[key] = first_value.clone()
                continue

            # 对当前参数做加权平均。
            aggregated_state[key] = torch.zeros_like(first_value)
            for update, weight in zip(client_updates, weights):
                aggregated_state[key] += update[key].detach().cpu() * (weight / denominator)

        return aggregated_state

    def _get_fisher_score(self, client_stats, layer_id, expert_id):
        # 从客户端统计信息中取出指定 layer、指定 expert 的 Fisher score。
        score = self._get_layer_expert_value(
            client_stats=client_stats,
            field_name="expert_fisher_score_by_layer",
            layer_id=layer_id,
            expert_id=expert_id,
        )

        # 缺失值视为 0，表示该客户端对该 expert 没有有效 Fisher 贡献。
        if score is None:
            return 0.0

        # Fisher 权重不允许为负，负数直接截断为 0。
        return max(score, 0.0)

    def _get_layer_expert_value(self, client_stats, field_name, layer_id, expert_id):
        # 只处理 dict 格式的客户端统计信息。
        if not isinstance(client_stats, dict):
            return None

        # 读取按 layer 存储的统计字段，例如 expert_fisher_score_by_layer。
        value_by_layer = client_stats.get(field_name, {})
        layer_values = value_by_layer.get(str(layer_id))
        if layer_values is None:
            return None

        if torch.is_tensor(layer_values):
            # 如果 layer_values 是 tensor，则先转 CPU 并拉平成一维。
            flat_values = layer_values.detach().cpu().flatten()
            if expert_id >= flat_values.numel():
                return None
            value = flat_values[expert_id].item()
        else:
            # 如果 layer_values 是 list / tuple，则直接用 expert_id 取值。
            if expert_id >= len(layer_values):
                return None
            value = layer_values[expert_id]

        # 转成 Python float，方便后续权重计算。
        value = float(value)

        # NaN / inf 这类非法数值不参与聚合。
        if not math.isfinite(value):
            return None

        return value


def build_aggregator(args):
    # 根据配置中的 agg_method 构造对应聚合器。
    if args.agg_method == "fedwolf_fisher_only":
        return FisherOnlyAggregator(args)
    if args.agg_method == "equal_avg":
        return EqualAvgAggregator()
    if args.agg_method == "expert_equal_avg":
        return ExpertEqualAvgAggregator()
    if args.agg_method == "expert_fedavg":
        return ExpertFedAvgAggregator()
    if args.agg_method == "fedavg":
        return FedAvgAggregator()

    # 如果配置里写了未注册的聚合方法，直接报错。
    raise ValueError(f"Unknown aggregation method: {args.agg_method}")