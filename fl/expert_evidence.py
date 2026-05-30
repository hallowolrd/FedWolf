import math

import torch
import torch.nn.functional as F


# Fisher score 模式别名表。
# 目的：兼容旧配置名 / 不同写法，统一映射到内部使用的标准模式。
FISHER_SCORE_MODE_ALIASES = {
    "mean_diag": "mean_diag",
    "mean_param_total_sample": "mean_diag",
    "mean_param_batch": "mean_diag",
    "mean_diag_active": "mean_diag_active",
    "trace_per_sample": "trace_per_sample",
    "sum_per_sample": "trace_per_sample",
    "sum_per_batch": "trace_per_sample",
    "trace_per_active_sample": "trace_per_active_sample",
    "sum_per_active_sample": "trace_per_active_sample",
    "sum_per_active_batch": "trace_per_active_sample",
    "trace_raw": "trace_raw",
    "sum_raw": "trace_raw",
}

# Fisher evidence 计算器别名表。
FISHER_ESTIMATOR_ALIASES = {
    "per_sample_backward": "per_sample_backward",
    "per_sample": "per_sample_backward",
    "linear_hook_token_fast": "linear_hook_token_fast",
    "token_fast": "linear_hook_token_fast",
    "linear_hook_sample_fast": "linear_hook_sample_fast",
    "sample_fast": "linear_hook_sample_fast",
}

# 内部真正支持的 Fisher score 标准模式。
FISHER_SCORE_MODES = (
    "mean_diag",
    "mean_diag_active",
    "trace_per_sample",
    "trace_per_active_sample",
    "trace_raw",
)

# 每种 Fisher score 模式对应的归一化方式说明。
# 主要用于 diagnostics 日志，方便确认当前 score 是怎么除的。
FISHER_SCORE_NORMALIZATION = {
    "mean_diag": "param_count_times_total_samples",
    "mean_diag_active": "param_count_times_active_samples",
    "trace_per_sample": "total_samples",
    "trace_per_active_sample": "active_samples",
    "trace_raw": "raw_grad_square_sum",
}


def parse_expert_param_ref(name):
    """ 解析 expert 参数名。
    目标参数名格式大致为：
        blocks.{layer}.ffn.experts.{expert_id}.*
    例如：
        blocks.1.ffn.experts.0.net.0.weight
    返回：
        (layer_id, expert_id)
        layer_id: str 类型，例如 "1"
        expert_id: int 类型，例如 0
    如果当前参数名不是 expert 参数，则返回 None。"""

    # 按 "." 拆开参数名，方便定位 blocks 和 experts。
    parts = name.split(".")

    # 如果参数名里没有 blocks 或 experts，就不是 MoE expert 参数。
    if "blocks" not in parts or "experts" not in parts:
        return None

    # 找到 blocks 和 experts 在参数名中的位置。
    blocks_idx = parts.index("blocks")
    experts_idx = parts.index("experts")

    # blocks 后面应该跟 layer id，experts 后面应该跟 expert id。
    if blocks_idx + 1 >= len(parts) or experts_idx + 1 >= len(parts):
        return None

    # layer id 和 expert id 必须是数字，否则说明参数名格式不符合预期。
    if not parts[blocks_idx + 1].isdigit() or not parts[experts_idx + 1].isdigit():
        return None

    # layer_id 用 str，便于后续作为字典 key；
    # expert_id 用 int，便于后续作为张量 / 列表下标。
    return str(parts[blocks_idx + 1]), int(parts[experts_idx + 1])


def _collect_expert_parameter_entries(model, num_experts):
    # 收集模型中所有 trainable expert 参数。
    # 返回：
    # - expert_entries: 按 layer_id 和 expert_id 组织的参数列表；
    # - param_counts: 每个 layer-expert 的参数量；
    # - matched_param_names: 成功匹配到的 expert 参数名列表。
    expert_entries = {}
    param_counts = {}
    matched_param_names = []

    for name, param in model.named_parameters():
        # 不需要梯度的参数不参与 Fisher 计算。
        if not param.requires_grad:
            continue

        # 判断当前参数是否属于某个 expert。
        expert_ref = parse_expert_param_ref(name)
        if expert_ref is None:
            continue

        layer_id, expert_id = expert_ref

        # 如果解析到的 expert_id 超出配置 num_experts，直接跳过。
        if expert_id >= num_experts:
            continue

        # 按 layer_id -> expert_id -> [(name, param), ...] 存储。
        expert_entries.setdefault(layer_id, {}).setdefault(expert_id, []).append((name, param))

        # 统计该 expert 的参数总数。
        param_counts[(layer_id, expert_id)] = param_counts.get((layer_id, expert_id), 0) + param.numel()

        # 保存匹配到的参数名，便于 diagnostics 检查匹配情况。
        matched_param_names.append(name)

    return expert_entries, param_counts, matched_param_names


def _collect_expert_parameters(model, num_experts):
    # 简化版 expert 参数收集函数。
    # 只返回纯参数列表，不返回参数名。
    expert_entries, param_counts, _ = _collect_expert_parameter_entries(model, num_experts)

    # 去掉 name，只保留 param。
    expert_params = {
        layer_id: {
            expert_id: [param for _, param in entries]
            for expert_id, entries in experts.items()
        }
        for layer_id, experts in expert_entries.items()
    }

    return expert_params, param_counts


def _scientific_list(values):
    # 将数值列表转成科学计数法字符串，方便日志中查看非常小的 Fisher score。
    return [f"{float(value):.12e}" for value in values]


def canonicalize_fisher_score_mode(score_mode):
    # 将用户配置的 fisher_score_mode 规范化成内部标准模式。
    # 如果 score_mode=None，则默认使用 mean_diag。
    raw_mode = "mean_diag" if score_mode is None else str(score_mode).strip().lower()

    # 根据别名表映射到标准模式。
    canonical_mode = FISHER_SCORE_MODE_ALIASES.get(raw_mode)

    # 如果配置不在支持范围内，直接报错，并列出所有可选项。
    if canonical_mode is None:
        supported = ", ".join(sorted(FISHER_SCORE_MODE_ALIASES))
        raise ValueError(
            f"fedwolf_fisher_score_mode must be one of: {supported}. "
            f"Got {score_mode!r}."
        )

    return canonical_mode


def canonicalize_fisher_estimator(estimator):
    # 将用户配置的 Fisher evidence 计算器规范化成内部标准取值。
    raw_estimator = "per_sample_backward" if estimator is None else str(estimator).strip().lower()
    canonical_estimator = FISHER_ESTIMATOR_ALIASES.get(raw_estimator)
    if canonical_estimator is None:
        supported = ", ".join(sorted(FISHER_ESTIMATOR_ALIASES))
        raise ValueError(
            f"fedwolf_fisher_estimator must be one of: {supported}. "
            f"Got {estimator!r}."
        )

    return canonical_estimator


def compute_fisher_scalar_from_sums(
    grad_square_sum,
    param_count,
    total_samples,
    num_samples_with_grad,
    mode,
):
    # 根据累计的梯度平方和，计算某个 expert 的标量 Fisher score。
    # 不同 mode 代表不同归一化方式。
    grad_square_sum = float(grad_square_sum)
    param_count = int(param_count)
    total_samples = int(total_samples)
    num_samples_with_grad = int(num_samples_with_grad)

    # 梯度平方和非法或非正时，该 expert 的 Fisher score 记为 0。
    if not math.isfinite(grad_square_sum) or grad_square_sum <= 0.0:
        return 0.0

    if mode == "mean_diag":
        # 平均到每个参数、每个总样本：
        # grad_square_sum / (参数量 * 总样本数)。
        if param_count <= 0 or total_samples <= 0:
            return 0.0
        value = grad_square_sum / float(param_count * total_samples)
    elif mode == "mean_diag_active":
        # 平均到每个参数、每个对该 expert 有梯度的样本：
        # grad_square_sum / (参数量 * active 样本数)。
        if param_count <= 0 or num_samples_with_grad <= 0:
            return 0.0
        value = grad_square_sum / float(param_count * num_samples_with_grad)
    elif mode == "trace_per_sample":
        # 不除以参数量，只除以总样本数。
        if total_samples <= 0:
            return 0.0
        value = grad_square_sum / float(total_samples)
    elif mode == "trace_per_active_sample":
        # 不除以参数量，只除以 active 样本数。
        if num_samples_with_grad <= 0:
            return 0.0
        value = grad_square_sum / float(num_samples_with_grad)
    elif mode == "trace_raw":
        # 不做归一化，直接使用原始梯度平方和。
        value = grad_square_sum
    else:
        # 理论上不会走到这里，除非传入了未规范化的非法 mode。
        raise ValueError(f"Unknown canonical Fisher score mode: {mode!r}.")

    # 归一化后的值仍然需要检查合法性。
    if not math.isfinite(value) or value <= 0.0:
        return 0.0

    return float(value)


def _reduce_loss_to_per_sample(per_element_losses, batch_size):
    # 将 criterion 返回的 unreduced loss 转成每个样本一个 loss。
    # 输出形状应该是 [batch_size]。
    if not torch.is_tensor(per_element_losses):
        raise TypeError("Per-sample Fisher requires a tensor loss output.")

    # 标量 loss 说明已经 reduction 过了，不适合逐样本 Fisher。
    if per_element_losses.ndim == 0:
        raise TypeError("Per-sample Fisher requires unreduced losses with a batch dimension.")

    # 第一维必须等于 batch_size。
    if per_element_losses.size(0) != batch_size:
        raise TypeError(
            "Per-sample Fisher expected unreduced losses whose first dimension "
            f"matches batch_size={batch_size}, got shape={tuple(per_element_losses.shape)}."
        )

    # 如果本来就是 [batch_size]，直接返回。
    if per_element_losses.ndim == 1:
        return per_element_losses

    # 如果 loss 还有额外维度，就把每个样本内部的 loss 平均成一个标量。
    return per_element_losses.reshape(batch_size, -1).mean(dim=1)


def _compute_per_sample_supervised_losses(criterion, outputs, labels):
    # 计算逐样本 supervised loss。
    # Fisher evidence 需要对每个样本单独 backward，所以不能使用 reduction='mean' 后的 batch loss。
    batch_size = labels.size(0)

    if isinstance(criterion, torch.nn.CrossEntropyLoss):
        # 对 CrossEntropyLoss 单独处理，避免直接修改 criterion 的 reduction。
        weight = criterion.weight
        if weight is not None:
            weight = weight.to(device=outputs.device, dtype=outputs.dtype)

        # reduction="none" 表示返回每个样本的 loss。
        per_element_losses = F.cross_entropy(
            outputs,
            labels,
            weight=weight,
            ignore_index=criterion.ignore_index,
            reduction="none",
            label_smoothing=getattr(criterion, "label_smoothing", 0.0),
        )

        return _reduce_loss_to_per_sample(per_element_losses, batch_size)

    # 对其他 criterion，要求它至少有 reduction 属性。
    if not hasattr(criterion, "reduction"):
        raise TypeError(
            "Per-sample Fisher currently supports torch.nn.CrossEntropyLoss, or a criterion "
            "that exposes a reduction attribute and can return reduction='none' losses."
        )

    # 尝试临时把 criterion.reduction 改成 "none"，计算逐样本 loss。
    original_reduction = criterion.reduction
    try:
        criterion.reduction = "none"
        per_element_losses = criterion(outputs, labels)
    except Exception as exc:
        raise TypeError(
            "Per-sample Fisher currently supports torch.nn.CrossEntropyLoss. "
            "Could not reliably compute reduction='none' losses for "
            f"{criterion.__class__.__name__}."
        ) from exc
    finally:
        # 无论是否成功，都恢复 criterion 原始 reduction，避免影响后续训练。
        criterion.reduction = original_reduction

    return _reduce_loss_to_per_sample(per_element_losses, batch_size)


def _collect_expert_linear_entries(model, num_experts):
    # 收集 expert 内部的 Linear 模块，用于 fast Fisher hook。
    linear_entries = []
    module_by_name = dict(model.named_modules())

    for name, module in module_by_name.items():
        if not isinstance(module, torch.nn.Linear):
            continue

        expert_ref = parse_expert_param_ref(name)
        if expert_ref is None:
            continue

        layer_id, expert_id = expert_ref
        if expert_id >= num_experts:
            continue

        parts = name.split(".")
        experts_idx = parts.index("experts")
        expert_module_name = ".".join(parts[: experts_idx + 2])
        expert_module = module_by_name[expert_module_name]
        block_prefix = ".".join(parts[experts_idx + 2:])

        linear_entries.append((name, module, layer_id, expert_id, expert_module, block_prefix))

    return linear_entries


def compute_expert_fisher_evidence(
    model,
    data_loader,
    criterion,
    device,
    num_experts,
    get_auxiliary_losses=None,
    return_diagnostics=False,
    model_mode="eval",
    score_mode="mean_diag",
    debug_batches=0,
    fisher_estimator="per_sample_backward",
):
    """Compute per-sample empirical diagonal Fisher scalar evidence for experts.

    For each expert parameter block Theta_k with d_k parameters, this computes:
        1 / (|D_m| * d_k) * sum_i ||grad_{Theta_k} loss_i||^2

    This pass only reads gradients from the trained local model. It does not
    update parameters or optimizer state.
    """

    # Fisher evidence 支持在 eval 或 train 模式下计算。
    # eval 更稳定，train 会保留 dropout 等训练行为。
    if model_mode not in {"eval", "train"}:
        raise ValueError(f"model_mode must be either 'eval' or 'train', got {model_mode!r}.")

    # 规范化 Fisher score 模式和 Fisher evidence 计算器。
    canonical_score_mode = canonicalize_fisher_score_mode(score_mode)
    canonical_fisher_estimator = canonicalize_fisher_estimator(fisher_estimator)

    # debug_batches 小于 0 时按 0 处理。
    debug_batches = max(int(debug_batches), 0)

    # 收集所有 expert 参数、每个 expert 的参数量，以及匹配到的参数名。
    expert_entries, param_counts, matched_param_names = _collect_expert_parameter_entries(
        model=model,
        num_experts=num_experts,
    )

    # diagnostics 保存 Fisher 计算过程中的诊断信息，方便排查 Fisher score 是否有效。
    diagnostics = {
        "matched_param_name_count": len(matched_param_names),
        "matched_param_count_by_layer": {
            str(layer_id): {
                str(expert_id): len(entries)
                for expert_id, entries in experts.items()
            }
            for layer_id, experts in expert_entries.items()
        },
        "param_count_by_layer": {},
        "batch_grad_status": [],
        "grad_square_sum_by_layer": {},
        "num_batches_with_grad_by_layer": "deprecated; use num_samples_with_grad_by_layer",
        "num_samples_with_grad_by_layer": {},
        "score_scientific_by_layer": {},
        "total_samples": 0,
        "num_batches": 0,
        "fisher_estimator": canonical_fisher_estimator,
        "fisher_estimator_raw": fisher_estimator,
        "fisher_estimator_impl": canonical_fisher_estimator,
        "fisher_score_mode": canonical_score_mode,
        "fisher_score_mode_raw": score_mode,
        "normalization": FISHER_SCORE_NORMALIZATION[canonical_score_mode],
        "model_mode": model_mode,
        "debug_batches": int(debug_batches),
        # Expert Fisher evidence 只基于逐样本 supervised CE loss。
        # router 辅助损失是 batch-level 标量，不是逐样本 loss，加入这里会重复计入。
        "fisher_loss_source": "supervised_cross_entropy_only",
        "auxiliary_loss_used_for_fisher": False,
        "auxiliary_loss_note": (
            "Skipped batch-level auxiliary losses for per-sample Fisher because they are not "
            "per-sample losses."
        ),
        "zero_score_reason": None,
        "score_mean_diag_by_layer": {},
        "score_mean_diag_active_by_layer": {},
        "score_trace_per_sample_by_layer": {},
        "score_trace_per_active_sample_by_layer": {},
        "score_trace_raw_by_layer": {},
        "fast_fisher_active_token_count_by_layer": {},
        "fast_fisher_active_sample_count_by_layer": {},
    }

    # 如果一个 expert 参数都没匹配到，说明模型结构或参数命名不符合预期。
    if not expert_entries:
        diagnostics["zero_score_reason"] = "No trainable expert parameters matched blocks.*.ffn.experts.* names."
        if return_diagnostics:
            return {}, {}, diagnostics
        return {}, {}

    # score_sums 保存每层每个 expert 的逐样本梯度平方和累计值。
    score_sums = {
        layer_id: torch.zeros(num_experts, dtype=torch.float64)
        for layer_id in expert_entries
    }

    # samples_with_grad 保存每层每个 expert 有非 None 梯度的样本数。
    samples_with_grad = {
        layer_id: torch.zeros(num_experts, dtype=torch.long)
        for layer_id in expert_entries
    }

    # 记录模型原本是 train 还是 eval，计算结束后要恢复。
    was_training = model.training

    # 根据配置切换模型模式。
    if model_mode == "eval":
        model.eval()
    else:
        model.train()

    # 统计实际计算 Fisher 的 batch 数和样本数。
    num_batches = 0
    total_samples = 0

    hook_handles = []
    activation_cache = {}
    fast_batch_token_counts = {}
    fast_sample_counted_experts = set()
    fast_active_token_counts = {
        layer_id: torch.zeros(num_experts, dtype=torch.long)
        for layer_id in expert_entries
    }
    fast_active_sample_counts = {
        layer_id: torch.zeros(num_experts, dtype=torch.long)
        for layer_id in expert_entries
    }

    try:
        if canonical_fisher_estimator in {"linear_hook_token_fast", "linear_hook_sample_fast"}:
            linear_entries = _collect_expert_linear_entries(model, num_experts)
            diagnostics["fast_fisher_hooked_linear_count"] = len(linear_entries)
            diagnostics["fast_fisher_hooked_linear_names"] = [
                name for name, _, _, _, _, _ in linear_entries
            ]
            experts_with_net0 = {
                (layer_id, expert_id)
                for _, _, layer_id, expert_id, _, block_prefix in linear_entries
                if block_prefix == "net.0"
            }

            def make_forward_hook(module_name, layer_id, expert_id):
                def forward_hook(module, inputs, output):
                    if not inputs or inputs[0] is None:
                        return

                    activation = inputs[0].detach()
                    if activation.numel() == 0:
                        return

                    activation = activation.reshape(-1, activation.shape[-1])
                    activation_cache.setdefault(module_name, []).append(activation)

                    expert_key = (layer_id, expert_id)
                    token_count = int(activation.size(0))
                    fast_batch_token_counts[expert_key] = max(
                        fast_batch_token_counts.get(expert_key, 0),
                        token_count,
                    )

                return forward_hook

            def make_backward_hook(module_name, layer_id, expert_id, expert_module, block_prefix):
                def backward_hook(module, grad_input, grad_output):
                    if not grad_output or grad_output[0] is None:
                        return

                    cached_activations = activation_cache.get(module_name)
                    if not cached_activations:
                        return

                    activation = cached_activations.pop()
                    delta = grad_output[0].detach()
                    if delta.numel() == 0:
                        return

                    delta = delta.reshape(-1, delta.shape[-1])
                    if delta.size(0) != activation.size(0):
                        raise RuntimeError(
                            f"{canonical_fisher_estimator} expected activation and grad_output to have "
                            f"the same token dimension for {module_name}, got "
                            f"{activation.size(0)} and {delta.size(0)}."
                        )

                    activation = activation.to(device=delta.device, dtype=delta.dtype)
                    with torch.no_grad():
                        if canonical_fisher_estimator == "linear_hook_token_fast":
                            weight_grad_square_sum = torch.einsum(
                                "to,ti->",
                                delta.pow(2),
                                activation.pow(2),
                            )
                            grad_square_sum = float(weight_grad_square_sum.detach().cpu())

                            if module.bias is not None:
                                grad_square_sum += float(delta.pow(2).sum().detach().cpu())

                            score_sums[layer_id][expert_id] += grad_square_sum
                            return

                        sample_ids = getattr(expert_module, "_fedwolf_accepted_sample_ids", None)
                        if sample_ids is None:
                            raise ValueError(
                                "linear_hook_sample_fast requires expert._fedwolf_accepted_sample_ids. "
                                "Please expose accepted sample ids in TokenSwitchFFN.forward before using this estimator."
                            )

                        sample_ids = sample_ids.detach().to(device=delta.device).reshape(-1)
                        if sample_ids.numel() != delta.size(0):
                            raise RuntimeError(
                                "linear_hook_sample_fast expected sample_ids to match token dimension for "
                                f"{module_name}, got {sample_ids.numel()} and {delta.size(0)}."
                            )

                        grad_square_sum = 0.0
                        unique_sample_ids = torch.unique(sample_ids, sorted=False)
                        for sample_id in unique_sample_ids:
                            sample_mask = sample_ids == sample_id
                            delta_s = delta[sample_mask]
                            activation_s = activation[sample_mask]
                            sample_grad_w = torch.einsum("to,ti->oi", delta_s, activation_s)
                            grad_square_sum += float(sample_grad_w.pow(2).sum().detach().cpu())
                            if module.bias is not None:
                                sample_grad_b = delta_s.sum(dim=0)
                                grad_square_sum += float(sample_grad_b.pow(2).sum().detach().cpu())

                        score_sums[layer_id][expert_id] += grad_square_sum

                        expert_key = (layer_id, expert_id)
                        if expert_key in experts_with_net0:
                            should_count = block_prefix == "net.0"
                        else:
                            should_count = expert_key not in fast_sample_counted_experts

                        if should_count:
                            fast_sample_counted_experts.add(expert_key)
                            token_count = int(sample_ids.numel())
                            sample_count = int(unique_sample_ids.numel())
                            samples_with_grad[layer_id][expert_id] += sample_count
                            fast_active_token_counts[layer_id][expert_id] += token_count
                            fast_active_sample_counts[layer_id][expert_id] += sample_count

                return backward_hook

            for module_name, module, layer_id, expert_id, expert_module, block_prefix in linear_entries:
                hook_handles.append(
                    module.register_forward_hook(
                        make_forward_hook(module_name, layer_id, expert_id)
                    )
                )
                hook_handles.append(
                    module.register_full_backward_hook(
                        make_backward_hook(module_name, layer_id, expert_id, expert_module, block_prefix)
                    )
                )

        for inputs, labels in data_loader:
            # 将数据移动到指定设备。
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 清空旧梯度，避免和之前训练或上一个 batch 的梯度混在一起。
            model.zero_grad(set_to_none=True)
            activation_cache.clear()
            fast_batch_token_counts.clear()
            fast_sample_counted_experts.clear()

            # 前向传播。
            result = model(inputs)

            # 兼容模型返回 dict 或直接返回 logits 两种情况。
            outputs = result["logits"] if isinstance(result, dict) else result

            # 计算逐样本 supervised loss。
            per_sample_losses = _compute_per_sample_supervised_losses(criterion, outputs, labels)

            batch_size = labels.size(0)
            num_batches += 1
            total_samples += batch_size

            if canonical_fisher_estimator == "per_sample_backward":
                # Important:
                # We intentionally compute grad(loss_i)^2 for each sample and then average.
                # This is different from grad(mean_i loss_i)^2, which underestimates Fisher
                # because gradients from different samples can cancel before squaring.
                # 这里逐样本 backward，得到 grad(loss_i)^2 后再累计。
                # 不能先对 batch loss 求平均再 backward，否则不同样本的梯度可能互相抵消。
                for sample_idx in range(batch_size):
                    # 每个样本单独 backward 前都要清空梯度。
                    model.zero_grad(set_to_none=True)

                    # 除最后一个样本外，都需要保留计算图，供后续样本继续 backward。
                    retain_graph = sample_idx < batch_size - 1

                    # 对单个样本 loss 做反向传播。
                    per_sample_losses[sample_idx].backward(retain_graph=retain_graph)

                    # 遍历每层每个 expert，累计该 expert 参数的梯度平方和。
                    for layer_id, experts in expert_entries.items():
                        for expert_id, entries in experts.items():
                            grad_square_sum = 0.0
                            has_grad_param_count = 0
                            none_grad_param_count = 0

                            # 当前 expert 可能包含多个参数张量，例如 Linear weight / bias。
                            for _, param in entries:
                                if param.grad is not None:
                                    # 累加当前参数张量的 grad^2 sum。
                                    grad_square_sum += float(param.grad.detach().pow(2).sum().cpu())
                                    has_grad_param_count += 1
                                else:
                                    # 记录没有梯度的参数数量，目前主要用于调试时理解代码。
                                    none_grad_param_count += 1

                            # 只要当前 expert 中至少一个参数有梯度，就认为该样本激活了该 expert 的梯度。
                            if has_grad_param_count > 0:
                                samples_with_grad[layer_id][expert_id] += 1

                            # 累加当前样本对该 expert 的梯度平方和。
                            score_sums[layer_id][expert_id] += grad_square_sum

                # 只在前 debug_batches 个 batch 中记录简要调试信息。
                if num_batches <= debug_batches:
                    diagnostics["batch_grad_status"].append(
                        {
                            "batch_index": num_batches,
                            "batch_size": int(batch_size),
                            "sample_count": int(batch_size),
                        }
                    )
            elif canonical_fisher_estimator in {"linear_hook_token_fast", "linear_hook_sample_fast"}:
                # fast Fisher 一个 batch 只 backward 一次，hook 内累计 Linear Fisher。
                per_sample_losses.sum().backward()

                if canonical_fisher_estimator == "linear_hook_token_fast":
                    for (layer_id, expert_id), token_count in fast_batch_token_counts.items():
                        samples_with_grad[layer_id][expert_id] += int(token_count)

                if num_batches <= debug_batches:
                    diagnostics["batch_grad_status"].append(
                        {
                            "batch_index": num_batches,
                            "batch_size": int(batch_size),
                            "sample_count": int(batch_size),
                        }
                    )

                activation_cache.clear()
                fast_batch_token_counts.clear()
                fast_sample_counted_experts.clear()
                model.zero_grad(set_to_none=True)
            else:
                raise ValueError(f"Unsupported Fisher estimator: {canonical_fisher_estimator!r}.")
    finally:
        for handle in hook_handles:
            handle.remove()
        activation_cache.clear()
        fast_batch_token_counts.clear()
        fast_sample_counted_experts.clear()

        # 无论中间是否报错，都清空梯度，避免污染后续训练或评估。
        model.zero_grad(set_to_none=True)

        # 恢复模型进入 Fisher 计算前的 train/eval 状态。
        if was_training:
            model.train()
        else:
            model.eval()

    # 写入总样本数和 batch 数。
    diagnostics["total_samples"] = int(total_samples)
    diagnostics["num_batches"] = int(num_batches)
    if canonical_fisher_estimator == "linear_hook_sample_fast":
        diagnostics["fast_fisher_active_token_count_by_layer"] = {
            str(layer_id): [int(value) for value in counts.tolist()]
            for layer_id, counts in fast_active_token_counts.items()
        }
        diagnostics["fast_fisher_active_sample_count_by_layer"] = {
            str(layer_id): [int(value) for value in counts.tolist()]
            for layer_id, counts in fast_active_sample_counts.items()
        }

    # 最终返回的 Fisher score。
    score_by_layer = {}

    # log1p 后的 Fisher score，主要用于日志查看或数值压缩。
    log_score_by_layer = {}

    for layer_id, scores in score_sums.items():
        # 同时计算所有标准模式的 score，方便 diagnostics 对比。
        mode_scores_by_layer = {
            mode: torch.zeros(num_experts, dtype=torch.float64)
            for mode in FISHER_SCORE_MODES
        }

        # layer_scores 保存当前配置 canonical_score_mode 对应的最终 score。
        layer_scores = torch.zeros(num_experts, dtype=torch.float64)

        for expert_id in range(num_experts):
            # 当前 layer-expert 的参数量。
            param_count = param_counts.get((layer_id, expert_id), 0)

            # 当前 layer-expert 有梯度的样本数。
            num_samples_with_grad = int(samples_with_grad[layer_id][expert_id].item())

            # 对每种 Fisher score 模式都计算一遍。
            for mode in FISHER_SCORE_MODES:
                mode_scores_by_layer[mode][expert_id] = compute_fisher_scalar_from_sums(
                    grad_square_sum=scores[expert_id].item(),
                    param_count=param_count,
                    total_samples=total_samples,
                    num_samples_with_grad=num_samples_with_grad,
                    mode=mode,
                )

            # 当前真正用于聚合的是 canonical_score_mode 对应的 score。
            layer_scores[expert_id] = mode_scores_by_layer[canonical_score_mode][expert_id]

        # 清理 NaN / inf，并保证 Fisher score 非负。
        layer_scores = torch.nan_to_num(layer_scores, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

        # 保存当前层最终 Fisher score。
        score_by_layer[str(layer_id)] = layer_scores.cpu()

        # 保存 log1p 版本 score。
        log_score_by_layer[str(layer_id)] = torch.log1p(layer_scores).cpu()

        # 记录当前层每个 expert 的参数量。
        diagnostics["param_count_by_layer"][str(layer_id)] = {
            str(expert_id): int(param_counts.get((layer_id, expert_id), 0))
            for expert_id in range(num_experts)
        }

        # 记录当前层每个 expert 的原始梯度平方和。
        diagnostics["grad_square_sum_by_layer"][str(layer_id)] = _scientific_list(scores.tolist())

        # 记录当前层每个 expert 有梯度的样本数。
        diagnostics["num_samples_with_grad_by_layer"][str(layer_id)] = [
            int(value) for value in samples_with_grad[layer_id].tolist()
        ]

        # 记录当前配置实际使用的 Fisher score。
        diagnostics["score_scientific_by_layer"][str(layer_id)] = _scientific_list(layer_scores.tolist())

        # 以下 diagnostics 同时保存不同归一化方式下的 score，方便后续分析哪个模式更合理。
        diagnostics["score_mean_diag_by_layer"][str(layer_id)] = _scientific_list(
            mode_scores_by_layer["mean_diag"].tolist()
        )
        diagnostics["score_mean_diag_active_by_layer"][str(layer_id)] = _scientific_list(
            mode_scores_by_layer["mean_diag_active"].tolist()
        )
        diagnostics["score_trace_per_sample_by_layer"][str(layer_id)] = _scientific_list(
            mode_scores_by_layer["trace_per_sample"].tolist()
        )
        diagnostics["score_trace_per_active_sample_by_layer"][str(layer_id)] = _scientific_list(
            mode_scores_by_layer["trace_per_active_sample"].tolist()
        )
        diagnostics["score_trace_raw_by_layer"][str(layer_id)] = _scientific_list(
            mode_scores_by_layer["trace_raw"].tolist()
        )

    # 汇总所有层所有 expert 的 score，用于判断是否全部为 0。
    all_scores = [
        float(value)
        for scores in score_by_layer.values()
        for value in scores.tolist()
    ]

    # 如果所有 Fisher score 都是 0，补充 zero_score_reason，帮助定位原因。
    if all_scores and all(score == 0.0 for score in all_scores):
        total_samples_with_grad = sum(
            sum(layer_counts)
            for layer_counts in diagnostics["num_samples_with_grad_by_layer"].values()
        )
        total_grad_square_sum = sum(
            float(value)
            for layer_scores in diagnostics["grad_square_sum_by_layer"].values()
            for value in layer_scores
        )

        if total_samples_with_grad == 0:
            diagnostics["zero_score_reason"] = (
                "All matched expert parameters had grad=None for every sample after per-sample backward. "
                "Experts may be disconnected from the loss or no samples reached them."
            )
        elif total_grad_square_sum == 0.0:
            diagnostics["zero_score_reason"] = (
                "Expert gradients existed for some samples, but every per-sample expert "
                "grad_square_sum was exactly 0."
            )
        else:
            diagnostics["zero_score_reason"] = (
                "Per-sample grad_square_sum was non-zero before normalization, but final scores became 0."
            )

    # 如果调用方需要 diagnostics，就一起返回。
    if return_diagnostics:
        return score_by_layer, log_score_by_layer, diagnostics

    # 默认只返回 Fisher score 和 log Fisher score。
    return score_by_layer, log_score_by_layer