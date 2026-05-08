import math

import torch
import torch.nn.functional as F

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

FISHER_SCORE_MODES = (
    "mean_diag",
    "mean_diag_active",
    "trace_per_sample",
    "trace_per_active_sample",
    "trace_raw",
)

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

    parts = name.split(".")
    if "blocks" not in parts or "experts" not in parts:
        return None

    blocks_idx = parts.index("blocks")
    experts_idx = parts.index("experts")
    if blocks_idx + 1 >= len(parts) or experts_idx + 1 >= len(parts):
        return None
    if not parts[blocks_idx + 1].isdigit() or not parts[experts_idx + 1].isdigit():
        return None

    return str(parts[blocks_idx + 1]), int(parts[experts_idx + 1])


def _collect_expert_parameter_entries(model, num_experts):
    expert_entries = {}
    param_counts = {}
    matched_param_names = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        expert_ref = parse_expert_param_ref(name)
        if expert_ref is None:
            continue

        layer_id, expert_id = expert_ref
        if expert_id >= num_experts:
            continue

        expert_entries.setdefault(layer_id, {}).setdefault(expert_id, []).append((name, param))
        param_counts[(layer_id, expert_id)] = param_counts.get((layer_id, expert_id), 0) + param.numel()
        matched_param_names.append(name)

    return expert_entries, param_counts, matched_param_names


def _collect_expert_parameters(model, num_experts):
    expert_entries, param_counts, _ = _collect_expert_parameter_entries(model, num_experts)
    expert_params = {
        layer_id: {
            expert_id: [param for _, param in entries]
            for expert_id, entries in experts.items()
        }
        for layer_id, experts in expert_entries.items()
    }
    return expert_params, param_counts


def _scientific_list(values):
    return [f"{float(value):.12e}" for value in values]


def canonicalize_fisher_score_mode(score_mode):
    raw_mode = "mean_diag" if score_mode is None else str(score_mode).strip().lower()
    canonical_mode = FISHER_SCORE_MODE_ALIASES.get(raw_mode)
    if canonical_mode is None:
        supported = ", ".join(sorted(FISHER_SCORE_MODE_ALIASES))
        raise ValueError(
            f"fedwolf_fisher_score_mode must be one of: {supported}. "
            f"Got {score_mode!r}."
        )
    return canonical_mode


def parse_optional_positive_int_limit(value, field_name):
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "none", "null"}:
            return None
        try:
            value = int(normalized)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a positive integer, 0/null/None, got {value!r}.") from exc
    elif isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer, 0/null/None, got {value!r}.")
    else:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be a positive integer, 0/null/None, got {value!r}.") from exc
        if not math.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(f"{field_name} must be a positive integer, 0/null/None, got {value!r}.")
        value = int(numeric_value)

    if value <= 0:
        return None
    return int(value)


def compute_fisher_scalar_from_sums(
    grad_square_sum,
    param_count,
    total_samples,
    num_samples_with_grad,
    mode,
):
    grad_square_sum = float(grad_square_sum)
    param_count = int(param_count)
    total_samples = int(total_samples)
    num_samples_with_grad = int(num_samples_with_grad)

    if not math.isfinite(grad_square_sum) or grad_square_sum <= 0.0:
        return 0.0

    if mode == "mean_diag":
        if param_count <= 0 or total_samples <= 0:
            return 0.0
        value = grad_square_sum / float(param_count * total_samples)
    elif mode == "mean_diag_active":
        if param_count <= 0 or num_samples_with_grad <= 0:
            return 0.0
        value = grad_square_sum / float(param_count * num_samples_with_grad)
    elif mode == "trace_per_sample":
        if total_samples <= 0:
            return 0.0
        value = grad_square_sum / float(total_samples)
    elif mode == "trace_per_active_sample":
        if num_samples_with_grad <= 0:
            return 0.0
        value = grad_square_sum / float(num_samples_with_grad)
    elif mode == "trace_raw":
        value = grad_square_sum
    else:
        raise ValueError(f"Unknown canonical Fisher score mode: {mode!r}.")

    if not math.isfinite(value) or value <= 0.0:
        return 0.0
    return float(value)


def _reduce_loss_to_per_sample(per_element_losses, batch_size):
    if not torch.is_tensor(per_element_losses):
        raise TypeError("Per-sample Fisher requires a tensor loss output.")
    if per_element_losses.ndim == 0:
        raise TypeError("Per-sample Fisher requires unreduced losses with a batch dimension.")
    if per_element_losses.size(0) != batch_size:
        raise TypeError(
            "Per-sample Fisher expected unreduced losses whose first dimension "
            f"matches batch_size={batch_size}, got shape={tuple(per_element_losses.shape)}."
        )
    if per_element_losses.ndim == 1:
        return per_element_losses
    return per_element_losses.reshape(batch_size, -1).mean(dim=1)


def _compute_per_sample_supervised_losses(criterion, outputs, labels):
    batch_size = labels.size(0)

    if isinstance(criterion, torch.nn.CrossEntropyLoss):
        weight = criterion.weight
        if weight is not None:
            weight = weight.to(device=outputs.device, dtype=outputs.dtype)
        per_element_losses = F.cross_entropy(
            outputs,
            labels,
            weight=weight,
            ignore_index=criterion.ignore_index,
            reduction="none",
            label_smoothing=getattr(criterion, "label_smoothing", 0.0),
        )
        return _reduce_loss_to_per_sample(per_element_losses, batch_size)

    if not hasattr(criterion, "reduction"):
        raise TypeError(
            "Per-sample Fisher currently supports torch.nn.CrossEntropyLoss, or a criterion "
            "that exposes a reduction attribute and can return reduction='none' losses."
        )

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
        criterion.reduction = original_reduction

    return _reduce_loss_to_per_sample(per_element_losses, batch_size)


def _is_cuda_device(device):
    return str(torch.device(device)).startswith("cuda")


def _move_batch_to_device(inputs, labels, device, pin_memory=False):
    non_blocking = bool(pin_memory) and _is_cuda_device(device)
    return (
        inputs.to(device, non_blocking=non_blocking),
        labels.to(device, non_blocking=non_blocking),
    )


def _zeros_for_experts(num_experts, dtype=torch.float64, device="cpu"):
    return torch.zeros(num_experts, dtype=dtype, device=device)


def _as_expert_vector(value, num_experts, device, dtype=torch.float64):
    vector = _zeros_for_experts(num_experts=num_experts, dtype=dtype, device=device)
    if value is None:
        return vector

    if torch.is_tensor(value):
        flat_value = value.detach().to(device=device, dtype=dtype).flatten()
    else:
        try:
            flat_value = torch.as_tensor(value, dtype=dtype, device=device).flatten()
        except (TypeError, ValueError):
            return vector

    usable_size = min(num_experts, flat_value.numel())
    if usable_size > 0:
        vector[:usable_size] = flat_value[:usable_size]
    return torch.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)


def _get_result_layer_stats(result):
    if not isinstance(result, dict):
        return {}

    layer_stats = result.get("expert_stats_by_layer")
    if isinstance(layer_stats, dict):
        return layer_stats

    activations_by_layer = result.get("expert_activations_by_layer")
    if isinstance(activations_by_layer, dict):
        return {
            layer_id: {"expert_activations": activations}
            for layer_id, activations in activations_by_layer.items()
        }

    return {}


def _accumulate_evidence_expert_stats(total_stats, result, num_experts, device):
    """Accumulate expert token usage observed during the evidence forward pass."""

    for layer_id, stats in _get_result_layer_stats(result).items():
        layer_key = str(layer_id)
        if not isinstance(stats, dict):
            stats = {"expert_activations": stats}

        if layer_key not in total_stats:
            total_stats[layer_key] = {
                "expert_activations": _zeros_for_experts(num_experts, device=device),
                "selected_counts": _zeros_for_experts(num_experts, device=device),
                "overflow_counts": _zeros_for_experts(num_experts, device=device),
                "avg_router_probs_sum": _zeros_for_experts(num_experts, device=device),
                "avg_router_probs_batches": 0,
                "capacity": 0,
            }

        layer_total = total_stats[layer_key]
        layer_total["expert_activations"] += _as_expert_vector(
            stats.get("expert_activations"),
            num_experts=num_experts,
            device=device,
        )
        layer_total["selected_counts"] += _as_expert_vector(
            stats.get("selected_counts"),
            num_experts=num_experts,
            device=device,
        )
        layer_total["overflow_counts"] += _as_expert_vector(
            stats.get("overflow_counts"),
            num_experts=num_experts,
            device=device,
        )

        avg_router_probs = stats.get("avg_router_probs")
        if avg_router_probs is not None:
            layer_total["avg_router_probs_sum"] += _as_expert_vector(
                avg_router_probs,
                num_experts=num_experts,
                device=device,
            )
            layer_total["avg_router_probs_batches"] += 1

        capacity = stats.get("capacity", layer_total["capacity"])
        if torch.is_tensor(capacity):
            capacity = capacity.detach().cpu().item()
        try:
            layer_total["capacity"] = max(int(capacity), int(layer_total["capacity"]))
        except (TypeError, ValueError):
            pass


def _finalize_evidence_expert_stats(total_stats):
    stats_by_layer = {}
    activations_by_layer = {}
    selected_counts_by_layer = {}
    overflow_counts_by_layer = {}

    for layer_id, stats in total_stats.items():
        avg_router_probs_batches = max(int(stats["avg_router_probs_batches"]), 1)
        avg_router_probs = stats["avg_router_probs_sum"] / avg_router_probs_batches

        expert_activations = [int(value) for value in stats["expert_activations"].detach().cpu().tolist()]
        selected_counts = [int(value) for value in stats["selected_counts"].detach().cpu().tolist()]
        overflow_counts = [int(value) for value in stats["overflow_counts"].detach().cpu().tolist()]

        stats_by_layer[str(layer_id)] = {
            "expert_activations": expert_activations,
            "selected_counts": selected_counts,
            "overflow_counts": overflow_counts,
            "avg_router_probs": _scientific_list(avg_router_probs.detach().cpu().tolist()),
            "capacity": int(stats["capacity"]),
        }
        activations_by_layer[str(layer_id)] = expert_activations
        selected_counts_by_layer[str(layer_id)] = selected_counts
        overflow_counts_by_layer[str(layer_id)] = overflow_counts

    return (
        stats_by_layer,
        activations_by_layer,
        selected_counts_by_layer,
        overflow_counts_by_layer,
    )


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
    max_samples=None,
    max_batches=None,
    pin_memory=False,
):
    """Compute per-sample empirical diagonal Fisher scalar evidence for experts.

    For each expert parameter block Theta_k with d_k parameters, this computes:
        1 / (|D_m| * d_k) * sum_i ||grad_{Theta_k} loss_i||^2

    This pass only reads gradients from the trained local model. It does not
    update parameters or optimizer state.
    """

    if model_mode not in {"eval", "train"}:
        raise ValueError(f"model_mode must be either 'eval' or 'train', got {model_mode!r}.")

    canonical_score_mode = canonicalize_fisher_score_mode(score_mode)
    debug_batches = max(int(debug_batches), 0)
    max_samples = parse_optional_positive_int_limit(max_samples, "fedwolf_fisher_max_samples")
    max_batches = parse_optional_positive_int_limit(max_batches, "fedwolf_fisher_max_batches")

    expert_entries, param_counts, matched_param_names = _collect_expert_parameter_entries(
        model=model,
        num_experts=num_experts,
    )

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
        "fisher_estimator": "per_sample_empirical_diagonal_fisher",
        "fisher_score_mode": canonical_score_mode,
        "fisher_score_mode_raw": score_mode,
        "normalization": FISHER_SCORE_NORMALIZATION[canonical_score_mode],
        "model_mode": model_mode,
        "debug_batches": int(debug_batches),
        "max_samples": max_samples,
        "max_batches": max_batches,
        "pin_memory": bool(pin_memory),
        "non_blocking_transfer": bool(pin_memory) and _is_cuda_device(device),
        "limit_reached": False,
        "stop_reason": None,
        "effective_total_samples": 0,
        "effective_num_batches": 0,
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
        "evidence_expert_stats_by_layer": {},
        "evidence_expert_activations_by_layer": {},
        "evidence_selected_counts_by_layer": {},
        "evidence_overflow_counts_by_layer": {},
    }

    if not expert_entries:
        diagnostics["zero_score_reason"] = "No trainable expert parameters matched blocks.*.ffn.experts.* names."
        if return_diagnostics:
            return {}, {}, diagnostics
        return {}, {}

    score_sums = {
        layer_id: torch.zeros(num_experts, dtype=torch.float64, device=device)
        for layer_id in expert_entries
    }
    samples_with_grad = {
        layer_id: torch.zeros(num_experts, dtype=torch.long, device=device)
        for layer_id in expert_entries
    }

    was_training = model.training
    if model_mode == "eval":
        model.eval()
    else:
        model.train()
    num_batches = 0
    total_samples = 0
    limit_reached = False
    stop_reason = None
    evidence_expert_stats_by_layer = {}

    try:
        for inputs, labels in data_loader:
            if max_batches is not None and num_batches >= max_batches:
                limit_reached = True
                stop_reason = "max_batches"
                break
            if max_samples is not None and total_samples >= max_samples:
                limit_reached = True
                stop_reason = "max_samples"
                break

            if max_samples is not None:
                remaining = max_samples - total_samples
                if remaining <= 0:
                    limit_reached = True
                    stop_reason = "max_samples"
                    break
                if labels.size(0) > remaining:
                    inputs = inputs[:remaining]
                    labels = labels[:remaining]

            if labels.size(0) <= 0:
                continue

            inputs, labels = _move_batch_to_device(
                inputs=inputs,
                labels=labels,
                device=device,
                pin_memory=pin_memory,
            )

            model.zero_grad(set_to_none=True)
            result = model(inputs)
            _accumulate_evidence_expert_stats(
                total_stats=evidence_expert_stats_by_layer,
                result=result,
                num_experts=num_experts,
                device=device,
            )
            outputs = result["logits"] if isinstance(result, dict) else result

            per_sample_losses = _compute_per_sample_supervised_losses(criterion, outputs, labels)
            batch_size = labels.size(0)
            num_batches += 1
            total_samples += batch_size

            # 重要：
            # 这里仍然逐样本 backward 并累计 grad(loss_i)^2，然后再平均。
            # 这不同于 grad(mean_i loss_i)^2；后者会先让不同样本梯度相互抵消，再平方。
            # 梯度平方和保留在 evidence device 上累计，避免内层循环里的 .cpu()/.item()/float(tensor)
            # 触发 GPU 同步；只有最后构造 CPU 返回值和 diagnostics 时才转 CPU。
            for sample_idx in range(batch_size):
                model.zero_grad(set_to_none=True)
                retain_graph = sample_idx < batch_size - 1
                per_sample_losses[sample_idx].backward(retain_graph=retain_graph)

                for layer_id, experts in expert_entries.items():
                    for expert_id, entries in experts.items():
                        grad_square_sum = None
                        has_grad_param_count = 0
                        none_grad_param_count = 0
                        for _, param in entries:
                            if param.grad is not None:
                                value = param.grad.detach().pow(2).sum().to(dtype=torch.float64)
                                grad_square_sum = value if grad_square_sum is None else grad_square_sum + value
                                has_grad_param_count += 1
                            else:
                                none_grad_param_count += 1

                        if has_grad_param_count > 0:
                            samples_with_grad[layer_id][expert_id] += 1
                            score_sums[layer_id][expert_id] += grad_square_sum

            if num_batches <= debug_batches:
                diagnostics["batch_grad_status"].append(
                    {
                        "batch_index": num_batches,
                        "batch_size": int(batch_size),
                        "sample_count": int(batch_size),
                    }
                )

            if max_samples is not None and total_samples >= max_samples:
                limit_reached = True
                stop_reason = "max_samples"
                break
            if max_batches is not None and num_batches >= max_batches:
                limit_reached = True
                stop_reason = "max_batches"
                break
    finally:
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()
        else:
            model.eval()

    diagnostics["total_samples"] = int(total_samples)
    diagnostics["num_batches"] = int(num_batches)
    diagnostics["effective_total_samples"] = int(total_samples)
    diagnostics["effective_num_batches"] = int(num_batches)
    diagnostics["limit_reached"] = bool(limit_reached)
    diagnostics["stop_reason"] = stop_reason
    (
        diagnostics["evidence_expert_stats_by_layer"],
        diagnostics["evidence_expert_activations_by_layer"],
        diagnostics["evidence_selected_counts_by_layer"],
        diagnostics["evidence_overflow_counts_by_layer"],
    ) = _finalize_evidence_expert_stats(evidence_expert_stats_by_layer)

    score_by_layer = {}
    log_score_by_layer = {}

    for layer_id, scores in score_sums.items():
        scores_cpu = scores.detach().cpu()
        samples_with_grad_cpu = samples_with_grad[layer_id].detach().cpu()
        mode_scores_by_layer = {
            mode: torch.zeros(num_experts, dtype=torch.float64)
            for mode in FISHER_SCORE_MODES
        }
        layer_scores = torch.zeros(num_experts, dtype=torch.float64)
        for expert_id in range(num_experts):
            param_count = param_counts.get((layer_id, expert_id), 0)
            num_samples_with_grad = int(samples_with_grad_cpu[expert_id].item())
            for mode in FISHER_SCORE_MODES:
                mode_scores_by_layer[mode][expert_id] = compute_fisher_scalar_from_sums(
                    grad_square_sum=scores_cpu[expert_id].item(),
                    param_count=param_count,
                    total_samples=total_samples,
                    num_samples_with_grad=num_samples_with_grad,
                    mode=mode,
                )
            layer_scores[expert_id] = mode_scores_by_layer[canonical_score_mode][expert_id]

        layer_scores = torch.nan_to_num(layer_scores, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        score_by_layer[str(layer_id)] = layer_scores.cpu()
        log_score_by_layer[str(layer_id)] = torch.log1p(layer_scores).cpu()
        diagnostics["param_count_by_layer"][str(layer_id)] = {
            str(expert_id): int(param_counts.get((layer_id, expert_id), 0))
            for expert_id in range(num_experts)
        }
        diagnostics["grad_square_sum_by_layer"][str(layer_id)] = _scientific_list(scores_cpu.tolist())
        diagnostics["num_samples_with_grad_by_layer"][str(layer_id)] = [
            int(value) for value in samples_with_grad_cpu.tolist()
        ]
        diagnostics["score_scientific_by_layer"][str(layer_id)] = _scientific_list(layer_scores.tolist())
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

    all_scores = [
        float(value)
        for scores in score_by_layer.values()
        for value in scores.tolist()
    ]
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

    if return_diagnostics:
        return score_by_layer, log_score_by_layer, diagnostics

    return score_by_layer, log_score_by_layer
