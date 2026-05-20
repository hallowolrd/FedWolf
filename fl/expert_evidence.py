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

FISHER_ESTIMATOR_ALIASES = {
    "per_sample_backward": "per_sample_backward",
    "per_sample": "per_sample_backward",
    "per_sample_empirical_diagonal_fisher": "per_sample_backward",
    "linear_hook_token_fast": "linear_hook_token_fast",
    "token_fast": "linear_hook_token_fast",
    "linear_hook_sample_fast": "linear_hook_sample_fast",
    "sample_fast": "linear_hook_sample_fast",
}

FISHER_ESTIMATORS = (
    "per_sample_backward",
    "linear_hook_token_fast",
    "linear_hook_sample_fast",
)


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


def parse_expert_param_block_ref(name):
    expert_ref = parse_expert_param_ref(name)
    if expert_ref is None:
        return None

    parts = name.split(".")
    experts_idx = parts.index("experts")
    block_parts = parts[experts_idx + 2:]
    if not block_parts:
        return None

    layer_id, expert_id = expert_ref
    return layer_id, expert_id, ".".join(block_parts)


def parse_expert_linear_module_ref(module_name):
    parts = module_name.split(".")
    if "blocks" not in parts or "experts" not in parts:
        return None

    blocks_idx = parts.index("blocks")
    experts_idx = parts.index("experts")
    if blocks_idx + 1 >= len(parts) or experts_idx + 1 >= len(parts):
        return None
    if not parts[blocks_idx + 1].isdigit() or not parts[experts_idx + 1].isdigit():
        return None

    block_parts = parts[experts_idx + 2:]
    if not block_parts:
        return None

    return str(parts[blocks_idx + 1]), int(parts[experts_idx + 1]), ".".join(block_parts)


def _collect_expert_linear_modules(model, num_experts, block_param_counts):
    linear_modules = []
    module_lookup = dict(model.named_modules())

    for module_name, module in module_lookup.items():
        if not isinstance(module, torch.nn.Linear):
            continue

        module_ref = parse_expert_linear_module_ref(module_name)
        if module_ref is None:
            continue

        layer_id, expert_id, block_prefix = module_ref
        if expert_id >= num_experts:
            continue

        parts = module_name.split(".")
        experts_idx = parts.index("experts")
        expert_module_name = ".".join(parts[:experts_idx + 2])
        expert_module = module_lookup.get(expert_module_name)

        layer_key = str(layer_id)
        expert_blocks = _get_expert_mapping(
            block_param_counts,
            layer_key,
            expert_id,
            default={},
        ) or {}
        weight_block_name = f"{block_prefix}.weight"
        bias_block_name = f"{block_prefix}.bias"
        has_weight_block = weight_block_name in expert_blocks
        has_bias_block = bias_block_name in expert_blocks
        if not has_weight_block and not has_bias_block:
            continue

        linear_modules.append(
            {
                "name": module_name,
                "module": module,
                "expert_module": expert_module,
                "expert_module_name": expert_module_name,
                "layer_id": layer_key,
                "expert_id": expert_id,
                "block_prefix": block_prefix,
                "weight_block_name": weight_block_name,
                "bias_block_name": bias_block_name,
                "has_weight_block": has_weight_block,
                "has_bias_block": has_bias_block,
            }
        )

    return linear_modules


def _flatten_linear_hook_tensor(value, module_name, tensor_name):
    if not torch.is_tensor(value):
        raise RuntimeError(
            "linear hook Fisher expected tensor "
            f"{tensor_name} for expert Linear module {module_name!r}."
        )
    if value.ndim == 0:
        raise RuntimeError(
            "linear hook Fisher expected non-scalar "
            f"{tensor_name} for expert Linear module {module_name!r}."
        )
    return value.reshape(-1, value.shape[-1])


def _compute_sample_grouped_linear_fisher_scalars(x_flat, delta_flat, sample_ids):
    x_work = x_flat.detach()
    delta_work = delta_flat.detach()
    if x_work.dtype in {torch.float16, torch.bfloat16}:
        x_work = x_work.to(dtype=torch.float32)
    if delta_work.dtype in {torch.float16, torch.bfloat16}:
        delta_work = delta_work.to(dtype=torch.float32)

    x_work = x_work.reshape(-1, x_work.shape[-1])
    delta_work = delta_work.reshape(-1, delta_work.shape[-1])
    sample_ids = sample_ids.detach().reshape(-1).to(device=x_work.device, dtype=torch.long)

    weight_grad_square_sum = torch.zeros((), dtype=torch.float64, device=x_work.device)
    bias_grad_square_sum = torch.zeros((), dtype=torch.float64, device=x_work.device)
    unique_sample_ids = torch.unique(sample_ids, sorted=True)

    for sample_id in unique_sample_ids:
        mask = sample_ids == sample_id
        x_s = x_work[mask]
        delta_s = delta_work[mask]
        sample_grad_w = torch.einsum("to,ti->oi", delta_s, x_s)
        sample_grad_b = delta_s.sum(dim=0)
        weight_grad_square_sum += sample_grad_w.pow(2).sum().to(dtype=torch.float64)
        bias_grad_square_sum += sample_grad_b.pow(2).sum().to(dtype=torch.float64)

    return (
        weight_grad_square_sum,
        bias_grad_square_sum,
        int(unique_sample_ids.numel()),
        int(sample_ids.numel()),
    )


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


def canonicalize_fisher_estimator(estimator):
    raw_estimator = "per_sample_backward" if estimator is None else str(estimator).strip().lower()
    canonical_estimator = FISHER_ESTIMATOR_ALIASES.get(raw_estimator)
    if canonical_estimator is None:
        supported = ", ".join(FISHER_ESTIMATORS)
        raise ValueError(
            f"fedwolf_fisher_estimator must be one of: {supported}. "
            f"Got {estimator!r}."
        )
    return canonical_estimator


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
    """累计 evidence forward pass 中观测到的 expert token usage。"""

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


def _tensor_scalar_to_float(value):
    if value is None:
        return 0.0
    if torch.is_tensor(value):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().cpu().item())
    return float(value)


def _count_to_int(value):
    if value is None:
        return 0
    if torch.is_tensor(value):
        if value.numel() == 0:
            return 0
        return int(value.detach().cpu().item())
    return int(value)


def _key_candidates(value):
    candidates = []
    for candidate in (value, str(value)):
        if candidate not in candidates:
            candidates.append(candidate)
    try:
        int_value = int(value)
    except (TypeError, ValueError):
        int_value = None
    if int_value is not None and int_value not in candidates:
        candidates.append(int_value)
    return candidates


def _get_layer_mapping(mapping, layer_id, default=None):
    if not isinstance(mapping, dict):
        return default
    for layer_key in _key_candidates(layer_id):
        if layer_key in mapping:
            return mapping[layer_key]
    return default


def _get_expert_mapping(mapping, layer_id, expert_id, default=None):
    layer_mapping = _get_layer_mapping(mapping, layer_id, default=None)
    if not isinstance(layer_mapping, dict):
        return default
    for expert_key in _key_candidates(expert_id):
        if expert_key in layer_mapping:
            return layer_mapping[expert_key]
    return default


def _get_param_count(param_counts, layer_id, expert_id):
    for layer_key in _key_candidates(layer_id):
        for expert_key in _key_candidates(expert_id):
            value = param_counts.get((layer_key, expert_key))
            if value is not None:
                return int(value)
    return 0


def _sum_numeric_tree(value):
    if value is None:
        return 0.0
    if torch.is_tensor(value):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().cpu().sum().item())
    if isinstance(value, dict):
        return sum(_sum_numeric_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_sum_numeric_tree(item) for item in value)
    return float(value)


def _ensure_default_fast_diagnostics(diagnostics, canonical_estimator):
    defaults = {
        "fast_fisher_sample_grouped": False,
        "fast_fisher_count_unit": None,
        "fast_fisher_token_count_unit": None,
        "fast_fisher_hooked_linear_count": 0,
        "fast_fisher_hooked_linear_names": [],
        "fast_fisher_token_count_by_layer": {},
        "fast_fisher_active_sample_count_by_layer": {},
        "fast_fisher_active_token_count_by_layer": {},
        "fast_fisher_note": None,
        "fast_fisher_unmatched_linear_block_count": 0,
    }

    if canonical_estimator == "linear_hook_token_fast":
        defaults.update(
            {
                "fast_fisher_sample_grouped": False,
                "fast_fisher_count_unit": "routed_tokens",
                "fast_fisher_token_count_unit": "accepted_routed_tokens",
                "fast_fisher_note": (
                    "linear_hook_token_fast uses token-level diagonal Fisher approximation: "
                    "sum_token grad_token^2, not sample-grouped (sum_token grad_token)^2."
                ),
            }
        )
    elif canonical_estimator == "linear_hook_sample_fast":
        defaults.update(
            {
                "fast_fisher_sample_grouped": True,
                "fast_fisher_count_unit": "active_original_samples",
                "fast_fisher_token_count_unit": "accepted_routed_tokens",
                "fast_fisher_note": (
                    "linear_hook_sample_fast groups routed token gradient contributions by original "
                    "sample before squaring: sum_sample (sum_token grad_token)^2."
                ),
            }
        )

    for key, value in defaults.items():
        diagnostics.setdefault(key, value)


def _infer_zero_score_reason(*, canonical_estimator, diagnostics, all_scores):
    if all_scores and any(float(score) != 0.0 for score in all_scores):
        return diagnostics.get("zero_score_reason")

    total_grad_square_sum = _sum_numeric_tree(diagnostics.get("grad_square_sum_by_layer", {}))

    if canonical_estimator == "linear_hook_token_fast":
        if int(diagnostics.get("fast_fisher_hooked_linear_count", 0)) == 0:
            return "No expert Linear modules were hooked for linear_hook_token_fast."
        active_token_count = _sum_numeric_tree(
            diagnostics.get("fast_fisher_active_token_count_by_layer")
            or diagnostics.get("num_samples_with_grad_by_layer", {})
        )
        if active_token_count == 0:
            return "No accepted routed tokens reached hooked expert Linear modules during evidence forward."
        if total_grad_square_sum == 0.0:
            return (
                "Accepted routed tokens reached hooked expert Linear modules, but token-level Fisher "
                "grad_square_sum was exactly 0."
            )
        return "Token-level Fisher grad_square_sum was non-zero before normalization, but final scores became 0."

    if canonical_estimator == "linear_hook_sample_fast":
        if int(diagnostics.get("fast_fisher_hooked_linear_count", 0)) == 0:
            return "No expert Linear modules were hooked for linear_hook_sample_fast."
        active_sample_count = _sum_numeric_tree(diagnostics.get("fast_fisher_active_sample_count_by_layer", {}))
        active_token_count = _sum_numeric_tree(diagnostics.get("fast_fisher_active_token_count_by_layer", {}))
        if active_sample_count == 0 and active_token_count == 0:
            return "No accepted routed tokens reached hooked expert Linear modules during evidence forward."
        if active_sample_count > 0 and total_grad_square_sum == 0.0:
            return (
                "Accepted routed tokens reached hooked expert Linear modules, but sample-grouped Fisher "
                "grad_square_sum was exactly 0."
            )
        return "Sample-grouped Fisher grad_square_sum was non-zero before normalization, but final scores became 0."

    total_samples_with_grad = _sum_numeric_tree(diagnostics.get("num_samples_with_grad_by_layer", {}))
    if total_samples_with_grad == 0:
        return "All matched expert parameters had grad=None for every sample after per-sample backward."
    if total_grad_square_sum == 0.0:
        return (
            "Matched expert parameters received gradients, but accumulated per-sample "
            "grad_square_sum was exactly 0."
        )
    return "Per-sample grad_square_sum was non-zero before normalization, but final scores became 0."


def _finalize_expert_fisher_outputs(
    *,
    score_sums,
    samples_with_grad,
    block_score_sums,
    block_samples_with_grad,
    param_counts,
    block_param_counts,
    num_experts,
    total_samples,
    canonical_score_mode,
    diagnostics,
):
    canonical_estimator = diagnostics.get("fisher_estimator", "per_sample_backward")
    score_by_layer = {}
    log_score_by_layer = {}
    expert_block_fisher_score_by_layer = {}
    positive_block_scores = []
    matched_block_count = 0

    for layer_id, experts in block_param_counts.items():
        layer_key = str(layer_id)
        layer_block_scores = {}
        for expert_id, blocks in experts.items():
            expert_key = str(expert_id)
            expert_block_scores = {}
            expert_block_score_sums = _get_expert_mapping(
                block_score_sums,
                layer_key,
                expert_id,
                default={},
            ) or {}
            expert_block_samples = _get_expert_mapping(
                block_samples_with_grad,
                layer_key,
                expert_id,
                default={},
            ) or {}
            for block_name, block_param_count in blocks.items():
                matched_block_count += 1
                grad_square_sum_value = _tensor_scalar_to_float(expert_block_score_sums.get(block_name))
                num_samples_with_grad = _count_to_int(expert_block_samples.get(block_name, 0))
                block_score = compute_fisher_scalar_from_sums(
                    grad_square_sum=grad_square_sum_value,
                    param_count=int(block_param_count),
                    total_samples=total_samples,
                    num_samples_with_grad=num_samples_with_grad,
                    mode=canonical_score_mode,
                )
                expert_block_scores[str(block_name)] = float(block_score)
                if block_score > 0.0:
                    positive_block_scores.append(float(block_score))
            layer_block_scores[expert_key] = expert_block_scores
        expert_block_fisher_score_by_layer[layer_key] = layer_block_scores

    diagnostics["expert_block_fisher_score_by_layer"] = expert_block_fisher_score_by_layer
    diagnostics["expert_block_fisher_matched_block_count"] = int(matched_block_count)
    diagnostics["expert_block_fisher_positive_block_count"] = int(len(positive_block_scores))
    diagnostics["expert_block_fisher_mean_positive"] = (
        sum(positive_block_scores) / len(positive_block_scores) if positive_block_scores else 0.0
    )
    diagnostics["expert_block_fisher_max_positive"] = max(positive_block_scores) if positive_block_scores else 0.0

    for layer_id, scores in score_sums.items():
        layer_key = str(layer_id)
        scores_cpu = scores.detach().cpu() if torch.is_tensor(scores) else torch.as_tensor(scores, dtype=torch.float64)
        raw_samples_with_grad = _get_layer_mapping(samples_with_grad, layer_id, default=None)
        if raw_samples_with_grad is None:
            samples_with_grad_cpu = torch.zeros(num_experts, dtype=torch.long)
        elif torch.is_tensor(raw_samples_with_grad):
            samples_with_grad_cpu = raw_samples_with_grad.detach().cpu()
        else:
            samples_with_grad_cpu = torch.as_tensor(raw_samples_with_grad, dtype=torch.long)

        mode_scores_by_layer = {
            mode: torch.zeros(num_experts, dtype=torch.float64)
            for mode in FISHER_SCORE_MODES
        }
        layer_scores = torch.zeros(num_experts, dtype=torch.float64)
        for expert_id in range(num_experts):
            param_count = _get_param_count(param_counts, layer_key, expert_id)
            num_samples_with_grad = int(samples_with_grad_cpu[expert_id].item())
            grad_square_sum = float(scores_cpu[expert_id].item())
            for mode in FISHER_SCORE_MODES:
                mode_scores_by_layer[mode][expert_id] = compute_fisher_scalar_from_sums(
                    grad_square_sum=grad_square_sum,
                    param_count=param_count,
                    total_samples=total_samples,
                    num_samples_with_grad=num_samples_with_grad,
                    mode=mode,
                )
            layer_scores[expert_id] = mode_scores_by_layer[canonical_score_mode][expert_id]

        layer_scores = torch.nan_to_num(layer_scores, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        score_by_layer[layer_key] = layer_scores.cpu()
        log_score_by_layer[layer_key] = torch.log1p(layer_scores).cpu()
        diagnostics["param_count_by_layer"][layer_key] = {
            str(expert_id): int(_get_param_count(param_counts, layer_key, expert_id))
            for expert_id in range(num_experts)
        }
        diagnostics["grad_square_sum_by_layer"][layer_key] = _scientific_list(scores_cpu.tolist())
        num_samples_with_grad_list = [int(value) for value in samples_with_grad_cpu.tolist()]
        diagnostics["num_samples_with_grad_by_layer"][layer_key] = num_samples_with_grad_list
        if canonical_estimator == "linear_hook_token_fast":
            diagnostics["fast_fisher_token_count_by_layer"][layer_key] = list(num_samples_with_grad_list)
            diagnostics["fast_fisher_active_token_count_by_layer"][layer_key] = list(num_samples_with_grad_list)
        diagnostics["score_scientific_by_layer"][layer_key] = _scientific_list(layer_scores.tolist())
        diagnostics["score_mean_diag_by_layer"][layer_key] = _scientific_list(
            mode_scores_by_layer["mean_diag"].tolist()
        )
        diagnostics["score_mean_diag_active_by_layer"][layer_key] = _scientific_list(
            mode_scores_by_layer["mean_diag_active"].tolist()
        )
        diagnostics["score_trace_per_sample_by_layer"][layer_key] = _scientific_list(
            mode_scores_by_layer["trace_per_sample"].tolist()
        )
        diagnostics["score_trace_per_active_sample_by_layer"][layer_key] = _scientific_list(
            mode_scores_by_layer["trace_per_active_sample"].tolist()
        )
        diagnostics["score_trace_raw_by_layer"][layer_key] = _scientific_list(
            mode_scores_by_layer["trace_raw"].tolist()
        )

    all_scores = [
        float(value)
        for scores in score_by_layer.values()
        for value in scores.tolist()
    ]
    diagnostics["zero_score_reason"] = _infer_zero_score_reason(
        canonical_estimator=canonical_estimator,
        diagnostics=diagnostics,
        all_scores=all_scores,
    )

    return score_by_layer, log_score_by_layer


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
    fisher_estimator="per_sample_backward",
    debug_batches=0,
    max_samples=None,
    max_batches=None,
    pin_memory=False,
):
    """计算 expert 的 Fisher 标量 evidence，不更新参数或 optimizer state。"""

    if model_mode not in {"eval", "train"}:
        raise ValueError(f"model_mode must be either 'eval' or 'train', got {model_mode!r}.")

    canonical_score_mode = canonicalize_fisher_score_mode(score_mode)
    canonical_estimator = canonicalize_fisher_estimator(fisher_estimator)
    is_linear_hook_estimator = canonical_estimator in {
        "linear_hook_token_fast",
        "linear_hook_sample_fast",
    }
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
        "num_samples_with_grad_semantics": (
            "active_routed_tokens_for_linear_hook_token_fast"
            if canonical_estimator == "linear_hook_token_fast"
            else (
                "active_original_samples_for_linear_hook_sample_fast"
                if canonical_estimator == "linear_hook_sample_fast"
                else "samples_with_non_none_expert_grad_for_per_sample_backward"
            )
        ),
        "score_scientific_by_layer": {},
        "total_samples": 0,
        "num_batches": 0,
        "fisher_estimator": canonical_estimator,
        "fisher_estimator_raw": fisher_estimator,
        "fisher_estimator_impl": (
            "linear_hook_token_diagonal_fisher"
            if canonical_estimator == "linear_hook_token_fast"
            else (
                "linear_hook_sample_grouped_diagonal_fisher"
                if canonical_estimator == "linear_hook_sample_fast"
                else "per_sample_empirical_diagonal_fisher"
            )
        ),
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
        "expert_block_fisher_score_by_layer": {},
        "expert_block_fisher_matched_block_count": 0,
        "expert_block_fisher_positive_block_count": 0,
        "expert_block_fisher_mean_positive": 0.0,
        "expert_block_fisher_max_positive": 0.0,
        "unmatched_block_name_count": 0,
        "evidence_expert_stats_by_layer": {},
        "evidence_expert_activations_by_layer": {},
        "evidence_selected_counts_by_layer": {},
        "evidence_overflow_counts_by_layer": {},
    }
    _ensure_default_fast_diagnostics(diagnostics, canonical_estimator)

    block_param_counts = {}
    unmatched_block_name_count = 0
    for layer_id, experts in expert_entries.items():
        layer_key = str(layer_id)
        for expert_id, entries in experts.items():
            expert_key = str(expert_id)
            for param_name, param in entries:
                block_ref = parse_expert_param_block_ref(param_name)
                if block_ref is None:
                    unmatched_block_name_count += 1
                    continue
                _, _, block_name = block_ref
                block_param_counts.setdefault(layer_key, {}).setdefault(expert_key, {})[block_name] = (
                    block_param_counts.setdefault(layer_key, {}).setdefault(expert_key, {}).get(block_name, 0)
                    + int(param.numel())
                )
    diagnostics["unmatched_block_name_count"] = int(unmatched_block_name_count)

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
    block_score_sums = {
        layer_id: {
            expert_id: {
                block_name: torch.zeros((), dtype=torch.float64, device=device)
                for block_name in (_get_expert_mapping(block_param_counts, layer_id, expert_id, default={}) or {})
            }
            for expert_id in experts
        }
        for layer_id, experts in expert_entries.items()
    }
    block_samples_with_grad = {
        layer_id: {
            expert_id: {
                block_name: 0
                for block_name in (_get_expert_mapping(block_param_counts, layer_id, expert_id, default={}) or {})
            }
            for expert_id in experts
        }
        for layer_id, experts in expert_entries.items()
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
    linear_modules = []
    activation_cache = {}
    hook_handles = []
    counted_experts_this_batch = set()
    fast_fisher_has_net0 = set()
    fast_active_token_counts = {}
    fast_active_sample_counts = {}

    if is_linear_hook_estimator:
        fast_active_token_counts = {
            layer_id: torch.zeros(num_experts, dtype=torch.long, device=device)
            for layer_id in expert_entries
        }
        fast_active_sample_counts = {
            layer_id: torch.zeros(num_experts, dtype=torch.long, device=device)
            for layer_id in expert_entries
        }
        linear_modules = _collect_expert_linear_modules(
            model=model,
            num_experts=num_experts,
            block_param_counts=block_param_counts,
        )
        fast_fisher_has_net0 = {
            (info["layer_id"], info["expert_id"])
            for info in linear_modules
            if info["block_prefix"] == "net.0"
        }
        diagnostics["fast_fisher_hooked_linear_count"] = int(len(linear_modules))
        diagnostics["fast_fisher_hooked_linear_names"] = [
            info["name"] for info in linear_modules[:64]
        ]
        diagnostics["fast_fisher_unmatched_linear_block_count"] = int(
            sum(1 for info in linear_modules if not info["has_weight_block"])
            + sum(
                1
                for info in linear_modules
                if info["module"].bias is not None and not info["has_bias_block"]
            )
        )

        def make_forward_hook(module_name):
            def forward_hook(module, inputs, output):
                if not inputs or not torch.is_tensor(inputs[0]):
                    raise RuntimeError(
                        f"{canonical_estimator} expected tensor input for expert "
                        f"Linear module {module_name!r}."
                    )
                activation_cache[id(module)] = inputs[0].detach()

            return forward_hook

        def make_backward_hook(info):
            def backward_hook(module, grad_input, grad_output):
                if not grad_output or not torch.is_tensor(grad_output[0]):
                    raise RuntimeError(
                        f"{canonical_estimator} expected tensor grad_output for expert "
                        f"Linear module {info['name']!r}."
                    )

                activation = activation_cache.get(id(module))
                if activation is None:
                    raise RuntimeError(
                        f"{canonical_estimator} could not find cached activation for expert "
                        f"Linear module {info['name']!r}."
                    )

                x_flat = _flatten_linear_hook_tensor(activation, info["name"], "input activation")
                delta_flat = _flatten_linear_hook_tensor(
                    grad_output[0].detach(),
                    info["name"],
                    "grad_output",
                )
                if x_flat.shape[0] != delta_flat.shape[0]:
                    raise RuntimeError(
                        f"{canonical_estimator} expected matching token counts for Linear input "
                        f"and grad_output in {info['name']!r}, got x_flat shape "
                        f"{tuple(x_flat.shape)} and delta_flat shape {tuple(delta_flat.shape)}."
                    )

                token_count = x_flat.shape[0]
                if token_count <= 0:
                    return

                layer_id = info["layer_id"]
                expert_id = info["expert_id"]
                expert_block_score_sums = _get_expert_mapping(
                    block_score_sums,
                    layer_id,
                    expert_id,
                    default={},
                ) or {}
                expert_block_samples = _get_expert_mapping(
                    block_samples_with_grad,
                    layer_id,
                    expert_id,
                    default={},
                ) or {}

                if canonical_estimator == "linear_hook_sample_fast":
                    expert_module = info.get("expert_module")
                    sample_ids = getattr(expert_module, "_fedwolf_accepted_sample_ids", None)
                    if sample_ids is None:
                        raise RuntimeError(
                            "linear_hook_sample_fast requires expert._fedwolf_accepted_sample_ids. "
                            "Please make sure TokenSwitchFFN.forward exposes accepted sample ids "
                            "before calling expert(...). "
                            "linear_hook_sample_fast depends on TokenSwitchFFN.forward exposing accepted "
                            "sample ids; expected attribute: _fedwolf_accepted_sample_ids. "
                            f"Linear module: {info['name']!r}; expert module: "
                            f"{info.get('expert_module_name')!r}."
                        )
                    sample_ids = sample_ids.detach().to(device=x_flat.device, dtype=torch.long).reshape(-1)
                    if sample_ids.numel() != token_count:
                        raise RuntimeError(
                            "linear_hook_sample_fast expected Linear input, grad_output, and accepted "
                            f"sample ids to have matching token counts for module {info['name']!r}; "
                            f"x_flat shape={tuple(x_flat.shape)}, delta_flat shape={tuple(delta_flat.shape)}, "
                            f"sample_ids shape={tuple(sample_ids.shape)}."
                        )
                    (
                        weight_grad_square_sum,
                        bias_grad_square_sum,
                        active_sample_count,
                        active_token_count,
                    ) = _compute_sample_grouped_linear_fisher_scalars(
                        x_flat=x_flat,
                        delta_flat=delta_flat,
                        sample_ids=sample_ids,
                    )
                    count_increment = active_sample_count
                else:
                    delta_square = delta_flat.pow(2)
                    weight_grad_square_sum = torch.einsum(
                        "to,ti->",
                        delta_square,
                        x_flat.pow(2),
                    ).to(dtype=torch.float64)
                    bias_grad_square_sum = delta_square.sum().to(dtype=torch.float64)
                    active_sample_count = 0
                    active_token_count = token_count
                    count_increment = token_count

                if info["has_weight_block"] and info["weight_block_name"] in expert_block_score_sums:
                    score_sums[layer_id][expert_id] += weight_grad_square_sum
                    expert_block_score_sums[info["weight_block_name"]] += weight_grad_square_sum
                    expert_block_samples[info["weight_block_name"]] += count_increment

                if (
                    module.bias is not None
                    and info["has_bias_block"]
                    and info["bias_block_name"] in expert_block_score_sums
                ):
                    score_sums[layer_id][expert_id] += bias_grad_square_sum
                    expert_block_score_sums[info["bias_block_name"]] += bias_grad_square_sum
                    expert_block_samples[info["bias_block_name"]] += count_increment

                count_key = (layer_id, expert_id)
                if count_key not in counted_experts_this_batch and (
                    info["block_prefix"] == "net.0" or count_key not in fast_fisher_has_net0
                ):
                    samples_with_grad[layer_id][expert_id] += count_increment
                    if canonical_estimator == "linear_hook_sample_fast":
                        fast_active_sample_counts[layer_id][expert_id] += active_sample_count
                    fast_active_token_counts[layer_id][expert_id] += active_token_count
                    counted_experts_this_batch.add(count_key)

            return backward_hook

    try:
        if is_linear_hook_estimator:
            for info in linear_modules:
                module = info["module"]
                hook_handles.append(module.register_forward_hook(make_forward_hook(info["name"])))
                hook_handles.append(module.register_full_backward_hook(make_backward_hook(info)))

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

            if is_linear_hook_estimator:
                counted_experts_this_batch.clear()
                per_sample_losses.sum().backward()
                model.zero_grad(set_to_none=True)
                activation_cache.clear()
            else:
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
                            block_grad_square_sums = {}
                            has_grad_param_count = 0
                            none_grad_param_count = 0
                            for param_name, param in entries:
                                if param.grad is not None:
                                    value = param.grad.detach().pow(2).sum().to(dtype=torch.float64)
                                    grad_square_sum = value if grad_square_sum is None else grad_square_sum + value
                                    block_ref = parse_expert_param_block_ref(param_name)
                                    if block_ref is not None:
                                        _, _, block_name = block_ref
                                        block_grad_square_sums[block_name] = (
                                            value
                                            if block_name not in block_grad_square_sums
                                            else block_grad_square_sums[block_name] + value
                                        )
                                    has_grad_param_count += 1
                                else:
                                    none_grad_param_count += 1

                            if has_grad_param_count > 0:
                                samples_with_grad[layer_id][expert_id] += 1
                                score_sums[layer_id][expert_id] += grad_square_sum
                                expert_block_score_sums = _get_expert_mapping(
                                    block_score_sums,
                                    layer_id,
                                    expert_id,
                                    default={},
                                ) or {}
                                expert_block_samples = _get_expert_mapping(
                                    block_samples_with_grad,
                                    layer_id,
                                    expert_id,
                                    default={},
                                ) or {}
                                for block_name, block_grad_square_sum in block_grad_square_sums.items():
                                    if block_name in expert_block_score_sums:
                                        expert_block_samples[block_name] += 1
                                        expert_block_score_sums[block_name] += block_grad_square_sum

            if num_batches <= debug_batches:
                batch_grad_status = {
                    "batch_index": num_batches,
                    "batch_size": int(batch_size),
                    "sample_count": int(batch_size),
                }
                if is_linear_hook_estimator:
                    batch_grad_status.update(
                        {
                            "fisher_estimator": canonical_estimator,
                            "hooked_linear_count": int(len(linear_modules)),
                        }
                    )
                diagnostics["batch_grad_status"].append(batch_grad_status)

            if max_samples is not None and total_samples >= max_samples:
                limit_reached = True
                stop_reason = "max_samples"
                break
            if max_batches is not None and num_batches >= max_batches:
                limit_reached = True
                stop_reason = "max_batches"
                break
    finally:
        for handle in hook_handles:
            handle.remove()
        activation_cache.clear()
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

    if canonical_estimator == "linear_hook_sample_fast":
        for layer_id in score_sums:
            layer_key = str(layer_id)
            token_counts = _get_layer_mapping(fast_active_token_counts, layer_id, default=None)
            sample_counts = _get_layer_mapping(fast_active_sample_counts, layer_id, default=None)
            if token_counts is not None:
                diagnostics["fast_fisher_active_token_count_by_layer"][layer_key] = [
                    int(value) for value in token_counts.detach().cpu().tolist()
                ]
            if sample_counts is not None:
                diagnostics["fast_fisher_active_sample_count_by_layer"][layer_key] = [
                    int(value) for value in sample_counts.detach().cpu().tolist()
                ]

    score_by_layer, log_score_by_layer = _finalize_expert_fisher_outputs(
        score_sums=score_sums,
        samples_with_grad=samples_with_grad,
        block_score_sums=block_score_sums,
        block_samples_with_grad=block_samples_with_grad,
        param_counts=param_counts,
        block_param_counts=block_param_counts,
        num_experts=num_experts,
        total_samples=total_samples,
        canonical_score_mode=canonical_score_mode,
        diagnostics=diagnostics,
    )

    if return_diagnostics:
        return score_by_layer, log_score_by_layer, diagnostics

    return score_by_layer, log_score_by_layer
