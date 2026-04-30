import torch
import torch.nn.functional as F


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


def compute_expert_fisher_evidence(
    model,
    data_loader,
    criterion,
    device,
    num_experts,
    get_auxiliary_losses=None,
    return_diagnostics=False,
):
    """Compute per-sample empirical diagonal Fisher scalar evidence for experts.

    For each expert parameter block Theta_k with d_k parameters, this computes:
        1 / (|D_m| * d_k) * sum_i ||grad_{Theta_k} loss_i||^2

    This pass only reads gradients from the trained local model. It does not
    update parameters or optimizer state.
    """

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
        "normalization": "param_count_times_total_samples",
        "auxiliary_loss_used_for_fisher": False,
        "auxiliary_loss_note": (
            "Skipped batch-level auxiliary losses for per-sample Fisher because they are not "
            "per-sample losses."
        ),
        "zero_score_reason": None,
    }

    if not expert_entries:
        diagnostics["zero_score_reason"] = "No trainable expert parameters matched blocks.*.ffn.experts.* names."
        if return_diagnostics:
            return {}, {}, diagnostics
        return {}, {}

    score_sums = {
        layer_id: torch.zeros(num_experts, dtype=torch.float64)
        for layer_id in expert_entries
    }
    samples_with_grad = {
        layer_id: torch.zeros(num_experts, dtype=torch.long)
        for layer_id in expert_entries
    }

    was_training = model.training
    model.eval()
    num_batches = 0
    total_samples = 0

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        model.zero_grad(set_to_none=True)
        result = model(inputs)
        outputs = result["logits"] if isinstance(result, dict) else result

        per_sample_losses = _compute_per_sample_supervised_losses(criterion, outputs, labels)
        batch_size = labels.size(0)
        num_batches += 1
        total_samples += batch_size

        # Important:
        # We intentionally compute grad(loss_i)^2 for each sample and then average.
        # This is different from grad(mean_i loss_i)^2, which underestimates Fisher
        # because gradients from different samples can cancel before squaring.
        for sample_idx in range(batch_size):
            model.zero_grad(set_to_none=True)
            retain_graph = sample_idx < batch_size - 1
            per_sample_losses[sample_idx].backward(retain_graph=retain_graph)

            for layer_id, experts in expert_entries.items():
                for expert_id, entries in experts.items():
                    grad_square_sum = 0.0
                    has_grad_param_count = 0
                    none_grad_param_count = 0
                    for _, param in entries:
                        if param.grad is not None:
                            grad_square_sum += float(param.grad.detach().pow(2).sum().cpu())
                            has_grad_param_count += 1
                        else:
                            none_grad_param_count += 1

                    if has_grad_param_count > 0:
                        samples_with_grad[layer_id][expert_id] += 1
                    score_sums[layer_id][expert_id] += grad_square_sum

        diagnostics["batch_grad_status"].append(
            {
                "batch_index": num_batches,
                "batch_size": int(batch_size),
                "sample_count": int(batch_size),
            }
        )

    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()

    diagnostics["total_samples"] = int(total_samples)
    diagnostics["num_batches"] = int(num_batches)

    denominator_samples = max(total_samples, 1)
    score_by_layer = {}
    log_score_by_layer = {}

    for layer_id, scores in score_sums.items():
        layer_scores = torch.zeros(num_experts, dtype=torch.float64)
        for expert_id in range(num_experts):
            param_count = param_counts.get((layer_id, expert_id), 0)
            if param_count > 0:
                layer_scores[expert_id] = scores[expert_id] / (param_count * denominator_samples)

        layer_scores = torch.nan_to_num(layer_scores, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        score_by_layer[str(layer_id)] = layer_scores.cpu()
        log_score_by_layer[str(layer_id)] = torch.log1p(layer_scores).cpu()
        diagnostics["param_count_by_layer"][str(layer_id)] = {
            str(expert_id): int(param_counts.get((layer_id, expert_id), 0))
            for expert_id in range(num_experts)
        }
        diagnostics["grad_square_sum_by_layer"][str(layer_id)] = _scientific_list(scores.tolist())
        diagnostics["num_samples_with_grad_by_layer"][str(layer_id)] = [
            int(value) for value in samples_with_grad[layer_id].tolist()
        ]
        diagnostics["score_scientific_by_layer"][str(layer_id)] = _scientific_list(layer_scores.tolist())

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
