import torch


def parse_expert_param_ref(name):
    """Parse blocks.{layer}.ffn.experts.{expert_id}.* parameter names."""

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


def compute_expert_fisher_evidence(
    model,
    data_loader,
    criterion,
    device,
    num_experts,
    get_auxiliary_losses=None,
    return_diagnostics=False,
):
    """Compute batch-approximate Fisher scalar evidence for expert parameters.

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
        "num_batches_with_grad_by_layer": {},
        "score_scientific_by_layer": {},
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
    batches_with_grad = {
        layer_id: torch.zeros(num_experts, dtype=torch.long)
        for layer_id in expert_entries
    }

    was_training = model.training
    model.eval()
    num_batches = 0

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        model.zero_grad(set_to_none=True)
        result = model(inputs)
        outputs = result["logits"] if isinstance(result, dict) else result

        loss = criterion(outputs, labels)
        if get_auxiliary_losses is not None and isinstance(result, dict):
            extra_loss, _, _ = get_auxiliary_losses(result)
            loss = loss + extra_loss

        loss.backward()
        num_batches += 1

        batch_status = {}
        for layer_id, experts in expert_entries.items():
            layer_status = {}
            for expert_id, entries in experts.items():
                grad_square_sum = 0.0
                none_grad_param_count = 0
                has_grad_param_count = 0
                for _, param in entries:
                    if param.grad is not None:
                        grad_square_sum += float(param.grad.detach().pow(2).sum().cpu())
                        has_grad_param_count += 1
                    else:
                        none_grad_param_count += 1

                if has_grad_param_count > 0:
                    batches_with_grad[layer_id][expert_id] += 1
                score_sums[layer_id][expert_id] += grad_square_sum
                layer_status[str(expert_id)] = {
                    "grad_is_none": has_grad_param_count == 0,
                    "has_grad_param_count": has_grad_param_count,
                    "none_grad_param_count": none_grad_param_count,
                    "grad_square_sum": f"{grad_square_sum:.12e}",
                }

            batch_status[str(layer_id)] = layer_status
        diagnostics["batch_grad_status"].append(
            {
                "batch_index": num_batches,
                "experts": batch_status,
            }
        )

    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()

    denominator_batches = max(num_batches, 1)
    score_by_layer = {}
    log_score_by_layer = {}

    for layer_id, scores in score_sums.items():
        layer_scores = torch.zeros(num_experts, dtype=torch.float64)
        for expert_id in range(num_experts):
            param_count = param_counts.get((layer_id, expert_id), 0)
            if param_count > 0:
                layer_scores[expert_id] = scores[expert_id] / (param_count * denominator_batches)

        layer_scores = torch.nan_to_num(layer_scores, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        score_by_layer[str(layer_id)] = layer_scores.cpu()
        log_score_by_layer[str(layer_id)] = torch.log1p(layer_scores).cpu()
        diagnostics["param_count_by_layer"][str(layer_id)] = {
            str(expert_id): int(param_counts.get((layer_id, expert_id), 0))
            for expert_id in range(num_experts)
        }
        diagnostics["grad_square_sum_by_layer"][str(layer_id)] = _scientific_list(scores.tolist())
        diagnostics["num_batches_with_grad_by_layer"][str(layer_id)] = [
            int(value) for value in batches_with_grad[layer_id].tolist()
        ]
        diagnostics["score_scientific_by_layer"][str(layer_id)] = _scientific_list(layer_scores.tolist())

    all_scores = [
        float(value)
        for scores in score_by_layer.values()
        for value in scores.tolist()
    ]
    if all_scores and all(score == 0.0 for score in all_scores):
        total_batches_with_grad = sum(
            sum(layer_counts)
            for layer_counts in diagnostics["num_batches_with_grad_by_layer"].values()
        )
        total_grad_square_sum = sum(
            float(value)
            for layer_scores in diagnostics["grad_square_sum_by_layer"].values()
            for value in layer_scores
        )
        if total_batches_with_grad == 0:
            diagnostics["zero_score_reason"] = (
                "All matched expert parameters had grad=None after backward. "
                "The selected experts may be disconnected from the loss or no tokens reached them."
            )
        elif total_grad_square_sum == 0.0:
            diagnostics["zero_score_reason"] = (
                "Expert gradients existed, but every expert grad_square_sum was exactly 0."
            )
        else:
            diagnostics["zero_score_reason"] = (
                "grad_square_sum was non-zero before normalization, but final scores became 0."
            )

    if return_diagnostics:
        return score_by_layer, log_score_by_layer, diagnostics

    return score_by_layer, log_score_by_layer
