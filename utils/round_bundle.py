import json
import os

import torch


def _to_cpu_detached(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {key: _to_cpu_detached(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu_detached(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu_detached(value) for value in obj)
    return obj


def _is_expert_key(key):
    try:
        from fl.aggregators import parse_expert_ref_from_key

        if callable(parse_expert_ref_from_key):
            return parse_expert_ref_from_key(str(key)) is not None
    except Exception:
        pass

    lowered = str(key).lower()
    return any(token in lowered for token in ("expert", "experts", "moe", "ffn"))


def _filter_state_dict_by_scope(state_dict, scope):
    if scope == "full":
        return _to_cpu_detached(state_dict)
    if scope == "expert_only":
        if not isinstance(state_dict, dict):
            return _to_cpu_detached(state_dict)
        return {
            key: _to_cpu_detached(value)
            for key, value in state_dict.items()
            if _is_expert_key(key)
        }
    raise ValueError("round_bundle_scope must be 'expert_only' or 'full'")


def _snapshot_args(args):
    if args is None:
        return {}

    fields = (
        "agg_method",
        "fedwolf_fusion_mode",
        "fedwolf_update_fusion_variant",
        "fedwolf_precision_granularity",
        "fedwolf_fisher_precision_granularity",
        "fedwolf_eps",
        "fedwolf_robust_eps",
        "fedwolf_min_expert_usage",
        "fedwolf_fisher_weight_power",
        "fedwolf_fisher_weight_clip_min",
        "fedwolf_fisher_weight_clip_max",
        "fedwolf_history_eta",
        "fedwolf_history_init",
        "fedwolf_history_min_factor",
        "fedwolf_history_max_factor",
        "fedwolf_history_use_usage",
        "model",
        "dataset",
        "num_clients",
        "client_fraction",
        "server_epochs",
        "seed",
        "model_save_path",
    )
    return {
        field: _to_cpu_detached(getattr(args, field))
        for field in fields
        if hasattr(args, field)
    }


def get_round_bundle_dir(args):
    base_dir = getattr(args, "model_save_path", "./outputs")
    bundle_dir_name = getattr(args, "round_bundle_dir", "round_bundles")
    bundle_dir = os.path.join(base_dir, bundle_dir_name)
    os.makedirs(bundle_dir, exist_ok=True)
    return bundle_dir


def _sequence_length(value):
    if value is None:
        return 0
    try:
        return len(value)
    except TypeError:
        return 0


def _log_round_bundle(message, logger=None):
    if logger is not None:
        logger.info(message)
    else:
        print(message)


def save_round_bundle(
    args,
    round_id,
    sampled_client_ids,
    server_state_before,
    client_states,
    client_sizes,
    client_stats,
    extra_payload=None,
    logger=None,
):
    if not bool(getattr(args, "save_round_bundle", False)):
        return None

    interval = int(getattr(args, "round_bundle_interval", 1))
    round_id = int(round_id)
    if interval <= 0 or round_id % interval != 0:
        return None

    max_rounds = getattr(args, "round_bundle_max_rounds", None)
    if max_rounds is not None and round_id > int(max_rounds):
        return None

    scope = str(getattr(args, "round_bundle_scope", "expert_only")).strip().lower()
    if scope not in {"expert_only", "full"}:
        raise ValueError("round_bundle_scope must be 'expert_only' or 'full'")

    bundle_dir = get_round_bundle_dir(args)
    path = os.path.join(bundle_dir, f"round_{round_id:04d}.pt")

    if client_states is None:
        filtered_client_states = None
    else:
        filtered_client_states = [
            _filter_state_dict_by_scope(client_state, scope)
            if isinstance(client_state, dict)
            else _to_cpu_detached(client_state)
            for client_state in client_states
        ]

    bundle = {
        "round": round_id,
        "sampled_client_ids": (
            list(sampled_client_ids) if sampled_client_ids is not None else None
        ),
        "scope": scope,
        "server_state_before": (
            _filter_state_dict_by_scope(server_state_before, scope)
            if server_state_before is not None
            else None
        ),
        "client_states": filtered_client_states,
        "client_sizes": _to_cpu_detached(client_sizes),
        "client_stats": _to_cpu_detached(client_stats),
        "args_snapshot": _snapshot_args(args),
        "online_agg_method": getattr(args, "agg_method", None),
        "online_fedwolf_update_fusion_variant": getattr(
            args,
            "fedwolf_update_fusion_variant",
            None,
        ),
        "extra_payload": _to_cpu_detached(extra_payload or {}),
    }

    torch.save(bundle, path)
    size_mb = os.path.getsize(path) / (1024.0 * 1024.0)

    manifest_entry = {
        "round": round_id,
        "path": os.path.basename(path),
        "scope": scope,
        "num_clients": _sequence_length(client_states),
        "num_client_sizes": _sequence_length(client_sizes),
        "num_client_stats": _sequence_length(client_stats),
        "has_server_state_before": server_state_before is not None,
        "has_client_states": client_states is not None,
        "has_client_stats": client_stats is not None,
        "size_mb": size_mb,
    }
    manifest_path = os.path.join(bundle_dir, "manifest.jsonl")
    with open(manifest_path, "a", encoding="utf-8") as manifest_file:
        manifest_file.write(json.dumps(manifest_entry, sort_keys=True) + "\n")

    _log_round_bundle(
        "[RoundBundle] saved "
        f"round={round_id} path={path} scope={scope} size_mb={size_mb:.3f}",
        logger=logger,
    )
    return path
