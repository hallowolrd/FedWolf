"""Project configuration loader backed by PyYAML."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from types import SimpleNamespace

import yaml


_CONFIG_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CONFIG_DIR.parent
DEFAULT_CONFIG_PATH = "configs/moefedavg_baseline/config.yaml"
_REQUIRED_SECTIONS = ("data", "model", "train")
_REQUIRED_CONFIG_KEYS = (
    "data_name",
    "data_path",
    "batch_size",
    "alpha",
    "seed",
    "num_workers",
    "pin_memory",
    "partition_impl",
    "model_type",
    "num_experts",
    "top_k",
    "learning_rate",
    "dropout",
    "router_aux_loss_coef",
    "router_z_loss_coef",
    "router_balance_loss_coef",
    "num_clients",
    "server_epochs",
    "client_epochs",
    "device",
    "save_root",
    "run_name",
    "allow_overwrite",
    "save_client_models",
    "aggregation_mode",
    "non_expert_agg_method",
    "expert_agg_method",
    "optimizer",
    "momentum",
    "weight_decay",
    "grad_clip_norm",
    "label_smooth",
    "fedwolf_eps",
)

_DEFAULTS = {
    "partition_meta_name": "partition_meta.pt",
    "partition_stats_name": "partition_stats.json",
    "resume": False,
    "fedwolf_evidence_loader_mode": "deterministic",
    "fedwolf_evidence_model_mode": "eval",
    "fedwolf_fisher_estimator": "linear_hook_token_fast",
    "fedwolf_fisher_score_mode": "trace_per_active_sample",
    "fedwolf_fisher_debug_batches": 0,
    "fedwolf_fisher_debug": False,
    "history_wolf_min_usage": 1,
    "history_wolf_score_momentum": 0.5,
}

_NON_EXPERT_AGG_METHODS = {
    "equal_avg",
    "sample_weighted_avg",
}
_EXPERT_AGG_METHODS = {
    "equal_avg",
    "sample_weighted_avg",
    "fisher_raw_score",
    "history_wolf_filter",
}
_AGGREGATION_MODE_ALIASES = {
    "whole_model_uniform_avg": "whole_model_uniform_avg",
    "split_expert": "split_expert",
    "split": "split_expert",
}


def _resolve_config_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return _PROJECT_ROOT / path


def _load_yaml_mapping(config_path: str | Path) -> dict:
    config_path = _resolve_config_path(str(config_path))
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            data = yaml.safe_load(config_file)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML syntax in config file: {config_path}") from exc
    except OSError as exc:
        raise OSError(f"Failed to read config file: {config_path}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {config_path}")
    return data


def _raise_if_duplicate_keys(named_configs: list[tuple[str, dict]]) -> None:
    duplicate_details = []
    for index, (left_name, left_cfg) in enumerate(named_configs):
        for right_name, right_cfg in named_configs[index + 1:]:
            duplicate_keys = sorted(set(left_cfg) & set(right_cfg))
            if duplicate_keys:
                duplicate_details.append(
                    f"{left_name} and {right_name}: {duplicate_keys}"
                )

    if duplicate_details:
        raise ValueError(
            "Duplicate config keys found across data/model/train sections. "
            + "; ".join(duplicate_details)
            + ". Please keep keys unique across sections before flattening."
        )


def _raise_if_missing_required_keys(merged_config: dict) -> None:
    missing_keys = sorted(key for key in _REQUIRED_CONFIG_KEYS if key not in merged_config)
    if missing_keys:
        raise ValueError(
            f"Missing required config keys: {missing_keys}. "
            "Please check the config.yaml passed by --config."
        )


def _normalize_aggregation_config(merged_config: dict) -> None:
    raw_mode = merged_config.get("aggregation_mode")
    aggregation_mode = str(raw_mode).strip().lower()
    if aggregation_mode not in _AGGREGATION_MODE_ALIASES:
        valid_values = sorted(_AGGREGATION_MODE_ALIASES)
        raise ValueError(
            f"`aggregation_mode` must be one of {valid_values}, got {raw_mode!r}."
        )
    aggregation_mode = _AGGREGATION_MODE_ALIASES[aggregation_mode]

    non_expert_method = str(merged_config.get("non_expert_agg_method", "")).strip().lower()
    expert_method = str(merged_config.get("expert_agg_method", "")).strip().lower()

    if non_expert_method not in _NON_EXPERT_AGG_METHODS:
        valid_values = sorted(_NON_EXPERT_AGG_METHODS)
        raise ValueError(
            f"`non_expert_agg_method` must be one of {valid_values}, "
            f"got {merged_config.get('non_expert_agg_method')!r}."
        )

    if expert_method not in _EXPERT_AGG_METHODS:
        valid_values = sorted(_EXPERT_AGG_METHODS)
        raise ValueError(
            f"`expert_agg_method` must be one of {valid_values}, "
            f"got {merged_config.get('expert_agg_method')!r}."
        )

    merged_config["aggregation_mode"] = aggregation_mode
    merged_config["non_expert_agg_method"] = non_expert_method
    merged_config["expert_agg_method"] = expert_method
    if aggregation_mode == "whole_model_uniform_avg":
        merged_config["agg_method"] = "whole_model_uniform_avg"
    else:
        merged_config["agg_method"] = (
            f"split_expert_nonexpert_{non_expert_method}_expert_{expert_method}"
        )


def _sanitize_run_name(run_name: object) -> str:
    run_name = str(run_name).strip()
    run_name = re.sub(r"\s+", "_", run_name)
    run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name)
    run_name = run_name.strip("._-")
    if not run_name:
        raise ValueError(
            "Missing required config key: run_name. "
            "Please set a non-empty run_name in the train section of config.yaml."
        )
    return run_name


def _coerce_bool(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"`{field_name}` must be a boolean value.")


def _derive_output_paths(merged_config: dict) -> None:
    run_name = _sanitize_run_name(merged_config["run_name"])
    save_root = str(merged_config["save_root"]).strip() or "save"
    allow_overwrite = _coerce_bool(
        merged_config["allow_overwrite"],
        "allow_overwrite",
    )
    save_client_models = _coerce_bool(
        merged_config["save_client_models"],
        "save_client_models",
    )
    resume = _coerce_bool(merged_config.get("resume", False), "resume")
    run_root = Path(save_root) / run_name

    merged_config["save_root"] = save_root
    merged_config["run_name"] = run_name
    merged_config["allow_overwrite"] = allow_overwrite
    merged_config["save_client_models"] = save_client_models
    merged_config["resume"] = resume
    merged_config["data_save_path"] = str(run_root / "data")
    merged_config["model_save_path"] = str(run_root / "model")
    merged_config["save_result"] = str(run_root / "result")


def _is_nonempty_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def validate_output_paths(args: SimpleNamespace, stage: str) -> None:
    if args.allow_overwrite:
        return

    if stage == "data":
        targets = [Path(args.data_save_path)]
    elif stage == "train":
        if getattr(args, "resume", False):
            return
        targets = [Path(args.model_save_path), Path(args.save_result)]
    else:
        raise ValueError("stage must be either 'data' or 'train'")

    nonempty_targets = [str(path) for path in targets if _is_nonempty_dir(path)]
    if nonempty_targets:
        raise FileExistsError(
            "Output directory already exists and is not empty: "
            f"{nonempty_targets}. Change `run_name`, or set `allow_overwrite: true` "
            "in the train section of config.yaml if you intentionally want to reuse these outputs."
        )


def add_config_path_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=(
            "Path to one nested experiment config.yaml with data/model/train sections. "
            f"Default: {DEFAULT_CONFIG_PATH}"
        ),
    )
    return parser


def load_args(config_path: str):
    try:
        resolved_path = _resolve_config_path(config_path)
        config = _load_yaml_mapping(resolved_path)
    except FileNotFoundError:
        raise
    except ValueError:
        raise
    except Exception as exc:
        resolved_path = _resolve_config_path(config_path)
        raise RuntimeError(f"Failed to load YAML config from `{resolved_path}`.") from exc

    missing_sections = [section for section in _REQUIRED_SECTIONS if section not in config]
    if missing_sections:
        raise ValueError(
            f"Missing required config sections: {missing_sections}. "
            "Please check the config.yaml passed by --config."
        )

    section_items = []
    for section in _REQUIRED_SECTIONS:
        section_config = config[section]
        if not isinstance(section_config, dict):
            raise ValueError(
                f"The `{section}` section in config.yaml must be a mapping. "
                "Please check the config.yaml passed by --config."
            )
        section_items.append((f"{section} section", section_config))

    _raise_if_duplicate_keys(section_items)

    merged_config = dict(_DEFAULTS)
    for _, section_config in section_items:
        merged_config.update(section_config)

    _raise_if_missing_required_keys(merged_config)
    _normalize_aggregation_config(merged_config)
    _derive_output_paths(merged_config)

    return SimpleNamespace(**merged_config)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "add_config_path_arguments",
    "load_args",
    "validate_output_paths",
]
