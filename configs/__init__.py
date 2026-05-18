"""基于 PyYAML 的项目配置加载器。"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from types import SimpleNamespace

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CONFIG_DIR.parent
DEFAULT_CONFIG_PATH = "configs/config.yaml"
_REQUIRED_SECTIONS = ("data", "model", "train")
_REQUIRED_CONFIG_KEYS = (
    "data_name",
    "data_path",
    "batch_size",
    "min_datasize",
    "alpha",
    "seed",
    "partition_meta_name",
    "partition_stats_name",
    "num_workers",
    "pin_memory",
    "num_clients",
    "server_epochs",
    "client_epochs",
    "device",
    "run_name",
    "agg_method",
    "model_type",
    "num_experts",
    "dropout",
    "learning_rate",
    "embed_dim",
    "num_heads",
    "mlp_ratio",
    "depth",
    "num_layers",
    "moe_layers",
    "top_k",
    "router_aux_loss_coef",
    "router_z_loss_coef",
    "router_jitter_noise",
    "capacity_factor",
    "min_capacity",
    "drop_tokens",
    "stem_channels",
    "token_grid_size",
    "use_cls_token",
)
_DEFAULT_SAVE_ROOT = "save"
_DEFAULT_ALLOW_OVERWRITE = False


def _resolve_config_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return _PROJECT_ROOT / path


def _load_yaml_mapping(config_path: str | Path) -> dict:
    """加载一个 YAML 文件，并要求顶层是 mapping。"""

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
            + ". Please keep keys unique across the sections before flattening."
        )


def _raise_if_missing_required_keys(merged_config: dict) -> None:
    missing_keys = sorted(key for key in _REQUIRED_CONFIG_KEYS if key not in merged_config)
    if missing_keys:
        raise ValueError(
            f"Missing required config keys: {missing_keys}. "
            "Please check the config.yaml passed by --config."
        )


def _validate_training_hparams(merged_config: dict) -> None:
    try:
        learning_rate = float(merged_config["learning_rate"])
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError("learning_rate must be a positive number.") from exc
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be a positive number.")
    merged_config["learning_rate"] = learning_rate

    min_learning_rate = merged_config.get("min_learning_rate", None)
    if min_learning_rate is not None:
        try:
            min_learning_rate = float(min_learning_rate)
        except (TypeError, ValueError) as exc:
            raise ValueError("min_learning_rate must be a non-negative number.") from exc
        if not math.isfinite(min_learning_rate) or min_learning_rate < 0.0:
            raise ValueError("min_learning_rate must be non-negative.")
        if min_learning_rate > learning_rate:
            raise ValueError("min_learning_rate must be <= learning_rate.")
        merged_config["min_learning_rate"] = min_learning_rate

    optimizer = str(merged_config.get("optimizer", "adam")).strip().lower()
    if optimizer not in {"adam", "adamw", "sgd"}:
        raise ValueError(
            "optimizer must be one of {'adam', 'adamw', 'sgd'}, "
            f"got {optimizer!r}."
        )
    merged_config["optimizer"] = optimizer

    try:
        weight_decay = float(merged_config.get("weight_decay", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("weight_decay must be a non-negative number.") from exc
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative.")
    merged_config["weight_decay"] = weight_decay

    try:
        momentum = float(merged_config.get("momentum", 0.9))
    except (TypeError, ValueError) as exc:
        raise ValueError("momentum must be a non-negative number.") from exc
    if not math.isfinite(momentum) or momentum < 0.0:
        raise ValueError("momentum must be non-negative.")
    merged_config["momentum"] = momentum

    warmup_rounds = merged_config.get("warmup_rounds", 0)
    if isinstance(warmup_rounds, bool) or not isinstance(warmup_rounds, int):
        raise ValueError("warmup_rounds must be an int.")
    if warmup_rounds < 0:
        raise ValueError("warmup_rounds must be non-negative.")
    merged_config["warmup_rounds"] = warmup_rounds

    warmup_start_learning_rate = merged_config.get("warmup_start_learning_rate", None)
    if warmup_start_learning_rate is not None:
        try:
            warmup_start_learning_rate = float(warmup_start_learning_rate)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "warmup_start_learning_rate must be a non-negative number or None."
            ) from exc
        if not math.isfinite(warmup_start_learning_rate) or warmup_start_learning_rate < 0.0:
            raise ValueError("warmup_start_learning_rate must be non-negative.")
        if warmup_start_learning_rate > learning_rate:
            raise ValueError("warmup_start_learning_rate must be <= learning_rate.")
        merged_config["warmup_start_learning_rate"] = warmup_start_learning_rate

    schedule = str(merged_config.get("lr_schedule", "constant")).strip().lower()
    if schedule not in {
        "constant",
        "none",
        "cosine",
        "warmup_cosine",
        "cosine_warmup",
        "linear_warmup_cosine",
    }:
        raise ValueError(
            "lr_schedule must be one of {'constant', 'none', 'cosine', "
            "'warmup_cosine', 'cosine_warmup', 'linear_warmup_cosine'}, "
            f"got {schedule!r}."
        )
    merged_config["lr_schedule"] = schedule

    if schedule in {"warmup_cosine", "cosine_warmup", "linear_warmup_cosine"}:
        try:
            total_rounds = int(merged_config["server_epochs"])
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("server_epochs must be a positive integer.") from exc
        if total_rounds <= 0:
            raise ValueError("server_epochs must be positive.")
        if warmup_rounds >= total_rounds:
            raise ValueError(
                "warmup_rounds must be smaller than server_epochs for warmup_cosine."
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
    """根据 save_root/run_name 派生所有实验输出路径。"""

    run_name = _sanitize_run_name(merged_config["run_name"])
    save_root = str(merged_config.get("save_root", _DEFAULT_SAVE_ROOT)).strip() or _DEFAULT_SAVE_ROOT
    allow_overwrite = _coerce_bool(
        merged_config.get("allow_overwrite", _DEFAULT_ALLOW_OVERWRITE),
        "allow_overwrite",
    )
    run_root = Path(save_root) / run_name

    merged_config["save_root"] = save_root
    merged_config["run_name"] = run_name
    merged_config["allow_overwrite"] = allow_overwrite
    merged_config["data_save_path"] = str(run_root / "data")
    merged_config["model_save_path"] = str(run_root / "model")
    merged_config["save_result"] = str(run_root / "result")


def _is_nonempty_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def validate_output_paths(args: SimpleNamespace, stage: str) -> None:
    """在某个阶段写入非空实验输出目录前快速失败。"""

    if args.allow_overwrite:
        return

    if stage == "data":
        targets = [Path(args.data_save_path)]
    elif stage == "train":
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
    """添加入口使用的单个实验配置路径参数。"""

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
    """加载一个嵌套 config.yaml，并返回扁平 args 风格 namespace。"""

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

    merged_config = {}
    for _, section_config in section_items:
        merged_config.update(section_config)

    _raise_if_missing_required_keys(merged_config)
    merged_config.setdefault("optimizer", "adam")
    merged_config.setdefault("weight_decay", 0.0)
    merged_config.setdefault("momentum", 0.9)
    merged_config.setdefault("warmup_rounds", 0)
    merged_config.setdefault("warmup_start_learning_rate", None)
    _validate_training_hparams(merged_config)
    _derive_output_paths(merged_config)

    return SimpleNamespace(**merged_config)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "add_config_path_arguments",
    "load_args",
    "validate_output_paths",
]
