import os
import random

import numpy as np
import torch


def get_checkpoint_dir(args):
    checkpoint_dir = os.path.join(args.model_save_path, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def get_latest_checkpoint_path(args):
    return os.path.join(get_checkpoint_dir(args), "latest.pth")


def resolve_checkpoint_path(args):
    checkpoint_path = getattr(args, "resume_checkpoint_path", "latest")
    if checkpoint_path is None or checkpoint_path == "" or checkpoint_path == "latest":
        return get_latest_checkpoint_path(args)
    return checkpoint_path


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(rng_state):
    if not rng_state:
        return

    if "python" in rng_state and rng_state["python"] is not None:
        random.setstate(rng_state["python"])
    if "numpy" in rng_state and rng_state["numpy"] is not None:
        np.random.set_state(rng_state["numpy"])
    if "torch" in rng_state and rng_state["torch"] is not None:
        torch.set_rng_state(rng_state["torch"])

    cuda_state = rng_state.get("cuda")
    if torch.cuda.is_available() and cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


def _atomic_torch_save(payload, final_path):
    tmp_path = f"{final_path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, final_path)


def save_training_checkpoint(args, model, completed_round, logger=None):
    checkpoint_dir = get_checkpoint_dir(args)
    completed_round = int(completed_round)
    checkpoint = {
        "checkpoint_version": 1,
        "completed_round": completed_round,
        "server_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "rng_state": capture_rng_state(),
        "args": vars(args),
    }

    round_path = os.path.join(checkpoint_dir, f"checkpoint_round_{completed_round:06d}.pth")
    latest_path = get_latest_checkpoint_path(args)
    _atomic_torch_save(checkpoint, round_path)
    _atomic_torch_save(checkpoint, latest_path)

    if logger is not None:
        logger.info(f"--checkpoint_saved : round={completed_round} path={latest_path}\n")


def load_training_checkpoint(args, model, logger=None):
    checkpoint_path = resolve_checkpoint_path(args)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Resume checkpoint does not exist: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["server_state_dict"])

    if bool(getattr(args, "restore_rng_state", True)):
        restore_rng_state(checkpoint.get("rng_state"))

    completed_round = int(checkpoint["completed_round"])
    if logger is not None:
        logger.info(f"--resume_checkpoint_path : {checkpoint_path}\n")
        logger.info(f"--resume_completed_round : {completed_round}\n")
        logger.info(f"--resume_next_round : {completed_round + 1}\n")

    return completed_round
