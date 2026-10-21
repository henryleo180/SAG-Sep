from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import torch

from sagsep_train.config import RECIPE_SCHEMA_VERSION

__all__ = [
    "CHECKPOINT_STEMS",
    "checkpoint_path",
    "save_checkpoint",
    "load_checkpoint",
    "strip_orig_mod",
    "restore_training_state",
]

CHECKPOINT_STEMS: dict[str, str] = {
    "latest": "subset_prob_latest.pt",
    "best": "subset_prob_best.pt",
}

_ORIG_MOD_PREFIX = "_orig_mod."


def checkpoint_path(run_dir: str | os.PathLike[str], kind: str) -> Path:
    try:
        stem = CHECKPOINT_STEMS[kind]
    except KeyError:
        raise KeyError(
            f"unknown checkpoint kind {kind!r}; valid: {', '.join(sorted(CHECKPOINT_STEMS))}"
        ) from None
    return Path(run_dir) / stem


def save_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    global_step: int,
    epoch: int,
    config: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "global_step": int(global_step),
        "epoch": int(epoch),
        "config": dict(config),
        "recipe_schema_version": RECIPE_SCHEMA_VERSION,
    }
    if extra:
        for key, value in extra.items():
            if key in payload:
                raise ValueError(f"save_checkpoint: extra key {key!r} collides with a payload key")
            payload[str(key)] = value

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()
    return target


def load_checkpoint(path: str | os.PathLike[str]) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: a checkpoint must hold a mapping, got {type(payload).__name__}")
    return payload


def strip_orig_mod(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    return {
        (key[len(_ORIG_MOD_PREFIX):] if key.startswith(_ORIG_MOD_PREFIX) else key): value
        for key, value in state_dict.items()
    }


def restore_training_state(
    payload: Mapping[str, Any],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
) -> tuple[int, int]:
    if "model" not in payload:
        raise KeyError("checkpoint has no 'model' state dict")
    model.load_state_dict(payload["model"])

    opt_state = payload.get("optimizer")
    if optimizer is not None and opt_state is not None:
        optimizer.load_state_dict(opt_state)

    sched_state = payload.get("scheduler")
    if scheduler is not None and sched_state is not None:
        scheduler.load_state_dict(sched_state)

    global_step = int(payload.get("global_step", 0))
    epoch = int(payload.get("epoch", 0))
    return global_step, epoch
