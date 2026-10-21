from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch

__all__ = [
    "CHECKPOINT_KINDS",
    "COMPILE_PREFIX",
    "checkpoint_filename",
    "load_run_state_dict",
    "strip_compile_prefix",
    "tensor_hashes",
    "tensor_specs",
    "write_checkpoint",
]

COMPILE_PREFIX = "_orig_mod."

CHECKPOINT_KINDS: Dict[str, str] = {
    "best": "subset_prob_best.pt",
    "latest": "subset_prob_latest.pt",
}


def checkpoint_filename(checkpoint: str) -> str:
    try:
        return CHECKPOINT_KINDS[checkpoint]
    except KeyError:
        raise ValueError(
            f"unknown checkpoint kind {checkpoint!r}; "
            f"expected one of {sorted(CHECKPOINT_KINDS)}"
        ) from None


def strip_compile_prefix(
    state_dict: Mapping[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], bool]:
    keys = list(state_dict)
    prefixed = [k for k in keys if k.startswith(COMPILE_PREFIX)]
    if not prefixed:
        return dict(state_dict), False
    if len(prefixed) != len(keys):
        plain = [k for k in keys if not k.startswith(COMPILE_PREFIX)]
        raise ValueError(
            f"checkpoint mixes {COMPILE_PREFIX!r}-prefixed and plain keys "
            f"({len(prefixed)} vs {len(plain)}); first plain key: {plain[0]!r}"
        )
    out: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        stripped = key[len(COMPILE_PREFIX):]
        if stripped in out:
            raise ValueError(f"stripping {COMPILE_PREFIX!r} collides on key {stripped!r}")
        out[stripped] = value
    return out, True


def load_run_state_dict(
    run_dir: str | Path, checkpoint: str = "best"
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    run_path = Path(run_dir)
    path = run_path / checkpoint_filename(checkpoint)
    if not path.exists():
        raise FileNotFoundError(
            f"checkpoint {checkpoint!r} not found in run directory: {path.name} is missing"
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} is not a dict, got {type(payload).__name__}")

    meta: Dict[str, Any] = {"checkpoint_kind": checkpoint, "checkpoint_file": path.name}
    if "model" in payload:
        raw = payload["model"]
        meta["payload_kind"] = "training_payload"
        for key in ("global_step", "epoch"):
            if payload.get(key) is not None:
                meta[key] = int(payload[key])
    else:
        raw = payload
        meta["payload_kind"] = "state_dict"
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path.name} carries no state dict")

    state_dict, stripped = strip_compile_prefix(raw)
    meta["compile_prefix_stripped"] = stripped
    meta["n_tensors"] = len(state_dict)
    return state_dict, meta


def write_checkpoint(path: str | Path, state_dict: Mapping[str, torch.Tensor]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": dict(state_dict)}, out)
    return out


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def tensor_hashes(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, str]:
    return {name: _tensor_sha256(tensor) for name, tensor in state_dict.items()}


def tensor_specs(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name, tensor in state_dict.items():
        out[name] = {
            "sha256": _tensor_sha256(tensor),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "shape": [int(s) for s in tensor.shape],
        }
    return out
