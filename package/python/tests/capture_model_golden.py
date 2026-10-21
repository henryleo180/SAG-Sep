from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import torch

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_golden_cases import CHUNK_ENVS, EDGE_CHUNKS, golden_cases
from sagsep.model import load_policy_model


def _set_env(echunk: str, kchunk: str) -> None:
    os.environ["SAGSEP_ECHUNK"] = echunk
    os.environ["SAGSEP_KCHUNK"] = kchunk


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    repo_root = _PKG_ROOT.parents[1]
    cases = golden_cases(repo_root)
    out: dict = {}
    for echunk, kchunk in CHUNK_ENVS:
        _set_env(echunk, kchunk)
        model = load_policy_model(args.ckpt, device="cpu")
        for name, feats in cases.items():
            probs = model.forward_probs(feats)
            out[f"{name}/probs/e{echunk}_k{kchunk}"] = np.asarray(probs, dtype=np.float32)
    _set_env("auto", "auto")
    model = load_policy_model(args.ckpt, device="cpu")
    net = model._net
    for name, feats in cases.items():
        out[f"{name}/in/edge_index"] = np.asarray(feats.edge_index, dtype=np.int64)
        out[f"{name}/in/edge_attr"] = np.asarray(feats.edge_attr, dtype=np.float32)
        out[f"{name}/in/node_attr"] = np.asarray(feats.node_attr, dtype=np.float32)
        ei = torch.from_numpy(np.array(feats.edge_index))
        ef = torch.from_numpy(np.array(feats.edge_attr))
        nf = torch.from_numpy(np.array(feats.node_attr).transpose(1, 0, 2).copy())
        with torch.no_grad():
            for c in EDGE_CHUNKS:
                logits = net.forward_replicas(ei, nf, ef, c)
                out[f"{name}/logits/chunk{c}"] = logits.detach().cpu().numpy().astype(np.float32)
    sd = net.state_dict()
    keys = sorted(sd.keys())
    out["state_dict/keys"] = np.asarray(keys)
    out["state_dict/sha256"] = np.asarray(
        [hashlib.sha256(sd[k].detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k in keys]
    )
    out["meta/torch"] = np.asarray(torch.__version__)
    out["meta/numpy"] = np.asarray(np.__version__)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"wrote {args.out}: {len(cases)} cases, {len(keys)} tensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
