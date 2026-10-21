from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_golden_cases import CHUNK_ENVS, EDGE_CHUNKS, golden_cases
from sagsep.model import GraphFeatures, load_policy_model

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "model_golden"
MODELS = ("cvrp-two-index-best", "cvrp-spf-best")


def _fixture(name: str):
    path = FIXTURE_DIR / f"{name}.npz"
    if not path.exists():
        pytest.skip(f"golden fixture not captured: {path}")
    return np.load(path, allow_pickle=False)


def _cases_match_fixture(cases, fx, name):
    for case, feats in cases.items():
        assert fx[f"{case}/in/edge_index"].tobytes() == np.asarray(feats.edge_index, dtype=np.int64).tobytes(), case
        assert fx[f"{case}/in/edge_attr"].tobytes() == np.asarray(feats.edge_attr, dtype=np.float32).tobytes(), case
        assert fx[f"{case}/in/node_attr"].tobytes() == np.asarray(feats.node_attr, dtype=np.float32).tobytes(), case


@pytest.mark.parametrize("model_name", MODELS)
def test_state_dict_unchanged(models_dir, model_name):
    fx = _fixture(model_name)
    ckpt = models_dir / f"{model_name}.ckpt"
    if not ckpt.exists():
        pytest.skip(f"released checkpoint not present: {ckpt}")
    model = load_policy_model(ckpt, device="cpu")
    sd = model._net.state_dict()
    keys = sorted(sd.keys())
    assert list(fx["state_dict/keys"]) == keys
    hashes = [hashlib.sha256(sd[k].detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k in keys]
    assert list(fx["state_dict/sha256"]) == hashes
    assert not model._net.training
    assert all(not p.requires_grad for p in model._net.parameters())


@pytest.mark.parametrize("model_name", MODELS)
def test_forward_probs_bitwise(models_dir, model_name, repo_root, monkeypatch):
    fx = _fixture(model_name)
    ckpt = models_dir / f"{model_name}.ckpt"
    if not ckpt.exists():
        pytest.skip(f"released checkpoint not present: {ckpt}")
    torch.set_num_threads(1)
    cases = golden_cases(repo_root)
    _cases_match_fixture(cases, fx, model_name)
    for echunk, kchunk in CHUNK_ENVS:
        monkeypatch.setenv("SAGSEP_ECHUNK", echunk)
        monkeypatch.setenv("SAGSEP_KCHUNK", kchunk)
        model = load_policy_model(ckpt, device="cpu")
        for case, feats in cases.items():
            probs = np.asarray(model.forward_probs(feats), dtype=np.float32)
            want = fx[f"{case}/probs/e{echunk}_k{kchunk}"]
            assert probs.shape == want.shape, case
            assert probs.tobytes() == want.tobytes(), f"{model_name} {case} e{echunk} k{kchunk}"


@pytest.mark.parametrize("model_name", MODELS)
def test_forward_replicas_bitwise(models_dir, model_name, repo_root, monkeypatch):
    fx = _fixture(model_name)
    ckpt = models_dir / f"{model_name}.ckpt"
    if not ckpt.exists():
        pytest.skip(f"released checkpoint not present: {ckpt}")
    torch.set_num_threads(1)
    monkeypatch.setenv("SAGSEP_ECHUNK", "auto")
    monkeypatch.setenv("SAGSEP_KCHUNK", "auto")
    model = load_policy_model(ckpt, device="cpu")
    net = model._net
    for case, feats in golden_cases(repo_root).items():
        ei = torch.from_numpy(np.array(feats.edge_index))
        ef = torch.from_numpy(np.array(feats.edge_attr))
        nf = torch.from_numpy(np.array(feats.node_attr).transpose(1, 0, 2).copy())
        with torch.no_grad():
            for c in EDGE_CHUNKS:
                logits = net.forward_replicas(ei, nf, ef, c).detach().cpu().numpy().astype(np.float32)
                assert logits.tobytes() == fx[f"{case}/logits/chunk{c}"].tobytes(), f"{case} chunk{c}"
