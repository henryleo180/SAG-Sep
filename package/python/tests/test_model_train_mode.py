from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_golden_cases import golden_cases
from sagsep.model import (
    GraphFeatures,
    PolicyModel,
    SubsetProbNet,
    build_policy_network,
    load_policy_model,
)

MODEL_NAME = "cvrp-two-index-best"
UNION_CASES = ("rand_n12_k3", "sparse_n25_k6", "inst_n50_cd1")
UNUSED_PREFIX = "backbone.backbone.edge_update."


@pytest.fixture(scope="module", autouse=True)
def _deterministic_threads() -> None:
    torch.set_num_threads(1)


@pytest.fixture(scope="module")
def ckpt(models_dir: Path) -> Path:
    path = models_dir / f"{MODEL_NAME}.ckpt"
    if not path.exists():
        pytest.skip(f"released checkpoint not present: {path}")
    return path


@pytest.fixture(scope="module")
def manifest(models_dir: Path) -> dict:
    path = models_dir / f"{MODEL_NAME}.yaml"
    if not path.exists():
        pytest.skip(f"released manifest not present: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="module")
def cases(repo_root: Path) -> Dict[str, GraphFeatures]:
    return golden_cases(repo_root)


def _released_state_dict(ckpt: Path) -> Dict[str, "torch.Tensor"]:
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    return payload


def _trained_net(manifest: dict, ckpt: Path) -> SubsetProbNet:
    net = build_policy_network(manifest["model"])
    net.load_state_dict(_released_state_dict(ckpt))
    return net


def _replica_tensors(feats: GraphFeatures) -> Tuple["torch.Tensor", ...]:
    edge_index = torch.from_numpy(np.array(feats.edge_index))
    edge_attr = torch.from_numpy(np.array(feats.edge_attr))
    node_attr = torch.from_numpy(np.array(feats.node_attr).transpose(1, 0, 2).copy())
    return edge_index, node_attr, edge_attr


def _union_tensors(feats: GraphFeatures) -> Tuple["torch.Tensor", ...]:
    edge_index = np.array(feats.edge_index)
    edge_attr = np.array(feats.edge_attr)
    node_attr = np.array(feats.node_attr)
    n_rep, n_nodes, node_dim = node_attr.shape
    union_edge_index = np.concatenate(
        [edge_index + k * n_nodes for k in range(n_rep)], axis=1
    ).astype(np.int64)
    union_edge_attr = np.concatenate([edge_attr] * n_rep, axis=0).astype(np.float32)
    union_node_attr = node_attr.reshape(n_rep * n_nodes, node_dim).astype(np.float32)
    return (
        torch.from_numpy(union_edge_index),
        torch.from_numpy(union_node_attr),
        torch.from_numpy(union_edge_attr),
    )


def _bits(tensor: "torch.Tensor") -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


# ---------------------------------------------------------------------------
# 1. the trainer builds the released architecture, not a lookalike
# ---------------------------------------------------------------------------


def test_build_policy_network_matches_the_released_checkpoint(manifest, ckpt) -> None:
    net = build_policy_network(manifest["model"])
    assert isinstance(net, SubsetProbNet)
    assert net.training is True, "a freshly built net is the trainer's, in train mode"
    assert all(p.requires_grad for p in net.parameters())

    released = _released_state_dict(ckpt)
    assert sorted(net.state_dict().keys()) == sorted(released.keys())
    incompatible = net.load_state_dict(released, strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys


def test_build_policy_network_keeps_the_strict_config_whitelist(manifest) -> None:
    bad = dict(manifest["model"])
    bad["not_a_model_key"] = 1
    with pytest.raises(ValueError, match="Unsupported model config keys"):
        build_policy_network(bad)


# ---------------------------------------------------------------------------
# 2. dropout: live in train(), identity in eval()
# ---------------------------------------------------------------------------


def test_dropout_is_live_in_train_and_identity_in_eval(manifest, ckpt, cases) -> None:
    net = _trained_net(manifest, ckpt)
    feats = cases["rand_n12_k3"]
    edge_index, node_attr, edge_attr = _union_tensors(feats)

    net.train()
    with torch.no_grad():
        torch.manual_seed(0)
        first = net.forward_batch(edge_index, node_attr, edge_attr)
        torch.manual_seed(1)
        second = net.forward_batch(edge_index, node_attr, edge_attr)
    assert first.shape == (node_attr.shape[0], 1)
    assert _bits(first) != _bits(second), "dropout must perturb a train-mode forward"

    net.eval()
    with torch.no_grad():
        torch.manual_seed(0)
        third = net.forward_batch(edge_index, node_attr, edge_attr)
        torch.manual_seed(1)
        fourth = net.forward_batch(edge_index, node_attr, edge_attr)
        stacked = net.forward_replicas(
            edge_index, node_attr[:, None, :], edge_attr, 0
        ).reshape(node_attr.shape[0], 1)
    assert _bits(third) == _bits(fourth), "eval() must not depend on the RNG"
    assert _bits(third) == _bits(stacked)


# ---------------------------------------------------------------------------
# 3. the training layout and the serving layout are the same function
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_name", UNION_CASES)
def test_forward_batch_union_equals_forward_replicas(manifest, ckpt, cases, case_name) -> None:
    feats = cases[case_name]
    net = _trained_net(manifest, ckpt)
    net.eval()
    n_rep, n_nodes = feats.vehicle_count, feats.n_nodes

    with torch.no_grad():
        union = net.forward_batch(*_union_tensors(feats))
        edge_index, node_attr, edge_attr = _replica_tensors(feats)
        stacked = net.forward_replicas(edge_index, node_attr, edge_attr, 0)
    graph_major = stacked.permute(1, 0, 2).reshape(n_rep * n_nodes, 1)
    assert union.shape == graph_major.shape
    assert _bits(union) == _bits(graph_major), case_name


# ---------------------------------------------------------------------------
# 4. a training step actually trains
# ---------------------------------------------------------------------------


def test_train_step_produces_gradients_everywhere_the_forward_reaches(manifest, cases) -> None:
    net = build_policy_network(manifest["model"])
    net.train()
    torch.manual_seed(7)
    edge_index, node_attr, edge_attr = _union_tensors(cases["rand_n12_k3"])

    logits = net.forward_batch(edge_index, node_attr, edge_attr)
    target = torch.zeros_like(logits)
    target[::2] = 1.0
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()

    assert torch.isfinite(loss) and float(loss.detach()) > 0.0
    for name, param in net.named_parameters():
        if name.startswith(UNUSED_PREFIX):
            assert param.grad is None, name
            continue
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        assert float(param.grad.abs().sum()) > 0.0, name


# ---------------------------------------------------------------------------
# 5. edge chunking is an eval-only device
# ---------------------------------------------------------------------------


def test_edge_chunking_is_disabled_while_training(manifest, ckpt, cases) -> None:
    net = _trained_net(manifest, ckpt)
    net.train()
    edge_index, node_attr, edge_attr = _replica_tensors(cases["rand_n12_k3"])
    with torch.no_grad():
        torch.manual_seed(11)
        chunked = net.forward_replicas(edge_index, node_attr, edge_attr, 7)
        torch.manual_seed(11)
        whole = net.forward_replicas(edge_index, node_attr, edge_attr, 0)
    assert _bits(chunked) == _bits(whole)


# ---------------------------------------------------------------------------
# 6. from_network: the caller keeps ownership of the net
# ---------------------------------------------------------------------------


def test_from_network_keeps_mode_and_matches_the_loader(manifest, ckpt, cases) -> None:
    reference = load_policy_model(ckpt, device="cpu")
    net = _trained_net(manifest, ckpt)
    net.train()

    wrapper = PolicyModel.from_network(net, device="cpu")
    assert net.training is True, "from_network must not switch the net to eval()"
    assert all(p.requires_grad for p in net.parameters())
    assert wrapper.checkpoint_path == "<in-memory>"
    assert wrapper.device == "cpu"

    net.eval()
    for name, feats in cases.items():
        got = np.asarray(wrapper.forward_probs(feats), dtype=np.float32)
        want = np.asarray(reference.forward_probs(feats), dtype=np.float32)
        assert got.tobytes() == want.tobytes(), name
    net.train()
    assert net.training is True


# ---------------------------------------------------------------------------
# 7. the optional input_contract manifest section
# ---------------------------------------------------------------------------


GOOD_CONTRACT = {
    "graph_builder": "partial_2hop_max_min_v1",
    "m_value_closed_interval": True,
    "partial_connect_cross_depot": True,
}


@pytest.fixture(scope="module")
def manifest_dir(tmp_path_factory, ckpt: Path) -> Path:
    target = tmp_path_factory.mktemp("manifest")
    shutil.copy2(ckpt, target / "m.ckpt")
    return target


def _write_manifest(manifest_dir: Path, manifest: dict, **sections: object) -> Path:
    payload = {"model": manifest["model"]}
    payload.update(sections)
    path = manifest_dir / "m.yaml"
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return manifest_dir / "m.ckpt"


def test_manifest_with_input_contract_loads(manifest_dir, manifest, cases, ckpt) -> None:
    path = _write_manifest(
        manifest_dir,
        manifest,
        artifact_schema_version=1,
        input_contract=dict(GOOD_CONTRACT),
        provenance={"trained_by": "sagsep_train", "run": "unit-test"},
    )
    model = load_policy_model(path, device="cpu", partial_connect_cross_depot=True)
    assert not model._net.training
    assert all(not p.requires_grad for p in model._net.parameters())

    reference = load_policy_model(ckpt, device="cpu")
    feats = cases["rand_n12_k3"]
    assert model.forward_probs(feats).tobytes() == reference.forward_probs(feats).tobytes()

    load_policy_model(path, device="cpu")


def test_manifest_rejects_a_conflicting_cross_depot_flag(manifest_dir, manifest) -> None:
    path = _write_manifest(manifest_dir, manifest, input_contract=dict(GOOD_CONTRACT))
    with pytest.raises(ValueError, match="partial_connect_cross_depot"):
        load_policy_model(path, device="cpu", partial_connect_cross_depot=False)


def test_manifest_rejects_a_foreign_graph_builder(manifest_dir, manifest) -> None:
    contract = dict(GOOD_CONTRACT, graph_builder="complete_graph_v0")
    path = _write_manifest(manifest_dir, manifest, input_contract=contract)
    with pytest.raises(ValueError, match="graph_builder"):
        load_policy_model(path, device="cpu")


def test_manifest_rejects_an_open_m_interval(manifest_dir, manifest) -> None:
    contract = dict(GOOD_CONTRACT, m_value_closed_interval=False)
    path = _write_manifest(manifest_dir, manifest, input_contract=contract)
    with pytest.raises(ValueError, match="m_value_closed_interval"):
        load_policy_model(path, device="cpu")


def test_manifest_rejects_a_non_mapping_contract(manifest_dir, manifest) -> None:
    path = _write_manifest(manifest_dir, manifest, input_contract=["not", "a", "mapping"])
    with pytest.raises(ValueError, match="must be a mapping"):
        load_policy_model(path, device="cpu")


def test_manifest_without_input_contract_is_unaffected(manifest_dir, manifest, cases, ckpt) -> None:
    path = _write_manifest(manifest_dir, manifest, artifact_schema_version=1)
    for flag in (None, True, False):
        model = load_policy_model(path, device="cpu", partial_connect_cross_depot=flag)
        reference = load_policy_model(ckpt, device="cpu")
        feats = cases["rand_n12_k3"]
        assert model.forward_probs(feats).tobytes() == reference.forward_probs(feats).tobytes()
