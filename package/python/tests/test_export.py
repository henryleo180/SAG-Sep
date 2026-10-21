from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_golden_cases import golden_cases

from sagsep.model import build_policy_network, load_policy_model
from sagsep_train.export import (
    ARTIFACT_SCHEMA_VERSION,
    GRAPH_BUILDER,
    REPORT_SCHEMA_VERSION,
    ExportReport,
    builtin_cases,
    compare,
    compare_onnx_files,
    export_run,
    load_run_state_dict,
    ort_session,
    project_manifest,
    tensor_hashes,
    write_checkpoint,
    write_manifest,
)
from sagsep_train.export import parity as parity_mod
from sagsep_train.export.checkpoint import (
    CHECKPOINT_KINDS,
    COMPILE_PREFIX,
    checkpoint_filename,
    strip_compile_prefix,
    tensor_specs,
)

REF_RUN_DIR = os.environ.get("SAGSEP_REF_RUN_DIR") or ""
SPF_RUN_DIR = os.environ.get("SAGSEP_SPF_RUN_DIR") or ""

MEMORY_CEILING = 8 * 1024**3


def _have(module: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


HAVE_ONNX = _have("onnx") and _have("onnxruntime")
needs_onnx = pytest.mark.skipif(not HAVE_ONNX, reason="onnx / onnxruntime not installed")
needs_ref = pytest.mark.skipif(not REF_RUN_DIR, reason="SAGSEP_REF_RUN_DIR is not set")
needs_spf = pytest.mark.skipif(not SPF_RUN_DIR, reason="SAGSEP_SPF_RUN_DIR is not set")


# ---------------------------------------------------------------------------
# a synthetic run directory: small enough to build in every test session
# ---------------------------------------------------------------------------

TINY_MODEL_CFG: Dict[str, Any] = {
    "node_dim": 2,
    "edge_dim": 2,
    "latent_dim": 16,
    "n_layer": 1,
    "backbone_type": "transformer",
    "transformer_params": {"num_heads": 2, "dropout": 0.1, "use_edge_features": True},
    "policy_params": {"hidden": [8], "out_act": "Identity"},
    "debug": False,
}

TINY_CASES = {"tiny": parity_mod.random_features(3, n_nodes=9, vehicles=2, density=0.4)}


def _tiny_state_dict(seed: int = 0) -> Dict[str, "torch.Tensor"]:
    torch.manual_seed(seed)
    return dict(build_policy_network(TINY_MODEL_CFG).state_dict())


def _write_run(
    run_dir: Path,
    *,
    kind: str = "best",
    cross_depot: bool = True,
    m_closed: bool = True,
    bare_state_dict: bool = False,
    prefix: str = "",
    global_step: int = 7,
    epoch: int = 2,
    seed: int = 0,
) -> Dict[str, "torch.Tensor"]:
    run_dir.mkdir(parents=True, exist_ok=True)
    state_dict = _tiny_state_dict(seed)
    stored = {prefix + key: value for key, value in state_dict.items()}
    payload: Any = stored
    if not bare_state_dict:
        payload = {
            "model": stored,
            "optimizer": {"state": {}},
            "scheduler": {"last_epoch": epoch},
            "global_step": global_step,
            "epoch": epoch,
            "config": {"trainer": {"seed": 42}},
        }
    torch.save(payload, run_dir / checkpoint_filename(kind))
    snapshot = {
        "model": dict(TINY_MODEL_CFG),
        "data": {
            "dataset_path": "somewhere",
            "partial_connect": 2,
            "partial_connect_cross_depot": cross_depot,
            "m_value_closed_interval": m_closed,
        },
        "optimizer": {"name": "Adam", "lr": 0.001},
        "trainer": {"seed": 42, "max_epochs": 3},
    }
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8"
    )
    return state_dict


# ---------------------------------------------------------------------------
# checkpoint.py
# ---------------------------------------------------------------------------


def test_checkpoint_kinds_cover_the_trainer_grid() -> None:
    assert set(CHECKPOINT_KINDS) == {"best", "latest"}
    for kind, filename in CHECKPOINT_KINDS.items():
        assert checkpoint_filename(kind) == filename
        assert filename.startswith("subset_prob_") and filename.endswith(".pt")
    with pytest.raises(ValueError, match="unknown checkpoint kind"):
        checkpoint_filename("second_best")


def test_strip_compile_prefix_is_all_or_nothing() -> None:
    plain = {"a.weight": torch.zeros(2), "b.bias": torch.ones(3)}
    same, stripped = strip_compile_prefix(plain)
    assert stripped is False and same == plain

    compiled = {COMPILE_PREFIX + key: value for key, value in plain.items()}
    out, stripped = strip_compile_prefix(compiled)
    assert stripped is True and list(out) == list(plain)

    mixed = {COMPILE_PREFIX + "a.weight": torch.zeros(2), "b.bias": torch.ones(3)}
    with pytest.raises(ValueError, match="mixes"):
        strip_compile_prefix(mixed)


@pytest.mark.parametrize("prefix", ["", COMPILE_PREFIX])
@pytest.mark.parametrize("bare", [False, True])
def test_load_run_state_dict_normalises_every_shape(tmp_path, prefix, bare) -> None:
    want = _write_run(tmp_path / "run", prefix=prefix, bare_state_dict=bare)
    state_dict, meta = load_run_state_dict(tmp_path / "run", "best")

    assert list(state_dict) == list(want)
    assert tensor_hashes(state_dict) == tensor_hashes(want)
    assert meta["compile_prefix_stripped"] is bool(prefix)
    assert meta["checkpoint_kind"] == "best"
    assert meta["n_tensors"] == len(want)
    if bare:
        assert "global_step" not in meta and meta["payload_kind"] == "state_dict"
    else:
        assert meta["global_step"] == 7 and meta["epoch"] == 2


def test_load_run_state_dict_reports_a_missing_checkpoint(tmp_path) -> None:
    _write_run(tmp_path / "run")
    with pytest.raises(FileNotFoundError, match="subset_prob_latest.pt"):
        load_run_state_dict(tmp_path / "run", "latest")


def test_tensor_hashes_are_content_addressed(tmp_path) -> None:
    state_dict = _tiny_state_dict()
    before = tensor_hashes(state_dict)
    key = next(iter(state_dict))
    perturbed = dict(state_dict)
    changed = state_dict[key].clone()
    changed.view(-1)[0] += 1.0
    perturbed[key] = changed
    after = tensor_hashes(perturbed)
    assert [k for k in before if before[k] != after[k]] == [key]

    specs = tensor_specs(state_dict)
    assert specs[key]["sha256"] == before[key]
    assert specs[key]["dtype"] == "float32"
    assert specs[key]["shape"] == list(state_dict[key].shape)


def test_write_checkpoint_produces_the_released_shape(tmp_path) -> None:
    state_dict = _tiny_state_dict()
    path = write_checkpoint(tmp_path / "nested" / "m.ckpt", state_dict)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert list(payload) == ["model"]
    assert tensor_hashes(payload["model"]) == tensor_hashes(state_dict)


# ---------------------------------------------------------------------------
# manifest.py
# ---------------------------------------------------------------------------


def _snapshot(**data: Any) -> Dict[str, Any]:
    return {"model": dict(TINY_MODEL_CFG), "data": data, "trainer": {"seed": 1}}


def test_project_manifest_keeps_exactly_four_sections() -> None:
    manifest = project_manifest(
        _snapshot(partial_connect_cross_depot=False, m_value_closed_interval=True),
        run_id="run-1",
        global_step=1234,
        checkpoint_kind="best",
    )
    assert list(manifest) == [
        "artifact_schema_version",
        "model",
        "input_contract",
        "provenance",
    ]
    assert manifest["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert list(manifest["model"]) == list(TINY_MODEL_CFG)
    assert manifest["model"] == TINY_MODEL_CFG
    assert manifest["input_contract"] == {
        "graph_builder": GRAPH_BUILDER,
        "partial_connect_cross_depot": False,
        "m_value_closed_interval": True,
    }
    assert manifest["provenance"] == {
        "run_id": "run-1",
        "global_step": 1234,
        "checkpoint_kind": "best",
    }


def test_project_manifest_defaults_the_contract_when_the_snapshot_is_silent() -> None:
    manifest = project_manifest(
        {"model": dict(TINY_MODEL_CFG)},
        run_id="r",
        global_step=None,
        checkpoint_kind="latest",
    )
    contract = manifest["input_contract"]
    assert contract["partial_connect_cross_depot"] is True
    assert contract["m_value_closed_interval"] is True
    assert manifest["provenance"]["global_step"] is None


def test_project_manifest_refuses_what_cannot_be_served() -> None:
    with pytest.raises(ValueError, match="closed interval"):
        project_manifest(
            _snapshot(m_value_closed_interval=False),
            run_id="r",
            global_step=None,
            checkpoint_kind="best",
        )
    with pytest.raises(ValueError, match="no model"):
        project_manifest(
            {"data": {}}, run_id="r",
            global_step=None, checkpoint_kind="best",
        )
    with pytest.raises(ValueError, match="must be a boolean"):
        project_manifest(
            _snapshot(partial_connect_cross_depot="false"),
            run_id="r",
            global_step=None, checkpoint_kind="best",
        )


@pytest.mark.parametrize("run_id", ["/absolute/run", "parent/run", "", "."])
def test_project_manifest_run_id_is_a_base_name(run_id) -> None:
    with pytest.raises(ValueError, match="base name"):
        project_manifest(
            _snapshot(), run_id=run_id,
            global_step=None, checkpoint_kind="best",
        )


def test_write_manifest_header_and_round_trip(tmp_path) -> None:
    manifest = project_manifest(
        _snapshot(partial_connect_cross_depot=True),
        run_id="r", global_step=5, checkpoint_kind="best",
    )
    path = write_manifest(tmp_path / "m.yaml", manifest, name="m")
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == (
        "# Model manifest for m.ckpt, projected from the training snapshot "
        "by sagsep_train.export."
    )
    assert lines[1].startswith("# The model section is the exact key set the loader reads")
    parsed = yaml.safe_load(text)
    assert parsed == manifest
    assert list(parsed) == list(manifest)
    assert list(parsed["model"]) == list(TINY_MODEL_CFG)


# ---------------------------------------------------------------------------
# export_run on a synthetic run
# ---------------------------------------------------------------------------


def test_export_run_round_trips_a_synthetic_run(tmp_path) -> None:
    want = _write_run(tmp_path / "run-42")
    report = export_run(
        tmp_path / "run-42",
        name="tiny",
        out_dir=tmp_path / "out",
        onnx=False,
        parity_cases=TINY_CASES,
    )
    assert isinstance(report, ExportReport)
    assert report.name == "tiny" and report.checkpoint_kind == "best"
    assert report.run_id == "run-42" and report.global_step == 7 and report.epoch == 2
    assert report.onnx_path is None and report.onnx_sha256 is None and report.parity is None
    assert report.tensor_sha256 == tensor_hashes(want)

    assert report.ckpt_path.name == "tiny.ckpt"
    assert report.manifest_path.name == "tiny.yaml"
    assert report.report_path is not None and report.report_path.name == "tiny.export.json"

    payload = json.loads(report.report_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == REPORT_SCHEMA_VERSION
    assert payload["artifacts"] == {
        "checkpoint": "tiny.ckpt", "manifest": "tiny.yaml", "onnx": None
    }
    assert payload["provenance"]["run_id"] == "run-42"
    assert payload["n_tensors"] == len(want)
    assert payload["tensor_sha256"] == tensor_hashes(want)
    assert "torch" in payload["versions"]
    blob = json.dumps(payload)
    assert str(tmp_path) not in blob and "/" not in payload["artifacts"]["checkpoint"]

    policy = load_policy_model(report.ckpt_path, device="cpu", partial_connect_cross_depot=True)
    probs = policy.forward_probs(TINY_CASES["tiny"])
    assert probs.shape == (2, 9) and np.isfinite(probs).all()


def test_export_run_refuses_a_run_without_a_snapshot(tmp_path) -> None:
    run = tmp_path / "run"
    _write_run(run)
    (run / "config_snapshot.yaml").unlink()
    with pytest.raises(FileNotFoundError, match="config_snapshot.yaml"):
        export_run(run, name="tiny", out_dir=tmp_path / "out", onnx=False)


def test_export_run_carries_the_cross_depot_setting_into_the_contract(tmp_path) -> None:
    _write_run(tmp_path / "run", cross_depot=False)
    report = export_run(
        tmp_path / "run", name="tiny", out_dir=tmp_path / "out", onnx=False,
        parity_cases=TINY_CASES,
    )
    manifest = yaml.safe_load(report.manifest_path.read_text(encoding="utf-8"))
    assert manifest["input_contract"]["partial_connect_cross_depot"] is False
    load_policy_model(report.ckpt_path, device="cpu", partial_connect_cross_depot=False)
    with pytest.raises(ValueError, match="partial_connect_cross_depot"):
        load_policy_model(report.ckpt_path, device="cpu", partial_connect_cross_depot=True)


def test_export_run_without_onnx_needs_no_runtime(tmp_path, monkeypatch) -> None:
    _write_run(tmp_path / "run")
    for module in ("onnx", "onnxruntime"):
        monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(ImportError):
        import onnxruntime

    report = export_run(
        tmp_path / "run", name="tiny", out_dir=tmp_path / "out", onnx=False,
        parity_cases=TINY_CASES,
    )
    assert report.onnx_path is None
    payload = json.loads(report.report_path.read_text(encoding="utf-8"))
    assert payload["onnx"] is None and payload["parity"] is None
    assert set(payload["versions"]) == {"torch"}


def test_export_run_rejects_an_unknown_case_selector(tmp_path) -> None:
    _write_run(tmp_path / "run")
    with pytest.raises(ValueError, match="parity_cases"):
        export_run(
            tmp_path / "run", name="tiny", out_dir=tmp_path / "out", onnx=False,
            parity_cases="all-of-them",
        )


def test_builtin_cases_are_deterministic_and_well_formed() -> None:
    first, second = builtin_cases(), builtin_cases()
    assert set(first) == set(second) and first
    for name, feats in first.items():
        other = second[name]
        assert feats.edge_index.tobytes() == other.edge_index.tobytes(), name
        assert feats.node_attr.tobytes() == other.node_attr.tobytes(), name
        assert feats.edge_attr.tobytes() == other.edge_attr.tobytes(), name
        assert feats.node_attr.shape == (feats.vehicle_count, feats.n_nodes, 2), name
        assert feats.edge_attr.shape == (feats.edge_index.shape[1], 2), name


# ---------------------------------------------------------------------------
# the released models, re-exported from the runs they came from
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def export_out_dir(tmp_path_factory) -> Path:
    override = os.environ.get("SAGSEP_EXPORT_OUT_DIR")
    if override:
        path = Path(override)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return tmp_path_factory.mktemp("export")


@pytest.fixture(scope="session")
def ref_export(export_out_dir, tmp_path_factory) -> ExportReport:
    if not REF_RUN_DIR:
        pytest.skip("SAGSEP_REF_RUN_DIR is not set")
    source = Path(REF_RUN_DIR).resolve()
    staged = tmp_path_factory.mktemp("ref-run")
    (staged / "config_snapshot.yaml").symlink_to(source / "config_snapshot.yaml")
    ref_ckpt = os.environ.get("SAGSEP_REF_CHECKPOINT_FILE", checkpoint_filename("best"))
    (staged / checkpoint_filename("best")).symlink_to(source / ref_ckpt)
    return export_run(
        staged,
        name="cvrp-two-index-best",
        out_dir=export_out_dir / "cvrp-two-index-best",
        checkpoint="best",
        onnx=HAVE_ONNX,
        verbose=True,
    )


@pytest.fixture(scope="session")
def spf_export(export_out_dir) -> ExportReport:
    if not SPF_RUN_DIR:
        pytest.skip("SAGSEP_SPF_RUN_DIR is not set")
    return export_run(
        SPF_RUN_DIR,
        name="cvrp-spf-best",
        out_dir=export_out_dir / "cvrp-spf-best",
        checkpoint="best",
        onnx=HAVE_ONNX,
        verbose=True,
    )


def _shipped_hashes(models_dir: Path, name: str) -> Dict[str, str]:
    payload = torch.load(models_dir / f"{name}.ckpt", map_location="cpu", weights_only=True)
    return tensor_hashes(payload["model"])


def _assert_reproduces_shipped(report: ExportReport, models_dir: Path, name: str) -> None:
    shipped = _shipped_hashes(models_dir, name)
    produced = tensor_hashes(
        torch.load(report.ckpt_path, map_location="cpu", weights_only=True)["model"]
    )
    assert list(produced) == list(shipped), "state-dict key order changed"
    differing = [k for k in shipped if produced.get(k) != shipped[k]]
    assert not differing, f"{len(differing)}/{len(shipped)} tensors differ, first {differing[:3]}"
    assert produced == report.tensor_sha256
    print(f"[{name}] {len(shipped)} tensors: per-tensor sha256 identical to the shipped ckpt")


@needs_ref
def test_reference_export_reproduces_the_shipped_checkpoint(ref_export, models_dir) -> None:
    _assert_reproduces_shipped(ref_export, models_dir, "cvrp-two-index-best")
    assert len(ref_export.tensor_sha256) == 96
    assert ref_export.checkpoint_kind == "best"
    assert ref_export.global_step is not None


@needs_ref
def test_reference_manifest_model_section_matches_the_shipped_one(
    ref_export, models_dir
) -> None:
    produced = yaml.safe_load(ref_export.manifest_path.read_text(encoding="utf-8"))
    shipped = yaml.safe_load((models_dir / "cvrp-two-index-best.yaml").read_text("utf-8"))
    assert produced["model"] == shipped["model"]
    assert list(produced["model"]) == list(shipped["model"])
    assert produced["input_contract"]["partial_connect_cross_depot"] is True
    assert produced["input_contract"]["graph_builder"] == GRAPH_BUILDER
    assert set(produced["provenance"]) == set(shipped["provenance"])
    assert "/" not in produced["provenance"]["run_id"]


@needs_ref
def test_reference_checkpoint_loads_under_its_own_contract(ref_export) -> None:
    ckpt = ref_export.ckpt_path
    assert load_policy_model(ckpt, device="cpu") is not None
    assert load_policy_model(ckpt, device="cpu", partial_connect_cross_depot=True) is not None
    with pytest.raises(ValueError, match="partial_connect_cross_depot"):
        load_policy_model(ckpt, device="cpu", partial_connect_cross_depot=False)


@needs_spf
def test_spf_export_reproduces_the_shipped_checkpoint(spf_export, models_dir) -> None:
    _assert_reproduces_shipped(spf_export, models_dir, "cvrp-spf-best")
    assert len(spf_export.tensor_sha256) == 96
    assert spf_export.checkpoint_kind == "best"


@needs_spf
def test_spf_manifest_pins_the_no_cross_depot_contract(spf_export, models_dir) -> None:
    produced = yaml.safe_load(spf_export.manifest_path.read_text(encoding="utf-8"))
    shipped = yaml.safe_load((models_dir / "cvrp-spf-best.yaml").read_text("utf-8"))
    assert produced["model"] == shipped["model"]
    assert produced["input_contract"]["partial_connect_cross_depot"] is False

    ckpt = spf_export.ckpt_path
    assert load_policy_model(ckpt, device="cpu", partial_connect_cross_depot=False) is not None
    with pytest.raises(ValueError, match="partial_connect_cross_depot"):
        load_policy_model(ckpt, device="cpu", partial_connect_cross_depot=True)


# ---------------------------------------------------------------------------
# the ONNX artefact
# ---------------------------------------------------------------------------


def _report_histograms(name: str, produced: Dict[str, int], shipped: Dict[str, int]) -> None:
    print(f"[{name}] op histogram, produced vs shipped:")
    for op in sorted(set(produced) | set(shipped)):
        a, b = produced.get(op, 0), shipped.get(op, 0)
        print(f"    {op:<22}{a:>6}{b:>6}{'' if a == b else '   <-- DIFFERS'}")


@needs_onnx
@pytest.mark.parametrize("fixture_name,model_name", [
    ("ref_export", "cvrp-two-index-best"),
    ("spf_export", "cvrp-spf-best"),
])
def test_onnx_op_histogram_matches_the_shipped_graph(
    request, models_dir, fixture_name, model_name
) -> None:
    from sagsep_train.export.onnx import REFERENCE_OPSET, graph_meta

    report = request.getfixturevalue(fixture_name)
    produced = graph_meta(report.onnx_path)
    shipped = graph_meta(models_dir / f"{model_name}.onnx")
    _report_histograms(model_name, produced["op_histogram"], shipped["op_histogram"])

    assert produced["opset"] == shipped["opset"] == REFERENCE_OPSET
    assert produced["inputs"] == ["edge_index", "node_attr", "edge_attr"]
    assert produced["outputs"] == ["probs"]
    assert produced["inputs"] == shipped["inputs"] and produced["outputs"] == shipped["outputs"]
    assert produced["op_histogram"] == shipped["op_histogram"]
    assert produced["op_histogram"].get("ScatterND", 0) > 0
    assert "Expand" not in produced["op_histogram"]
    assert "ScatterElements" not in produced["op_histogram"]


@needs_onnx
@pytest.mark.parametrize("fixture_name,model_name", [
    ("ref_export", "cvrp-two-index-best"),
    ("spf_export", "cvrp-spf-best"),
])
def test_onnx_output_is_bitwise_identical_to_the_shipped_graph(
    request, models_dir, repo_root, fixture_name, model_name
) -> None:
    report = request.getfixturevalue(fixture_name)
    cases = dict(builtin_cases())
    cases.update({f"golden_{k}": v for k, v in golden_cases(repo_root).items()})
    assert len(cases) >= 9 + 2

    result = compare_onnx_files(report.onnx_path, models_dir / f"{model_name}.onnx", cases)
    print(f"[{model_name}] produced vs shipped ONNX on {result['n_cases']} cases: "
          f"bitwise={result['bitwise']} max|delta|={result['max_abs_diff']:.3e}")
    if not result["bitwise"]:
        assert result["max_abs_diff"] < 1e-6, result["cases"]
        pytest.fail(
            f"produced graph is not bit-identical to the shipped one "
            f"(max|delta| = {result['max_abs_diff']:.3e}); investigate before releasing"
        )


@needs_onnx
@pytest.mark.parametrize("fixture_name,model_name", [
    ("ref_export", "cvrp-two-index-best"),
    ("spf_export", "cvrp-spf-best"),
])
def test_onnx_parity_against_torch(request, repo_root, fixture_name, model_name) -> None:
    from sagsep_train.export.onnx import load_export_network

    report = request.getfixturevalue(fixture_name)
    net = load_export_network(report.ckpt_path)
    cases = dict(builtin_cases())
    cases.update({f"golden_{k}": v for k, v in golden_cases(repo_root).items()})

    result = compare(net, ort_session(report.onnx_path), cases)
    print(f"[{model_name}] runtime vs torch, {len(result['cases'])} cases "
          f"-> {result['verdict']}, worst a = {result['max_abs_diff']:.3e}")
    for name, row in result["cases"].items():
        print(f"    {name:<24} K,N={tuple(row['shape'])!s:>10} E={row['n_edges']:>6}  "
              f"a={row['ort_vs_torch32']:.3e} b={row['torch32_vs_torch64']:.3e} "
              f"c={row['ort_vs_torch64']:.3e}  {row['rule']}")
    failed = sorted(name for name, row in result["cases"].items() if not row["ok"])
    assert not failed, failed
    assert result["ok"] is True

    assert report.parity is not None and report.parity["ok"] is True


MEMORY_PROBE = """
import json, resource, sys
import numpy as np
import onnxruntime as ort

# Imports nothing but numpy and the runtime, on purpose: what is being measured
# is what *running the graph* costs, which is what the C engine pays, and torch
# alone would add hundreds of megabytes of resident set that the graph never
# touches.  That is also why the three inputs are built here rather than through
# the case helpers -- those live next to the torch-backed model.
#
# The session options mirror export.parity.ort_session, and through it the C
# engine: one intra-op thread, sequential execution, memory arena on.  Measuring
# under a lighter allocator would answer a question nobody asks.
N, K, DENSITY, SEED = 400, 30, 0.45, 101
rng = np.random.default_rng(SEED)
iu, ju = np.triu_indices(N, 1)
keep = rng.random(iu.size) < DENSITY
u, v = iu[keep].astype(np.int64), ju[keep].astype(np.int64)
E = 2 * u.size
edge_index = np.empty((2, E), dtype=np.int64)          # both arcs of a pair, adjacent
edge_index[0, 0::2], edge_index[1, 0::2] = u, v
edge_index[0, 1::2], edge_index[1, 1::2] = v, u
x = rng.uniform(0.05, 1.0, u.size).astype(np.float32)
hop = rng.integers(0, 2, u.size).astype(np.float32)
edge_attr = np.empty((E, 2), dtype=np.float32)         # [x, hop], shared by both arcs
edge_attr[0::2, 0] = edge_attr[1::2, 0] = x
edge_attr[0::2, 1] = edge_attr[1::2, 1] = hop
demand = rng.uniform(0.02, 0.4, N).astype(np.float32)
demand[0] = 0.0                                        # node 0 is the depot
node_attr = np.empty((K, N, 2), dtype=np.float32)      # [demand/Q, scaled_M]
node_attr[:, :, 0] = demand
node_attr[:, :, 1] = (np.arange(K, dtype=np.float64) / (K - 1)).astype(np.float32)[:, None]

before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
options.log_severity_level = 3
session = ort.InferenceSession(sys.argv[1], options, providers=["CPUExecutionProvider"])
out = session.run(
    ["probs"],
    {"edge_index": edge_index, "node_attr": node_attr, "edge_attr": edge_attr},
)[0]
after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
print(json.dumps({
    "shape": [int(s) for s in out.shape],
    "n_edges": int(E),
    "finite": bool(np.isfinite(out).all()),
    "rss_before": before,
    "rss_peak": after,
}))
"""


@needs_ref
@needs_onnx
def test_onnx_stays_within_memory_at_n400(ref_export) -> None:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(MEMORY_PROBE), str(ref_export.onnx_path)],
        capture_output=True, text=True, env=env, timeout=3600,
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stdout}\n{proc.stderr}"
    stats = json.loads(proc.stdout.strip().splitlines()[-1])
    peak, before = stats["rss_peak"], stats["rss_before"]
    index_gib = stats["n_edges"] * 30 * 256 * 8 / 1024**3
    print(f"[memory] n=400 K=30 E={stats['n_edges']} -> probs{tuple(stats['shape'])}; "
          f"RSS before the session {before / 1024**3:.3f} GiB, peak {peak / 1024**3:.3f} GiB "
          f"= {peak / 1e9:.3f} GB ({peak} B); the forward itself adds "
          f"{(peak - before) / 1024**3:.3f} GiB. Ceiling {MEMORY_CEILING / 1024**3:.0f} GiB, "
          f"used {100.0 * peak / MEMORY_CEILING:.1f}% of it. For contrast, the "
          f"Expand + ScatterElements form this graph replaced would materialise a "
          f"{index_gib:.2f} GiB int64 index per layer on top of the same activations, "
          f"six layers with no sharing.")
    assert stats["shape"] == [30, 400]
    assert stats["finite"] is True
    assert peak < MEMORY_CEILING, f"peak RSS {peak / 1024**3:.3f} GiB exceeds the ceiling"
