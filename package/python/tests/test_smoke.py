from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from sagsep_train import smoke
from sagsep_train.cli import EXIT_OK, EXIT_SKIPPED, CapabilityMissing


@pytest.fixture(scope="module", autouse=True)
def _single_thread() -> None:
    torch.set_num_threads(1)


def _gurobi_available() -> bool:
    try:
        smoke._require_gurobi()
    except CapabilityMissing:
        return False
    return True


def _onnx_available() -> bool:
    try:
        smoke._require_onnx()
    except CapabilityMissing:
        return False
    return True


# ---------------------------------------------------------------------------
# the packaged sample and the two recipe copies
# ---------------------------------------------------------------------------


def test_sample_is_packaged() -> None:
    sample = smoke.sample_dir()
    assert (sample / "shard_0001.npz").is_file()
    assert (sample / "dataset-manifest.json").is_file()
    assert (sample / smoke.SMOKE_RECIPE_NAME).is_file()
    assert (sample / "manifest.csv").is_file()
    assert len(sorted(sample.glob("*.vrp"))) == 8
    assert len(sorted((sample / "labels").glob("*.labels.json"))) == 8


def test_the_two_smoke_recipes_differ_in_one_line(repo_root: Path) -> None:
    inside = (smoke.sample_dir() / smoke.SMOKE_RECIPE_NAME).read_text(encoding="utf-8")
    outside_path = repo_root / "configs" / "train" / smoke.SMOKE_RECIPE_NAME
    if not outside_path.is_file():
        pytest.skip(f"not a source checkout: {outside_path} is missing")
    outside = outside_path.read_text(encoding="utf-8")

    a, b = inside.splitlines(), outside.splitlines()
    assert len(a) == len(b), "the two smoke recipes must have the same number of lines"
    differing = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
    assert len(differing) == 1, f"expected one differing line, got {differing}"
    _index, sample_line, repo_line = differing[0]
    assert sample_line == 'dataset_dir = "."'
    assert repo_line == 'dataset_dir = "../../package/python/sagsep_train/sample"'


# ---------------------------------------------------------------------------
# the tiers
# ---------------------------------------------------------------------------


def test_base_tier(tmp_path: Path) -> None:
    workdir = tmp_path / "base"
    assert smoke.run(workdir=workdir, licensed=False, onnx=False) == EXIT_OK
    report = json.loads((workdir / smoke.REPORT_FILENAME).read_text(encoding="utf-8"))
    assert report["ok"] is True and report["skipped"] is None
    assert report["tiers"] == ["base"]
    assert [step["name"] for step in report["steps"]] == [
        "sample", "recipe", "train", "export", "serve"
    ]
    assert all(step["ok"] for step in report["steps"])
    assert report["total_seconds"] < 180.0
    for name in smoke.EXPECTED_RUN_FILES:
        assert (workdir / "run" / name).is_file(), name
    assert (workdir / "export" / "smoke.ckpt").is_file()
    assert not (workdir / "export" / "smoke.onnx").exists()


@pytest.mark.skipif(not _gurobi_available(), reason="no usable Gurobi")
def test_licensed_tier(tmp_path: Path) -> None:
    workdir = tmp_path / "licensed"
    assert smoke.run(workdir=workdir, licensed=True, onnx=False) == EXIT_OK
    report = json.loads((workdir / smoke.REPORT_FILENAME).read_text(encoding="utf-8"))
    assert report["ok"] is True and report["tiers"] == ["base", "licensed"]
    names = [step["name"] for step in report["steps"] if step["tier"] == "licensed"]
    assert names == [
        "solver", "generate", "label", "pack", "validate", "released-model"
    ]
    assert all(step["ok"] for step in report["steps"])


@pytest.mark.skipif(_gurobi_available(), reason="Gurobi is available, nothing to skip")
def test_licensed_tier_skips_cleanly(tmp_path: Path) -> None:
    workdir = tmp_path / "licensed-skip"
    assert smoke.run(workdir=workdir, licensed=True, onnx=False) == EXIT_SKIPPED
    report = json.loads((workdir / smoke.REPORT_FILENAME).read_text(encoding="utf-8"))
    assert report["skipped"] and "Gurobi" in report["skipped"]


@pytest.mark.skipif(not _onnx_available(), reason="onnx/onnxruntime are not installed")
def test_onnx_tier(tmp_path: Path) -> None:
    workdir = tmp_path / "onnx"
    assert smoke.run(workdir=workdir, licensed=False, onnx=True) == EXIT_OK
    report = json.loads((workdir / smoke.REPORT_FILENAME).read_text(encoding="utf-8"))
    assert report["ok"] is True and report["tiers"] == ["base", "onnx"]
    steps = {step["name"]: step for step in report["steps"] if step["tier"] == "onnx"}
    assert set(steps) == {"runtime", "export-onnx", "c-engine"}
    assert all(step["ok"] for step in steps.values())
    assert (workdir / "export-onnx" / "smoke.onnx").is_file()


@pytest.mark.skipif(_onnx_available(), reason="onnxruntime is available, nothing to skip")
def test_onnx_tier_skips_cleanly(tmp_path: Path) -> None:
    workdir = tmp_path / "onnx-skip"
    assert smoke.run(workdir=workdir, licensed=False, onnx=True) == EXIT_SKIPPED
    report = json.loads((workdir / smoke.REPORT_FILENAME).read_text(encoding="utf-8"))
    assert report["skipped"] and "onnx" in report["skipped"]
