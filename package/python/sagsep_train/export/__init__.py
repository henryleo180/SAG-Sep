from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import torch

from sagsep.model import GraphFeatures, build_policy_network, load_policy_model

from sagsep_train.export.checkpoint import (
    CHECKPOINT_KINDS,
    load_run_state_dict,
    tensor_hashes,
    tensor_specs,
    write_checkpoint,
)
from sagsep_train.export.manifest import (
    ARTIFACT_SCHEMA_VERSION,
    GRAPH_BUILDER,
    project_manifest,
    write_manifest,
)
from sagsep_train.export.parity import (
    builtin_cases,
    compare,
    compare_onnx_files,
    ort_session,
    torch_probs,
)

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "CHECKPOINT_KINDS",
    "GRAPH_BUILDER",
    "REPORT_SCHEMA_VERSION",
    "SNAPSHOT_FILENAME",
    "ExportReport",
    "builtin_cases",
    "compare",
    "compare_onnx_files",
    "export_run",
    "load_run_state_dict",
    "ort_session",
    "project_manifest",
    "tensor_hashes",
    "write_checkpoint",
    "write_manifest",
]

SNAPSHOT_FILENAME = "config_snapshot.yaml"

REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ExportReport:
    name: str
    ckpt_path: Path
    manifest_path: Path
    onnx_path: Optional[Path]
    tensor_sha256: Dict[str, str]
    snapshot_sha256: str
    onnx_sha256: Optional[str]
    parity: Optional[Dict[str, Any]]
    global_step: Optional[int]
    checkpoint_kind: str
    run_id: str = ""
    epoch: Optional[int] = None
    ckpt_sha256: str = ""
    tensors: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    manifest: Dict[str, Any] = field(default_factory=dict)
    onnx_graph: Optional[Dict[str, Any]] = None
    versions: Dict[str, str] = field(default_factory=dict)
    report_path: Optional[Path] = None

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "name": self.name,
            "artifacts": {
                "checkpoint": self.ckpt_path.name,
                "manifest": self.manifest_path.name,
                "onnx": None if self.onnx_path is None else self.onnx_path.name,
            },
            "provenance": {
                "run_id": self.run_id,
                "checkpoint_kind": self.checkpoint_kind,
                "global_step": self.global_step,
                "epoch": self.epoch,
                "training_snapshot_sha256": self.snapshot_sha256,
            },
            "input_contract": dict(self.manifest.get("input_contract", {})),
            "model": dict(self.manifest.get("model", {})),
            "checkpoint_sha256": self.ckpt_sha256,
            "onnx_sha256": self.onnx_sha256,
            "n_tensors": len(self.tensor_sha256),
            "tensor_sha256": dict(self.tensor_sha256),
            "tensors": dict(self.tensors),
            "onnx": self.onnx_graph,
            "parity": self.parity,
            "versions": dict(self.versions),
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _versions(with_onnx: bool) -> Dict[str, str]:
    out = {"torch": str(torch.__version__)}
    if with_onnx:
        import onnx
        import onnxruntime

        out["onnx"] = str(onnx.__version__)
        out["onnxruntime"] = str(onnxruntime.__version__)
    return out


def _resolve_cases(
    parity_cases: Union[str, Mapping[str, GraphFeatures]]
) -> Dict[str, GraphFeatures]:
    if isinstance(parity_cases, str):
        if parity_cases == "builtin":
            return builtin_cases()
        if parity_cases == "none":
            return {}
        raise ValueError(
            f"parity_cases must be 'builtin', 'none' or a mapping, got {parity_cases!r}"
        )
    return dict(parity_cases)


def _roundtrip(
    ckpt_path: Path,
    net: Any,
    source_hashes: Mapping[str, str],
    manifest: Mapping[str, Any],
    cases: Mapping[str, GraphFeatures],
) -> None:
    cross_depot = bool(manifest["input_contract"]["partial_connect_cross_depot"])
    policy = load_policy_model(
        ckpt_path, device="cpu", partial_connect_cross_depot=cross_depot
    )

    reloaded = build_policy_network(dict(manifest["model"]))
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    reloaded.load_state_dict(payload["model"])
    round_hashes = tensor_hashes(reloaded.state_dict())
    if round_hashes != dict(source_hashes):
        changed = sorted(
            name
            for name in set(round_hashes) | set(source_hashes)
            if round_hashes.get(name) != source_hashes.get(name)
        )
        raise RuntimeError(
            f"the written checkpoint does not reproduce the run's weights; "
            f"{len(changed)} tensor(s) differ, first: {changed[0]!r}"
        )

    for name, feats in cases.items():
        want = torch_probs(net, feats)
        got = policy.forward_probs(feats)
        if want.tobytes() != got.tobytes():
            diff = float(np.abs(want.astype(np.float64) - got.astype(np.float64)).max())
            raise RuntimeError(
                f"the exported artefact does not serve what was trained: case {name} "
                f"differs from the in-memory network (max|delta| = {diff:.3e})"
            )


def export_run(
    run_dir: str | Path,
    *,
    name: str,
    out_dir: str | Path,
    checkpoint: str = "best",
    onnx: bool = True,
    parity_cases: Union[str, Mapping[str, GraphFeatures]] = "builtin",
    write_report: bool = True,
    verbose: bool = False,
) -> ExportReport:
    import yaml

    run_path = Path(run_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    snapshot_path = run_path / SNAPSHOT_FILENAME
    if not snapshot_path.exists():
        raise FileNotFoundError(
            f"run directory has no {SNAPSHOT_FILENAME}: cannot describe the artefact"
        )
    snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    snapshot_sha256 = _sha256_file(snapshot_path)

    state_dict, meta = load_run_state_dict(run_path, checkpoint)
    source_specs = tensor_specs(state_dict)
    source_hashes = {name_: spec["sha256"] for name_, spec in source_specs.items()}

    manifest = project_manifest(
        snapshot,
        run_id=run_path.resolve().name,
        global_step=meta.get("global_step"),
        checkpoint_kind=checkpoint,
    )

    ckpt_path = write_checkpoint(out_path / f"{name}.ckpt", state_dict)
    manifest_path = write_manifest(out_path / f"{name}.yaml", manifest, name=name)
    if verbose:
        print(f"[export] wrote {ckpt_path.name} ({len(state_dict)} tensors) and "
              f"{manifest_path.name}")

    net = build_policy_network(dict(manifest["model"]))
    net.load_state_dict(state_dict)
    net.eval()
    net.requires_grad_(False)

    cases = _resolve_cases(parity_cases)
    _roundtrip(ckpt_path, net, source_hashes, manifest, cases)
    if verbose:
        print(f"[export] round-trip through load_policy_model on {len(cases)} cases: bitwise OK")

    onnx_path: Optional[Path] = None
    onnx_sha256: Optional[str] = None
    onnx_graph: Optional[Dict[str, Any]] = None
    parity: Optional[Dict[str, Any]] = None
    if onnx:
        from sagsep_train.export.onnx import export_onnx, graph_meta

        stats: Dict[str, Any] = {}
        onnx_path = export_onnx(
            net, out_path / f"{name}.onnx", cases=cases or None, verbose=verbose, stats=stats
        )
        onnx_sha256 = _sha256_file(onnx_path)
        onnx_graph = graph_meta(onnx_path)
        onnx_graph["scatternd"] = stats.get("scatternd")
        onnx_graph["bytes"] = int(onnx_path.stat().st_size)
        if cases:
            parity = compare(net, ort_session(onnx_path), cases)
            if not parity["ok"]:
                failed = sorted(k for k, v in parity["cases"].items() if not v["ok"])
                raise RuntimeError(
                    f"ONNX parity failed on {failed}: max|ORT - torch| = "
                    f"{parity['max_abs_diff']:.3e}"
                )
            if verbose:
                print(f"[export] ONNX parity: PASS, worst |ORT - torch| = "
                      f"{parity['max_abs_diff']:.3e}")

    report = ExportReport(
        name=name,
        ckpt_path=ckpt_path,
        manifest_path=manifest_path,
        onnx_path=onnx_path,
        tensor_sha256=source_hashes,
        snapshot_sha256=snapshot_sha256,
        onnx_sha256=onnx_sha256,
        parity=parity,
        global_step=meta.get("global_step"),
        checkpoint_kind=checkpoint,
        run_id=manifest["provenance"]["run_id"],
        epoch=meta.get("epoch"),
        ckpt_sha256=_sha256_file(ckpt_path),
        tensors=source_specs,
        manifest=dict(manifest),
        onnx_graph=onnx_graph,
        versions=_versions(onnx),
    )
    if write_report:
        report_path = out_path / f"{name}.export.json"
        report_path.write_text(
            json.dumps(report.to_json_dict(), indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        report = replace(report, report_path=report_path)
        if verbose:
            print(f"[export] wrote {report_path.name}")
    return report
