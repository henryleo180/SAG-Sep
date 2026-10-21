from __future__ import annotations

from pathlib import Path, PurePath
from typing import Any, Dict, Mapping, Optional

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "GRAPH_BUILDER",
    "MANIFEST_SECTIONS",
    "project_manifest",
    "write_manifest",
]

ARTIFACT_SCHEMA_VERSION = 1

GRAPH_BUILDER = "partial_2hop_max_min_v1"

MANIFEST_SECTIONS = ("artifact_schema_version", "model", "input_contract", "provenance")


def _data_flag(snapshot: Mapping[str, Any], key: str, default: bool) -> bool:
    data = snapshot.get("data")
    if data is None:
        return default
    if not isinstance(data, Mapping):
        raise ValueError(f"snapshot data: section must be a mapping, got {type(data).__name__}")
    if key not in data or data[key] is None:
        return default
    value = data[key]
    if not isinstance(value, bool):
        raise ValueError(
            f"snapshot data.{key} must be a boolean, got {value!r} ({type(value).__name__})"
        )
    return value


def project_manifest(
    snapshot: Mapping[str, Any],
    *,
    run_id: str,
    global_step: Optional[int],
    checkpoint_kind: str,
) -> Dict[str, Any]:
    model = snapshot.get("model")
    if not isinstance(model, Mapping) or not model:
        raise ValueError("snapshot has no model: section -- cannot describe the artefact")

    if not run_id or run_id != PurePath(run_id).name or run_id in (".", ".."):
        raise ValueError(
            f"run_id must be a run directory's base name, got {run_id!r}: a manifest "
            "records which run produced the weights, never where that run lived"
        )

    m_closed = _data_flag(snapshot, "m_value_closed_interval", True)
    if m_closed is not True:
        raise ValueError(
            "snapshot data.m_value_closed_interval is false: this package's vehicle "
            "replica sweep is the closed interval m = 1..K, so these weights cannot "
            "be served and must not be exported"
        )

    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model": dict(model),
        "input_contract": {
            "graph_builder": GRAPH_BUILDER,
            "partial_connect_cross_depot": _data_flag(
                snapshot, "partial_connect_cross_depot", True
            ),
            "m_value_closed_interval": m_closed,
        },
        "provenance": {
            "run_id": str(run_id),
            "global_step": None if global_step is None else int(global_step),
            "checkpoint_kind": str(checkpoint_kind),
        },
    }


def write_manifest(path: str | Path, manifest: Mapping[str, Any], *, name: str) -> Path:
    import yaml

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Model manifest for {name}.ckpt, projected from the training snapshot "
        "by sagsep_train.export.\n"
        "# The model section is the exact key set the loader reads; training "
        "configuration is not shipped.\n"
    )
    body = yaml.safe_dump(dict(manifest), sort_keys=False, default_flow_style=False)
    out.write_text(header + body, encoding="utf-8")
    return out
