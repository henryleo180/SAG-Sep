from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sagsep_train.data.format import (
    FEATURE_CONTRACT,
    SUPPORT_RULE,
    read_shard_counts,
    sha256_file,
)

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "MANIFEST_FILENAME",
    "CREATED_BY_PACKAGE",
    "SHARD_GLOB",
    "SOURCE_KINDS",
    "REQUIRED_LABELER_KEYS",
    "ShardEntry",
    "DatasetManifest",
    "write_manifest",
    "read_manifest",
    "build_manifest",
    "validate_source",
    "validate_labeler",
]

MANIFEST_SCHEMA_VERSION = 1

MANIFEST_FILENAME = "dataset-manifest.json"

CREATED_BY_PACKAGE = "sagsep_train"

SHARD_GLOB = "shard_*.npz"

SOURCE_KINDS: dict[str, tuple[str, ...]] = {
    "generator": ("name", "seed"),
    "external": ("archive", "archive_sha256"),
}

REQUIRED_LABELER_KEYS: dict[str, tuple[type, ...]] = {
    "version": (str,),
    "solver": (str,),
    "solver_version": (str,),
    "threads": (int,),
    "lp_method": (int,),
    "tolerances": (dict,),
    "min_violation": (int, float),
    "max_iter": (int, type(None)),
    "require_optimal": (bool,),
}

SPLITS: tuple[str, ...] = ("train", "val")


def validate_source(source: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        raise TypeError(f"source must be a mapping, got {type(source).__name__}")
    kind = source.get("kind")
    if kind not in SOURCE_KINDS:
        raise ValueError(f"source.kind must be one of {sorted(SOURCE_KINDS)}, got {kind!r}")
    missing = [key for key in SOURCE_KINDS[kind] if key not in source]
    if missing:
        raise ValueError(f"source (kind={kind}) is missing key(s): {', '.join(missing)}")
    return dict(source)


def validate_labeler(labeler: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(labeler, Mapping):
        raise TypeError(f"labeler must be a mapping, got {type(labeler).__name__}")
    missing = [key for key in REQUIRED_LABELER_KEYS if key not in labeler]
    if missing:
        raise ValueError(f"labeler is missing key(s): {', '.join(missing)}")
    for key, types in REQUIRED_LABELER_KEYS.items():
        value = labeler[key]
        if isinstance(value, bool) and bool not in types:
            raise ValueError(f"labeler.{key} must be of type {types[0].__name__}, got bool")
        if not isinstance(value, types):
            names = " or ".join("null" if t is type(None) else t.__name__ for t in types)
            raise ValueError(
                f"labeler.{key} must be {names}, got {type(value).__name__}"
            )
    return dict(labeler)


@dataclass
class ShardEntry:
    file: str
    sha256: str
    bytes: int
    n_instances: int
    n_states: int
    split: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "sha256": self.sha256,
            "bytes": int(self.bytes),
            "n_instances": int(self.n_instances),
            "n_states": int(self.n_states),
            "split": self.split,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ShardEntry":
        missing = [
            key
            for key in ("file", "sha256", "bytes", "n_instances", "n_states")
            if key not in payload
        ]
        if missing:
            raise ValueError(f"shard entry is missing key(s): {', '.join(missing)}")
        split = payload.get("split")
        if split is not None and split not in SPLITS:
            raise ValueError(f"shard entry {payload['file']!r}: split must be one of {list(SPLITS)} or null, got {split!r}")
        return cls(
            file=str(payload["file"]),
            sha256=str(payload["sha256"]),
            bytes=int(payload["bytes"]),
            n_instances=int(payload["n_instances"]),
            n_states=int(payload["n_states"]),
            split=split,
        )


@dataclass
class DatasetManifest:
    created_by_version: str
    source: dict[str, Any]
    labeler: dict[str, Any]
    shards: list[ShardEntry] = field(default_factory=list)
    n_instances: int = 0
    n_states: int = 0
    schema_version: int = MANIFEST_SCHEMA_VERSION
    feature_contract: str = FEATURE_CONTRACT
    support_rule: str = SUPPORT_RULE
    created_by_package: str = CREATED_BY_PACKAGE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "created_by": {
                "package": self.created_by_package,
                "version": str(self.created_by_version),
            },
            "feature_contract": self.feature_contract,
            "support_rule": self.support_rule,
            "source": dict(self.source),
            "labeler": dict(self.labeler),
            "shards": [entry.to_dict() for entry in self.shards],
            "counts": {"instances": int(self.n_instances), "states": int(self.n_states)},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DatasetManifest":
        if not isinstance(payload, Mapping):
            raise TypeError(f"manifest must be a JSON object, got {type(payload).__name__}")
        version = payload.get("schema_version")
        if version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"manifest schema_version {version!r} is not supported "
                f"(this build reads version {MANIFEST_SCHEMA_VERSION})"
            )
        contract = payload.get("feature_contract")
        if contract != FEATURE_CONTRACT:
            raise ValueError(
                f"manifest feature_contract {contract!r} does not match {FEATURE_CONTRACT!r}"
            )
        rule = payload.get("support_rule")
        if rule != SUPPORT_RULE:
            raise ValueError(
                f"manifest support_rule {rule!r} does not match {SUPPORT_RULE!r}"
            )
        created_by = payload.get("created_by")
        if not isinstance(created_by, Mapping) or "version" not in created_by:
            raise ValueError("manifest created_by must be an object with a 'version' key")
        counts = payload.get("counts")
        if not isinstance(counts, Mapping) or "instances" not in counts or "states" not in counts:
            raise ValueError("manifest counts must be an object with 'instances' and 'states'")
        shards_payload = payload.get("shards")
        if not isinstance(shards_payload, list):
            raise ValueError("manifest shards must be a list")
        return cls(
            created_by_version=str(created_by["version"]),
            source=validate_source(payload.get("source", {})),
            labeler=validate_labeler(payload.get("labeler", {})),
            shards=[ShardEntry.from_dict(entry) for entry in shards_payload],
            n_instances=int(counts["instances"]),
            n_states=int(counts["states"]),
            created_by_package=str(created_by.get("package", CREATED_BY_PACKAGE)),
        )


def write_manifest(path: str | os.PathLike[str], manifest: DatasetManifest) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=target.name + ".", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def read_manifest(path: str | os.PathLike[str]) -> DatasetManifest:
    src = Path(path)
    with open(src, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    try:
        return DatasetManifest.from_dict(payload)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{src}: {exc}") from exc


def build_manifest(
    dataset_dir: str | os.PathLike[str],
    *,
    source: Mapping[str, Any],
    labeler: Mapping[str, Any],
    splits: Mapping[str, str] | None = None,
    created_by_version: str,
) -> DatasetManifest:
    directory = Path(dataset_dir)
    if not directory.is_dir():
        raise ValueError(f"{directory}: not a directory")
    checked_source = validate_source(source)
    checked_labeler = validate_labeler(labeler)

    split_map: dict[str, str] = {}
    if splits:
        for key, value in splits.items():
            if value not in SPLITS:
                raise ValueError(f"splits[{key!r}]: split must be one of {list(SPLITS)}, got {value!r}")
            split_map[str(key)] = value

    files = sorted(directory.glob(SHARD_GLOB))
    entries: list[ShardEntry] = []
    used: set[str] = set()
    for shard_path in files:
        name = shard_path.name
        split = None
        for key in (name, shard_path.stem):
            if key in split_map:
                split = split_map[key]
                used.add(key)
                break
        n_instances, n_states = read_shard_counts(shard_path)
        entries.append(
            ShardEntry(
                file=name,
                sha256=sha256_file(shard_path),
                bytes=shard_path.stat().st_size,
                n_instances=n_instances,
                n_states=n_states,
                split=split,
            )
        )

    unknown = sorted(set(split_map) - used)
    if unknown:
        raise ValueError(f"splits names no such shard: {', '.join(unknown)}")

    return DatasetManifest(
        created_by_version=str(created_by_version),
        source=checked_source,
        labeler=checked_labeler,
        shards=entries,
        n_instances=sum(entry.n_instances for entry in entries),
        n_states=sum(entry.n_states for entry in entries),
    )
