from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_format import make_demo_shard, single_instance_shards

from sagsep_train.data.format import (
    FEATURE_CONTRACT,
    SUPPORT_RULE,
    sha256_file,
    write_shard,
)
from sagsep_train.data.manifest import (
    CREATED_BY_PACKAGE,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    DatasetManifest,
    ShardEntry,
    build_manifest,
    read_manifest,
    write_manifest,
)

SOURCE = {
    "kind": "generator",
    "name": "uniform_cvrp",
    "seed": 20260901,
    "params": {"n_nodes": [50, 100], "capacity": 50},
}

LABELER = {
    "version": "1.0.0",
    "solver": "gurobi",
    "solver_version": "13.0.0",
    "threads": 4,
    "lp_method": 1,
    "tolerances": {"feasibility": 1e-9, "optimality": 1e-9, "integrality": 1e-9},
    "min_violation": 0.01,
    "max_iter": 30,
    "require_optimal": True,
}


def make_dataset(tmp_path: Path) -> Path:
    directory = tmp_path / "dataset"
    directory.mkdir()
    for i, shard in enumerate(single_instance_shards(make_demo_shard()), start=1):
        write_shard(directory / f"shard_{i:04d}.npz", shard)
    return directory


def test_manifest_round_trip(tmp_path):
    manifest = DatasetManifest(
        created_by_version="0.1.0",
        source=dict(SOURCE),
        labeler=dict(LABELER),
        shards=[
            ShardEntry("shard_0001.npz", "a" * 64, 123, 2, 7, "train"),
            ShardEntry("shard_0002.npz", "b" * 64, 456, 1, 3, "val"),
            ShardEntry("shard_0003.npz", "c" * 64, 789, 1, 2, None),
        ],
        n_instances=4,
        n_states=12,
    )
    path = tmp_path / MANIFEST_FILENAME
    write_manifest(path, manifest)
    assert sorted(item.name for item in tmp_path.iterdir()) == [MANIFEST_FILENAME]

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert payload["created_by"] == {"package": CREATED_BY_PACKAGE, "version": "0.1.0"}
    assert payload["feature_contract"] == FEATURE_CONTRACT
    assert payload["support_rule"] == SUPPORT_RULE
    assert payload["counts"] == {"instances": 4, "states": 12}
    assert [entry["split"] for entry in payload["shards"]] == ["train", "val", None]

    back = read_manifest(path)
    assert back == manifest
    assert back.to_dict() == manifest.to_dict()


def test_build_manifest_scans_directory(tmp_path):
    directory = make_dataset(tmp_path)
    manifest = build_manifest(
        directory,
        source=SOURCE,
        labeler=LABELER,
        splits={"shard_0001.npz": "train", "shard_0003": "val"},
        created_by_version="0.2.0",
    )
    assert [entry.file for entry in manifest.shards] == [
        "shard_0001.npz",
        "shard_0002.npz",
        "shard_0003.npz",
    ]
    assert [entry.split for entry in manifest.shards] == ["train", None, "val"]
    assert [entry.n_instances for entry in manifest.shards] == [1, 1, 1]
    assert [entry.n_states for entry in manifest.shards] == [3, 2, 4]
    assert manifest.n_instances == 3
    assert manifest.n_states == 9
    for entry in manifest.shards:
        path = directory / entry.file
        assert entry.sha256 == sha256_file(path)
        assert entry.bytes == path.stat().st_size

    path = directory / MANIFEST_FILENAME
    write_manifest(path, manifest)
    assert read_manifest(path) == manifest


def test_build_manifest_rejects_unknown_split_key(tmp_path):
    directory = make_dataset(tmp_path)
    with pytest.raises(ValueError, match="no such shard"):
        build_manifest(
            directory,
            source=SOURCE,
            labeler=LABELER,
            splits={"shard_0009.npz": "train"},
            created_by_version="0.2.0",
        )


def test_build_manifest_rejects_unknown_split_name(tmp_path):
    directory = make_dataset(tmp_path)
    with pytest.raises(ValueError, match="split must be one of"):
        build_manifest(
            directory,
            source=SOURCE,
            labeler=LABELER,
            splits={"shard_0001.npz": "test"},
            created_by_version="0.2.0",
        )


def test_build_manifest_on_empty_directory(tmp_path):
    directory = tmp_path / "empty"
    directory.mkdir()
    manifest = build_manifest(
        directory, source=SOURCE, labeler=LABELER, splits=None, created_by_version="0.2.0"
    )
    assert manifest.shards == []
    assert (manifest.n_instances, manifest.n_states) == (0, 0)


@pytest.mark.parametrize("missing", sorted(LABELER))
def test_labeler_requires_every_key(tmp_path, missing):
    labeler = {key: value for key, value in LABELER.items() if key != missing}
    with pytest.raises(ValueError, match=f"labeler is missing key\\(s\\): {missing}"):
        build_manifest(
            tmp_path, source=SOURCE, labeler=labeler, splits=None, created_by_version="0.1.0"
        )


def test_labeler_type_check(tmp_path):
    labeler = dict(LABELER, threads="four")
    with pytest.raises(ValueError, match="labeler.threads must be int"):
        build_manifest(
            tmp_path, source=SOURCE, labeler=labeler, splits=None, created_by_version="0.1.0"
        )


def test_labeler_allows_null_max_iter(tmp_path):
    directory = tmp_path / "empty"
    directory.mkdir()
    manifest = build_manifest(
        directory,
        source=SOURCE,
        labeler=dict(LABELER, max_iter=None),
        splits=None,
        created_by_version="0.1.0",
    )
    assert manifest.labeler["max_iter"] is None


def test_source_kinds(tmp_path):
    directory = tmp_path / "empty"
    directory.mkdir()
    external = {"kind": "external", "archive": "instances.tar.gz", "archive_sha256": "d" * 64}
    manifest = build_manifest(
        directory, source=external, labeler=LABELER, splits=None, created_by_version="0.1.0"
    )
    assert manifest.source == external

    with pytest.raises(ValueError, match="source.kind must be one of"):
        build_manifest(
            directory,
            source={"kind": "magic"},
            labeler=LABELER,
            splits=None,
            created_by_version="0.1.0",
        )
    with pytest.raises(ValueError, match=r"source \(kind=external\) is missing key"):
        build_manifest(
            directory,
            source={"kind": "external", "archive": "x.tar"},
            labeler=LABELER,
            splits=None,
            created_by_version="0.1.0",
        )
    with pytest.raises(ValueError, match=r"source \(kind=generator\) is missing key"):
        build_manifest(
            directory,
            source={"kind": "generator", "name": "uniform_cvrp"},
            labeler=LABELER,
            splits=None,
            created_by_version="0.1.0",
        )


def test_read_manifest_refuses_other_schema_version(tmp_path):
    path = tmp_path / MANIFEST_FILENAME
    manifest = DatasetManifest(created_by_version="0.1.0", source=dict(SOURCE), labeler=dict(LABELER))
    write_manifest(path, manifest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = MANIFEST_SCHEMA_VERSION + 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        read_manifest(path)


def test_read_manifest_refuses_other_feature_contract(tmp_path):
    path = tmp_path / MANIFEST_FILENAME
    manifest = DatasetManifest(created_by_version="0.1.0", source=dict(SOURCE), labeler=dict(LABELER))
    write_manifest(path, manifest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["feature_contract"] = "some_other_contract_v9"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="feature_contract"):
        read_manifest(path)


def test_shard_entry_rejects_unknown_split():
    with pytest.raises(ValueError, match="split must be one of"):
        ShardEntry.from_dict(
            {
                "file": "shard_0001.npz",
                "sha256": "a" * 64,
                "bytes": 1,
                "n_instances": 1,
                "n_states": 1,
                "split": "holdout",
            }
        )
