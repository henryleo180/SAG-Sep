from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_format import make_demo_shard, single_instance_shards
from test_manifest import LABELER, SOURCE

from sagsep_train.data.format import (
    SHARD_SCHEMA_VERSION,
    read_shard,
    write_shard,
)
from sagsep_train.data.manifest import (
    MANIFEST_FILENAME,
    build_manifest,
    read_manifest,
    write_manifest,
)
from sagsep_train.data.validate import (
    ValidationError,
    assert_valid_dataset,
    assert_valid_shard,
    validate_dataset,
    validate_shard,
)


def write_demo(tmp_path: Path) -> Path:
    path = tmp_path / "shard_0001.npz"
    write_shard(path, make_demo_shard())
    return path


def corrupt(path: Path, mutate) -> Path:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    mutate(arrays)
    np.savez_compressed(path, **arrays)
    return path


def make_dataset(tmp_path: Path, splits=None) -> Path:
    directory = tmp_path / "dataset"
    directory.mkdir()
    for i, shard in enumerate(single_instance_shards(make_demo_shard()), start=1):
        write_shard(directory / f"shard_{i:04d}.npz", shard)
    manifest = build_manifest(
        directory,
        source=SOURCE,
        labeler=LABELER,
        splits=splits,
        created_by_version="0.1.0",
    )
    write_manifest(directory / MANIFEST_FILENAME, manifest)
    return directory


# --------------------------------------------------------------------------
# clean input
# --------------------------------------------------------------------------


def test_valid_shard_reports_nothing(tmp_path):
    path = write_demo(tmp_path)
    assert validate_shard(path) == []
    assert validate_shard(read_shard(path)) == []
    with np.load(path, allow_pickle=False) as archive:
        assert validate_shard({key: archive[key] for key in archive.files}) == []
    assert_valid_shard(path)


def test_valid_dataset_reports_nothing(tmp_path):
    directory = make_dataset(tmp_path, splits={"shard_0001.npz": "train", "shard_0002.npz": "val"})
    assert validate_dataset(directory) == []
    assert_valid_dataset(directory)


def test_missing_file_is_reported(tmp_path):
    assert validate_shard(tmp_path / "nothing.npz") == ["shard: file is missing: " + str(tmp_path / "nothing.npz")]


def test_assert_valid_shard_raises_with_problems(tmp_path):
    path = corrupt(write_demo(tmp_path), lambda a: a.__setitem__("sup_x", a["sup_x"] * 0.0))
    with pytest.raises(ValidationError) as excinfo:
        assert_valid_shard(path)
    assert excinfo.value.problems
    assert any("x > 0" in problem for problem in excinfo.value.problems)


# --------------------------------------------------------------------------
# shard corruptions
# --------------------------------------------------------------------------


def _set(key, index, value):
    def mutate(arrays):
        arrays[key][index] = value

    return mutate


def _swap_first_edge(arrays):
    u, v = arrays["sup_u"], arrays["sup_v"]
    u[0], v[0] = v[0], u[0]


def _drop_key(key):
    def mutate(arrays):
        arrays.pop(key)

    return mutate


CORRUPTIONS = [
    (
        "wrong dtype",
        lambda a: a.__setitem__("inst_capacity", a["inst_capacity"].astype(np.int32)),
        "inst_capacity: dtype must be int64",
    ),
    (
        "wrong shape",
        lambda a: a.__setitem__("inst_coords", a["inst_coords"][:, :1]),
        "inst_coords: must have shape",
    ),
    ("missing key", _drop_key("m_rhs"), "m_rhs: key is missing"),
    (
        "unexpected key",
        lambda a: a.__setitem__("bonus", np.zeros(3, dtype=np.int32)),
        "bonus: unexpected key",
    ),
    (
        "wrong schema version",
        lambda a: a.__setitem__(
            "schema_version", np.asarray(SHARD_SCHEMA_VERSION + 1, dtype=np.int32)
        ),
        "schema_version: must be 1",
    ),
    (
        "wrong feature contract",
        lambda a: a.__setitem__("feature_contract", np.asarray("other_v9", dtype=np.str_)),
        "feature_contract: must be",
    ),
    (
        "non-monotone offsets",
        lambda a: a["sup_offsets"].__setitem__(1, int(a["sup_offsets"][2]) + 1),
        "sup_offsets: must be non-decreasing",
    ),
    (
        "offsets short of the array length",
        lambda a: a["lab_offsets"].__setitem__(-1, int(a["lab_offsets"][-1]) - 1),
        "lab_offsets: must end at the array length",
    ),
    ("u >= v", _swap_first_edge, "sup_u: support requires u < v"),
    ("v out of range", _set("sup_v", 0, 99), "sup_v: must be < n_nodes"),
    ("x <= 0", _set("sup_x", 0, 0.0), "sup_x: support requires x > 0"),
    (
        "unordered support",
        lambda a: (
            a["sup_u"].__setitem__(1, a["sup_u"][0]),
            a["sup_v"].__setitem__(1, a["sup_v"][0]),
        ),
        "strictly increasing in lexicographic",
    ),
    ("s_hat = 2", _set("lab_s_hat", -1, 2), "lab_s_hat: values must be 0/1"),
    ("s_hat depot row set", _set("lab_s_hat", 0, 1), "lab_s_hat: the depot row"),
    ("wrong k_label", _set("inst_k_label", 0, 9), "inst_k_label: instance 0"),
    ("depot demand != 0", _set("inst_demand", 0, 4), "inst_demand: depot demand must be 0"),
    ("negative demand", _set("inst_demand", 1, -3), "inst_demand: demand must be non-negative"),
    ("two terminals", _set("st_terminal", 0, 1), "has 2 terminal states"),
    ("missing terminal", _set("st_terminal", 2, 0), "has no terminal state"),
    (
        "terminal not last",
        lambda a: (a["st_terminal"].__setitem__(2, 0), a["st_terminal"].__setitem__(0, 1)),
        "must be the last one",
    ),
    ("iteration gap", _set("st_iteration", 1, 3), "st_iteration: instance 0"),
    ("instance row out of range", _set("st_inst_row", 0, 99), "st_inst_row: instance row 99"),
    (
        "states not grouped",
        lambda a: (a["st_inst_row"].__setitem__(0, 1), a["st_inst_row"].__setitem__(1, 0)),
        "must be grouped by instance",
    ),
    ("bad source digest", _set("inst_source_sha256", 0, "abc"), "inst_source_sha256: not a 64-character"),
    ("duplicate instance id", lambda a: a["inst_id"].__setitem__(1, a["inst_id"][0]), "duplicate instance id"),
    ("bad termination code", _set("inst_termination", 0, 9), "inst_termination: code must be 0..4"),
    ("slot_violated = 2", _set("m_slot_violated", 0, 2), "m_slot_violated: must be 0/1"),
    ("negative rhs", _set("m_rhs", 0, -1), "m_rhs: must be non-negative"),
    ("non-finite lp value", _set("st_lp_value", 0, np.nan), "st_lp_value: contains non-finite"),
    ("negative cuts_added", _set("st_cuts_added", 0, -2), "st_cuts_added: must be non-negative"),
]


@pytest.mark.parametrize(
    "mutate,expected", [case[1:] for case in CORRUPTIONS], ids=[case[0] for case in CORRUPTIONS]
)
def test_corruption_is_reported(tmp_path, mutate, expected):
    path = corrupt(write_demo(tmp_path), mutate)
    problems = validate_shard(path)
    assert problems, "corruption went unreported"
    assert any(expected in problem for problem in problems), problems


# --------------------------------------------------------------------------
# dataset corruptions
# --------------------------------------------------------------------------


def test_manifest_sha_mismatch_is_reported(tmp_path):
    directory = make_dataset(tmp_path)
    path = directory / MANIFEST_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["shards"][1]["sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    problems = validate_dataset(directory)
    assert any("sha256 mismatch" in problem for problem in problems), problems


def test_manifest_bytes_mismatch_is_reported(tmp_path):
    directory = make_dataset(tmp_path)
    path = directory / MANIFEST_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["shards"][0]["bytes"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    problems = validate_dataset(directory)
    assert any("bytes mismatch" in problem for problem in problems), problems


def test_manifest_counts_mismatch_is_reported(tmp_path):
    directory = make_dataset(tmp_path)
    path = directory / MANIFEST_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["counts"]["states"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    problems = validate_dataset(directory)
    assert any("counts.states mismatch" in problem for problem in problems), problems


def test_manifest_shard_counts_mismatch_is_reported(tmp_path):
    directory = make_dataset(tmp_path)
    path = directory / MANIFEST_FILENAME
    manifest = read_manifest(path)
    manifest.shards[0].n_states += 1
    manifest.n_states += 1
    write_manifest(path, manifest)
    problems = validate_dataset(directory)
    assert any("n_states mismatch" in problem for problem in problems), problems


def test_missing_shard_file_is_reported(tmp_path):
    directory = make_dataset(tmp_path)
    (directory / "shard_0002.npz").unlink()
    problems = validate_dataset(directory)
    assert any("shards[1] 'shard_0002.npz': file is missing" in problem for problem in problems), problems


def test_unlisted_shard_file_is_reported(tmp_path):
    directory = make_dataset(tmp_path)
    write_shard(directory / "shard_0009.npz", make_demo_shard())
    problems = validate_dataset(directory)
    assert any("not listed: shard_0009.npz" in problem for problem in problems), problems


def test_shard_problem_is_reported_with_its_file(tmp_path):
    directory = make_dataset(tmp_path)
    corrupt(directory / "shard_0003.npz", _set("sup_x", 0, -1.0))
    problems = validate_dataset(directory)
    assert any(
        problem.startswith("shards[2] 'shard_0003.npz': ") and "x > 0" in problem
        for problem in problems
    ), problems


def test_duplicate_instance_across_shards_is_reported(tmp_path):
    directory = make_dataset(tmp_path)
    shards = single_instance_shards(make_demo_shard())
    write_shard(directory / "shard_0003.npz", shards[0])
    manifest = build_manifest(
        directory, source=SOURCE, labeler=LABELER, splits=None, created_by_version="0.1.0"
    )
    write_manifest(directory / MANIFEST_FILENAME, manifest)
    problems = validate_dataset(directory)
    assert any("also appears in" in problem for problem in problems), problems


def test_missing_manifest_is_reported(tmp_path):
    directory = make_dataset(tmp_path)
    (directory / MANIFEST_FILENAME).unlink()
    problems = validate_dataset(directory)
    assert len(problems) == 1 and "manifest: file is missing" in problems[0]


def test_unreadable_manifest_is_reported(tmp_path):
    directory = make_dataset(tmp_path)
    (directory / MANIFEST_FILENAME).write_text("{not json", encoding="utf-8")
    problems = validate_dataset(directory)
    assert len(problems) == 1 and problems[0].startswith("manifest: ")


def test_dataset_dir_must_exist(tmp_path):
    problems = validate_dataset(tmp_path / "nowhere")
    assert problems == [f"dataset: not a directory: {tmp_path / 'nowhere'}"]


def test_assert_valid_dataset_raises(tmp_path):
    directory = make_dataset(tmp_path)
    (directory / "shard_0001.npz").unlink()
    with pytest.raises(ValidationError) as excinfo:
        assert_valid_dataset(directory)
    assert excinfo.value.problems
