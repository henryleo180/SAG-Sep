from __future__ import annotations

import json
import math

import numpy as np
import pytest

from sagsep import features
from sagsep_train.instances import generate_set
from sagsep_train.labeler import (
    GUROBI_METHOD,
    label_instance,
    label_set,
    labeled_to_shard_data,
    write_labeled,
)
from sagsep_train.oracle import k_label

gurobi_c = pytest.importorskip("sagsep.gurobi_c")


@pytest.fixture(scope="module", autouse=True)
def _require_gurobi():
    try:
        handle = gurobi_c.GurobiEnv()
    except Exception as exc:
        pytest.skip(f"Gurobi environment unavailable: {exc}")
    handle.dispose()


@pytest.fixture(scope="module")
def p_n20_path(instances_dir):
    return instances_dir / "random" / "P-n20-k2.vrp"


@pytest.fixture(scope="module")
def labeled(p_n20_path):
    return label_instance(p_n20_path, vehicles=2, max_iter=None)


@pytest.fixture(scope="module")
def tiny_set(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tiny")
    rows = generate_set(directory, n_customers=12, count=4, seed=20260904)
    return directory, rows


def test_label_instance_is_consistent(labeled, p_n20_path) -> None:
    instance = labeled.instance
    assert labeled.name == "P-n20-k2"
    assert len(labeled.source_sha256) == 64
    assert labeled.k_fleet == 2
    assert labeled.k_label == k_label(instance.demand, int(instance.capacity))
    assert labeled.k_label == features.vehicle_replica_count(instance)
    assert instance.vehicles == labeled.k_fleet

    result = labeled.result
    assert result.termination == "closed"
    assert labeled.states is result.states
    assert len(labeled.states) == result.iterations
    assert result.lower_bound > result.base_lower_bound

    for state in labeled.states:
        payload = state.payload
        assert payload is not None
        assert payload["s_hat"].shape == (instance.n_nodes, labeled.k_label)
        assert payload["objective"].shape == (labeled.k_label,)
        assert payload["rhs"].shape == (labeled.k_label,)
        assert payload["slot_violated"].shape == (labeled.k_label,)

    info = labeled.solver_info
    assert info["gurobi_version"] == list(gurobi_c.version())
    assert info["gurobi_method"] == GUROBI_METHOD
    assert info["gurobi_threads"] == 1
    assert info["min_violation"] == 1e-6
    assert info["oracle_tol"] == 1e-6
    assert info["support_rule"] == "x_gt_0_float64"


def test_vehicles_default_is_k_label(p_n20_path) -> None:
    labeled = label_instance(p_n20_path, vehicles=None, max_iter=1)
    assert labeled.k_fleet == labeled.k_label
    assert labeled.instance.vehicles == labeled.k_label
    assert labeled.result.termination == "iteration_cap"
    assert len(labeled.states) == 1


def test_shard_round_trip(labeled) -> None:
    fmt = pytest.importorskip("sagsep_train.data.format")
    data = labeled_to_shard_data(labeled)
    assert data.n_instances == 1
    assert data.n_states == len(labeled.states)

    record = data.instance(0)
    assert record.id == labeled.name
    assert record.source_sha256 == labeled.source_sha256
    assert record.k_fleet == labeled.k_fleet
    assert record.k_label == labeled.k_label
    assert record.termination == labeled.result.termination
    assert np.array_equal(record.demand, labeled.instance.demand)
    assert np.array_equal(record.coords, labeled.instance.coords)
    assert fmt.TERMINATION_CODES[record.termination] == int(data.inst_termination[0])

    for row, state in enumerate(labeled.states):
        stored = data.state_record(row)
        assert stored.iteration == state.iteration
        assert stored.lp_value == pytest.approx(state.lp_value)
        assert stored.cuts_added == state.cuts_added
        assert stored.terminal == (row == len(labeled.states) - 1)
        assert np.array_equal(stored.support_u, state.support_u)
        assert np.array_equal(stored.support_x, state.support_x)
        assert np.array_equal(stored.s_hat, state.payload["s_hat"])
        assert np.array_equal(stored.objective, state.payload["objective"])


def test_write_labeled_writes_both_files(labeled, tmp_path) -> None:
    fmt = pytest.importorskip("sagsep_train.data.format")
    npz_path = write_labeled(labeled, tmp_path)
    json_path = tmp_path / f"{labeled.name}.labels.json"

    assert npz_path == tmp_path / f"{labeled.name}.labels.npz"
    assert npz_path.exists() and json_path.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == [json_path.name, npz_path.name]

    data = fmt.read_shard(npz_path)
    assert data.n_instances == 1
    assert data.n_states == len(labeled.states)
    assert data.instance(0).id == labeled.name

    sidecar = json.loads(json_path.read_text(encoding="utf-8"))
    assert sidecar["name"] == labeled.name
    assert sidecar["termination"] == labeled.result.termination
    assert sidecar["iterations"] == labeled.result.iterations
    assert sidecar["n_states"] == len(labeled.states)
    assert sidecar["k_fleet"] == labeled.k_fleet
    assert sidecar["k_label"] == labeled.k_label
    assert sidecar["base_lower_bound"] == pytest.approx(labeled.result.base_lower_bound)
    assert sidecar["lower_bound"] == pytest.approx(labeled.result.lower_bound)
    assert sidecar["runtime_s"] > 0.0
    assert sidecar["solver_info"] == labeled.solver_info

    again = write_labeled(labeled, tmp_path)
    assert again == npz_path
    assert sorted(p.name for p in tmp_path.iterdir()) == [json_path.name, npz_path.name]


def test_label_set_sharding_is_stable_and_resumable(tiny_set, tmp_path) -> None:
    pytest.importorskip("sagsep_train.data.format")
    directory, rows = tiny_set
    manifest = directory / "manifest.csv"
    names = sorted(row["instance"] for row in rows)

    whole = label_set(
        directory,
        tmp_path / "whole",
        shard="1/1",
        vehicles_from=manifest,
        max_iter=2,
    )
    assert [p.name for p in whole] == [f"{name}.labels.npz" for name in names]

    first = label_set(directory, tmp_path / "a", shard="1/2", vehicles_from=manifest, max_iter=2)
    second = label_set(directory, tmp_path / "b", shard="2/2", vehicles_from=manifest, max_iter=2)
    first_names = [p.name for p in first]
    second_names = [p.name for p in second]
    assert set(first_names).isdisjoint(second_names)
    assert sorted(first_names + second_names) == [f"{name}.labels.npz" for name in names]
    assert first_names == [f"{names[i]}.labels.npz" for i in (0, 2)]
    assert second_names == [f"{names[i]}.labels.npz" for i in (1, 3)]

    stamps = {p: p.stat().st_mtime_ns for p in first}
    again = label_set(directory, tmp_path / "a", shard="1/2", vehicles_from=manifest, max_iter=2)
    assert again == first
    assert {p: p.stat().st_mtime_ns for p in again} == stamps


def test_label_set_pins_vehicles_from_manifest(tiny_set, tmp_path) -> None:
    fmt = pytest.importorskip("sagsep_train.data.format")
    directory, rows = tiny_set
    paths = label_set(
        directory,
        tmp_path / "out",
        shard="1/1",
        vehicles_from=directory / "manifest.csv",
        max_iter=1,
    )
    by_name = {row["instance"]: row for row in rows}
    for path in paths:
        data = fmt.read_shard(path)
        record = data.instance(0)
        row = by_name[record.id]
        assert record.k_fleet == int(row["vehicles"])
        loaded = features.load_instance(directory / f"{record.id}.vrp", vehicles=1)
        total = int(loaded.demand.sum())
        assert record.k_label == math.ceil(total / int(loaded.capacity))


def test_label_set_rejects_unknown_shard(tiny_set, tmp_path) -> None:
    directory, _rows = tiny_set
    for shard in ["0/2", "3/2", "1", "a/b"]:
        with pytest.raises(ValueError):
            label_set(directory, tmp_path / "x", shard=shard, max_iter=1)


def test_label_set_reports_missing_manifest_row(tiny_set, tmp_path) -> None:
    directory, rows = tiny_set
    manifest = tmp_path / "partial.csv"
    manifest.write_text("instance,vehicles\nnot-an-instance,3\n", encoding="utf-8")
    with pytest.raises(ValueError):
        label_set(directory, tmp_path / "y", shard="1/1", vehicles_from=manifest, max_iter=1)
