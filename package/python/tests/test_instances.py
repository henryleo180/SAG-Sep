from __future__ import annotations

import csv
import hashlib
import math

import numpy as np
import pytest

from sagsep import features
from sagsep_train.instances import (
    CLUSTER_SEED_RANGE,
    COORD_MAX,
    MANIFEST_COLUMNS,
    ROUTE_SIZE_INTERVALS,
    GeneratedInstance,
    XmlSpec,
    all_specs,
    generate_instance,
    generate_set,
    set_seeds,
    spec_code,
    write_vrp,
)

_DEMAND_BOUNDS = {
    1: (1, 1),
    2: (1, 10),
    3: (5, 10),
    4: (1, 100),
    5: (50, 100),
    6: (1, 100),
    7: (1, 100),
}


def _check_instance(inst: GeneratedInstance) -> None:
    spec = inst.spec
    n = int(spec.n_customers)
    coords = inst.coords
    demand = inst.demand

    assert coords.shape == (n + 1, 2)
    assert demand.shape == (n + 1,)
    assert coords.dtype == np.int64
    assert demand.dtype == np.int64

    assert int(coords.min()) >= 0
    assert int(coords.max()) <= COORD_MAX
    points = {(int(x), int(y)) for x, y in coords}
    assert len(points) == n + 1

    if spec.depot_pos == 2:
        assert tuple(coords[0]) == (COORD_MAX // 2, COORD_MAX // 2)
    elif spec.depot_pos == 3:
        assert tuple(coords[0]) == (0, 0)

    assert int(demand[0]) == 0
    low, high = _DEMAND_BOUNDS[spec.demand_type]
    assert int(demand[1:].min()) >= low
    assert int(demand[1:].max()) <= high
    if spec.demand_type == 6:
        half = COORD_MAX // 2
        xs, ys = coords[1:, 0], coords[1:, 1]
        even = ((xs < half) & (ys < half)) | ((xs >= half) & (ys >= half))
        assert int(demand[1:][even].min(initial=51)) >= 51
        assert int(demand[1:][even].max(initial=51)) <= 100
        assert int(demand[1:][~even].min(initial=1)) >= 1
        assert int(demand[1:][~even].max(initial=1)) <= 50
    if spec.demand_type == 7:
        small = (demand[1:] >= 1) & (demand[1:] <= 10)
        large = (demand[1:] >= 50) & (demand[1:] <= 100)
        assert np.all(small | large)
        fraction = float(small.sum()) / n
        assert 0.70 - 1.0 / n <= fraction <= 0.95 + 1.0 / n

    total = int(demand.sum())
    max_demand = int(demand.max())
    assert inst.capacity >= max_demand
    assert inst.k_fleet == -(-total // inst.capacity)
    assert inst.k_fleet >= 1

    lo, hi = ROUTE_SIZE_INTERVALS[spec.route_size]
    implied = inst.capacity * n / total
    assert implied >= lo - 1e-9
    if inst.capacity > max_demand:
        assert implied < hi + n / total + 1e-9


@pytest.mark.parametrize("demand_type", [1, 2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("route_size", [1, 2, 3, 4, 5, 6])
def test_demand_and_route_classes(demand_type: int, route_size: int) -> None:
    for index, (depot_pos, customer_pos) in enumerate([(1, 1), (2, 2), (3, 3)]):
        n = 15 + 5 * index + demand_type
        spec = XmlSpec(n, depot_pos, customer_pos, demand_type, route_size)
        inst = generate_instance(spec, seed=1000 * demand_type + 10 * route_size + index, name="probe")
        _check_instance(inst)


@pytest.mark.parametrize("customer_pos", [1, 2, 3])
def test_customer_position_classes(customer_pos: int) -> None:
    spec = XmlSpec(30, 1, customer_pos, 2, 3)
    inst = generate_instance(spec, seed=99, name="probe")
    _check_instance(inst)


def test_clustered_is_more_concentrated_than_uniform() -> None:
    def mean_nearest(coords: np.ndarray) -> float:
        pts = coords[1:].astype(np.float64)
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.hypot(diff[..., 0], diff[..., 1])
        np.fill_diagonal(dist, np.inf)
        return float(dist.min(axis=1).mean())

    uniform = [
        mean_nearest(generate_instance(XmlSpec(60, 1, 1, 2, 3), seed=s, name="u").coords)
        for s in range(5)
    ]
    clustered = [
        mean_nearest(generate_instance(XmlSpec(60, 1, 2, 2, 3), seed=s, name="c").coords)
        for s in range(5)
    ]
    assert float(np.mean(clustered)) < float(np.mean(uniform))


def test_cluster_seed_count_range_is_documented() -> None:
    assert CLUSTER_SEED_RANGE == (3, 8)


def test_spec_code_and_grid() -> None:
    assert spec_code(XmlSpec(10, 1, 2, 1, 1)) == "1211"
    assert spec_code(XmlSpec(10, 3, 3, 7, 6)) == "3376"
    grid = all_specs(10)
    assert len(grid) == 3 * 3 * 7 * 6
    assert len({spec_code(spec) for spec in grid}) == len(grid)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_customers": 0},
        {"depot_pos": 0},
        {"depot_pos": 4},
        {"customer_pos": 4},
        {"demand_type": 8},
        {"route_size": 7},
    ],
)
def test_spec_validation(kwargs: dict) -> None:
    base = {
        "n_customers": 10,
        "depot_pos": 1,
        "customer_pos": 1,
        "demand_type": 1,
        "route_size": 1,
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        XmlSpec(**base)


def test_write_vrp_round_trip(tmp_path) -> None:
    spec = XmlSpec(24, 1, 3, 6, 2)
    inst = generate_instance(spec, seed=4242, name="uchoa-n24-1362-000")
    path = tmp_path / f"{inst.name}.vrp"
    write_vrp(path, inst)

    loaded = features.load_instance(path, vehicles=inst.k_fleet)
    assert loaded.name == inst.name
    assert loaded.n_nodes == inst.coords.shape[0]
    assert int(loaded.capacity) == inst.capacity
    assert np.array_equal(loaded.coords, inst.coords.astype(np.float64))
    assert np.array_equal(loaded.demand, inst.demand)
    assert loaded.vehicles == inst.k_fleet

    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == f"NAME : {inst.name}"
    assert "spec 1362" in text
    assert "seed 4242" in text
    assert text.endswith("EOF\n")


def test_determinism_same_seed_same_bytes(tmp_path) -> None:
    spec = XmlSpec(20, 1, 2, 7, 4)
    first = generate_instance(spec, seed=7, name="probe")
    second = generate_instance(spec, seed=7, name="probe")
    assert np.array_equal(first.coords, second.coords)
    assert np.array_equal(first.demand, second.demand)
    assert first.capacity == second.capacity

    path_a = tmp_path / "a.vrp"
    path_b = tmp_path / "b.vrp"
    write_vrp(path_a, first)
    write_vrp(path_b, second)
    assert path_a.read_bytes() == path_b.read_bytes()

    other = generate_instance(spec, seed=8, name="probe")
    assert not np.array_equal(first.coords, other.coords)


def test_set_seeds_recovery_recipe() -> None:
    seeds = set_seeds(2024, 5)
    assert len(seeds) == 5
    assert len(set(seeds)) == 5
    children = np.random.SeedSequence(2024).spawn(5)
    for index, child in enumerate(children):
        words = child.generate_state(2, dtype=np.uint32)
        assert seeds[index] == (int(words[0]) | (int(words[1]) << 32))


def test_generate_set_writes_manifest(tmp_path) -> None:
    rows = generate_set(tmp_path, n_customers=18, count=6, seed=31337)
    assert len(rows) == 6

    manifest = tmp_path / "manifest.csv"
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == MANIFEST_COLUMNS
        table = list(reader)
    assert len(table) == 6

    seeds = set_seeds(31337, 6)
    for index, (row, record) in enumerate(zip(rows, table)):
        name = row["instance"]
        assert name == record["instance"]
        assert name.startswith(f"uchoa-n18-{row['spec']}-")
        assert name.endswith(f"-{index:03d}")
        assert int(row["seed"]) == seeds[index]
        assert row["optimal"] == ""

        path = tmp_path / f"{name}.vrp"
        assert path.exists()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        assert record["sha256"] == row["sha256"]

        loaded = features.load_instance(path, vehicles=int(row["vehicles"]))
        total = int(loaded.demand.sum())
        assert int(row["vehicles"]) == math.ceil(total / int(loaded.capacity))
        assert int(row["n"]) == loaded.n_nodes - 1


def test_generate_set_is_reproducible(tmp_path) -> None:
    first = generate_set(tmp_path / "a", n_customers=15, count=4, seed=5)
    second = generate_set(tmp_path / "b", n_customers=15, count=4, seed=5)
    assert [row["sha256"] for row in first] == [row["sha256"] for row in second]
    assert [row["spec"] for row in first] == [row["spec"] for row in second]

    different = generate_set(tmp_path / "c", n_customers=15, count=4, seed=6)
    assert [row["sha256"] for row in first] != [row["sha256"] for row in different]


def test_generate_set_with_explicit_specs(tmp_path) -> None:
    specs = [XmlSpec(99, 2, 1, 3, 2), XmlSpec(99, 3, 2, 5, 4)]
    rows = generate_set(tmp_path, n_customers=16, count=4, seed=11, specs=specs)
    assert [row["spec"] for row in rows] == ["2132", "3254", "2132", "3254"]
    for row in rows:
        loaded = features.load_instance(tmp_path / f"{row['instance']}.vrp", vehicles=1)
        assert loaded.n_nodes == 17
