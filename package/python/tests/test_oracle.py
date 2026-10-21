from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from sagsep import features
from sagsep.gurobi_c import GRB_OPTIMAL
from sagsep.loop import build_root_lp
from sagsep_train.oracle import (
    ExactSeparator,
    OracleError,
    exact_rounded_capacity_labels,
    k_label,
)

gurobi_c = pytest.importorskip("sagsep.gurobi_c")


@pytest.fixture(scope="module")
def env():
    try:
        handle = gurobi_c.GurobiEnv()
    except Exception as exc:
        pytest.skip(f"Gurobi environment unavailable: {exc}")
    yield handle
    handle.dispose()


def _boundary(x_dense: np.ndarray, subset) -> float:
    nodes = np.asarray(list(subset), dtype=np.int64)
    if nodes.size == 0:
        return 0.0
    degree = x_dense.sum(axis=1)
    inside = float(x_dense[np.ix_(nodes, nodes)].sum())
    return float(degree[nodes].sum()) - inside


def _brute_force(x_dense: np.ndarray, demand: np.ndarray, capacity: int, slot: int):
    n = int(demand.shape[0])
    best = None
    for size in range(1, n):
        for subset in itertools.combinations(range(1, n), size):
            if int(demand[list(subset)].sum()) >= slot * capacity + 1:
                value = _boundary(x_dense, subset)
                if best is None or value < best - 1e-15:
                    best = value
    return best


def _random_point(rng: np.random.Generator, n: int) -> np.ndarray:
    x = np.zeros((n, n), dtype=np.float64)
    iu = np.triu_indices(n, k=1)
    x[iu] = rng.choice([0.0, 0.25, 0.5, 1.0], size=iu[0].size)
    return x + x.T


@pytest.mark.parametrize("seed", range(6))
def test_matches_brute_force(env, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 11))
    demand = np.zeros(n, dtype=np.int64)
    demand[1:] = rng.integers(1, 12, size=n - 1)
    capacity = int(rng.integers(int(demand.max()), int(demand.sum()) + 1))
    x = _random_point(rng, n)

    result = exact_rounded_capacity_labels(x, demand, capacity, env=env)
    assert result.k_label == k_label(demand, capacity)
    assert result.objective.shape == (result.k_label,)
    assert result.rhs.shape == (result.k_label,)
    assert result.slot_violated.shape == (result.k_label,)

    for slot in range(result.k_label):
        expected = _brute_force(x, demand, capacity, slot)
        assert expected is not None
        assert result.objective[slot] == pytest.approx(expected, abs=1e-9)

        subset = result.subsets[slot]
        assert subset.dtype == np.int32
        assert subset.size > 0
        assert np.all(np.diff(subset) > 0)
        assert int(subset.min()) >= 1
        assert _boundary(x, subset) == pytest.approx(expected, abs=1e-9)
        subset_demand = int(demand[subset.astype(np.int64)].sum())
        assert subset_demand >= slot * capacity + 1
        assert int(result.rhs[slot]) == math.ceil(subset_demand / capacity)
        assert int(result.rhs[slot]) >= slot + 1
        assert bool(result.slot_violated[slot]) == (expected < 2.0 * (slot + 1) - 1e-6)


def test_crossing_edges_are_counted_once(env) -> None:
    n = 5
    x = np.zeros((n, n), dtype=np.float64)

    def put(i, j, value):
        x[i, j] = value
        x[j, i] = value

    put(0, 1, 2.0)
    put(0, 2, 0.1)
    put(0, 3, 0.3)
    put(0, 4, 2.0)
    put(1, 2, 1.0)
    put(2, 3, 0.75)
    put(3, 4, 1.0)

    demand = np.array([0, 1, 1, 1, 1], dtype=np.int64)
    capacity = 4
    result = exact_rounded_capacity_labels(x, demand, capacity, env=env)

    assert result.k_label == 1
    assert _boundary(x, [2]) == pytest.approx(1.85)
    assert result.objective[0] == pytest.approx(1.85, abs=1e-9)
    assert result.subsets[0].tolist() == [2]
    assert int(result.rhs[0]) == 1
    assert bool(result.slot_violated[0]) is True


def test_tie_returns_one_of_the_optima(env) -> None:
    n = 5
    x = np.zeros((n, n), dtype=np.float64)

    def put(i, j, value):
        x[i, j] = value
        x[j, i] = value

    put(0, 1, 0.5)
    put(0, 2, 0.5)
    put(1, 2, 1.5)
    put(0, 3, 0.5)
    put(0, 4, 0.5)
    put(3, 4, 1.5)

    demand = np.array([0, 1, 1, 1, 1], dtype=np.int64)
    capacity = 4
    assert _boundary(x, [1, 2]) == pytest.approx(1.0)
    assert _boundary(x, [3, 4]) == pytest.approx(1.0)

    result = exact_rounded_capacity_labels(x, demand, capacity, env=env)
    assert result.k_label == 1
    assert result.objective[0] == pytest.approx(1.0, abs=1e-9)
    assert result.subsets[0].tolist() in ([1, 2], [3, 4])
    assert int(result.rhs[0]) == 1
    assert bool(result.slot_violated[0]) is True


def test_s_hat_layout(env) -> None:
    demand = np.array([0, 3, 3, 3, 3], dtype=np.int64)
    capacity = 4
    x = _random_point(np.random.default_rng(0), 5)
    result = exact_rounded_capacity_labels(x, demand, capacity, env=env)
    s_hat = result.s_hat(5)
    assert s_hat.shape == (5, result.k_label)
    assert s_hat.dtype == np.uint8
    assert int(s_hat[0].max()) == 0
    for slot, subset in enumerate(result.subsets):
        assert np.array_equal(np.flatnonzero(s_hat[:, slot]), subset.astype(np.int64))


@pytest.mark.parametrize(
    "demand, capacity",
    [
        (np.array([5, 1, 1], dtype=np.int64), 3),
        (np.array([0, -1, 2], dtype=np.int64), 3),
        (np.array([0, 0, 0], dtype=np.int64), 3),
        (np.array([0, 1, 2], dtype=np.int64), 0),
        (np.array([0, 1.5, 2.0]), 3),
    ],
)
def test_k_label_validation(demand, capacity) -> None:
    with pytest.raises(ValueError):
        k_label(demand, capacity)


def test_k_label_integer_arithmetic() -> None:
    assert k_label(np.array([0, 10, 10, 10], dtype=np.int64), 10) == 3
    assert k_label(np.array([0, 10, 10, 1], dtype=np.int64), 10) == 3
    assert k_label(np.array([0, 1], dtype=np.int64), 10**9) == 1
    big = np.array([0, 3 * 10**15], dtype=np.int64)
    assert k_label(big, 10**15) == 3


def test_negative_point_is_rejected(env) -> None:
    x = np.zeros((4, 4), dtype=np.float64)
    x[1, 2] = x[2, 1] = -1.0
    with pytest.raises(ValueError):
        exact_rounded_capacity_labels(x, np.array([0, 1, 1, 1], dtype=np.int64), 2, env=env)


def test_non_integral_parameter_is_rejected(env) -> None:
    x = _random_point(np.random.default_rng(3), 5)
    demand = np.array([0, 1, 1, 1, 1], dtype=np.int64)
    with pytest.raises(OracleError):
        exact_rounded_capacity_labels(x, demand, 4, env=env, params={"MIPGap": 0.5})


def test_labels_at_the_root_lp_point(env, instances_dir) -> None:
    path = instances_dir / "random" / "P-n20-k2.vrp"
    instance = features.load_instance(path, vehicles=2)
    n = instance.n_nodes

    model = gurobi_c.GurobiModel("root_lp", env=env)
    try:
        model.set_param_int("OutputFlag", 0)
        model.set_param_int("Method", 1)
        model.set_param_int("Threads", 1)
        weight = features.distance_matrix(instance)
        edge_i, edge_j, _var_index = build_root_lp(model, instance, weight, instance.vehicles)
        model.optimize()
        assert model.status == GRB_OPTIMAL
        values = np.array(model.get_solution(), dtype=np.float64, copy=True)
    finally:
        model.dispose()

    x = np.zeros((n, n), dtype=np.float64)
    x[edge_i, edge_j] = values
    x[edge_j, edge_i] = values

    result = exact_rounded_capacity_labels(x, instance.demand, int(instance.capacity), env=env)
    assert result.k_label == features.vehicle_replica_count(instance)
    assert result.k_label == k_label(instance.demand, int(instance.capacity))

    for slot in range(result.k_label):
        subset = result.subsets[slot]
        assert subset.size > 0
        assert result.objective[slot] == pytest.approx(_boundary(x, subset), abs=1e-9)
        subset_demand = int(instance.demand[subset.astype(np.int64)].sum())
        assert subset_demand >= slot * int(instance.capacity) + 1
        assert int(result.rhs[slot]) == math.ceil(subset_demand / int(instance.capacity))
    assert bool(result.slot_violated.any())


def test_exact_separator_payload(env, instances_dir) -> None:
    path = instances_dir / "random" / "P-n20-k2.vrp"
    instance = features.load_instance(path, vehicles=2)
    n = instance.n_nodes
    rng = np.random.default_rng(0)
    x = _random_point(rng, n)

    separator = ExactSeparator.from_instance(instance, env=env)
    result = separator.separate(x, iteration=1)
    payload = result.payload
    assert set(payload) == {"objective", "rhs", "slot_violated", "s_hat", "subsets"}
    assert payload["s_hat"].shape == (n, separator.k_label)
    assert payload["objective"].shape == (separator.k_label,)
    assert len(result.cuts) == int(np.count_nonzero(payload["slot_violated"]))
    for cut in result.cuts:
        assert cut.customers.dtype == np.int32
        assert np.all(np.diff(cut.customers) > 0)
        assert int(cut.customers.min()) >= 1
        subset_demand = int(instance.demand[cut.customers.astype(np.int64)].sum())
        assert cut.rhs == math.ceil(subset_demand / int(instance.capacity))
    own = ExactSeparator.from_instance(instance)
    own.close()
    assert own.env is None
