from __future__ import annotations

import csv
import math

import numpy as np
import pytest

from sagsep import features
from sagsep_train.instances import XmlSpec, generate_instance, write_vrp
from sagsep_train.oracle import ExactSeparator
from sagsep_train.rootloop import Cut, SeparationResult, run_root_loop

gurobi_c = pytest.importorskip("sagsep.gurobi_c")

MIN_VIOLATION = 1e-6


@pytest.fixture(scope="module")
def env():
    try:
        handle = gurobi_c.GurobiEnv()
    except Exception as exc:
        pytest.skip(f"Gurobi environment unavailable: {exc}")
    yield handle
    handle.dispose()


def _manifest_vehicles(instances_dir, name: str) -> int:
    with (instances_dir / "random" / "manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["instance"] == name:
                return int(row["vehicles"])
    raise AssertionError(f"{name} missing from the manifest")


@pytest.fixture(scope="module")
def p_n20(instances_dir):
    vehicles = _manifest_vehicles(instances_dir, "P-n20-k2")
    assert vehicles == 2
    return features.load_instance(instances_dir / "random" / "P-n20-k2.vrp", vehicles=vehicles)


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    directory = tmp_path_factory.mktemp("generated")
    out = []
    for index, spec in enumerate([XmlSpec(15, 1, 1, 2, 2), XmlSpec(15, 2, 3, 4, 1)]):
        inst = generate_instance(spec, seed=900 + index, name=f"gen-{index:03d}")
        path = directory / f"{inst.name}.vrp"
        write_vrp(path, inst)
        out.append(features.load_instance(path, vehicles=inst.k_fleet))
    return out


def _dense_from_state(state, n: int) -> tuple[np.ndarray, np.ndarray]:
    dense = np.zeros((n, n), dtype=np.float64)
    dense[state.support_u, state.support_v] = state.support_x
    dense[state.support_v, state.support_u] = state.support_x
    return dense, dense.sum(axis=1)


def _check_trajectory(instance, result, *, min_violation: float = MIN_VIOLATION) -> None:
    n = int(instance.n_nodes)
    demand = np.asarray(instance.demand, dtype=np.int64)
    capacity = int(instance.capacity)

    assert result.iterations == len(result.states)
    assert [state.iteration for state in result.states] == list(range(1, len(result.states) + 1))

    seen: set[frozenset] = set()
    previous_lp = -math.inf
    for state in result.states:
        assert state.lp_value >= previous_lp - 1e-6
        previous_lp = state.lp_value

        assert state.support_x.size == 0 or float(state.support_x.min()) > 0.0
        key = state.support_u.astype(np.int64) * n + state.support_v.astype(np.int64)
        assert np.all(np.diff(key) > 0)
        assert np.all(state.support_u < state.support_v)

        dense, degree = _dense_from_state(state, n)
        payload = state.payload
        assert payload is not None

        admitted = 0
        for slot, subset in enumerate(payload["subsets"]):
            if not bool(payload["slot_violated"][slot]) or subset.size == 0:
                continue
            nodes = subset.astype(np.int64)
            cut_key = frozenset(int(node) for node in nodes)
            if cut_key in seen:
                continue
            rhs = int(payload["rhs"][slot])
            assert rhs == math.ceil(int(demand[nodes].sum()) / capacity)
            inside = float(dense[np.ix_(nodes, nodes)].sum())
            violation = 2.0 * rhs - (float(degree[nodes].sum()) - inside)
            if violation <= min_violation:
                continue
            assert violation > min_violation
            seen.add(cut_key)
            admitted += 1
        assert admitted == state.cuts_added

    assert sum(state.cuts_added for state in result.states) == result.total_cuts
    assert result.base_lower_bound == result.states[0].lp_value
    assert result.lower_bound == result.states[-1].lp_value
    assert result.lower_bound >= result.base_lower_bound - 1e-9
    assert result.runtime_s > 0.0


def test_closure_on_p_n20(env, p_n20) -> None:
    separator = ExactSeparator.from_instance(p_n20, env=env)
    result = run_root_loop(
        p_n20,
        separator=separator,
        vehicles=p_n20.vehicles,
        max_iter=None,
        min_violation=MIN_VIOLATION,
    )
    assert result.termination == "closed"
    assert result.states[-1].cuts_added == 0
    assert result.total_cuts > 0
    assert result.lower_bound > result.base_lower_bound
    _check_trajectory(p_n20, result)


@pytest.mark.parametrize("index", [0, 1])
def test_closure_on_generated(env, generated, index: int) -> None:
    instance = generated[index]
    separator = ExactSeparator.from_instance(instance, env=env)
    result = run_root_loop(
        instance,
        separator=separator,
        vehicles=instance.vehicles,
        max_iter=None,
        min_violation=MIN_VIOLATION,
    )
    assert result.termination == "closed"
    assert result.states[-1].cuts_added == 0
    assert result.lower_bound >= result.base_lower_bound
    _check_trajectory(instance, result)


def test_iteration_cap(env, p_n20) -> None:
    separator = ExactSeparator.from_instance(p_n20, env=env)
    result = run_root_loop(
        p_n20,
        separator=separator,
        vehicles=p_n20.vehicles,
        max_iter=2,
        min_violation=MIN_VIOLATION,
    )
    assert result.termination == "iteration_cap"
    assert result.iterations == 2
    assert len(result.states) == 2
    assert result.states[-1].cuts_added > 0
    _check_trajectory(p_n20, result)


class _RepeatSeparator:
    def __init__(self, inner) -> None:
        self._inner = inner
        self._first = None
        self.calls = 0

    def separate(self, x_dense, *, iteration: int) -> SeparationResult:
        self.calls += 1
        if self._first is None:
            self._first = self._inner.separate(x_dense, iteration=iteration)
        return SeparationResult(cuts=list(self._first.cuts), payload=self._first.payload)


def test_stalled_duplicate(env, p_n20) -> None:
    separator = _RepeatSeparator(ExactSeparator.from_instance(p_n20, env=env))
    result = run_root_loop(
        p_n20,
        separator=separator,
        vehicles=p_n20.vehicles,
        max_iter=None,
        min_violation=MIN_VIOLATION,
    )
    assert result.termination == "stalled_duplicate"
    assert separator.calls == 2
    assert result.iterations == 2
    assert result.states[0].cuts_added > 0
    assert result.states[1].cuts_added == 0


class _EmptySeparator:
    def separate(self, x_dense, *, iteration: int) -> SeparationResult:
        return SeparationResult(cuts=[], payload={"iteration": iteration})


def test_closed_without_any_cut(env, p_n20) -> None:
    result = run_root_loop(
        p_n20,
        separator=_EmptySeparator(),
        vehicles=p_n20.vehicles,
        max_iter=None,
    )
    assert result.termination == "closed"
    assert result.iterations == 1
    assert result.total_cuts == 0
    assert result.base_lower_bound == result.lower_bound
    assert result.states[0].payload == {"iteration": 1}


class _BadCutSeparator:
    def __init__(self, cut: Cut) -> None:
        self._cut = cut

    def separate(self, x_dense, *, iteration: int) -> SeparationResult:
        return SeparationResult(cuts=[self._cut], payload=None)


@pytest.mark.parametrize(
    "cut",
    [
        Cut(customers=np.array([0, 1], dtype=np.int32), rhs=1),
        Cut(customers=np.array([3, 2], dtype=np.int32), rhs=1),
        Cut(customers=np.array([], dtype=np.int32), rhs=1),
        Cut(customers=np.array([1, 2], dtype=np.int32), rhs=0),
    ],
)
def test_malformed_cuts_are_rejected(env, p_n20, cut: Cut) -> None:
    with pytest.raises(ValueError):
        run_root_loop(
            p_n20,
            separator=_BadCutSeparator(cut),
            vehicles=p_n20.vehicles,
            max_iter=None,
        )


def test_time_limit(env, p_n20) -> None:
    separator = ExactSeparator.from_instance(p_n20, env=env)
    result = run_root_loop(
        p_n20,
        separator=separator,
        vehicles=p_n20.vehicles,
        max_iter=None,
        min_violation=MIN_VIOLATION,
        time_limit=1e-6,
    )
    assert result.termination == "time_limit"
    assert result.iterations == 1
    _check_trajectory(p_n20, result)
