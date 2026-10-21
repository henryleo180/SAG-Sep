from __future__ import annotations

import numpy as np
import pytest

from sagsep import features, loop
from sagsep.loop import add_cut_rows, build_cut_row, build_root_lp

try:
    from sagsep import gurobi_c

    gurobi_c.version()
except Exception as exc:
    pytest.skip(f"Gurobi C library unavailable: {exc}", allow_module_level=True)

INSTANCE = "P-n20-k2.vrp"
VEHICLES = 2


@pytest.fixture(scope="module")
def instance(instances_dir):
    path = instances_dir / "random" / INSTANCE
    if not path.exists():
        pytest.skip(f"instance not present: {path}")
    return features.load_instance(path, vehicles=VEHICLES)


def _dense_and_degree(values, edge_i, edge_j, n):
    dense = np.zeros((n, n), dtype=np.float64)
    dense[edge_i, edge_j] = values
    dense[edge_j, edge_i] = values
    degree = np.zeros(n, dtype=np.float64)
    np.add.at(degree, edge_i, values)
    np.add.at(degree, edge_j, values)
    return dense, degree


def _rhs(instance, nodes0) -> float:
    demand = np.asarray(instance.demand, dtype=np.int64)[nodes0].sum()
    capacity = np.int64(instance.capacity)
    return float((demand + capacity - np.int64(1)) // capacity)


def _violation(dense, degree, instance, nodes0) -> float:
    inside = float(dense[np.ix_(nodes0, nodes0)].sum())
    boundary = float(degree[nodes0].sum()) - inside
    return 2.0 * _rhs(instance, nodes0) - boundary


def _most_violated_subset(dense, degree, instance):
    n = instance.n_nodes
    best = (np.array([1], dtype=np.int64), _rhs(instance, [1]), -np.inf)
    for seed in range(1, n):
        chosen = [seed]
        remaining = [c for c in range(1, n) if c != seed]
        while remaining and len(chosen) < n - 2:
            link = [float(dense[np.ix_(chosen, [c])].sum()) for c in remaining]
            chosen.append(remaining.pop(int(np.argmax(link))))
            nodes0 = np.array(sorted(chosen), dtype=np.int64)
            violation = _violation(dense, degree, instance, nodes0)
            if violation > best[2]:
                best = (nodes0, _rhs(instance, nodes0), violation)
    return best


def test_wrappers_delegate_to_the_root_loop_implementations(monkeypatch) -> None:
    sentinel_lp = object()
    sentinel_row = object()
    monkeypatch.setattr(loop, "_build_lp", lambda *a, **k: sentinel_lp)
    monkeypatch.setattr(loop, "_build_cut_row", lambda *a, **k: sentinel_row)
    monkeypatch.setattr(loop, "_add_rows_csr", lambda *a, **k: 12345)
    assert build_root_lp(None, None, None, 0) is sentinel_lp
    assert build_cut_row([1], 1.0, None, 0, {}) is sentinel_row
    assert add_cut_rows(None, []) == 12345


def test_root_lp_and_cut_row_end_to_end(instance) -> None:
    n = instance.n_nodes
    weight = features.distance_matrix(instance)
    env = gurobi_c.GurobiEnv()
    model_lp = gurobi_c.GurobiModel("wrapper_root_lp", env=env)
    try:
        model_lp.set_param_int("OutputFlag", 0)
        model_lp.set_param_int("Method", 1)
        model_lp.set_param_int("Threads", 1)

        edge_i, edge_j, var_index = build_root_lp(model_lp, instance, weight, VEHICLES)
        assert edge_i.size == n * (n - 1) // 2
        assert np.all(edge_i < edge_j), "variable order is the external (i<j) upper triangle"
        assert var_index.shape == (n, n)
        assert model_lp.num_constrs == n

        model_lp.optimize()
        assert model_lp.status == gurobi_c.GRB_OPTIMAL
        objective_before = model_lp.obj_val
        rows_before = model_lp.num_constrs

        values = np.array(model_lp.get_solution(), dtype=np.float64)
        dense, degree = _dense_and_degree(values, edge_i, edge_j, n)
        nodes0, rhs, violation = _most_violated_subset(dense, degree, instance)
        assert violation > 0.0, "the degree-only relaxation should leave a violated RCC"

        nodes_ext = (nodes0 + 1).tolist()
        cache = {}
        cols, coeffs, bound = build_cut_row(nodes_ext, rhs, var_index, n - 1, cache)
        assert cols.dtype == np.int32 and coeffs.dtype == np.float64
        assert cols.size == coeffs.size > 0
        assert np.all(cols >= 0) and np.all(cols < edge_i.size)

        nnz = add_cut_rows(model_lp, [(cols, coeffs, bound)])
        model_lp.update()
        assert nnz == cols.size
        assert model_lp.num_constrs == rows_before + 1

        model_lp.optimize()
        assert model_lp.status == gurobi_c.GRB_OPTIMAL
        objective_after = model_lp.obj_val
        assert objective_after >= objective_before - 1e-9
        assert objective_after > objective_before + 1e-9

        new_values = np.array(model_lp.get_solution(), dtype=np.float64)
        assert float(new_values[cols].dot(coeffs)) <= bound + 1e-6
    finally:
        model_lp.dispose()
        env.dispose()


def test_build_cut_row_matches_the_private_builder(instance) -> None:
    n = instance.n_nodes
    var_index = np.full((n, n), -1, dtype=np.int32)
    iu, ju = np.triu_indices(n, k=1)
    ids = np.arange(iu.size, dtype=np.int32)
    var_index[iu, ju] = ids
    var_index[ju, iu] = ids
    for subset in ([2, 3, 4], list(range(2, n + 1)), [5]):
        wrapped = build_cut_row(subset, 2.0, var_index, n - 1, {})
        private = loop._build_cut_row(subset, 2.0, var_index, n - 1, {})
        assert wrapped[0].tobytes() == private[0].tobytes()
        assert wrapped[1].tobytes() == private[1].tobytes()
        assert wrapped[2] == private[2]
