from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence

import numpy as np

from sagsep.gurobi_c import GRB_OPTIMAL, GurobiEnv, GurobiModel
from sagsep.types import Instance
from sagsep_train.rootloop import Cut, SeparationResult

__all__ = [
    "OracleError",
    "OracleResult",
    "k_label",
    "exact_rounded_capacity_labels",
    "ExactSeparator",
]


class OracleError(RuntimeError):
    pass


@dataclass
class OracleResult:
    k_label: int
    subsets: List[np.ndarray]
    objective: np.ndarray
    rhs: np.ndarray
    slot_violated: np.ndarray

    def s_hat(self, n_nodes: int) -> np.ndarray:
        matrix = np.zeros((int(n_nodes), int(self.k_label)), dtype=np.uint8)
        for slot, subset in enumerate(self.subsets):
            matrix[np.asarray(subset, dtype=np.int64), slot] = 1
        return matrix


def _clean_demand(demand: Sequence[int] | np.ndarray) -> np.ndarray:
    array = np.asarray(demand)
    if array.ndim != 1 or array.shape[0] < 2:
        raise ValueError(f"demand must be a 1-d array of length >= 2, got shape {array.shape}")
    if array.dtype.kind == "f":
        if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
            raise ValueError("demand must hold integer values")
        array = array.astype(np.int64)
    elif array.dtype.kind in "iub":
        array = array.astype(np.int64)
    else:
        raise ValueError(f"demand must be an integer array, got dtype {array.dtype}")
    if int(array[0]) != 0:
        raise ValueError(f"depot demand must be 0, got {int(array[0])}")
    if int(array.min()) < 0:
        raise ValueError(f"demand must be non-negative, got min {int(array.min())}")
    return array


def k_label(demand: Sequence[int] | np.ndarray, capacity: int) -> int:
    array = _clean_demand(demand)
    total = int(array.sum())
    quota = int(capacity)
    if quota <= 0:
        raise ValueError(f"capacity must be positive, got {capacity}")
    if total <= 0:
        raise ValueError(f"total customer demand must be positive, got {total}")
    return (total + quota - 1) // quota


def _ordered_pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.repeat(np.arange(n, dtype=np.int64), n)
    cols = np.tile(np.arange(n, dtype=np.int64), n)
    keep = rows != cols
    return rows[keep], cols[keep]


def exact_rounded_capacity_labels(
    x_dense: np.ndarray,
    demand: Sequence[int] | np.ndarray,
    capacity: int,
    *,
    env: GurobiEnv,
    threads: int = 1,
    tol: float = 1e-6,
    params: Optional[Mapping[str, float]] = None,
) -> OracleResult:
    matrix = np.asarray(x_dense, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"x_dense must be a square matrix, got shape {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("x_dense must be finite")
    if matrix.size and float(matrix.min()) < 0.0:
        raise ValueError(f"x_dense must be non-negative, got min {float(matrix.min())}")

    demand_arr = _clean_demand(demand)
    n = int(matrix.shape[0])
    if demand_arr.shape[0] != n:
        raise ValueError(
            f"demand length {demand_arr.shape[0]} does not match x_dense dimension {n}"
        )
    quota = int(capacity)
    slots = k_label(demand_arr, quota)

    pair_i, pair_j = _ordered_pairs(n)
    n_pairs = int(pair_i.size)
    obj = np.ascontiguousarray(matrix[pair_i, pair_j], dtype=np.float64)
    y_start = n_pairs

    w_indptr = np.arange(0, 3 * n_pairs + 1, 3, dtype=np.int64)
    w_cols = np.empty(3 * n_pairs, dtype=np.int32)
    w_cols[0::3] = np.arange(n_pairs, dtype=np.int32)
    w_cols[1::3] = (y_start + pair_j).astype(np.int32)
    w_cols[2::3] = (y_start + pair_i).astype(np.int32)
    w_vals = np.tile(np.array([1.0, -1.0, 1.0], dtype=np.float64), n_pairs)
    w_rhs = np.zeros(n_pairs, dtype=np.float64)

    demand_cols = np.arange(y_start + 1, y_start + n, dtype=np.int32)
    demand_coeffs = demand_arr[1:].astype(np.float64)

    extra = dict(params or {})
    subsets: List[np.ndarray] = []
    objectives = np.zeros(slots, dtype=np.float64)
    rhs_values = np.zeros(slots, dtype=np.int32)
    violated = np.zeros(slots, dtype=bool)

    for slot in range(slots):
        model = GurobiModel("exact_capacity", env=env)
        try:
            model.set_param_int("OutputFlag", 0)
            if threads and int(threads) > 0:
                model.set_param_int("Threads", int(threads))
            for key, value in extra.items():
                if float(value) != int(value):
                    raise OracleError(
                        f"Gurobi parameter {key!r}={value!r} is not integral; the package's "
                        "C binding can only set integer parameters"
                    )
                try:
                    model.set_param_int(str(key), int(value))
                except RuntimeError as exc:
                    raise OracleError(f"could not set Gurobi parameter {key!r}: {exc}") from exc

            model.add_vars(n_pairs, obj=obj)
            model.add_vars(n, vtype="B")
            model.set_minimize()
            model.update()

            model.add_constrs_csr(w_indptr, w_cols, w_vals, b">", w_rhs)
            model.add_constr([y_start], [1.0], "=", 0.0, "fix_depot")
            model.add_constr(
                demand_cols,
                demand_coeffs,
                ">",
                float(slot) * float(quota) + 1.0,
                "demand_floor",
            )
            model.update()
            model.optimize()

            status = model.status
            if status != GRB_OPTIMAL:
                raise OracleError(
                    f"label MILP for M={slot} ended with Gurobi status {status} "
                    "(only GRB_OPTIMAL is accepted)"
                )

            value = float(model.obj_val)
            y_sol = np.array(model.get_solution(first=y_start, count=n), copy=True)
        finally:
            model.dispose()

        subset = np.flatnonzero(np.rint(y_sol) == 1.0).astype(np.int32)
        subset = subset[subset >= 1]
        subset_demand = int(demand_arr[subset.astype(np.int64)].sum())
        objectives[slot] = value
        rhs_values[slot] = (subset_demand + quota - 1) // quota
        violated[slot] = value < 2.0 * (slot + 1) - float(tol)
        subsets.append(subset)

    return OracleResult(
        k_label=slots,
        subsets=subsets,
        objective=objectives,
        rhs=rhs_values,
        slot_violated=violated,
    )


class ExactSeparator:
    def __init__(
        self,
        demand: Sequence[int] | np.ndarray,
        capacity: int,
        *,
        env: Optional[GurobiEnv] = None,
        threads: int = 1,
        tol: float = 1e-6,
        params: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.demand = _clean_demand(demand)
        self.capacity = int(capacity)
        self.k_label = k_label(self.demand, self.capacity)
        self.threads = int(threads)
        self.tol = float(tol)
        self.params = dict(params or {})
        self._owns_env = env is None
        self.env = GurobiEnv() if env is None else env

    @classmethod
    def from_instance(cls, instance: Instance, **kwargs) -> "ExactSeparator":
        return cls(instance.demand, int(instance.capacity), **kwargs)

    def separate(self, x_dense: np.ndarray, *, iteration: int) -> SeparationResult:
        result = exact_rounded_capacity_labels(
            x_dense,
            self.demand,
            self.capacity,
            env=self.env,
            threads=self.threads,
            tol=self.tol,
            params=self.params,
        )
        cuts = [
            Cut(customers=result.subsets[slot], rhs=int(result.rhs[slot]))
            for slot in range(result.k_label)
            if bool(result.slot_violated[slot]) and result.subsets[slot].size
        ]
        payload = {
            "objective": result.objective,
            "rhs": result.rhs,
            "slot_violated": result.slot_violated,
            "s_hat": result.s_hat(int(np.asarray(x_dense).shape[0])),
            "subsets": result.subsets,
        }
        return SeparationResult(cuts=cuts, payload=payload)

    def close(self) -> None:
        if self._owns_env and self.env is not None:
            self.env.dispose()
            self.env = None
            self._owns_env = False

    def __enter__(self) -> "ExactSeparator":
        return self

    def __exit__(self, *args) -> None:
        self.close()
