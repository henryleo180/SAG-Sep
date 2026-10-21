from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np

from sagsep import features
from sagsep.gurobi_c import GRB_OPTIMAL, GurobiEnv, GurobiModel
from sagsep.loop import add_cut_rows, build_cut_row, build_root_lp
from sagsep.types import Instance

__all__ = [
    "Cut",
    "SeparationResult",
    "Separator",
    "StateRecord",
    "RootLoopResult",
    "TERMINATIONS",
    "run_root_loop",
]

TERMINATIONS: tuple[str, ...] = (
    "closed",
    "iteration_cap",
    "stalled_duplicate",
    "time_limit",
    "error",
)


@dataclass(frozen=True)
class Cut:
    customers: np.ndarray
    rhs: int


@dataclass
class SeparationResult:
    cuts: List[Cut]
    payload: Optional[dict] = None


class Separator(Protocol):
    def separate(self, x_dense: np.ndarray, *, iteration: int) -> SeparationResult:
        pass


@dataclass
class StateRecord:
    iteration: int
    lp_value: float
    support_u: np.ndarray
    support_v: np.ndarray
    support_x: np.ndarray
    cuts_added: int = 0
    payload: Optional[dict] = None


@dataclass
class RootLoopResult:
    base_lower_bound: float
    lower_bound: float
    iterations: int
    total_cuts: int
    termination: str
    states: List[StateRecord] = field(default_factory=list)
    runtime_s: float = 0.0


def _normalise_cut(cut: Cut, n_nodes: int) -> Tuple[np.ndarray, int]:
    nodes = np.asarray(cut.customers, dtype=np.int64).reshape(-1)
    if nodes.size == 0:
        raise ValueError("Cut.customers must be non-empty")
    if np.any(nodes < 1) or np.any(nodes >= n_nodes):
        raise ValueError(
            f"Cut.customers must be 0-based internal node ids in 1..{n_nodes - 1} "
            f"(the depot is excluded), got min {int(nodes.min())} max {int(nodes.max())}"
        )
    if np.any(np.diff(nodes) <= 0):
        raise ValueError("Cut.customers must be strictly ascending")
    rhs = int(cut.rhs)
    if rhs != cut.rhs or rhs < 1:
        raise ValueError(f"Cut.rhs must be a positive integer, got {cut.rhs!r}")
    return nodes, rhs


def run_root_loop(
    instance: Instance,
    *,
    separator: Separator,
    vehicles: int,
    max_iter: Optional[int],
    min_violation: float = 1e-6,
    gurobi_method: int = 1,
    gurobi_threads: int = 1,
    time_limit: Optional[float] = None,
) -> RootLoopResult:
    n = int(instance.n_nodes)
    n_customers = n - 1
    if max_iter is not None and int(max_iter) < 1:
        raise ValueError(f"max_iter must be None or >= 1, got {max_iter}")
    if time_limit is not None and float(time_limit) <= 0.0:
        raise ValueError(f"time_limit must be None or positive, got {time_limit}")

    states: List[StateRecord] = []
    total_cuts = 0
    termination = "error"
    base_lower_bound = float("nan")
    lower_bound = float("nan")

    start_time = perf_counter()
    env = GurobiEnv()
    model_lp = GurobiModel("cvrp_root_relaxation", env=env)
    try:
        model_lp.set_param_int("OutputFlag", 0)
        model_lp.set_param_int("Method", int(gurobi_method))
        if gurobi_threads and int(gurobi_threads) > 0:
            model_lp.set_param_int("Threads", int(gurobi_threads))

        weight = features.distance_matrix(instance)
        edge_i, edge_j, var_index = build_root_lp(model_lp, instance, weight, int(vehicles))

        dense = np.zeros((n, n), dtype=np.float64)
        degree = np.zeros(n, dtype=np.float64)
        upper_tri_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        seen: set[frozenset] = set()

        iteration = 0
        while True:
            iteration += 1
            model_lp.optimize()
            if model_lp.status != GRB_OPTIMAL:
                termination = "error"
                break

            lp_value = float(model_lp.obj_val)
            values = np.array(model_lp.get_solution(), dtype=np.float64, copy=True)
            dense[:] = 0.0
            dense[edge_i, edge_j] = values
            dense[edge_j, edge_i] = values
            degree[:] = 0.0
            np.add.at(degree, edge_i, values)
            np.add.at(degree, edge_j, values)

            support = values > 0.0
            state = StateRecord(
                iteration=iteration,
                lp_value=lp_value,
                support_u=np.ascontiguousarray(edge_i[support], dtype=np.int32),
                support_v=np.ascontiguousarray(edge_j[support], dtype=np.int32),
                support_x=np.ascontiguousarray(values[support], dtype=np.float64),
            )
            states.append(state)
            if len(states) == 1:
                base_lower_bound = lp_value
            lower_bound = lp_value

            result = separator.separate(dense, iteration=iteration)
            state.payload = result.payload

            if not result.cuts:
                termination = "closed"
                break

            admitted: List[Tuple[np.ndarray, int, frozenset]] = []
            round_keys: set[frozenset] = set()
            for cut in result.cuts:
                nodes, rhs = _normalise_cut(cut, n)
                key = frozenset(int(node) for node in nodes)
                if key in seen or key in round_keys:
                    continue
                inside = float(dense[np.ix_(nodes, nodes)].sum())
                boundary = float(degree[nodes].sum()) - inside
                violation = 2.0 * float(rhs) - boundary
                if violation <= float(min_violation):
                    continue
                round_keys.add(key)
                admitted.append((nodes, rhs, key))

            if not admitted:
                termination = "stalled_duplicate"
                break

            rows = [
                build_cut_row(nodes + 1, float(rhs), var_index, n_customers, upper_tri_cache)
                for nodes, rhs, _key in admitted
            ]
            add_cut_rows(model_lp, rows)
            for _nodes, _rhs, key in admitted:
                seen.add(key)
            state.cuts_added = len(admitted)
            total_cuts += len(admitted)

            if max_iter is not None and iteration >= int(max_iter):
                termination = "iteration_cap"
                break
            if time_limit is not None and perf_counter() - start_time >= float(time_limit):
                termination = "time_limit"
                break
    finally:
        model_lp.dispose()
        env.dispose()

    return RootLoopResult(
        base_lower_bound=base_lower_bound,
        lower_bound=lower_bound,
        iterations=len(states),
        total_cuts=total_cuts,
        termination=termination,
        states=states,
        runtime_s=perf_counter() - start_time,
    )
