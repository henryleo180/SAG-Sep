from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from sagsep_bc_common.gurobi_capi import BranchCutModel, GurobiEnv

from .instance import DirectedInstance

__all__ = [
    "DirectedModel",
    "build_directed_model",
    "add_rows_csr",
    "canonical_subset",
    "rcc_rhs",
    "violation_from_dense",
]

INTEGRALITY_TOLERANCE = 0.5

_VTYPE_CONTINUOUS = ord("C")
_VTYPE_BINARY = ord("B")


@dataclass
class DirectedModel:
    model: BranchCutModel
    instance: DirectedInstance
    vehicles: int
    open_routes: bool
    n_model_nodes: int
    terminal: int
    arc_tail: np.ndarray
    arc_head: np.ndarray
    arc_sep_tail: np.ndarray
    arc_sep_head: np.ndarray
    n_arc_vars: int
    pair_u: np.ndarray
    pair_v: np.ndarray
    pair_forward: np.ndarray
    pair_reverse: np.ndarray

    @property
    def n_sep_nodes(self) -> int:
        return int(self.instance.n_nodes)

    @property
    def n_customers(self) -> int:
        return int(self.instance.n_nodes) - 1

    def relax_to_continuous(self) -> None:
        self.model.set_vtypes(
            0, np.full(self.n_arc_vars, _VTYPE_CONTINUOUS, dtype=np.uint8)
        )
        self.model.update()

    def restore_integer_types(self) -> None:
        self.model.set_vtypes(0, np.full(self.n_arc_vars, _VTYPE_BINARY, dtype=np.uint8))
        self.model.update()

    def project(self, values: np.ndarray) -> np.ndarray:
        return values[self.pair_forward] + values[self.pair_reverse]

    def support(
        self, y: np.ndarray, threshold: float = 1e-8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask = y > threshold
        return (
            self.pair_u[mask],
            self.pair_v[mask],
            np.ascontiguousarray(y[mask], dtype=np.float64),
        )

    def dense_support(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = self.n_sep_nodes
        dense = np.zeros((n, n), dtype=np.float64)
        dense[self.pair_u, self.pair_v] = y
        dense[self.pair_v, self.pair_u] = y
        degree = np.zeros(n, dtype=np.float64)
        np.add.at(degree, self.pair_u, y)
        np.add.at(degree, self.pair_v, y)
        return dense, degree

    def cut_row(
        self, nodes_ext: Sequence[int], rhs: float
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        inside = np.zeros(self.n_sep_nodes, dtype=bool)
        inside[np.asarray(nodes_ext, dtype=np.int64) - 1] = True
        crossing = inside[self.arc_sep_tail] ^ inside[self.arc_sep_head]
        cols = np.flatnonzero(crossing).astype(np.int32, copy=False)
        return cols, np.ones(cols.size, dtype=np.float64), 2.0 * float(rhs)

    def rcc_rhs(self, nodes_ext: Iterable[int]) -> float:
        return rcc_rhs(nodes_ext, self.instance)

    def detect_violations(self, values: np.ndarray) -> List[Tuple[List[int], float]]:
        used = values > INTEGRALITY_TOLERANCE
        successor = np.full(self.n_model_nodes, -1, dtype=np.int64)
        customer_arcs = np.flatnonzero(used & (self.arc_tail != 0))
        successor[self.arc_tail[customer_arcs]] = self.arc_head[customer_arcs]
        depot_arcs = np.flatnonzero(used & (self.arc_tail == 0))
        starts = self.arc_head[depot_arcs]

        n = self.n_sep_nodes
        visited = np.zeros(self.n_model_nodes, dtype=bool)
        routes: List[List[int]] = []
        for start in starts.tolist():
            route: List[int] = []
            node = int(start)
            while node != self.terminal and node != 0:
                if visited[node]:
                    break
                visited[node] = True
                route.append(node + 1)
                node = int(successor[node])
                if node < 0:
                    break
            if route:
                routes.append(route)

        violations: List[Tuple[List[int], float]] = []
        for node in range(1, n):
            if visited[node]:
                continue
            cycle: List[int] = []
            walk = node
            while 0 < walk < n and not visited[walk]:
                visited[walk] = True
                cycle.append(walk + 1)
                walk = int(successor[walk])
            if cycle:
                violations.append((sorted(cycle), rcc_rhs(cycle, self.instance)))
        for route in routes:
            required = rcc_rhs(route, self.instance)
            if required > 1.0:
                violations.append((sorted(route), required))
        return violations


def build_directed_model(
    env: GurobiEnv,
    instance: DirectedInstance,
    vehicles: int,
    *,
    open_routes: bool,
    name: str = "directed_two_index",
) -> DirectedModel:
    if vehicles <= 0:
        raise ValueError(f"vehicles must be positive, got {vehicles}")
    n = int(instance.n_nodes)
    n_customers = n - 1
    if vehicles > n_customers:
        raise ValueError(
            f"a fixed fleet of {vehicles} cannot serve {n_customers} customers: "
            "every route must visit at least one"
        )
    sink = n if open_routes else -1
    n_model_nodes = n + 1 if open_routes else n
    terminal = sink if open_routes else 0

    tails: List[int] = []
    heads: List[int] = []
    for tail in range(n):
        for head in range(n):
            if head == tail or (open_routes and head == 0):
                continue
            tails.append(tail)
            heads.append(head)
        if open_routes and tail != 0:
            tails.append(tail)
            heads.append(sink)
    arc_tail = np.asarray(tails, dtype=np.int64)
    arc_head = np.asarray(heads, dtype=np.int64)
    n_arcs = int(arc_tail.size)

    obj = np.zeros(n_arcs, dtype=np.float64)
    physical = arc_head != sink
    obj[physical] = instance.cost[arc_tail[physical], arc_head[physical]]

    model = BranchCutModel(name, env=env)
    model.add_vars(
        n_arcs,
        lb=np.zeros(n_arcs, dtype=np.float64),
        ub=np.ones(n_arcs, dtype=np.float64),
        obj=obj,
        vtypes=np.full(n_arcs, _VTYPE_BINARY, dtype=np.uint8),
    )
    model.set_minimize()
    model.update()

    columns = np.arange(n_arcs, dtype=np.int32)
    for node in range(1, n):
        out_cols = columns[arc_tail == node]
        model.add_constr(out_cols, np.ones(out_cols.size), b"=", 1.0, f"out_{node + 1}")
        in_cols = columns[arc_head == node]
        model.add_constr(in_cols, np.ones(in_cols.size), b"=", 1.0, f"in_{node + 1}")
    source_cols = columns[arc_tail == 0]
    model.add_constr(
        source_cols, np.ones(source_cols.size), b"=", float(vehicles), "fleet_out"
    )
    terminal_cols = columns[arc_head == terminal]
    model.add_constr(
        terminal_cols, np.ones(terminal_cols.size), b"=", float(vehicles), "fleet_in"
    )
    model.update()

    fold = np.arange(n_model_nodes, dtype=np.int64)
    if open_routes:
        fold[sink] = 0
    arc_sep_tail = fold[arc_tail]
    arc_sep_head = fold[arc_head]

    pair_u, pair_v = (idx.astype(np.int64) for idx in np.triu_indices(n, k=1))
    pair_id = np.full((n, n), -1, dtype=np.int64)
    pair_id[pair_u, pair_v] = np.arange(pair_u.size, dtype=np.int64)
    pair_id[pair_v, pair_u] = np.arange(pair_u.size, dtype=np.int64)

    low = np.minimum(arc_sep_tail, arc_sep_head)
    high = np.maximum(arc_sep_tail, arc_sep_head)
    if np.any(low == high):
        raise AssertionError("an arc projects onto a single separator node")
    pair_of_arc = pair_id[low, high]
    pair_forward = np.full(pair_u.size, -1, dtype=np.int64)
    pair_reverse = np.full(pair_u.size, -1, dtype=np.int64)
    ascending = arc_sep_tail < arc_sep_head
    pair_forward[pair_of_arc[ascending]] = columns[ascending]
    pair_reverse[pair_of_arc[~ascending]] = columns[~ascending]
    if np.any(pair_forward < 0) or np.any(pair_reverse < 0):
        raise AssertionError("a separator pair is not covered by two arcs")

    return DirectedModel(
        model=model,
        instance=instance,
        vehicles=int(vehicles),
        open_routes=bool(open_routes),
        n_model_nodes=n_model_nodes,
        terminal=terminal,
        arc_tail=arc_tail,
        arc_head=arc_head,
        arc_sep_tail=arc_sep_tail,
        arc_sep_head=arc_sep_head,
        n_arc_vars=n_arcs,
        pair_u=pair_u,
        pair_v=pair_v,
        pair_forward=pair_forward,
        pair_reverse=pair_reverse,
    )


def add_rows_csr(
    model: BranchCutModel,
    rows: Sequence[Tuple[np.ndarray, np.ndarray, float]],
) -> int:
    n_rows = len(rows)
    if n_rows == 0:
        return 0
    sizes = np.fromiter((row[0].size for row in rows), dtype=np.int64, count=n_rows)
    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(sizes, out=indptr[1:])
    nnz = int(indptr[-1])
    cols = np.empty(nnz, dtype=np.int32)
    vals = np.empty(nnz, dtype=np.float64)
    rhs = np.empty(n_rows, dtype=np.float64)
    for row_idx, (row_cols, row_vals, bound) in enumerate(rows):
        cols[indptr[row_idx] : indptr[row_idx + 1]] = row_cols
        vals[indptr[row_idx] : indptr[row_idx + 1]] = row_vals
        rhs[row_idx] = bound
    model.add_constrs_csr(indptr, cols, vals, b">", rhs)
    return nnz


def canonical_subset(nodes_ext: Iterable[int], n_sep_nodes: int) -> List[int]:
    seen: set[int] = set()
    nodes: List[int] = []
    for raw in nodes_ext:
        node = int(raw)
        if node != raw:
            raise ValueError(f"separator proposed a non-integral node id {raw!r}")
        if node == 1:
            raise ValueError(
                "separator proposed a subset containing the depot; a cut set is a "
                "set of customers, and dropping the depot from it would silently "
                "change which inequality is added"
            )
        if not 2 <= node <= n_sep_nodes:
            raise ValueError(
                f"separator proposed node {node}, outside the customer range "
                f"2..{n_sep_nodes}"
            )
        if node in seen:
            raise ValueError(f"separator proposed node {node} twice in one subset")
        seen.add(node)
        nodes.append(node)
    if not nodes:
        raise ValueError("separator proposed an empty subset")
    nodes.sort()
    return nodes


def rcc_rhs(nodes_ext: Iterable[int], instance: DirectedInstance) -> float:
    demand = instance.demand
    total = 0
    for node in nodes_ext:
        total += int(demand[int(node) - 1])
    capacity = int(instance.capacity)
    return float(max(1, -(-total // capacity)))


def violation_from_dense(
    nodes_ext: Sequence[int],
    rhs: float,
    dense: np.ndarray,
    degree: np.ndarray,
) -> float:
    if len(nodes_ext) == 0:
        return 0.0
    nodes0 = np.asarray(nodes_ext, dtype=np.int64) - 1
    inside = float(dense[np.ix_(nodes0, nodes0)].sum())
    boundary = float(degree[nodes0].sum()) - inside
    return 2.0 * float(rhs) - boundary
