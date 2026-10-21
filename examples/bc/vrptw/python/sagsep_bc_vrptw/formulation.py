from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sagsep.types import Instance, mask_key_for_ext_nodes, n_mask_words

from sagsep_bc_common.gurobi_capi import BranchCutModel, GurobiEnv

from .solomon import PreprocessedArcs, SolomonInstance, check_customer_degrees

__all__ = [
    "VrptwModel",
    "build_vrptw_model",
    "aggregated_instance",
    "add_rows_csr",
    "canonical_customers",
    "cut_mask_key",
    "rcc_rhs",
    "violation_from_dense",
]

INTEGRALITY_TOLERANCE = 0.5
SUPPORT_THRESHOLD = 1e-8

_VTYPE_CONTINUOUS = ord("C")
_VTYPE_BINARY = ord("B")


def aggregated_instance(
    instance: SolomonInstance, vehicles: int, optimal: float | None
) -> Instance:
    return Instance(
        name=instance.name,
        n_nodes=instance.n_customers + 1,
        depot=0,
        coords=instance.coords,
        demand=instance.demand,
        capacity=int(instance.capacity),
        vehicles=int(vehicles),
        optimal=optimal,
    )


@dataclass
class VrptwModel:
    model: BranchCutModel
    instance: SolomonInstance
    arcs: PreprocessedArcs
    vehicles: int
    n_arc_vars: int
    n_time_vars: int

    y_u: np.ndarray
    y_v: np.ndarray
    y_arc_a: np.ndarray
    y_arc_b: np.ndarray
    y_index: np.ndarray

    _triu_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict, repr=False
    )

    @property
    def n_customers(self) -> int:
        return int(self.instance.n_customers)

    @property
    def n_agg_nodes(self) -> int:
        return int(self.instance.n_customers) + 1

    def relax_to_continuous(self) -> None:
        self.model.set_vtypes(
            0, np.full(self.n_arc_vars, _VTYPE_CONTINUOUS, dtype=np.uint8)
        )
        self.model.update()

    def restore_integer_types(self) -> None:
        self.model.set_vtypes(0, np.full(self.n_arc_vars, _VTYPE_BINARY, dtype=np.uint8))
        self.model.update()

    def aggregate(self, values: np.ndarray) -> np.ndarray:
        padded = np.empty(self.n_arc_vars + 1, dtype=np.float64)
        padded[: self.n_arc_vars] = values
        padded[self.n_arc_vars] = 0.0
        return padded[self.y_arc_a] + padded[self.y_arc_b]

    def support(self, y_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask = y_values > SUPPORT_THRESHOLD
        return (
            self.y_u[mask],
            self.y_v[mask],
            np.ascontiguousarray(y_values[mask], dtype=np.float64),
        )

    def dense_support(self, y_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = self.n_agg_nodes
        dense = np.zeros((n, n), dtype=np.float64)
        dense[self.y_u, self.y_v] = y_values
        dense[self.y_v, self.y_u] = y_values
        degree = np.zeros(n, dtype=np.float64)
        np.add.at(degree, self.y_u, y_values)
        np.add.at(degree, self.y_v, y_values)
        return dense, degree

    def _agg_inner(self, customers: np.ndarray) -> np.ndarray:
        size = int(customers.size)
        if size < 2:
            return np.empty(0, dtype=np.int64)
        pairs = self._triu_cache.get(size)
        if pairs is None:
            pairs = np.triu_indices(size, k=1)
            self._triu_cache[size] = pairs
        block = self.y_index[np.ix_(customers, customers)]
        edges = block[pairs[0], pairs[1]]
        return edges[edges >= 0]

    def _agg_depot(self, customers: np.ndarray) -> np.ndarray:
        edges = self.y_index[0, customers]
        return edges[edges >= 0]

    def _expand(
        self, blocks: Sequence[Tuple[np.ndarray, float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        cols: List[np.ndarray] = []
        coeffs: List[np.ndarray] = []
        for edges, coefficient in blocks:
            if edges.size == 0:
                continue
            arcs = np.stack((self.y_arc_a[edges], self.y_arc_b[edges]), axis=1).ravel()
            arcs = arcs[arcs >= 0]
            if arcs.size:
                cols.append(arcs.astype(np.int32, copy=False))
                coeffs.append(np.full(arcs.size, coefficient, dtype=np.float64))
        if not cols:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)
        return np.concatenate(cols), np.concatenate(coeffs)

    def cut_row(
        self, customers: Sequence[int], rhs: float
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        subset = np.asarray(customers, dtype=np.int64)
        n_customers = self.n_customers

        if subset.size > n_customers / 2:
            inside = np.zeros(n_customers + 1, dtype=bool)
            inside[subset] = True
            complement = np.flatnonzero(~inside)[1:]
            cols, coeffs = self._expand(
                (
                    (self._agg_inner(complement), 1.0),
                    (self._agg_depot(complement), 0.5),
                    (self._agg_depot(subset), -0.5),
                )
            )
            return cols, coeffs, float(complement.size) - float(rhs)

        cols, coeffs = self._expand(((self._agg_inner(subset), 1.0),))
        return cols, coeffs, float(subset.size) - float(rhs)

    def detect_capacity_violations(
        self, values: np.ndarray
    ) -> List[Tuple[List[int], float]]:
        arcs = self.arcs
        n = self.n_customers
        sink = n + 1
        used = np.flatnonzero(values > INTEGRALITY_TOLERANCE)
        tails = arcs.tail[used]
        heads = arcs.head[used]

        successor = np.full(n + 2, -1, dtype=np.int64)
        starts: List[int] = []
        for tail, head in zip(tails.tolist(), heads.tolist()):
            if tail == 0:
                starts.append(head)
            else:
                successor[tail] = head

        violations: List[Tuple[List[int], float]] = []
        visited = np.zeros(n + 2, dtype=bool)
        for first in starts:
            if visited[first]:
                continue
            route: List[int] = []
            node = int(first)
            while node != sink and node > 0:
                if visited[node]:
                    break
                visited[node] = True
                route.append(node)
                node = int(successor[node])
            if not route:
                continue
            required = rcc_rhs(route, self.instance)
            if required > 1.0:
                violations.append((sorted(route), required))

        unreached = [
            customer for customer in range(1, n + 1) if not visited[customer]
        ]
        if unreached:
            violations.extend(self._unreached_components(unreached, tails, heads))
        return violations

    def _unreached_components(
        self, unreached: Sequence[int], tails: np.ndarray, heads: np.ndarray
    ) -> List[Tuple[List[int], float]]:
        member = set(int(customer) for customer in unreached)
        adjacency: Dict[int, List[int]] = {customer: [] for customer in member}
        for tail, head in zip(tails.tolist(), heads.tolist()):
            if tail in member and head in member:
                adjacency[tail].append(head)
                adjacency[head].append(tail)
        seen: set[int] = set()
        components: List[Tuple[List[int], float]] = []
        for customer in unreached:
            if customer in seen:
                continue
            component: List[int] = []
            stack = [int(customer)]
            while stack:
                node = stack.pop()
                if node in seen:
                    continue
                seen.add(node)
                component.append(node)
                stack.extend(nb for nb in adjacency[node] if nb not in seen)
            component.sort()
            components.append((component, rcc_rhs(component, self.instance)))
        return components


def build_vrptw_model(
    env: GurobiEnv,
    instance: SolomonInstance,
    arcs: PreprocessedArcs,
    vehicles: int,
    *,
    name: str = "vrptw_two_index",
) -> VrptwModel:
    if vehicles <= 0:
        raise ValueError(f"vehicles must be positive, got {vehicles}")
    check_customer_degrees(arcs, instance.n_customers)

    n = instance.n_customers
    sink = n + 1
    n_nodes = arcs.n_model_nodes
    n_arcs = arcs.n_arcs

    model = BranchCutModel(name, env=env)
    model.add_vars(
        n_arcs,
        lb=np.zeros(n_arcs, dtype=np.float64),
        ub=np.ones(n_arcs, dtype=np.float64),
        obj=np.ascontiguousarray(arcs.cost, dtype=np.float64),
        vtypes=np.full(n_arcs, _VTYPE_BINARY, dtype=np.uint8),
    )
    model.add_vars(
        n_nodes,
        lb=np.ascontiguousarray(arcs.lower, dtype=np.float64),
        ub=np.ascontiguousarray(arcs.upper, dtype=np.float64),
        obj=np.zeros(n_nodes, dtype=np.float64),
        vtypes=np.full(n_nodes, _VTYPE_CONTINUOUS, dtype=np.uint8),
    )
    model.set_minimize()
    model.update()

    time_column = np.arange(n_arcs, n_arcs + n_nodes, dtype=np.int32)
    arc_index = np.full((n_nodes, n_nodes), -1, dtype=np.int32)
    arc_index[arcs.tail, arcs.head] = np.arange(n_arcs, dtype=np.int32)

    out_arcs: List[List[int]] = [[] for _ in range(n_nodes)]
    in_arcs: List[List[int]] = [[] for _ in range(n_nodes)]
    for column, (tail, head) in enumerate(zip(arcs.tail.tolist(), arcs.head.tolist())):
        out_arcs[tail].append(column)
        in_arcs[head].append(column)

    for customer in range(1, n + 1):
        for group, label in ((out_arcs[customer], "out"), (in_arcs[customer], "in")):
            indices = np.asarray(group, dtype=np.int32)
            model.add_constr(
                indices,
                np.ones(indices.size, dtype=np.float64),
                b"=",
                1.0,
                f"deg_{label}_{customer}",
            )
    for group, label in ((out_arcs[0], "source"), (in_arcs[sink], "sink")):
        indices = np.asarray(group, dtype=np.int32)
        model.add_constr(
            indices,
            np.ones(indices.size, dtype=np.float64),
            b"=",
            float(vehicles),
            f"fleet_{label}",
        )

    pair_rows: List[Tuple[np.ndarray, np.ndarray, float]] = []
    for i in range(1, n + 1):
        forward = arc_index[i, i + 1 : n + 1]
        backward = arc_index[i + 1 : n + 1, i]
        both = np.flatnonzero((forward >= 0) & (backward >= 0))
        for offset in both.tolist():
            pair_rows.append(
                (
                    np.asarray([forward[offset], backward[offset]], dtype=np.int32),
                    np.ones(2, dtype=np.float64),
                    1.0,
                )
            )
    add_rows_csr(model, pair_rows, b"<")

    service = arcs.service
    time_rows: List[Tuple[np.ndarray, np.ndarray, float]] = []
    for column in range(n_arcs):
        tail = int(arcs.tail[column])
        head = int(arcs.head[column])
        big_m = float(arcs.big_m[column])
        bound = big_m - service[tail] - float(arcs.time[column])
        if big_m > 0.0:
            time_rows.append(
                (
                    np.asarray(
                        [column, time_column[tail], time_column[head]], dtype=np.int32
                    ),
                    np.asarray([big_m, 1.0, -1.0], dtype=np.float64),
                    bound,
                )
            )
        else:
            time_rows.append(
                (
                    np.asarray([time_column[tail], time_column[head]], dtype=np.int32),
                    np.asarray([1.0, -1.0], dtype=np.float64),
                    bound,
                )
            )
    add_rows_csr(model, time_rows, b"<")
    model.update()

    y_u, y_v, y_arc_a, y_arc_b, y_index = _build_aggregation(arc_index, n, sink)

    return VrptwModel(
        model=model,
        instance=instance,
        arcs=arcs,
        vehicles=int(vehicles),
        n_arc_vars=n_arcs,
        n_time_vars=n_nodes,
        y_u=y_u,
        y_v=y_v,
        y_arc_a=y_arc_a,
        y_arc_b=y_arc_b,
        y_index=y_index,
    )


def _build_aggregation(
    arc_index: np.ndarray, n_customers: int, sink: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_agg = n_customers + 1
    y_u: List[int] = []
    y_v: List[int] = []
    y_arc_a: List[int] = []
    y_arc_b: List[int] = []
    y_index = np.full((n_agg, n_agg), -1, dtype=np.int32)
    for u in range(n_agg):
        for v in range(u + 1, n_agg):
            if u == 0:
                first = int(arc_index[0, v])
                second = int(arc_index[v, sink])
            else:
                first = int(arc_index[u, v])
                second = int(arc_index[v, u])
            if first < 0 and second < 0:
                continue
            y_index[u, v] = y_index[v, u] = len(y_u)
            y_u.append(u)
            y_v.append(v)
            y_arc_a.append(first)
            y_arc_b.append(second)
    return (
        np.asarray(y_u, dtype=np.int64),
        np.asarray(y_v, dtype=np.int64),
        np.asarray(y_arc_a, dtype=np.int64),
        np.asarray(y_arc_b, dtype=np.int64),
        y_index,
    )


def add_rows_csr(
    model: BranchCutModel,
    rows: Sequence[Tuple[np.ndarray, np.ndarray, float]],
    sense: bytes,
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
    model.add_constrs_csr(indptr, cols, vals, sense, rhs)
    return nnz


def cut_mask_key(customers: Sequence[int], n_agg_nodes: int) -> bytes:
    return mask_key_for_ext_nodes(
        [int(customer) + 1 for customer in customers],
        n_mask_words(int(n_agg_nodes)),
        strip_depot=True,
    )


def rcc_rhs(customers: Iterable[int], instance: SolomonInstance) -> float:
    total = 0
    for customer in customers:
        total += int(instance.demand[int(customer)])
    capacity = int(instance.capacity)
    required = (total + capacity - 1) // capacity
    return float(required if required > 1 else 1)


def canonical_customers(subset: Iterable[object], n_customers: int) -> List[int]:
    customers: List[int] = []
    seen = set()
    for raw in subset:
        value = int(raw)
        if value != raw:
            raise ValueError(f"separator returned a non-integral node id {raw!r}")
        if value == 1:
            raise ValueError(
                "separator returned a cut set containing the depot (external id 1); "
                "this driver's contract is customer sets only"
            )
        if not 2 <= value <= n_customers + 1:
            raise ValueError(
                f"separator returned external node id {value}, outside 2..{n_customers + 1}"
            )
        if value in seen:
            raise ValueError(f"separator returned a repeated node id {value}")
        seen.add(value)
        customers.append(value - 1)
    customers.sort()
    return customers


def violation_from_dense(
    customers: Sequence[int],
    rhs: float,
    dense: np.ndarray,
    degree: np.ndarray,
) -> float:
    if len(customers) == 0:
        return 0.0
    nodes = np.asarray(customers, dtype=np.int64)
    inside = float(dense[np.ix_(nodes, nodes)].sum())
    boundary = float(degree[nodes].sum()) - inside
    return 2.0 * float(rhs) - boundary
