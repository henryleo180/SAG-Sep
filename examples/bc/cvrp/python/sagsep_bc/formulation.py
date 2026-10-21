from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sagsep import features
from sagsep.types import Instance

from sagsep_bc_common.gurobi_capi import BranchCutModel, GurobiEnv

__all__ = [
    "TwoIndexModel",
    "build_two_index_model",
    "build_cut_row",
    "add_rows_csr",
    "canonical_customers",
    "rcc_rhs",
    "violation_from_dense",
    "detect_subtours",
    "detect_capacity_violations",
]

INTEGRALITY_TOLERANCE = 0.5

_VTYPE_CONTINUOUS = ord("C")
_VTYPE_INTEGER = ord("I")
_VTYPE_BINARY = ord("B")


@dataclass
class TwoIndexModel:
    model: BranchCutModel
    instance: Instance
    vehicles: int
    edge_i: np.ndarray
    edge_j: np.ndarray
    var_index: np.ndarray
    n_edge_vars: int
    _triu_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict, repr=False
    )

    @property
    def n_nodes(self) -> int:
        return int(self.instance.n_nodes)

    @property
    def n_customers(self) -> int:
        return int(self.instance.n_nodes) - 1

    def integer_vtypes(self) -> np.ndarray:
        vtypes = np.full(self.n_edge_vars, _VTYPE_BINARY, dtype=np.uint8)
        vtypes[self.edge_i == 0] = _VTYPE_INTEGER
        return vtypes

    def relax_to_continuous(self) -> None:
        self.model.set_vtypes(0, np.full(self.n_edge_vars, _VTYPE_CONTINUOUS, dtype=np.uint8))
        self.model.update()

    def restore_integer_types(self) -> None:
        self.model.set_vtypes(0, self.integer_vtypes())
        self.model.update()

    def support(self, values: np.ndarray, threshold: float = 1e-8) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray
    ]:
        mask = values > threshold
        return (
            self.edge_i[mask],
            self.edge_j[mask],
            np.ascontiguousarray(values[mask], dtype=np.float64),
        )

    def dense_support(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = self.n_nodes
        dense = np.zeros((n, n), dtype=np.float64)
        dense[self.edge_i, self.edge_j] = values
        dense[self.edge_j, self.edge_i] = values
        degree = np.zeros(n, dtype=np.float64)
        np.add.at(degree, self.edge_i, values)
        np.add.at(degree, self.edge_j, values)
        return dense, degree

    def cut_row(self, nodes_ext: Sequence[int], rhs: float) -> Tuple[
        np.ndarray, np.ndarray, float
    ]:
        return build_cut_row(
            nodes_ext, rhs, self.var_index, self.n_customers, self._triu_cache
        )


def build_two_index_model(
    env: GurobiEnv,
    instance: Instance,
    vehicles: int,
    *,
    name: str = "cvrp_two_index",
) -> TwoIndexModel:
    n = int(instance.n_nodes)
    n_customers = n - 1
    if not 1 <= vehicles <= n_customers:
        raise ValueError(
            f"vehicles must be between 1 and {n_customers}, got {vehicles}"
        )
    weight = features.distance_matrix(instance)

    edge_i, edge_j = (idx.astype(np.int64) for idx in np.triu_indices(n, k=1))
    n_edges = int(edge_i.size)

    is_depot_edge = edge_i == 0
    obj = np.ascontiguousarray(weight[edge_i, edge_j], dtype=np.float64)
    lb = np.zeros(n_edges, dtype=np.float64)
    ub = np.where(is_depot_edge, 2.0, 1.0)
    vtypes = np.where(is_depot_edge, _VTYPE_INTEGER, _VTYPE_BINARY).astype(np.uint8)

    model = BranchCutModel(name, env=env)
    model.add_vars(n_edges, lb=lb, ub=ub, obj=obj, vtypes=vtypes)
    model.set_minimize()
    model.update()

    var_index = np.full((n, n), -1, dtype=np.int32)
    columns = np.arange(n_edges, dtype=np.int32)
    var_index[edge_i, edge_j] = columns
    var_index[edge_j, edge_i] = columns

    ones = np.ones(n, dtype=np.float64)
    for node in range(1, n):
        others = np.concatenate(
            (np.arange(0, node, dtype=np.int64), np.arange(node + 1, n, dtype=np.int64))
        )
        indices = np.ascontiguousarray(var_index[node, others], dtype=np.int32)
        model.add_constr(indices, ones[: indices.size], b"=", 2.0, f"deg_{node + 1}")
    depot_indices = np.ascontiguousarray(var_index[0, 1:n], dtype=np.int32)
    model.add_constr(
        depot_indices, ones[: depot_indices.size], b"=", 2.0 * float(vehicles), "deg_depot"
    )
    model.update()

    return TwoIndexModel(
        model=model,
        instance=instance,
        vehicles=int(vehicles),
        edge_i=edge_i,
        edge_j=edge_j,
        var_index=var_index,
        n_edge_vars=n_edges,
    )


def build_cut_row(
    nodes_ext: Sequence[int],
    rhs: float,
    var_index: np.ndarray,
    n_customers: int,
    triu_cache: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, float]:
    nodes0 = np.asarray(nodes_ext, dtype=np.int64) - 1
    n = var_index.shape[0]

    def inner(subset0: np.ndarray) -> np.ndarray:
        size = int(subset0.size)
        if size < 2:
            return np.empty(0, dtype=np.int32)
        pairs = triu_cache.get(size)
        if pairs is None:
            pairs = np.triu_indices(size, k=1)
            triu_cache[size] = pairs
        block = var_index[np.ix_(subset0, subset0)]
        cols = block[pairs[0], pairs[1]]
        return cols[cols >= 0].astype(np.int32, copy=False)

    if nodes0.size > n_customers / 2:
        in_subset = np.zeros(n, dtype=bool)
        in_subset[nodes0] = True
        complement = np.flatnonzero(~in_subset)[1:]
        if complement.size == 0:
            cols = inner(nodes0)
            return cols, np.ones(cols.size, dtype=np.float64), float(nodes0.size) - float(rhs)
        col_blocks = [inner(complement)]
        coeff_blocks = [np.ones(col_blocks[0].size, dtype=np.float64)]
        depot_comp = var_index[0, complement]
        depot_comp = depot_comp[depot_comp >= 0].astype(np.int32, copy=False)
        if depot_comp.size:
            col_blocks.append(depot_comp)
            coeff_blocks.append(np.full(depot_comp.size, 0.5, dtype=np.float64))
        depot_sub = var_index[0, nodes0]
        depot_sub = depot_sub[depot_sub >= 0].astype(np.int32, copy=False)
        if depot_sub.size:
            col_blocks.append(depot_sub)
            coeff_blocks.append(np.full(depot_sub.size, -0.5, dtype=np.float64))
        return (
            np.concatenate(col_blocks),
            np.concatenate(coeff_blocks),
            float(complement.size) - float(rhs),
        )

    cols = inner(nodes0)
    return cols, np.ones(cols.size, dtype=np.float64), float(nodes0.size) - float(rhs)


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
    model.add_constrs_csr(indptr, cols, vals, b"<", rhs)
    return nnz


def canonical_customers(
    nodes_ext: Iterable[int], n_customers: int
) -> List[int]:
    upper = int(n_customers) + 1
    customers: List[int] = []
    seen: set[int] = set()
    for raw_node in nodes_ext:
        try:
            node = operator.index(raw_node)
        except TypeError as exc:
            raise ValueError(
                f"cut node ids must be integral, got {raw_node!r}"
            ) from exc
        if node == 1:
            raise ValueError("cut subset must not contain the depot (external id 1)")
        if node < 2 or node > upper:
            raise ValueError(
                f"cut node id must be between 2 and {upper}, got {node}"
            )
        if node in seen:
            raise ValueError(f"cut subset contains duplicate node id {node}")
        seen.add(node)
        customers.append(node)
    customers.sort()
    return customers


def rcc_rhs(nodes_ext: Iterable[int], instance: Instance) -> float:
    total_demand = sum(
        int(instance.demand[node - 1])
        for node in nodes_ext
    )
    capacity = int(instance.capacity)
    return float((total_demand + capacity - 1) // capacity)


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


def _adjacency(
    values: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    n_nodes: int,
) -> List[List[int]]:
    used = np.flatnonzero(values > INTEGRALITY_TOLERANCE)
    adjacency: List[List[int]] = [[] for _ in range(n_nodes)]
    for u, v in zip(edge_i[used].tolist(), edge_j[used].tolist()):
        adjacency[u].append(v)
        adjacency[v].append(u)
    return adjacency


def detect_subtours(
    values: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    n_nodes: int,
) -> List[List[int]]:
    adjacency = _adjacency(values, edge_i, edge_j, n_nodes)
    visited = np.zeros(n_nodes, dtype=bool)
    subtours: List[List[int]] = []
    for start in range(n_nodes):
        if visited[start]:
            continue
        component: List[int] = []
        stack = [start]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            component.append(node)
            stack.extend(nb for nb in adjacency[node] if not visited[nb])
        if 0 not in component:
            subtours.append(sorted(node + 1 for node in component))
    return subtours


def _trace_routes(
    values: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    n_nodes: int,
) -> List[List[int]]:
    adjacency = _adjacency(values, edge_i, edge_j, n_nodes)
    visited = np.zeros(n_nodes, dtype=bool)
    routes: List[List[int]] = []
    for first in adjacency[0]:
        if visited[first]:
            continue
        route: List[int] = []
        previous, current = 0, first
        while current != 0:
            if visited[current]:
                break
            visited[current] = True
            route.append(current + 1)
            following = next((nb for nb in adjacency[current] if nb != previous), None)
            if following is None:
                break
            previous, current = current, following
        if route:
            routes.append(route)
    return routes


def detect_capacity_violations(
    values: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    instance: Instance,
) -> List[Tuple[List[int], float]]:
    violations: List[Tuple[List[int], float]] = []
    for route in _trace_routes(values, edge_i, edge_j, int(instance.n_nodes)):
        required = rcc_rhs(route, instance)
        if required > 1.0:
            violations.append((sorted(route), required))
    return violations
