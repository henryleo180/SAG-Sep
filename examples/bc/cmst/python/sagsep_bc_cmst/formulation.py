from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, NamedTuple, Sequence, Tuple

import numpy as np
from sagsep_bc_common.gurobi_capi import BranchCutModel, GurobiEnv

from .instance import CmstInstance, min_root_degree

__all__ = [
    "TreeModel",
    "build_tree_model",
    "LazyRow",
    "canonical_subset",
    "build_inner_row",
    "build_cutset_row",
    "add_rows_csr",
    "capacity_rhs",
    "inner_violation_from_dense",
    "cutset_violation_from_dense",
    "detect_violations",
]

INTEGRALITY_TOLERANCE = 0.5

_VTYPE_CONTINUOUS = ord("C")
_VTYPE_BINARY = ord("B")


class LazyRow(NamedTuple):
    nodes: List[int]
    rhs: int
    inner: bool


@dataclass
class TreeModel:
    model: BranchCutModel
    instance: CmstInstance
    root_degree: int
    edge_i: np.ndarray
    edge_j: np.ndarray
    var_index: np.ndarray
    n_edge_vars: int
    fixed_pairs: int = 0
    _triu_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict, repr=False
    )

    @property
    def n_nodes(self) -> int:
        return int(self.instance.n_nodes)

    @property
    def n_customers(self) -> int:
        return int(self.instance.n_nodes) - 1

    def relax_to_continuous(self) -> None:
        self.model.set_vtypes(
            0, np.full(self.n_edge_vars, _VTYPE_CONTINUOUS, dtype=np.uint8)
        )
        self.model.update()

    def restore_integer_types(self) -> None:
        self.model.set_vtypes(0, np.full(self.n_edge_vars, _VTYPE_BINARY, dtype=np.uint8))
        self.model.update()

    def dense_support(self, values: np.ndarray) -> np.ndarray:
        n = self.n_nodes
        dense = np.zeros((n, n), dtype=np.float64)
        dense[self.edge_i, self.edge_j] = values
        dense[self.edge_j, self.edge_i] = values
        return dense

    def inner_row(self, nodes_ext: Sequence[int], rhs: int) -> Tuple[
        np.ndarray, np.ndarray, float
    ]:
        return build_inner_row(nodes_ext, rhs, self.var_index, self._triu_cache)

    def cutset_row(self, nodes_ext: Sequence[int], rhs: int) -> Tuple[
        np.ndarray, np.ndarray, float
    ]:
        return build_cutset_row(nodes_ext, rhs, self.var_index)


def build_tree_model(
    env: GurobiEnv,
    instance: CmstInstance,
    *,
    name: str = "cmst_tree",
) -> TreeModel:
    n = int(instance.n_nodes)
    capacity = int(instance.capacity)
    demand = np.asarray(instance.demand, dtype=np.int64)

    edge_i, edge_j = (idx.astype(np.int64) for idx in np.triu_indices(n, k=1))
    n_edges = int(edge_i.size)

    obj = np.ascontiguousarray(instance.weight[edge_i, edge_j], dtype=np.float64)
    lb = np.zeros(n_edges, dtype=np.float64)
    incompatible = (edge_i > 0) & (demand[edge_i] > capacity - demand[edge_j])
    ub = np.where(incompatible, 0.0, 1.0)
    vtypes = np.full(n_edges, _VTYPE_BINARY, dtype=np.uint8)

    model = BranchCutModel(name, env=env)
    model.add_vars(n_edges, lb=lb, ub=ub, obj=obj, vtypes=vtypes)
    model.set_minimize()
    model.update()

    var_index = np.full((n, n), -1, dtype=np.int32)
    columns = np.arange(n_edges, dtype=np.int32)
    var_index[edge_i, edge_j] = columns
    var_index[edge_j, edge_i] = columns

    ones = np.ones(n_edges, dtype=np.float64)
    model.add_constr(columns, ones, b"=", float(n - 1), "edge_count")
    root_degree = min_root_degree(demand, capacity)
    root_indices = np.ascontiguousarray(var_index[0, 1:n], dtype=np.int32)
    model.add_constr(
        root_indices, ones[: root_indices.size], b">", float(root_degree), "root_degree"
    )
    for node in range(1, n):
        others = np.concatenate(
            (np.arange(0, node, dtype=np.int64), np.arange(node + 1, n, dtype=np.int64))
        )
        indices = np.ascontiguousarray(var_index[node, others], dtype=np.int32)
        model.add_constr(indices, ones[: indices.size], b">", 1.0, f"deg_{node + 1}")
    model.update()

    return TreeModel(
        model=model,
        instance=instance,
        root_degree=root_degree,
        edge_i=edge_i,
        edge_j=edge_j,
        var_index=var_index,
        n_edge_vars=n_edges,
        fixed_pairs=int(incompatible.sum()),
    )


def build_inner_row(
    nodes_ext: Sequence[int],
    rhs: int,
    var_index: np.ndarray,
    triu_cache: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, float]:
    nodes0 = np.asarray(nodes_ext, dtype=np.int64) - 1
    size = int(nodes0.size)
    bound = float(size) - float(rhs)
    if size < 2:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64), bound
    pairs = triu_cache.get(size)
    if pairs is None:
        pairs = np.triu_indices(size, k=1)
        triu_cache[size] = pairs
    block = var_index[np.ix_(nodes0, nodes0)]
    cols = block[pairs[0], pairs[1]].astype(np.int32, copy=False)
    return cols, np.ones(cols.size, dtype=np.float64), bound


def build_cutset_row(
    nodes_ext: Sequence[int],
    rhs: int,
    var_index: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    nodes0 = np.asarray(nodes_ext, dtype=np.int64) - 1
    n = var_index.shape[0]
    inside = np.zeros(n, dtype=bool)
    inside[nodes0] = True
    outside = np.flatnonzero(~inside)
    cols = var_index[np.ix_(nodes0, outside)].reshape(-1).astype(np.int32, copy=False)
    return cols, np.ones(cols.size, dtype=np.float64), float(rhs)


def add_rows_csr(
    model: BranchCutModel,
    rows: Sequence[Tuple[np.ndarray, np.ndarray, float, bytes]],
) -> int:
    n_rows = len(rows)
    if n_rows == 0:
        return 0
    nnz_total = 0
    start = 0
    while start < n_rows:
        sense = rows[start][3]
        stop = start + 1
        while stop < n_rows and rows[stop][3] == sense:
            stop += 1
        run = rows[start:stop]
        sizes = np.fromiter(
            (row[0].size for row in run), dtype=np.int64, count=len(run)
        )
        indptr = np.zeros(len(run) + 1, dtype=np.int64)
        np.cumsum(sizes, out=indptr[1:])
        nnz = int(indptr[-1])
        cols = np.empty(nnz, dtype=np.int32)
        vals = np.empty(nnz, dtype=np.float64)
        rhs = np.empty(len(run), dtype=np.float64)
        for row_idx, (row_cols, row_vals, bound, _sense) in enumerate(run):
            cols[indptr[row_idx] : indptr[row_idx + 1]] = row_cols
            vals[indptr[row_idx] : indptr[row_idx + 1]] = row_vals
            rhs[row_idx] = bound
        model.add_constrs_csr(indptr, cols, vals, sense, rhs)
        nnz_total += nnz
        start = stop
    return nnz_total


def canonical_subset(subset: Iterable[object], n_nodes: int) -> List[int]:
    nodes: List[int] = []
    for raw in subset:
        node = int(raw)
        if node != raw:
            raise ValueError(f"separator proposed a non-integral node id {raw!r}")
        if node == 1:
            raise ValueError(
                "separator proposed a set containing the root (external id 1); "
                "capacity inequalities are indexed by root-free sets only"
            )
        if not 2 <= node <= int(n_nodes):
            raise ValueError(
                f"separator proposed node id {node}, outside 2..{int(n_nodes)}"
            )
        nodes.append(node)
    if not nodes:
        raise ValueError("separator proposed an empty set")
    nodes.sort()
    for index in range(1, len(nodes)):
        if nodes[index] == nodes[index - 1]:
            raise ValueError(f"separator proposed node id {nodes[index]} twice")
    return nodes


def capacity_rhs(nodes_ext: Iterable[int], instance: CmstInstance) -> int:
    total = 0
    demand = instance.demand
    for node in nodes_ext:
        total += int(demand[int(node) - 1])
    return -(-total // int(instance.capacity))


def inner_violation_from_dense(
    nodes_ext: Sequence[int],
    rhs: int,
    dense: np.ndarray,
) -> float:
    if len(nodes_ext) == 0:
        return 0.0
    nodes0 = np.asarray(nodes_ext, dtype=np.int64) - 1
    inside = float(dense[np.ix_(nodes0, nodes0)].sum())
    return 0.5 * inside - float(len(nodes_ext)) + float(rhs)


def cutset_violation_from_dense(
    nodes_ext: Sequence[int],
    rhs: int,
    dense: np.ndarray,
) -> float:
    if len(nodes_ext) == 0:
        return 0.0
    nodes0 = np.asarray(nodes_ext, dtype=np.int64) - 1
    inside = np.zeros(int(dense.shape[0]), dtype=bool)
    inside[nodes0] = True
    outside = np.flatnonzero(~inside)
    crossing = float(dense[np.ix_(nodes0, outside)].sum())
    return float(rhs) - crossing


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


def _component(adjacency: Sequence[Sequence[int]], start: int, visited: np.ndarray) -> List[int]:
    component: List[int] = []
    stack = [start]
    while stack:
        node = stack.pop()
        if visited[node]:
            continue
        visited[node] = True
        component.append(node)
        stack.extend(nb for nb in adjacency[node] if not visited[nb])
    component.sort()
    return component


def detect_violations(
    values: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    instance: CmstInstance,
) -> List[LazyRow]:
    n_nodes = int(instance.n_nodes)
    adjacency = _adjacency(values, edge_i, edge_j, n_nodes)

    visited = np.zeros(n_nodes, dtype=bool)
    rows: List[LazyRow] = []
    for start in range(n_nodes):
        if visited[start]:
            continue
        component = _component(adjacency, start, visited)
        if component[0] == 0:
            continue
        nodes_ext = [node + 1 for node in component]
        rows.append(LazyRow(nodes_ext, capacity_rhs(nodes_ext, instance), inner=False))
    if rows:
        return rows

    visited = np.zeros(n_nodes, dtype=bool)
    visited[0] = True
    for first in sorted(adjacency[0]):
        if visited[first]:
            continue
        branch = _component(adjacency, first, visited)
        nodes_ext = [node + 1 for node in branch]
        rhs = capacity_rhs(nodes_ext, instance)
        if rhs > 1:
            rows.append(LazyRow(nodes_ext, rhs, inner=True))
    return rows
