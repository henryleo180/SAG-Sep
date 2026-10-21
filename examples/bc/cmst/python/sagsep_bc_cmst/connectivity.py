from __future__ import annotations

import math
from dataclasses import dataclass
from typing import AbstractSet, Dict, List, Sequence, TYPE_CHECKING

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from .instance import CmstInstance

__all__ = [
    "CONNECTIVITY_RESIDUAL_THRESHOLD",
    "CONNECTIVITY_SUPPORT_THRESHOLD",
    "ConnectivityCut",
    "separate_connectivity",
]

CONNECTIVITY_SUPPORT_THRESHOLD = 1e-9

CONNECTIVITY_RESIDUAL_THRESHOLD = 1e-12


@dataclass(frozen=True, slots=True)
class ConnectivityCut:
    nodes: List[int]
    rhs: int
    cut_value: float
    key: bytes

    @property
    def violation(self) -> float:
        return float(self.rhs) - self.cut_value


def _mask_key(nodes_internal: Sequence[int], n_nodes: int) -> bytes:
    words = [0] * ((int(n_nodes) + 63) // 64)
    for node in nodes_internal:
        words[int(node) >> 6] |= 1 << (int(node) & 63)
    return b"".join(word.to_bytes(8, byteorder="little", signed=False) for word in words)


def _capacity_rhs(nodes_internal: Sequence[int], instance: "CmstInstance") -> int:
    capacity = int(instance.capacity)
    if capacity <= 0:
        raise ValueError(f"capacity must be positive, got {capacity}")
    total = 0
    demand = instance.demand
    for node in nodes_internal:
        total += int(demand[int(node)])
    return max(1, -(-total // capacity))


def _reachable_from_root(
    graph: nx.DiGraph,
    flow_dict: Dict[int, Dict[int, float]],
    n_nodes: int,
) -> np.ndarray:
    reachable = np.zeros(n_nodes, dtype=bool)
    reachable[0] = True
    queue = [0]
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        outgoing = flow_dict[node]
        for raw_next in sorted(graph.successors(node)):
            next_node = int(raw_next)
            if reachable[next_node]:
                continue
            residual = (
                float(graph[node][next_node]["capacity"])
                - float(outgoing.get(next_node, 0.0))
                + float(flow_dict[next_node].get(node, 0.0))
            )
            if residual <= CONNECTIVITY_RESIDUAL_THRESHOLD:
                continue
            reachable[next_node] = True
            queue.append(next_node)
    return reachable


def separate_connectivity(
    values: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    instance: "CmstInstance",
    *,
    min_violation: float,
    blocked: AbstractSet[bytes] = frozenset(),
) -> List[ConnectivityCut]:
    tolerance = float(min_violation)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            f"min_violation must be a finite non-negative number, got {min_violation}"
        )

    values = np.asarray(values, dtype=np.float64)
    edge_i = np.asarray(edge_i)
    edge_j = np.asarray(edge_j)
    if values.ndim != 1 or edge_i.ndim != 1 or edge_j.ndim != 1:
        raise ValueError("values, edge_i and edge_j must be one-dimensional")
    if not (values.size == edge_i.size == edge_j.size):
        raise ValueError(
            "values, edge_i and edge_j must describe the same number of edges"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("the LP edge solution contains a non-finite value")

    n_nodes = int(instance.n_nodes)
    if n_nodes < 1:
        raise ValueError(f"instance must contain at least the root, got {n_nodes} nodes")
    if len(instance.demand) != n_nodes:
        raise ValueError(
            f"instance has {n_nodes} nodes but {len(instance.demand)} demand values"
        )
    if n_nodes == 1:
        return []

    cutoff = 1.0 - tolerance
    if cutoff <= 0.0:
        return []

    in_support = values > CONNECTIVITY_SUPPORT_THRESHOLD
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n_nodes))
    for raw_u, raw_v, raw_capacity, keep in zip(
        edge_i.tolist(), edge_j.tolist(), values.tolist(), in_support.tolist()
    ):
        u = int(raw_u)
        v = int(raw_v)
        if u != raw_u or v != raw_v:
            raise ValueError(f"edge endpoints must be integral, got ({raw_u}, {raw_v})")
        if not (0 <= u < v < n_nodes):
            raise ValueError(
                f"edge ({u}, {v}) is outside the canonical node range 0..{n_nodes - 1}"
            )
        if not keep:
            continue
        capacity = float(raw_capacity)
        graph.add_edge(u, v, capacity=capacity)
        graph.add_edge(v, u, capacity=capacity)

    covered = np.zeros(n_nodes, dtype=bool)
    inside = np.zeros(n_nodes, dtype=bool)
    seen = set(blocked)
    cuts: List[ConnectivityCut] = []

    for target in range(1, n_nodes):
        if covered[target]:
            continue

        _value, flow_dict = nx.maximum_flow(graph, 0, target, capacity="capacity")
        reachable = _reachable_from_root(graph, flow_dict, n_nodes)
        if not reachable[0] or reachable[target]:
            raise RuntimeError(
                "the residual graph of a maximum flow reaches customer "
                f"{target} from the root"
            )

        nodes_internal = [node for node in range(1, n_nodes) if not reachable[node]]
        if not nodes_internal:
            raise RuntimeError("minimum-cut target shore is unexpectedly empty")

        inside[:] = False
        inside[nodes_internal] = True
        crossing = in_support & (inside[edge_i] != inside[edge_j])
        cut_value = float(values[crossing].sum())
        if not cut_value < cutoff:
            continue

        key = _mask_key(nodes_internal, n_nodes)

        if key in seen:
            for node in nodes_internal:
                covered[node] = True
            continue

        nodes_ext = [node + 1 for node in nodes_internal]
        cuts.append(
            ConnectivityCut(
                nodes=nodes_ext,
                rhs=_capacity_rhs(nodes_internal, instance),
                cut_value=cut_value,
                key=key,
            )
        )
        seen.add(key)
        for node in nodes_internal:
            covered[node] = True

    return cuts
