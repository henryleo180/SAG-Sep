from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

__all__ = ["CompressionState", "build_compression_state"]


@dataclass(frozen=True, slots=True)
class CompressionState:
    labels: np.ndarray
    frozen: np.ndarray
    member: np.ndarray
    aggregate: np.ndarray
    demand: np.ndarray
    sizes: np.ndarray
    internal_x: np.ndarray
    one_edge_customer_supernodes: int


def _normalise_labels(parent: np.ndarray) -> np.ndarray:
    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    roots = np.asarray([find(node) for node in range(parent.size)], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int64, copy=False)


def _one_edge_partition(adj_view: np.ndarray, eps: float) -> np.ndarray:
    adj = np.asarray(adj_view)
    n = int(adj.shape[0])
    parent = np.arange(n, dtype=np.int64)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    for tail, head in zip(*np.nonzero(adj >= (1.0 - float(eps)))):
        tail_i = int(tail)
        head_i = int(head)
        if tail_i == 0 or head_i == 0 or tail_i >= head_i:
            continue
        root_tail = find(tail_i)
        root_head = find(head_i)
        if root_tail == root_head:
            continue
        parent[root_head] = root_tail
    return _normalise_labels(parent)


def _partition_data(
    adj_view: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    adj = np.asarray(adj_view, dtype=np.float64)
    n_super = int(labels.max()) + 1 if labels.size else 0
    member = np.zeros((n_super, labels.size), dtype=np.float64)
    if labels.size:
        member[labels, np.arange(labels.size)] = 1.0
    aggregate_index = labels[:, None] * n_super + labels[None, :]
    aggregate = np.bincount(
        aggregate_index.ravel(),
        weights=adj.ravel(),
        minlength=n_super * n_super,
    ).reshape(n_super, n_super)
    sizes = np.bincount(labels, minlength=n_super).astype(np.int64, copy=False)
    internal = 0.5 * np.diag(aggregate).copy()
    return member, aggregate, sizes, internal


def _compress_partition(
    adj_view: np.ndarray,
    demand_view: np.ndarray,
    capacity: float,
    old_sets: Sequence[np.ndarray] = (),
    eps: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    adj = np.asarray(adj_view, dtype=np.float64).copy()
    np.fill_diagonal(adj, 0.0)
    demand = np.asarray(demand_view, dtype=np.float64)
    n = int(adj.shape[0])
    parent = np.arange(n, dtype=np.int64)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    def merge(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        if root_left > root_right:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left

    initial = _one_edge_partition(adj, eps)
    initial_representatives: Dict[int, int] = {}
    for node in range(1, n):
        label = int(initial[node])
        representative = initial_representatives.setdefault(label, node)
        merge(representative, node)

    tight_one_vehicle_sets = []
    if capacity > 0.0:
        for old_selection in old_sets:
            selection = np.asarray(old_selection)
            if selection.shape != (n,):
                continue
            customers = np.flatnonzero(selection > 0.5)
            customers = customers[customers != 0]
            if customers.size == 0:
                continue
            demand_sum = float(demand[customers].sum())
            if math.ceil(demand_sum / float(capacity)) != 1:
                continue
            old_adj = adj[np.ix_(customers, customers)]
            internal_sum = float(np.triu(old_adj, k=1).sum())
            slack = float(customers.size) - internal_sum - 1.0
            if slack <= 0.01:
                tight_one_vehicle_sets.append(customers)

    final_frozen = np.ones(1, dtype=bool)
    while True:
        labels = _normalise_labels(parent)
        member, aggregate, sizes, internal = _partition_data(adj, labels)
        n_super = int(member.shape[0])
        frozen = internal >= (sizes.astype(np.float64) - 1.0 + 0.01)
        if frozen.size:
            frozen[0] = True
        final_frozen = frozen
        representatives = np.asarray(
            [int(np.flatnonzero(labels == label)[0]) for label in range(n_super)],
            dtype=np.int64,
        )

        shrinkable = ~frozen
        if shrinkable.size:
            shrinkable[0] = False
        pair_mask = (
            (aggregate >= 0.999)
            & shrinkable[:, None]
            & shrinkable[None, :]
        )
        pair_links = np.argwhere(np.triu(pair_mask, k=1))
        if pair_links.size:
            for left, right in pair_links.tolist():
                merge(int(representatives[left]), int(representatives[right]))
            continue

        triple = None
        upper_values = aggregate[np.triu_indices(n_super, k=1)]
        top_count = min(3, int(upper_values.size))
        triple_upper_bound = (
            float(np.partition(upper_values, -top_count)[-top_count:].sum())
            if top_count == 3
            else 0.0
        )
        if triple_upper_bound >= 1.999:
            positive = (
                (aggregate > 0.01)
                & shrinkable[:, None]
                & shrinkable[None, :]
            )
            for first in range(1, n_super - 2):
                if frozen[first]:
                    continue
                for second in range(first + 1, n_super - 1):
                    if not positive[first, second]:
                        continue
                    third_mask = (
                        positive[first, second + 1 :]
                        & positive[second, second + 1 :]
                    )
                    third_offsets = np.flatnonzero(third_mask)
                    if third_offsets.size == 0:
                        continue
                    third_nodes = third_offsets + second + 1
                    totals = (
                        aggregate[first, second]
                        + aggregate[first, third_nodes]
                        + aggregate[second, third_nodes]
                    )
                    feasible_thirds = third_nodes[totals >= 1.999]
                    if feasible_thirds.size:
                        triple = (first, second, int(feasible_thirds[0]))
                        break
                if triple is not None:
                    break
        if triple is not None:
            first, second, third = triple
            merge(int(representatives[first]), int(representatives[second]))
            merge(int(representatives[first]), int(representatives[third]))
            continue

        old_components = None
        for customers in tight_one_vehicle_sets:
            components = np.unique(labels[customers])
            components = components[components != 0]
            if components.size <= 1:
                continue
            if np.any(frozen[components]):
                continue
            old_components = components
            break
        if old_components is not None:
            first = int(old_components[0])
            for component in old_components[1:]:
                merge(
                    int(representatives[first]),
                    int(representatives[int(component)]),
                )
            continue
        break

    return labels, final_frozen.astype(bool, copy=False)


def build_compression_state(
    adj_view: np.ndarray,
    demand_view: np.ndarray,
    capacity: float,
    old_sets: Sequence[np.ndarray] = (),
    eps: float = 0.01,
) -> CompressionState:
    one_edge_labels = _one_edge_partition(adj_view, eps)
    labels, frozen = _compress_partition(
        adj_view,
        demand_view,
        capacity,
        old_sets=old_sets,
        eps=eps,
    )
    member, aggregate, sizes, internal = _partition_data(adj_view, labels)
    member = member.astype(np.float32, copy=False)
    aggregate = aggregate.astype(np.float32, copy=False)
    demand = (member @ np.asarray(demand_view, dtype=np.float32)).astype(
        np.float32, copy=False
    )
    for value in (labels, frozen, member, aggregate, demand, sizes, internal):
        value.setflags(write=False)
    return CompressionState(
        labels=labels,
        frozen=frozen,
        member=member,
        aggregate=aggregate,
        demand=demand,
        sizes=sizes,
        internal_x=internal,
        one_edge_customer_supernodes=int(np.unique(one_edge_labels[1:]).size),
    )
