from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

from sagsep.families.tts.compress import CompressionState

__all__ = [
    "minimum_cut_sink_mask",
    "fcapfix_lite_subsets",
    "super_scores",
]


# ---------------------------------------------------------------------------
# Node scoring -> supernode scoring
# ---------------------------------------------------------------------------


def super_scores(
    compression: CompressionState, node_scores: np.ndarray, n_nodes: int
) -> np.ndarray:
    member = compression.member
    counts = member.sum(axis=1).clip(min=1.0)
    return (member @ np.asarray(node_scores[:n_nodes], dtype=np.float64)) / counts




def minimum_cut_sink_mask(
    capacity_matrix: np.ndarray,
    source: int,
    sink: int,
    *,
    force_source: Sequence[int],
    force_sink: Sequence[int],
    infinite_capacity: int,
) -> np.ndarray:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_flow

    capacities = np.asarray(capacity_matrix, dtype=np.int64).copy()
    inf_cap = int(infinite_capacity)
    for node in force_source:
        capacities[source, int(node)] = inf_cap
    for node in force_sink:
        capacities[int(node), sink] = inf_cap
    result = maximum_flow(csr_matrix(capacities), int(source), int(sink))
    residual = capacities - result.flow.toarray()
    reachable = np.zeros(capacities.shape[0], dtype=bool)
    reachable[source] = True
    stack = [int(source)]
    while stack:
        node = stack.pop()
        neighbors = np.flatnonzero((residual[node] > 0) & ~reachable)
        reachable[neighbors] = True
        stack.extend(neighbors.tolist())
    return ~reachable


def fcapfix_lite_subsets(
    adj_view: np.ndarray,
    capacity: float,
    *,
    rounds: int,
    seed_limit: int,
    seed_scores: np.ndarray,
    compression: CompressionState,
) -> np.ndarray:
    n = int(adj_view.shape[0])
    if n <= 2 or rounds <= 0 or capacity <= 0.0:
        return np.zeros((0, n), dtype=np.float32)
    member = compression.member
    adj_super = compression.aggregate
    demand_super = compression.demand
    n_super = int(member.shape[0])
    customer_adj = adj_super[1:, 1:]
    customer_demand = demand_super[1:]
    m = int(customer_demand.shape[0])
    if m <= 1:
        return np.zeros((0, n), dtype=np.float32)

    virtual_capacity = float(capacity)
    flow_scale = 1
    while virtual_capacity < 1000.0:
        virtual_capacity *= 10.0
        flow_scale *= 10
    source, sink = m, m + 1
    network = np.zeros((m + 2, m + 2), dtype=np.int64)
    support_u, support_v = np.nonzero(customer_adj > 0.0)
    upper = support_u < support_v
    edge_u = support_u[upper]
    edge_v = support_v[upper]
    if edge_u.size:
        arc_capacity = np.floor(
            customer_adj[edge_u, edge_v].astype(np.float64) * virtual_capacity + 1.0
        ).astype(np.int64)
        network[edge_u, edge_v] = arc_capacity
        network[edge_v, edge_u] = arc_capacity
    depot_edges = adj_super[0, 1:]
    unary = (
        depot_edges.astype(np.float64) * virtual_capacity
        + 1.0
        - 2.0 * float(flow_scale) * customer_demand.astype(np.float64)
    )
    pos_nodes = np.flatnonzero(unary > 0.0)
    neg_nodes = np.flatnonzero(~(unary > 0.0))
    network[source, pos_nodes] = np.floor(unary[pos_nodes]).astype(np.int64)
    network[neg_nodes, sink] = (-np.floor(unary[neg_nodes])).astype(np.int64)
    inf_cap = int(network[source].sum() + 2.0 * virtual_capacity)

    score_super = super_scores(compression, seed_scores, n)
    seed_order = np.argsort(-score_super[1:], kind="stable")
    if 0 < int(seed_limit) < m:
        seed_order = seed_order[: int(seed_limit)]
    history: List[Tuple[Tuple[int, ...], int]] = []
    use_seed = np.ones(m, dtype=bool)
    singleton = np.zeros(m, dtype=bool)
    out: List[np.ndarray] = []
    out_keys: set[bytes] = set()
    for _ in range(int(rounds)):
        for seed_value in seed_order:
            seed = int(seed_value)
            if not use_seed[seed]:
                continue
            fixed_source: set[int] = set()
            for prior_set, prior_seed in reversed(history):
                if len(prior_set) == 1 or seed not in prior_set:
                    continue
                if fixed_source.intersection(prior_set):
                    continue
                cover = prior_seed
                if cover == seed:
                    cover = next(node for node in prior_set if node != seed)
                fixed_source.add(int(cover))

            fixed_sink = [seed]
            if singleton[seed]:
                neighbors = np.flatnonzero(customer_adj[seed] > 0.0)
                neighbors = np.asarray(
                    [node for node in neighbors.tolist() if node not in fixed_source],
                    dtype=np.int64,
                )
                if neighbors.size == 0:
                    use_seed[seed] = False
                    continue
                values = customer_adj[seed, neighbors]
                fixed_sink.append(int(neighbors[int(np.argmax(values))]))

            sink_side = minimum_cut_sink_mask(
                network,
                source,
                sink,
                force_source=sorted(fixed_source),
                force_sink=fixed_sink,
                infinite_capacity=inf_cap,
            )[:m]
            nodes = np.flatnonzero(sink_side).tolist()
            if not nodes:
                continue
            node_set = set(nodes)
            demand_sum = float(customer_demand[nodes].sum())
            capacity_sum = float(capacity) * max(1, int(math.ceil(demand_sum / capacity)))
            outside = [node for node in range(m) if node not in node_set]
            max_outside_demand = max(
                (float(customer_demand[node]) for node in outside), default=0.0
            )
            if demand_sum + max_outside_demand > capacity_sum:
                while True:
                    connectivity = customer_adj[:, nodes].sum(axis=1)
                    candidates = [
                        node
                        for node in range(m)
                        if node not in node_set
                        and node not in fixed_source
                        and connectivity[node] > 0.0
                        and demand_sum + float(customer_demand[node]) > capacity_sum
                    ]
                    if not candidates:
                        break
                    values = connectivity[candidates]
                    add_node = int(candidates[int(np.argmax(values))])
                    nodes.append(add_node)
                    node_set.add(add_node)
                    demand_sum += float(customer_demand[add_node])
                    capacity_sum = float(capacity) * max(
                        1, int(math.ceil(demand_sum / capacity))
                    )

            if len(nodes) == 1:
                singleton[seed] = True
            ordered_nodes = tuple(sorted(nodes))
            history.append((ordered_nodes, seed))
            super_selection = np.zeros(n_super, dtype=np.float32)
            super_selection[np.asarray(nodes, dtype=np.int64) + 1] = 1.0
            full = (super_selection @ member).astype(np.float32, copy=False)
            full[0] = 0.0
            key = full.tobytes()
            if key not in out_keys:
                out_keys.add(key)
                out.append(full.copy())
    return np.stack(out, axis=0) if out else np.zeros((0, n), dtype=np.float32)
