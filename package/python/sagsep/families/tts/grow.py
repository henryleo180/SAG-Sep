from __future__ import annotations

import math
from typing import Dict

import numpy as np

__all__ = ["native_grow_subsets"]


# ---------------------------------------------------------------------------
# Grow state
# ---------------------------------------------------------------------------


class _GrowState:
    __slots__ = (
        "in_set",
        "selection",
        "order",
        "set_mask",
        "connectivity",
        "demand_sum",
        "original_size",
        "x_in_set",
        "vehicles",
        "source_outputs",
        "previous_size",
        "previous_vehicles",
    )


# ---------------------------------------------------------------------------
# Single-step state advance
# ---------------------------------------------------------------------------


def _apply_move(
    st: _GrowState,
    chosen: int,
    chosen_violation: float,
    *,
    adj: np.ndarray,
    demand: np.ndarray,
    sizes: np.ndarray,
    internal: np.ndarray,
    Q: float,
    min_violation: float,
) -> None:
    st.in_set[chosen] = True
    st.selection[chosen] = 1.0
    st.order.append(chosen)
    st.set_mask |= 1 << chosen
    st.demand_sum += float(demand[chosen])
    st.original_size += int(sizes[chosen])
    st.x_in_set += float(st.connectivity[chosen]) + float(internal[chosen])
    st.vehicles = int(math.ceil(st.demand_sum / Q))
    st.connectivity += adj[chosen]

    if chosen_violation > float(min_violation):
        if (
            int(sizes[chosen]) == 1
            and len(st.order) == st.previous_size + 1
            and st.vehicles == st.previous_vehicles + 1
            and st.source_outputs
        ):
            st.source_outputs.pop()
        st.source_outputs.append(st.selection.copy())
        st.previous_size = len(st.order)
        st.previous_vehicles = st.vehicles


# ---------------------------------------------------------------------------
# Greedy grow main loop
# ---------------------------------------------------------------------------


def _grow_walk(
    st: _GrowState,
    *,
    adj: np.ndarray,
    demand: np.ndarray,
    sizes: np.ndarray,
    internal: np.ndarray,
    Q: float,
    min_violation: float,
    max_prefix: int,
    anti_extensions: Dict[int, set],
) -> None:
    while True:
        if 0 < int(max_prefix) <= len(st.order):
            break
        candidates = np.flatnonzero((~st.in_set) & (st.connectivity > 0.0))
        if candidates.size == 0:
            break
        blocked_nodes = anti_extensions.get(st.set_mask)
        if blocked_nodes:
            allowed = np.fromiter(
                (
                    int(candidate) not in blocked_nodes
                    for candidate in candidates
                ),
                dtype=bool,
                count=int(candidates.size),
            )
            candidates = candidates[allowed]
            if candidates.size == 0:
                break
        proposed_demand = st.demand_sum + demand[candidates]
        proposed_vehicles = np.ceil(proposed_demand / Q).astype(
            np.int64, copy=False
        )
        proposed_size = st.original_size + sizes[candidates]
        proposed_x = (
            st.x_in_set + st.connectivity[candidates] + internal[candidates]
        )
        violation_arr = 2.0 * (
            proposed_x - proposed_size + proposed_vehicles
        )
        preferred = (
            (sizes[candidates] == 1)
            & (violation_arr > float(min_violation))
        )
        eligible = np.flatnonzero(preferred)
        if eligible.size == 0:
            eligible = np.arange(candidates.size)
        local_index = int(eligible[int(np.argmax(violation_arr[eligible]))])
        chosen = int(candidates[local_index])
        chosen_violation = float(violation_arr[local_index])

        _apply_move(
            st,
            chosen,
            chosen_violation,
            adj=adj,
            demand=demand,
            sizes=sizes,
            internal=internal,
            Q=Q,
            min_violation=min_violation,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def native_grow_subsets(
    adj_cust: np.ndarray,
    demand_cust: np.ndarray,
    capacity: float,
    supernode_sizes: np.ndarray,
    internal_x: np.ndarray,
    n_seeds: int,
    rng: np.random.Generator,
    *,
    min_violation: float = 1e-4,
    max_prefix: int = 0,
) -> np.ndarray:
    adj = np.asarray(adj_cust, dtype=np.float64)
    demand = np.asarray(demand_cust, dtype=np.float64)
    sizes = np.asarray(supernode_sizes, dtype=np.int64)
    internal = np.asarray(internal_x, dtype=np.float64)
    n_cust = int(adj.shape[0])
    if n_cust == 0 or n_seeds <= 0 or capacity <= 0.0:
        return np.zeros((0, n_cust), dtype=np.float32)
    if n_seeds >= n_cust:
        seeds = np.arange(n_cust, dtype=np.int64)
    else:
        seeds = rng.choice(n_cust, size=n_seeds, replace=False)

    Q = float(capacity)

    anti_extensions: Dict[int, set[int]] = {}
    blocked_sources: set[int] = set()
    outputs = []
    output_keys: set[bytes] = set()
    for seed_value in seeds:
        seed = int(seed_value)
        if seed in blocked_sources:
            continue
        st = _GrowState()
        st.in_set = np.zeros(n_cust, dtype=bool)
        st.in_set[seed] = True
        st.selection = np.zeros(n_cust, dtype=np.float32)
        st.selection[seed] = 1.0
        st.order = [seed]
        st.set_mask = 1 << seed
        st.connectivity = adj[seed].copy()
        st.demand_sum = float(demand[seed])
        st.original_size = int(sizes[seed])
        st.x_in_set = float(internal[seed])
        st.vehicles = int(math.ceil(st.demand_sum / Q))
        st.source_outputs = []
        st.previous_size = -1
        st.previous_vehicles = -1
        seed_violation = 2.0 * (st.x_in_set - st.original_size + st.vehicles)
        if seed_violation > float(min_violation):
            st.source_outputs.append(st.selection.copy())
        else:
            _grow_walk(
                st,
                adj=adj,
                demand=demand,
                sizes=sizes,
                internal=internal,
                Q=Q,
                min_violation=min_violation,
                max_prefix=max_prefix,
                anti_extensions=anti_extensions,
            )

            # ---- anti-set / blocked_sources write-back ----
            if len(st.order) == 1:
                blocked_sources.add(st.order[0])
            else:
                prefix_mask = 1 << st.order[0]
                for next_node in st.order[1:]:
                    anti_extensions.setdefault(prefix_mask, set()).add(
                        next_node
                    )
                    prefix_mask |= 1 << next_node

        for output in st.source_outputs:
            key = output.tobytes()
            if key in output_keys:
                continue
            output_keys.add(key)
            outputs.append(output)

    if not outputs:
        return np.zeros((0, n_cust), dtype=np.float32)
    return np.stack(outputs, axis=0)
