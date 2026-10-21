from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from sagsep.families.tts.compress import CompressionState
from sagsep.families.tts.config import TTSConfig
from sagsep.families.tts.masks import (
    first_seen_unique_rows,
    pack_selection_rows,
    popcount,
    row_key,
)
from sagsep.families.tts.proposal import Proposal
from sagsep.families.tts.score import score_subsets

__all__ = [
    "select_adddrop_seeds_from_bank",
    "native_adddrop_subsets",
    "adddrop_proposals",
]


def select_adddrop_seeds_from_bank(
    bank_dense: np.ndarray,
    *,
    adj_view: np.ndarray,
    deg: np.ndarray,
    demand_view: np.ndarray,
    capacity: float,
) -> List[np.ndarray]:
    n = int(adj_view.shape[0])
    if bank_dense.shape[0] == 0:
        return []
    canonical = (bank_dense > 0.5).astype(np.float32)
    canonical[:, 0] = 0.0
    masks = pack_selection_rows(canonical)
    nonempty = popcount(masks) > 0
    canonical = canonical[nonempty]
    masks = masks[nonempty]
    if canonical.shape[0] == 0:
        return []
    keep = first_seen_unique_rows(masks)
    pool = canonical[keep]
    violations = score_subsets(pool, adj_view, deg, demand_view, capacity)[0]
    order = np.argsort(-violations, kind="stable")
    return [pool[index] for index in order.tolist()]


def native_adddrop_subsets(
    adj_super: np.ndarray,
    demand_super: np.ndarray,
    capacity: float,
    member: np.ndarray,
    seed_sets: List[np.ndarray] | Tuple[np.ndarray, ...],
    *,
    min_violation: float = 1e-4,
    diagnostics: Optional[Dict[str, object]] = None,
) -> np.ndarray:
    adj = np.asarray(adj_super, dtype=np.float64).copy()
    np.fill_diagonal(adj, 0.0)
    demand = np.asarray(demand_super, dtype=np.float64)
    membership = np.asarray(member)
    n_super = int(adj.shape[0])
    empty = np.zeros((0, n_super), dtype=np.float32)
    summary = {
        "bank_size": len(seed_sets),
        "seed_count": len(seed_sets),
        "mapped_count": 0,
        "remove_moves": 0,
        "add_moves": 0,
        "exchange_moves": 0,
        "candidate_count": 0,
        "best_violation": None,
        "oversized_supernode": False,
    }

    def finish(result: np.ndarray) -> np.ndarray:
        if diagnostics is not None:
            diagnostics.update(summary)
        return result

    if (
        n_super <= 1
        or capacity <= 0.0
        or len(seed_sets) == 0
        or demand.shape != (n_super,)
        or membership.shape[0] != n_super
    ):
        return finish(empty)
    if np.any(demand[1:] > float(capacity)):
        summary["oversized_supernode"] = True
        return finish(empty)

    n_original = int(membership.shape[1])
    rows = [np.asarray(s) for s in seed_sets]
    rows = [r for r in rows if r.shape == (n_original,)]
    if not rows:
        return finish(empty)
    seeds_mat = (np.stack(rows, axis=0) > 0.5).astype(np.float64)
    proj = (seeds_mat @ membership.T.astype(np.float64)) > 0.0
    proj[:, 0] = False
    nonempty = proj[:, 1:].any(axis=1)
    proj = proj[nonempty]
    if proj.shape[0] == 0:
        summary["mapped_count"] = 0
        return finish(empty)
    keep = first_seen_unique_rows(pack_selection_rows(proj.astype(np.float32)))
    projected = [proj[index] for index in keep.tolist()]
    summary["mapped_count"] = len(projected)

    Q = float(capacity)
    outputs: List[np.ndarray] = []
    output_keys: set[bytes] = set()
    best_violation: Optional[float] = None
    for initial in projected:
        in_set = initial.copy()
        demand_sum = float(demand[in_set].sum())
        vehicles = max(1, int(math.ceil(demand_sum / Q)))
        capacity_sum = float(vehicles) * Q
        remaining_slack = demand_sum - float(vehicles - 1) * Q - 1.0
        x_to_set = adj[:, in_set].sum(axis=1)

        while True:
            removable = np.flatnonzero(
                in_set
                & (demand <= remaining_slack)
                & (x_to_set <= 0.99)
            )
            removable = removable[removable != 0]
            if removable.size:
                node = int(removable[int(np.argmin(x_to_set[removable]))])
                in_set[node] = False
                demand_sum -= float(demand[node])
                remaining_slack -= float(demand[node])
                x_to_set -= adj[:, node]
                summary["remove_moves"] = int(summary["remove_moves"]) + 1
                continue

            outside = np.flatnonzero(~in_set)
            outside = outside[outside != 0]
            if outside.size == 0:
                break
            crosses_boundary = (
                demand_sum + demand[outside] > capacity_sum
            )
            add_scores = x_to_set[outside] + crosses_boundary.astype(np.float64)
            best_offset = int(np.argmax(add_scores))
            new_node = int(outside[best_offset])
            best_score = float(add_scores[best_offset])

            if best_score >= 1.01:
                in_set[new_node] = True
                demand_sum += float(demand[new_node])
                vehicles = max(1, int(math.ceil(demand_sum / Q)))
                capacity_sum = float(vehicles) * Q
                remaining_slack = demand_sum - float(vehicles - 1) * Q - 1.0
                x_to_set += adj[:, new_node]
                summary["add_moves"] = int(summary["add_moves"]) + 1
                continue

            if best_score < 0.01 or bool(crosses_boundary[best_offset]):
                break
            inside = np.flatnonzero(
                in_set
                & ((demand - demand[new_node]) <= remaining_slack)
            )
            inside = inside[inside != 0]
            if inside.size == 0:
                break
            exchange_losses = x_to_set[inside] + adj[new_node, inside]
            remove_offset = int(np.argmin(exchange_losses))
            old_node = int(inside[remove_offset])
            if best_score < float(exchange_losses[remove_offset]) + 0.01:
                break

            in_set[new_node] = True
            x_to_set += adj[:, new_node]
            demand_sum += float(demand[new_node])
            in_set[old_node] = False
            x_to_set -= adj[:, old_node]
            demand_sum -= float(demand[old_node])
            vehicles = max(1, int(math.ceil(demand_sum / Q)))
            capacity_sum = float(vehicles) * Q
            remaining_slack = demand_sum - float(vehicles - 1) * Q - 1.0
            summary["exchange_moves"] = int(summary["exchange_moves"]) + 1

        boundary = float(adj[np.ix_(in_set, ~in_set)].sum())
        violation = 2.0 * float(math.ceil(demand_sum / Q)) - boundary
        best_violation = (
            violation if best_violation is None else max(best_violation, violation)
        )
        if violation <= float(min_violation):
            continue
        candidate = in_set.astype(np.float32, copy=False)
        key = candidate.tobytes()
        if key in output_keys:
            continue
        output_keys.add(key)
        outputs.append(candidate.copy())

    summary["candidate_count"] = len(outputs)
    summary["best_violation"] = best_violation
    if not outputs:
        return finish(empty)
    return finish(np.stack(outputs, axis=0))


def adddrop_proposals(
    seed_sets: List[np.ndarray],
    *,
    adj_view: np.ndarray,
    deg: np.ndarray,
    demand_view: np.ndarray,
    capacity: float,
    config: TTSConfig,
    seen_keys: set[bytes] | frozenset[bytes],
    compression: CompressionState,
    diagnostics: Optional[Dict[str, object]] = None,
) -> List[Proposal]:
    native_summary: Dict[str, object] = {}
    super_subsets = native_adddrop_subsets(
        compression.aggregate,
        compression.demand,
        capacity,
        compression.member,
        seed_sets,
        min_violation=config.min_violation,
        diagnostics=native_summary,
    )
    native_summary["seed_count"] = len(seed_sets)
    native_summary["seed_source_counts"] = {}
    native_summary["seed_from_old_bank"] = len(seed_sets)
    if diagnostics is not None:
        native_summary["compressed_supernodes"] = int(compression.member.shape[0] - 1)
        native_summary["exact_candidate_count"] = 0
        native_summary["emitted_count"] = 0
        diagnostics["adddrop"] = native_summary
    if super_subsets.shape[0] == 0:
        return []

    full_subsets = (super_subsets @ compression.member).astype(np.float32, copy=False)
    full_subsets[:, 0] = 0.0
    violations = score_subsets(full_subsets, adj_view, deg, demand_view, capacity)[0]
    sizes = full_subsets.sum(axis=1)
    masks = pack_selection_rows(full_subsets)
    max_size = float(config.max_size_frac) * float(adj_view.shape[0] - 1)
    proposals: List[Proposal] = []
    out_keys: set[bytes] = set()
    best_violation: Optional[float] = None
    for i in range(full_subsets.shape[0]):
        violation = float(violations[i])
        size = float(sizes[i])
        best_violation = (
            violation if best_violation is None else max(best_violation, violation)
        )
        if (
            violation <= float(config.min_violation)
            or size < float(config.min_size)
            or size > max_size
        ):
            continue
        key = row_key(masks[i])
        if key in seen_keys or key in out_keys:
            continue
        out_keys.add(key)
        score = violation / math.sqrt(max(size, 1.0))
        proposals.append(
            Proposal(
                full_subsets[i].copy(), masks[i].copy(), key, "adddrop", violation, score
            )
        )

    if diagnostics is not None:
        native_summary["exact_candidate_count"] = len(proposals)
        sources = diagnostics.setdefault("sources", {})
        assert isinstance(sources, dict)
        stats = sources.setdefault(
            "adddrop",
            {
                "raw_pool_count": 0,
                "dedup_count": 0,
                "raw_best_violation": None,
                "ls_best_violation": None,
                "ls_best_gain": None,
                "ls_max_steps": 0,
                "ls_hit_max_flips": False,
                "emitted_count": 0,
            },
        )
        stats["raw_pool_count"] = int(stats["raw_pool_count"]) + int(super_subsets.shape[0])
        stats["dedup_count"] = int(stats["dedup_count"]) + len(proposals)
        if best_violation is not None:
            previous = stats.get("raw_best_violation")
            if previous is None or float(best_violation) > float(previous):
                stats["raw_best_violation"] = float(best_violation)
    return proposals
