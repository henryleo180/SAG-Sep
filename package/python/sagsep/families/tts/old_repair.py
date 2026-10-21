from __future__ import annotations

import math
from dataclasses import replace
from typing import Dict, List, Optional, Sequence

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
from sagsep.families.tts.score import local_search_1flip, score_subsets

__all__ = ["old_repair_proposals"]


def _source_stats(diagnostics: Dict[str, object], source: str) -> Dict[str, object]:
    sources = diagnostics.setdefault("sources", {})
    assert isinstance(sources, dict)
    return sources.setdefault(
        source,
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


def _update_best(stats: Dict[str, object], key: str, value: float) -> None:
    previous = stats.get(key)
    if previous is None or float(value) > float(previous):
        stats[key] = float(value)


def _flat_old_repair(
    old_rows: Sequence[np.ndarray],
    *,
    adj_view: np.ndarray,
    deg: np.ndarray,
    demand_view: np.ndarray,
    capacity: float,
    config: TTSConfig,
    diagnostics: Optional[Dict[str, object]],
) -> List[Proposal]:
    n = int(adj_view.shape[0])
    n_cust = n - 1
    max_size = float(config.max_size_frac) * float(n_cust)

    rows = [np.asarray(row) for row in old_rows]
    rows = [row for row in rows if row.shape == (n,)]
    if rows:
        pool_all = (np.stack(rows, axis=0) > 0.5).astype(np.float32)
    else:
        pool_all = np.zeros((0, n), dtype=np.float32)
    if pool_all.shape[0]:
        pool_all[:, 0] = 0.0
        masks_all = pack_selection_rows(pool_all)
        sizes_all = popcount(masks_all).astype(np.float64)
        size_ok = (sizes_all >= float(config.min_size)) & (sizes_all <= max_size)
        pool_ok = pool_all[size_ok]
        masks_ok = masks_all[size_ok]
        keep = first_seen_unique_rows(masks_ok)
        old_pool = pool_ok[keep]
        old_masks = masks_ok[keep]
    else:
        old_pool = pool_all
        old_masks = np.zeros((0, 0), dtype=np.uint64)

    summary: Optional[Dict[str, object]] = None
    stats: Optional[Dict[str, object]] = None
    if diagnostics is not None:
        summary = diagnostics.setdefault(
            "old_repair",
            {
                "bank_size": len(old_rows),
                "eligible_count": int(old_pool.shape[0]),
                "top_b_count": 0,
                "still_violated_count": 0,
                "repair_attempt_count": 0,
                "repair_success_count": 0,
                "candidate_count": 0,
                "emitted_count": 0,
            },
        )
        stats = _source_stats(diagnostics, "old_repair")
        stats["raw_pool_count"] = int(stats["raw_pool_count"]) + int(old_pool.shape[0])
    if old_pool.shape[0] == 0 or int(config.old_repair_top_b) <= 0:
        return []

    violations = score_subsets(old_pool, adj_view, deg, demand_view, capacity)[0]
    top_count = min(int(config.old_repair_top_b), old_pool.shape[0])
    top_indices = np.argsort(-violations, kind="stable")[:top_count]
    if summary is not None:
        summary["top_b_count"] = top_count
    if stats is not None:
        _update_best(stats, "raw_best_violation", float(np.max(violations)))

    proposals: List[Proposal] = []
    out_keys: set[bytes] = set()
    for index in top_indices.tolist():
        seed = old_pool[index]
        seed_violation = float(violations[index])
        if seed_violation > float(config.min_violation):
            if summary is not None:
                summary["still_violated_count"] = int(summary["still_violated_count"]) + 1
            candidate = seed.copy()
            candidate_violation = seed_violation
        else:
            if summary is not None:
                summary["repair_attempt_count"] = int(summary["repair_attempt_count"]) + 1
            ls_diag: Dict[str, object] = {}
            candidate, candidate_violation = local_search_1flip(
                seed,
                adj_view,
                deg,
                demand_view,
                capacity,
                max_flips=config.max_flips,
                min_gain=config.min_gain,
                max_size_frac=config.max_size_frac,
                swap_max_pairs=config.ls_swap_max_pairs,
                min_keep=config.min_size,
                diagnostics=ls_diag if diagnostics is not None else None,
            )
            if stats is not None:
                _update_best(stats, "ls_best_violation", candidate_violation)
                _update_best(stats, "ls_best_gain", float(ls_diag.get("gain", 0.0)))
                stats["ls_max_steps"] = max(
                    int(stats["ls_max_steps"]), int(ls_diag.get("steps", 0))
                )
                stats["ls_hit_max_flips"] = bool(stats["ls_hit_max_flips"]) or bool(
                    ls_diag.get("hit_max_flips", False)
                )
            if candidate_violation <= float(config.min_violation):
                continue
            if summary is not None:
                summary["repair_success_count"] = int(summary["repair_success_count"]) + 1

        mask = pack_selection_rows(candidate[None, :])[0]
        key = row_key(mask)
        if key in out_keys:
            continue
        out_keys.add(key)
        score = candidate_violation / math.sqrt(max(float(candidate.sum()), 1.0))
        proposals.append(
            Proposal(candidate, mask, key, "old_repair", candidate_violation, score)
        )
    if stats is not None:
        stats["dedup_count"] = int(stats["dedup_count"]) + len(proposals)
    if summary is not None:
        summary["candidate_count"] = len(proposals)
    return proposals


def old_repair_proposals(
    bank_dense: np.ndarray,
    *,
    adj_view: np.ndarray,
    deg: np.ndarray,
    demand_view: np.ndarray,
    capacity: float,
    config: TTSConfig,
    compression: CompressionState,
    diagnostics: Optional[Dict[str, object]] = None,
) -> List[Proposal]:
    n = int(adj_view.shape[0])
    if int(compression.member.shape[0]) >= n:
        return _flat_old_repair(
            list(bank_dense),
            adj_view=adj_view,
            deg=deg,
            demand_view=demand_view,
            capacity=capacity,
            config=config,
            diagnostics=diagnostics,
        )

    n_super = int(compression.member.shape[0])
    if bank_dense.shape[0]:
        projected = (bank_dense @ compression.member.T > 0.0).astype(np.float32)
        projected[:, 0] = 0.0
        projected_bank = list(projected)
    else:
        projected_bank = []

    adj_super = compression.aggregate.copy()
    np.fill_diagonal(adj_super, 0.0)
    deg_super = adj_super.sum(axis=1).astype(np.float32, copy=False)
    super_proposals = _flat_old_repair(
        projected_bank,
        adj_view=adj_super,
        deg=deg_super,
        demand_view=compression.demand,
        capacity=capacity,
        config=replace(config, min_size=1, max_size_frac=1.0),
        diagnostics=diagnostics,
    )

    max_size = float(config.max_size_frac) * float(n - 1)
    proposals: List[Proposal] = []
    out_keys: set[bytes] = set()
    for proposal in super_proposals:
        full = (proposal.selection @ compression.member).astype(np.float32, copy=False)
        full[0] = 0.0
        size = float(full.sum())
        mask = pack_selection_rows(full[None, :])[0]
        key = row_key(mask)
        if size < float(config.min_size) or size > max_size or key in out_keys:
            continue
        violation = float(
            score_subsets(full[None, :], adj_view, deg, demand_view, capacity)[0][0]
        )
        if violation <= float(config.min_violation):
            continue
        out_keys.add(key)
        score = violation / math.sqrt(max(size, 1.0))
        proposals.append(Proposal(full.copy(), mask, key, "old_repair", violation, score))
    if diagnostics is not None:
        summary = diagnostics.get("old_repair")
        if isinstance(summary, dict):
            summary["compressed_supernodes"] = n_super - 1
            summary["candidate_count"] = len(proposals)
    return proposals
