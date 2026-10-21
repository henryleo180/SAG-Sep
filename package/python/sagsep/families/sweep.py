from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sagsep.types import (
    FAMILY_SWEEP,
    CandidateBatch,
    CandidateContext,
)

__all__ = ["SweepConfig", "generate_sweep"]


@dataclass(frozen=True)
class SweepConfig:
    threshold: float = 0.5
    min_size: int = 2
    min_violation: float = 1e-6


def generate_sweep(ctx: CandidateContext, cfg: SweepConfig) -> CandidateBatch:
    probs = ctx.probs
    k_count = int(probs.shape[0])
    n = int(probs.shape[1])
    if k_count == 0 or n <= 1:
        return CandidateBatch.empty(max(n, 1))

    n_prefix = n - 1
    sweep_ok = n > int(cfg.min_size) and float(ctx.capacity) > 0.0

    if sweep_ok:
        probs64 = probs.astype(np.float64)
        orders = np.argsort(-probs64[:, 1:], axis=1, kind="stable") + 1

        adj_batch = ctx.lp.dense[orders[:, :, None], orders[:, None, :]]
        intra = np.tril(adj_batch, k=-1).sum(axis=2)
        boundaries = np.cumsum(
            ctx.lp.degree[orders], axis=1, dtype=np.float64
        ) - 2.0 * np.cumsum(intra, axis=1, dtype=np.float64)

        demand_cumsum = np.cumsum(ctx.demand.astype(np.float64)[orders], axis=1)
        violations = (
            2.0 * np.ceil(demand_cumsum / float(ctx.capacity)) - boundaries
        )
        violations[:, : min(int(cfg.min_size) - 1, n_prefix)] = -np.inf

        best = np.argmax(violations, axis=1)
        best_viol = violations[np.arange(k_count), best]
        winner = ~(best_viol <= float(cfg.min_violation))
    else:
        orders = None
        winner = np.zeros(k_count, dtype=bool)

    selections = np.zeros((k_count, n), dtype=bool)
    if orders is not None and bool(winner.any()):
        prefix = np.arange(n_prefix, dtype=np.int64)[None, :] <= best[:, None]
        prefix &= winner[:, None]
        rows = np.broadcast_to(
            np.arange(k_count, dtype=np.int64)[:, None], orders.shape
        )
        selections[rows, orders] = prefix

    fallback = ~winner
    if bool(fallback.any()):
        above = probs[:, 1:] > float(cfg.threshold)
        any_above = above.any(axis=1)
        fb_above = fallback & any_above
        if bool(fb_above.any()):
            selections[:, 1:][fb_above] = above[fb_above]
        fb_argmax = fallback & ~any_above
        if bool(fb_argmax.any()):
            argmax_col = np.argmax(probs[:, 1:], axis=1) + 1
            selections[fb_argmax, argmax_col[fb_argmax]] = True

    variants = tuple(
        np.where(winner, "winner", "threshold_fallback").tolist()
    )
    return CandidateBatch.from_selections(
        selections,
        family=FAMILY_SWEEP,
        group_ids=np.arange(k_count, dtype=np.int32),
        variants=variants,
        proposal_scores=None,
    )
