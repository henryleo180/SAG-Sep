from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sagsep.types import FAMILY_ONION, CandidateBatch, CandidateContext

__all__ = ["OnionConfig", "generate_onion"]

_ONION_MODES = frozenset({"mean"})
_ONION_CUTPOINTS = frozenset({"band_first"})


@dataclass(frozen=True)
class OnionConfig:
    mode: str = "mean"
    min_violation: float = 1e-4
    cutpoint: str = "band_first"

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        if mode not in _ONION_MODES:
            raise ValueError(
                f"Unknown/unsupported mbp_onion_mode '{self.mode}'. "
                f"Valid in SAG-Sep: {sorted(_ONION_MODES)}"
            )
        object.__setattr__(self, "mode", mode)
        cutpoint = str(self.cutpoint).strip().lower()
        if cutpoint not in _ONION_CUTPOINTS:
            raise ValueError(
                f"Unknown/unsupported mbp_onion_cutpoint '{self.cutpoint}'. "
                f"Valid in SAG-Sep: {sorted(_ONION_CUTPOINTS)}"
            )
        object.__setattr__(self, "cutpoint", cutpoint)


def generate_onion(ctx: CandidateContext, cfg: OnionConfig) -> CandidateBatch:
    n = int(ctx.n_nodes)
    empty = CandidateBatch.empty(n)
    if ctx.probs.shape[0] == 0 or float(ctx.capacity) <= 0.0:
        return empty
    if n <= 1:
        return empty

    priority = ctx.probs[:, 1:].mean(axis=0)

    cust = np.arange(1, n, dtype=np.int64)
    order = np.argsort(-priority, kind="stable")
    ranked = cust[order]

    demand_f32 = ctx.demand.astype(np.float32)
    d_ranked = demand_f32[ranked].astype(np.float64, copy=False)
    cumsum = np.cumsum(d_ranked)
    Q = float(ctx.capacity)
    ceils = np.ceil(cumsum / max(Q, 1e-9)).astype(np.int64)

    m_cap = int(ceils[-1])
    if m_cap <= 0:
        return empty

    band_starts = np.flatnonzero(np.concatenate(([True], ceils[1:] != ceils[:-1])))
    band_vals = ceils[band_starts]
    valid = (band_vals > 0) & (band_vals <= m_cap)
    cand_idx = band_starts[valid]
    if cand_idx.size == 0:
        return empty

    memb = np.arange(ranked.size, dtype=np.int64)[None, :] <= cand_idx[:, None]
    sel_bool = np.zeros((cand_idx.size, n), dtype=bool)
    sel_bool[:, ranked] = memb
    sel_mat = sel_bool.astype(np.float32)

    adj_view = ctx.lp.dense
    deg = ctx.lp.degree
    sum_deg_S = sel_mat @ deg
    intra = ((sel_mat @ adj_view) * sel_mat).sum(axis=1)
    lhs = sum_deg_S - intra
    d_S = sel_mat @ demand_f32
    rhs = 2.0 * np.ceil(d_S / max(Q, 1e-9)).astype(np.float32, copy=False)
    viol = rhs - lhs
    keep = viol > float(cfg.min_violation)
    if not keep.any():
        return empty
    keep_rows = np.flatnonzero(keep)

    bands = ceils[cand_idx[keep_rows]]

    return CandidateBatch.from_selections(
        sel_bool[keep_rows],
        family=FAMILY_ONION,
        group_ids=bands.astype(np.int32, copy=False),
        variants=(cfg.cutpoint,) * int(keep_rows.size),
        proposal_scores=None,
    )
