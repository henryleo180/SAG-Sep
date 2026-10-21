from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from sagsep.families.tts.adddrop import (
    adddrop_proposals,
    select_adddrop_seeds_from_bank,
)
from sagsep.families.tts.compress import CompressionState, build_compression_state
from sagsep.families.tts.config import TTSConfig
from sagsep.families.tts.flow import FlowState, apply_cut_flow, flow_scale
from sagsep.families.tts.fcapfix import fcapfix_lite_subsets
from sagsep.families.tts.grow import native_grow_subsets
from sagsep.families.tts.masks import (
    first_seen_unique_rows,
    pack_selection_rows,
    row_key,
)
from sagsep.families.tts.old_repair import (
    _source_stats,
    old_repair_proposals,
)
from sagsep.families.tts.proposal import Proposal
from sagsep.families.tts.score import local_search_1flip, score_subsets
from sagsep.types import (
    FAMILY_TTS,
    CandidateBatch,
    CandidateContext,
)

__all__ = ["generate_tts"]


def _bank_dense(ctx: CandidateContext) -> np.ndarray:
    bank = ctx.bank
    b = bank.size
    dense = np.zeros((b, ctx.n_nodes), dtype=np.float32)
    if b:
        rows = np.repeat(np.arange(b), np.diff(bank.indptr))
        dense[rows, bank.indices] = 1.0
    return dense


def _finalize_proposals(
    proposals: List[Proposal],
    *,
    config: TTSConfig,
    flow_state: FlowState,
    diagnostics: Optional[Dict[str, object]],
) -> List[Proposal]:
    flow_diagnostics: Optional[Dict[str, object]] = (
        {} if diagnostics is not None else None
    )
    selected_proposals = apply_cut_flow(
        proposals,
        config=config,
        flow_state=flow_state,
        diagnostics=flow_diagnostics,
    )
    if diagnostics is not None:
        diagnostics["flow"] = {
            "input_count": len(proposals),
            "output_count": len(selected_proposals),
            "global_emitted": int(flow_state.emitted),
            "vehicle_scale": flow_scale(config, flow_state.vehicle_count),
            **(flow_diagnostics or {}),
        }
    if diagnostics is not None:
        for source in ("old_repair", "adddrop"):
            emitted = sum(item.source == source for item in selected_proposals)
            if emitted:
                stats = _source_stats(diagnostics, source)
                stats["emitted_count"] = int(stats["emitted_count"]) + emitted
        for source in ("old_repair", "adddrop"):
            if source not in diagnostics:
                continue
            summary = diagnostics[source]
            assert isinstance(summary, dict)
            summary["emitted_count"] = sum(
                item.source == source for item in selected_proposals
            )
    return selected_proposals


def _vehicle_proposals(
    vehicle: int,
    probs_np: np.ndarray,
    adj_view: np.ndarray,
    deg: np.ndarray,
    dem_view: np.ndarray,
    capacity: float,
    cfg: TTSConfig,
    ctx: CandidateContext,
    bank_dense: np.ndarray,
    compression: CompressionState,
    diag: Optional[Dict[str, object]],
) -> List[Proposal]:
    n = int(probs_np.shape[0])
    n_cust = n - 1
    source_parts: List[Tuple[str, np.ndarray]] = []

    # --- lp_grow (native rules, grown on the compression graph, all vehicles) ---
    if cfg.lp_grow_seeds > 0:
        grow_rng = ctx.rng(vehicle, "lp_grow", base_seed=cfg.seed)
        adj_grow = compression.aggregate.copy()
        np.fill_diagonal(adj_grow, 0.0)
        n_super = int(compression.member.shape[0])
        grow_customer_subs = native_grow_subsets(
            adj_grow[1:, 1:],
            compression.demand[1:],
            capacity,
            compression.sizes[1:],
            compression.internal_x[1:],
            cfg.lp_grow_seeds,
            grow_rng,
            min_violation=cfg.min_violation,
            max_prefix=cfg.max_prefix,
        )
        grow_super_subs = np.zeros(
            (grow_customer_subs.shape[0], n_super), dtype=np.float32
        )
        grow_super_subs[:, 1:] = grow_customer_subs
        grow_subs = (grow_super_subs @ compression.member).astype(
            np.float32, copy=False
        )
        source_parts.append(("lp_grow", grow_subs))

    # --- fcapfix (vehicle 0 only) ---
    if vehicle == 0:
        fcapfix_subs = fcapfix_lite_subsets(
            adj_view,
            capacity,
            rounds=cfg.fcapfix_rounds,
            seed_limit=cfg.fcapfix_seed_limit,
            seed_scores=probs_np[:n],
            compression=compression,
        )
        source_parts.append(("fcapfix", fcapfix_subs))

    # --- old_repair (vehicle 0 only, bank non-empty) ---
    old_props: List[Proposal] = []
    if vehicle == 0 and bank_dense.shape[0]:
        old_props = old_repair_proposals(
            bank_dense,
            adj_view=adj_view,
            deg=deg,
            demand_view=dem_view,
            capacity=capacity,
            config=cfg,
            compression=compression,
            diagnostics=diag,
        )

    # --- adddrop (vehicle 0 only, old-bank seed, full pool) ---
    adddrop_props: List[Proposal] = []
    if vehicle == 0:
        seeds: List[np.ndarray] = []
        if cfg.adddrop_seed_old_bank and bank_dense.shape[0]:
            seeds = select_adddrop_seeds_from_bank(
                bank_dense,
                adj_view=adj_view,
                deg=deg,
                demand_view=dem_view,
                capacity=capacity,
            )
        adddrop_props = adddrop_proposals(
            seeds,
            adj_view=adj_view,
            deg=deg,
            demand_view=dem_view,
            capacity=capacity,
            config=cfg,
            seen_keys={item.key for item in old_props},
            compression=compression,
            diagnostics=diag,
        )
    supplemental = adddrop_props + old_props

    nonempty_parts = [(s, p) for s, p in source_parts if p.shape[0] > 0]
    if not nonempty_parts:
        return supplemental
    pool = np.concatenate([part for _, part in nonempty_parts], axis=0)
    pool_sources = np.concatenate(
        [np.full(part.shape[0], source, dtype=object) for source, part in nonempty_parts]
    )
    pool[:, 0] = 0.0
    sizes = pool.sum(axis=1)
    size_ok = (sizes >= float(cfg.min_size)) & (
        sizes <= float(cfg.max_size_frac) * float(n_cust)
    )
    pool = pool[size_ok]
    pool_sources = pool_sources[size_ok]
    if pool.shape[0] == 0:
        return supplemental
    pool_masks = pack_selection_rows(pool)
    keep = first_seen_unique_rows(pool_masks)
    if keep.size == 0:
        return supplemental
    pool = pool[keep]
    pool_sources = pool_sources[keep]
    pool_masks = pool_masks[keep]
    viol, _lhs, _rhs, _d_s = score_subsets(pool, adj_view, deg, dem_view, capacity)
    nnz = pool.sum(axis=1).clip(min=1.0)
    score = viol / np.sqrt(nnz)
    order = np.argsort(-score, kind="stable")
    top_b = order[: int(max(1, cfg.top_b))]
    pool_top = pool[top_b]
    viol_top = viol[top_b]
    score_top = score[top_b]
    source_top = pool_sources[top_b]
    masks_top = pool_masks[top_b]

    proposals: List[Proposal] = list(supplemental)
    out_keys: set[bytes] = {item.key for item in supplemental}
    for i in range(pool_top.shape[0]):
        if viol_top[i] > float(cfg.min_violation):
            key = row_key(masks_top[i])
            if key not in out_keys:
                proposals.append(
                    Proposal(
                        pool_top[i].astype(np.float32, copy=True),
                        masks_top[i].copy(),
                        key,
                        str(source_top[i]),
                        float(viol_top[i]),
                        float(score_top[i]),
                    )
                )
                out_keys.add(key)
    ls_count = int(min(cfg.top_l, pool_top.shape[0]))
    for i in range(ls_count):
        sel_init = pool_top[i].copy()
        new_sel, new_viol = local_search_1flip(
            sel_init,
            adj_view,
            deg,
            dem_view,
            capacity,
            max_flips=cfg.max_flips,
            min_gain=cfg.min_gain,
            max_size_frac=cfg.max_size_frac,
            swap_max_pairs=cfg.ls_swap_max_pairs,
            min_keep=cfg.min_size,
            diagnostics=None,
        )
        source = str(source_top[i])
        if new_viol <= float(cfg.min_violation):
            continue
        if float(new_sel.sum()) < float(cfg.min_size):
            continue
        new_mask = pack_selection_rows(new_sel[None, :])[0]
        key = row_key(new_mask)
        if key in out_keys:
            continue
        new_score = float(new_viol) / math.sqrt(max(float(new_sel.sum()), 1.0))
        proposals.append(Proposal(new_sel, new_mask, key, source, float(new_viol), new_score))
        out_keys.add(key)
    return proposals


def generate_tts(
    ctx: CandidateContext,
    cfg: TTSConfig,
    *,
    collect_diagnostics: bool = False,
) -> tuple[CandidateBatch, tuple[dict, ...]]:
    n = int(ctx.n_nodes)
    capacity = float(ctx.capacity)
    if n <= 1 or capacity <= 0.0 or ctx.vehicle_count <= 0:
        return CandidateBatch.empty(max(n, 1)), ()

    adj_view = ctx.lp.dense
    deg = ctx.lp.degree
    dem_view = ctx.demand.astype(np.float32)
    bank_dense = _bank_dense(ctx)

    compression = build_compression_state(
        adj_view,
        dem_view,
        capacity,
        old_sets=bank_dense,
        eps=cfg.shrink_eps,
    )
    flow_state = FlowState(
        vehicle_count=int(ctx.vehicle_count),
        n_nodes=n,
        mask_words=ctx.mask_words,
    )

    selections: List[np.ndarray] = []
    group_ids: List[int] = []
    variants: List[str] = []
    diags: List[dict] = []
    max_replicas = int(cfg.max_replicas)
    n_replicas = int(ctx.vehicle_count)
    if 0 < max_replicas < n_replicas:
        n_replicas = max_replicas
    for vehicle in range(n_replicas):
        diag: Optional[Dict[str, object]] = None
        if collect_diagnostics:
            diag = {
                "compression": {
                    "mode": "full",
                    "one_edge_customer_supernodes": int(
                        compression.one_edge_customer_supernodes
                    ),
                    "full_customer_supernodes": int(compression.member.shape[0] - 1),
                    "frozen_customer_supernodes": int(
                        np.count_nonzero(compression.frozen[1:])
                    ),
                },
            }
        proposals = _vehicle_proposals(
            vehicle,
            np.ascontiguousarray(ctx.probs[vehicle]),
            adj_view,
            deg,
            dem_view,
            capacity,
            cfg,
            ctx,
            bank_dense,
            compression,
            diag,
        )
        emitted = _finalize_proposals(
            proposals,
            config=cfg,
            flow_state=flow_state,
            diagnostics=diag,
        )
        for item in emitted:
            selections.append(item.selection)
            group_ids.append(vehicle)
            variants.append(item.source)
        if diag is not None:
            diags.append(diag)

    if not selections:
        return CandidateBatch.empty(n), tuple(diags)
    batch = CandidateBatch.from_selections(
        np.stack(selections, axis=0) > 0.5,
        family=FAMILY_TTS,
        group_ids=group_ids,
        variants=variants,
    )
    return batch, tuple(diags)
