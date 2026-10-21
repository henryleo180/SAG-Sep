from __future__ import annotations

from collections import OrderedDict
from time import perf_counter
from typing import Dict, List, Mapping, Tuple

import numpy as np

from sagsep.types import (
    FAMILY_NAMES,
    FAMILY_ONION,
    FAMILY_SWEEP,
    FAMILY_TTS,
    CandidateBatch,
    CandidateContext,
    DedupResult,
    FamilyDiag,
    MaterializedCuts,
    OldCutBankSnapshot,
    PipelineDiagnostics,
    ScoredBatch,
    SelectionBudget,
    SelectionResult,
    freeze_array,
    n_mask_words,
)

__all__ = [
    "assemble",
    "dedup",
    "filter_blocked",
    "score",
    "select",
    "materialize",
    "OldCutBank",
    "run_spine",
]

_NAN_BITS = np.uint64(0x7FF8000000000000)


# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------


def _rows_view(arr: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(arr)
    if arr.shape[1] == 0:
        raise ValueError("cannot build a row view of a zero-width array")
    dtype = np.dtype([(f"f{i}", arr.dtype.str) for i in range(arr.shape[1])])
    return arr.view(dtype).reshape(arr.shape[0])


def _take_rows(batch: CandidateBatch, order: np.ndarray) -> CandidateBatch:
    order = np.asarray(order, dtype=np.int64)
    sizes = np.diff(batch.indptr)
    new_sizes = sizes[order]
    new_indptr = np.zeros(order.size + 1, dtype=np.int64)
    np.cumsum(new_sizes, out=new_indptr[1:])
    total = int(new_indptr[-1])
    starts = batch.indptr[:-1][order]
    src = np.repeat(starts - new_indptr[:-1], new_sizes) + np.arange(total, dtype=np.int64)
    order_list = order.tolist()
    return CandidateBatch(
        n_nodes=batch.n_nodes,
        indptr=freeze_array(new_indptr),
        indices=freeze_array(batch.indices[src]),
        mask=freeze_array(batch.mask[order]),
        family=freeze_array(batch.family[order]),
        group_id=freeze_array(batch.group_id[order]),
        variant=tuple(batch.variant[i] for i in order_list),
        proposal_score=freeze_array(batch.proposal_score[order]),
    )


def _validate_family_batch(batch: CandidateBatch, code: int, *, vehicle_count: int, name: str) -> None:
    if batch.size == 0:
        return
    if not np.all(batch.family == np.int8(code)):
        raise ValueError(f"{name} batch contains rows with a foreign family code")
    if int(batch.group_id.min()) < 0 or int(batch.group_id.max()) >= int(vehicle_count):
        raise ValueError(f"{name} batch group_id out of range [0, {vehicle_count})")


# ---------------------------------------------------------------------------
# assemble -- the fixed family->spine interleave
# ---------------------------------------------------------------------------


def assemble(
    sweep: CandidateBatch,
    onion: CandidateBatch,
    tts: CandidateBatch,
    *,
    vehicle_count: int,
) -> CandidateBatch:
    n_nodes = sweep.n_nodes
    if onion.n_nodes != n_nodes or tts.n_nodes != n_nodes:
        raise ValueError("assemble: n_nodes mismatch across family batches")
    _validate_family_batch(sweep, FAMILY_SWEEP, vehicle_count=vehicle_count, name="sweep")
    _validate_family_batch(tts, FAMILY_TTS, vehicle_count=vehicle_count, name="tts")
    if onion.size and not np.all(onion.family == np.int8(FAMILY_ONION)):
        raise ValueError("onion batch contains rows with a foreign family code")

    head_parts = [b for b in (sweep, tts) if b.size > 0]
    if head_parts:
        head = head_parts[0] if len(head_parts) == 1 else CandidateBatch.concat(head_parts)
        group = np.concatenate(
            [sweep.group_id.astype(np.int64), tts.group_id.astype(np.int64)]
        )
        fam_rank = np.concatenate(
            [
                np.zeros(sweep.size, dtype=np.int64),
                np.ones(tts.size, dtype=np.int64),
            ]
        )
        pos = np.concatenate(
            [
                np.arange(sweep.size, dtype=np.int64),
                np.arange(tts.size, dtype=np.int64),
            ]
        )
        perm = np.lexsort((pos, fam_rank, group))
        head = _take_rows(head, perm)
        if onion.size:
            return CandidateBatch.concat([head, onion])
        return head
    if onion.size:
        return onion
    return CandidateBatch.empty(n_nodes)


# ---------------------------------------------------------------------------
# dedup -- exact set deduplication of the raw candidates
# ---------------------------------------------------------------------------


def dedup(raw: CandidateBatch) -> DedupResult:
    b = raw.size
    if b == 0:
        return DedupResult(
            unique=raw,
            origin_indptr=freeze_array(np.zeros(1, dtype=np.int64)),
            origin_raw=freeze_array(np.zeros(0, dtype=np.int32)),
            raw_count=0,
        )
    mask_view = _rows_view(raw.mask)
    _, first_idx, inverse = np.unique(mask_view, return_index=True, return_inverse=True)
    inverse = np.asarray(inverse, dtype=np.int64).reshape(-1)
    order = np.argsort(first_idx, kind="stable")
    rank = np.empty(order.size, dtype=np.int64)
    rank[order] = np.arange(order.size, dtype=np.int64)
    uid = rank[inverse]
    unique_rows = np.sort(first_idx).astype(np.int64)
    unique = _take_rows(raw, unique_rows)

    variant_arr = np.asarray(raw.variant)
    var_ids = np.unique(variant_arr, return_inverse=True)[1]
    var_ids = np.asarray(var_ids, dtype=np.int64).reshape(-1)
    score = raw.proposal_score.astype(np.float64, copy=True)
    score[score == 0.0] = 0.0
    bits = score.view(np.uint64)
    bits[np.isnan(score)] = _NAN_BITS
    comp = np.empty((b, 5), dtype=np.uint64)
    comp[:, 0] = uid.astype(np.uint64)
    comp[:, 1] = raw.family.astype(np.int64).astype(np.uint64)
    comp[:, 2] = raw.group_id.astype(np.int64).astype(np.uint64)
    comp[:, 3] = var_ids.astype(np.uint64)
    comp[:, 4] = bits
    _, keep_first = np.unique(_rows_view(comp), return_index=True)
    kept = np.sort(keep_first).astype(np.int64)
    kord = np.lexsort((kept, uid[kept]))
    origin_raw = kept[kord].astype(np.int32)
    counts = np.bincount(uid[kept], minlength=unique_rows.size)
    origin_indptr = np.zeros(unique_rows.size + 1, dtype=np.int64)
    np.cumsum(counts, out=origin_indptr[1:])
    return DedupResult(
        unique=unique,
        origin_indptr=freeze_array(origin_indptr),
        origin_raw=freeze_array(origin_raw),
        raw_count=b,
    )


# ---------------------------------------------------------------------------
# blocked filter -- drop deduplicated rows whose key is blocked
# ---------------------------------------------------------------------------


def filter_blocked(
    d: DedupResult, blocked: frozenset[bytes]
) -> Tuple[DedupResult, int]:
    u = d.unique.size
    if u == 0 or not blocked:
        return d, 0
    step = d.unique.mask.shape[1] * 8
    buf = d.unique.mask.tobytes()
    keep = np.fromiter(
        (buf[i * step : (i + 1) * step] not in blocked for i in range(u)),
        dtype=bool,
        count=u,
    )
    if keep.all():
        return d, 0
    kept_rows = np.flatnonzero(keep).astype(np.int64)
    unique = _take_rows(d.unique, kept_rows)
    sizes = np.diff(d.origin_indptr)
    new_sizes = sizes[kept_rows]
    new_indptr = np.zeros(kept_rows.size + 1, dtype=np.int64)
    np.cumsum(new_sizes, out=new_indptr[1:])
    total = int(new_indptr[-1])
    src = np.repeat(
        d.origin_indptr[:-1][kept_rows] - new_indptr[:-1], new_sizes
    ) + np.arange(total, dtype=np.int64)
    filtered = DedupResult(
        unique=unique,
        origin_indptr=freeze_array(new_indptr),
        origin_raw=freeze_array(d.origin_raw[src]),
        raw_count=d.raw_count,
    )
    return filtered, int(u - kept_rows.size)


# ---------------------------------------------------------------------------
# authoritative scoring -- boundary / rhs / violation / row cost
# ---------------------------------------------------------------------------


def score(batch: CandidateBatch, ctx: CandidateContext) -> ScoredBatch:
    n = int(ctx.n_nodes)
    if batch.n_nodes != n or ctx.lp.n_nodes != n:
        raise ValueError("score: n_nodes mismatch between batch, ctx and lp")
    b = batch.size
    if b == 0:
        empty = freeze_array(np.zeros(0, dtype=np.float64))
        return ScoredBatch(
            rhs=empty,
            boundary=empty,
            violation=empty,
            row_nnz=freeze_array(np.zeros(0, dtype=np.int64)),
            coef_l2=empty,
        )
    sizes = np.diff(batch.indptr)
    if (sizes <= 0).any():
        raise ValueError("candidate keys must contain only original 1-based customer ids")
    if batch.indices.size and (
        int(batch.indices.min()) < 1 or int(batch.indices.max()) >= n
    ):
        raise ValueError("candidate keys must contain only original 1-based customer ids")
    if int(ctx.capacity) <= 0:
        raise ValueError("score: capacity must be positive (int64 semantics)")

    rows = np.repeat(np.arange(b, dtype=np.int64), sizes)
    member = np.zeros((b, n), dtype=bool)
    member[rows, batch.indices] = True

    crossing = member[:, ctx.lp.edge_u] != member[:, ctx.lp.edge_v]
    cross_counts = crossing.sum(axis=1)
    offsets = np.zeros(b + 1, dtype=np.int64)
    np.cumsum(cross_counts, out=offsets[1:])
    flat_vals = ctx.lp.edge_x[crossing.nonzero()[1]]
    boundary = np.empty(b, dtype=np.float64)
    for i in range(b):
        boundary[i] = flat_vals[offsets[i] : offsets[i + 1]].sum(dtype=np.float64)

    demand_sum = np.add.reduceat(ctx.demand[batch.indices], batch.indptr[:-1])
    q = np.int64(ctx.capacity)
    rhs_int = (demand_sum + q - np.int64(1)) // q
    rhs = rhs_int.astype(np.float64)

    violation = 2.0 * rhs - boundary

    n_customers = np.int64(n - 1)
    s = sizes.astype(np.int64)
    if (s > n_customers).any():
        raise ValueError("subset size exceeds customer count")
    small = (2 * s <= n_customers) | (s == n_customers)
    comb_s = s * (s - 1) // 2
    comp = n_customers - s
    comb_c = comp * (comp - 1) // 2
    row_nnz = np.where(small, comb_s, comb_c + n_customers)
    l2_sq = np.where(
        small,
        row_nnz.astype(np.float64),
        comb_c.astype(np.float64) + 0.25 * float(n_customers),
    )
    coef_l2 = np.sqrt(l2_sq)

    return ScoredBatch(
        rhs=freeze_array(rhs),
        boundary=freeze_array(boundary),
        violation=freeze_array(violation),
        row_nnz=freeze_array(row_nnz),
        coef_l2=freeze_array(coef_l2),
    )


# ---------------------------------------------------------------------------
# selector -- passthrough selection + budget check
# ---------------------------------------------------------------------------


def select(scored: ScoredBatch, *, budget: SelectionBudget) -> SelectionResult:
    pool_rows = np.flatnonzero(scored.violation > 0.0).astype(np.int64)
    nonviolated = int(scored.violation.size - pool_rows.size)
    if budget.max_rows is not None and pool_rows.size > int(budget.max_rows):
        raise ValueError("selector exceeded max_rows budget")
    if budget.max_nnz is not None:
        selected_nnz = int(scored.row_nnz[pool_rows].sum())
        if selected_nnz > int(budget.max_nnz):
            raise ValueError("selector exceeded max_nnz budget")
    return SelectionResult(selected=freeze_array(pool_rows), nonviolated=nonviolated)


# ---------------------------------------------------------------------------
# materialize -- selected rows into external cut rows
# ---------------------------------------------------------------------------


def materialize(
    batch: CandidateBatch,
    scored: ScoredBatch,
    sel: SelectionResult,
    *,
    compute_cut_values: bool,
) -> MaterializedCuts:
    rows = sel.selected
    indptr = batch.indptr
    ext = batch.indices.astype(np.int64) + 1
    sets = tuple(
        tuple(ext[indptr[r] : indptr[r + 1]].tolist()) for r in rows.tolist()
    )
    rhs_values = tuple(scored.rhs[rows].tolist())
    set_sizes = tuple(int(v) for v in np.diff(indptr)[rows].tolist())
    cut_values = tuple(scored.boundary[rows].tolist()) if compute_cut_values else ()
    return MaterializedCuts(
        sets=sets,
        rhs_values=rhs_values,
        set_sizes=set_sizes,
        cut_values=cut_values,
        pool_rows=freeze_array(np.asarray(rows, dtype=np.int64)),
    )


# ---------------------------------------------------------------------------
# OldCutBank -- LRU OrderedDict of the emitted cut subsets
# ---------------------------------------------------------------------------


class OldCutBank:
    __slots__ = ("_n_nodes", "_words", "_limit", "_entries")

    def __init__(self, n_nodes: int, *, bank_size: int) -> None:
        self._n_nodes = int(n_nodes)
        self._words = n_mask_words(n_nodes)
        self._limit = int(bank_size)
        self._entries: "OrderedDict[bytes, Tuple[np.ndarray, np.ndarray]]" = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def remember(self, cuts: MaterializedCuts) -> None:
        n = self._n_nodes
        entries = self._entries
        for subset in cuts.sets:
            local = np.asarray(subset, dtype=np.int64) - 1
            local = local[(local > 0) & (local < n)]
            mask_row = np.zeros(self._words, dtype=np.uint64)
            np.bitwise_or.at(
                mask_row, local >> 6, np.uint64(1) << (local & 63).astype(np.uint64)
            )
            key = mask_row.tobytes()
            if key in entries:
                entries.move_to_end(key)
            else:
                entries[key] = (mask_row, local.astype(np.int32))
            while len(entries) > self._limit:
                entries.popitem(last=False)

    def snapshot(self) -> OldCutBankSnapshot:
        if not self._entries:
            return OldCutBankSnapshot.empty(self._n_nodes)
        masks: List[np.ndarray] = []
        index_rows: List[np.ndarray] = []
        for mask_row, indices in self._entries.values():
            masks.append(mask_row)
            index_rows.append(indices)
        sizes = np.asarray([row.size for row in index_rows], dtype=np.int64)
        indptr = np.zeros(sizes.size + 1, dtype=np.int64)
        np.cumsum(sizes, out=indptr[1:])
        return OldCutBankSnapshot(
            n_nodes=self._n_nodes,
            mask=freeze_array(np.vstack(masks)),
            indptr=freeze_array(indptr),
            indices=freeze_array(
                np.concatenate(index_rows) if index_rows else np.zeros(0, dtype=np.int32)
            ),
        )

# ---------------------------------------------------------------------------
# run_spine -- orchestration + admission diagnostics counts
# ---------------------------------------------------------------------------


def _family_first_seen_names(
    raw: CandidateBatch, family_elapsed: Mapping[str, float]
) -> List[str]:
    names: List[str] = []
    if raw.size:
        firsts = []
        for code in np.unique(raw.family):
            firsts.append((int(np.flatnonzero(raw.family == code)[0]), int(code)))
        firsts.sort()
        names = [FAMILY_NAMES[code] for _, code in firsts]
    for name in family_elapsed:
        if name not in names:
            names.append(name)
    return names


def run_spine(
    ctx: CandidateContext,
    sweep: CandidateBatch,
    onion: CandidateBatch,
    tts: CandidateBatch,
    *,
    blocked: frozenset[bytes],
    budget: SelectionBudget,
    vehicle_count: int,
    family_elapsed: Mapping[str, float],
    compute_cut_values: bool,
) -> Tuple[MaterializedCuts, PipelineDiagnostics]:
    raw = assemble(sweep, onion, tts, vehicle_count=vehicle_count)

    dedup_start = perf_counter()
    full = dedup(raw)
    dedup_time = perf_counter() - dedup_start
    pool, blocked_count = filter_blocked(full, blocked)

    scoring_start = perf_counter()
    scored = score(pool.unique, ctx)
    scoring_time = perf_counter() - scoring_start

    selector_start = perf_counter()
    sel = select(scored, budget=budget)
    selector_time = perf_counter() - selector_start

    materialize_start = perf_counter()
    cuts = materialize(pool.unique, scored, sel, compute_cut_values=compute_cut_values)
    materialize_time = perf_counter() - materialize_start

    diag = PipelineDiagnostics()
    diag.raw = raw.size
    diag.exact_unique = full.unique.size
    diag.blocked = blocked_count
    diag.nonviolated = sel.nonviolated
    diag.selector_pool = pool.unique.size - sel.nonviolated
    diag.selected = int(sel.selected.size)
    diag.materialized = len(cuts.sets)
    diag.selector_name = "passthrough"
    diag.budget = budget

    origin_fams = raw.family[pool.origin_raw] if pool.origin_raw.size else np.zeros(0, dtype=np.int8)
    owner = np.repeat(
        np.arange(pool.unique.size, dtype=np.int64), np.diff(pool.origin_indptr)
    )
    fam_bits = np.zeros(max(pool.unique.size, 1), dtype=np.uint8)
    if origin_fams.size:
        np.bitwise_or.at(
            fam_bits, owner, np.left_shift(np.uint8(1), origin_fams.astype(np.uint8))
        )
    selected_bits = fam_bits[sel.selected] if sel.selected.size else np.zeros(0, dtype=np.uint8)

    for name in _family_first_seen_names(raw, family_elapsed):
        fdiag = FamilyDiag(cpu_elapsed=float(family_elapsed.get(name, 0.0)))
        if name in FAMILY_NAMES:
            code = FAMILY_NAMES.index(name)
            fam_rows = np.flatnonzero(raw.family == code)
            fdiag.raw = int(fam_rows.size)
            if fam_rows.size:
                fdiag.within_family_unique = int(
                    np.unique(_rows_view(raw.mask[fam_rows])).size
                )
            fdiag.contributed_global_unique = int(
                np.count_nonzero(full.unique.family == code)
            )
            if selected_bits.size:
                fdiag.selected_with_origin = int(
                    np.count_nonzero((selected_bits >> np.uint8(code)) & np.uint8(1))
                )
        diag.families[name] = fdiag

    diag.timings = {
        "family_decode_wall": float(sum(family_elapsed.values())),
        "dedup": float(dedup_time),
        "scoring": float(scoring_time),
        "selector": float(selector_time),
        "materialize": float(materialize_time),
    }
    return cuts, diag
