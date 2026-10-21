from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "FAMILY_SWEEP",
    "FAMILY_ONION",
    "FAMILY_TTS",
    "FAMILY_NAMES",
    "POLICY_TIMING_KEYS",
    "LOOP_TIMING_KEYS",
    "freeze_array",
    "pack_masks",
    "mask_key_bytes",
    "n_mask_words",
    "mask_key_for_ext_nodes",
    "derive_source_rng",
    "Instance",
    "LPSupport",
    "OldCutBankSnapshot",
    "CandidateContext",
    "CandidateBatch",
    "DedupResult",
    "ScoredBatch",
    "SelectionBudget",
    "SelectionResult",
    "MaterializedCuts",
    "FamilyDiag",
    "PipelineDiagnostics",
]


# ---------------------------------------------------------------------------
# Family codes (fixed registry -- also the fixed spine assembly order)
# ---------------------------------------------------------------------------

FAMILY_SWEEP: int = 0
FAMILY_ONION: int = 1
FAMILY_TTS: int = 2
FAMILY_NAMES: Tuple[str, ...] = ("sweep", "onion", "tts")


# ---------------------------------------------------------------------------
# Timing bucket names (JSON schema compatibility with the reference implementation)
# ---------------------------------------------------------------------------

POLICY_TIMING_KEYS: Tuple[str, ...] = (
    "total",
    "prepare_total",
    "graph_build",
    "batch_materialize",
    "vehicle_unpack",
    "forward",
    "prob_export",
    "threshold",
    "subset_materialize",
    "rhs",
    "cut_value",
    "unified_prepare",
    "unified_sweep",
    "unified_onion",
    "unified_decode_wall",
    "unified_dedup",
    "unified_scoring",
    "unified_selector",
    "unified_build",
    "postprocess_total",
)

LOOP_TIMING_KEYS: Tuple[str, ...] = (
    "lp_solve",
    "solution_extract",
    "support_extract",
    "policy_call",
    "cvrpsep_wait",
    "iteration_hook",
    "candidate_prepare",
    "candidate_violation",
    "candidate_limit",
    "constraint_build",
    "constraint_add",
    "model_update",
    "iteration_total",
)


# ---------------------------------------------------------------------------
# Low-level helpers (contract-level: these definitions ARE the spec)
# ---------------------------------------------------------------------------


def freeze_array(arr: np.ndarray) -> np.ndarray:
    out = np.ascontiguousarray(arr)
    if out is arr and arr.flags.writeable:
        out = arr.copy()
    out.setflags(write=False)
    return out


def n_mask_words(n_nodes: int) -> int:
    return (int(n_nodes) + 63) >> 6


def mask_key_for_ext_nodes(
    nodes: Sequence[int], words: int, *, strip_depot: bool = False
) -> bytes:
    idx = np.asarray(nodes, dtype=np.int64) - 1
    if strip_depot:
        idx = idx[idx != 0]
    mask = np.zeros(words, dtype=np.uint64)
    if idx.size:
        if int(idx.min()) < 0 or int(idx.max()) >= words * 64:
            raise ValueError(
                f"external node id out of range for {words}-word mask key: "
                f"[{int(idx.min()) + 1}, {int(idx.max()) + 1}]"
            )
        np.bitwise_or.at(mask, idx >> 6, np.uint64(1) << (idx & 63).astype(np.uint64))
    return mask.tobytes()


def pack_masks(selections: np.ndarray) -> np.ndarray:
    sel = np.ascontiguousarray(selections.astype(bool, copy=False))
    if sel.ndim != 2:
        raise ValueError("selections must be 2-D (B, n_nodes)")
    b, n = sel.shape
    w = n_mask_words(n)
    bits = np.packbits(sel, axis=1, bitorder="little")
    padded = np.zeros((b, w * 8), dtype=np.uint8)
    padded[:, : bits.shape[1]] = bits
    return padded.view("<u8").reshape(b, w)


def mask_key_bytes(mask_row: np.ndarray) -> bytes:
    return np.ascontiguousarray(mask_row, dtype="<u8").tobytes()


def derive_source_rng(
    base_seed: int,
    *,
    instance: str,
    iteration: int,
    vehicle: int,
    source: str,
) -> np.random.Generator:
    payload = f"{int(base_seed)}\0{instance}\0{int(iteration)}\0{int(vehicle)}\0{source}"
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=16).digest()
    words = np.frombuffer(digest, dtype=np.uint32)
    return np.random.default_rng(np.random.SeedSequence(words.tolist()))


def _check_int24_precondition(demand: np.ndarray, capacity: int) -> None:
    limit = np.int64(1) << 24
    if int(capacity) >= limit:
        raise ValueError(
            f"capacity {capacity} >= 2**24 breaks the float32-exactness "
            "precondition documented in the types module docstring"
        )
    if demand.size:
        if int(demand.max()) >= limit or int(demand.sum()) >= limit:
            raise ValueError(
                "demand values/sum >= 2**24 break the float32-exactness "
                "precondition documented in the types module docstring"
            )


# ---------------------------------------------------------------------------
# Instance
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Instance:
    name: str
    n_nodes: int
    depot: int
    coords: np.ndarray
    demand: np.ndarray
    capacity: int
    vehicles: int
    optimal: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "coords", freeze_array(np.asarray(self.coords, dtype=np.float64)))
        object.__setattr__(self, "demand", freeze_array(np.asarray(self.demand, dtype=np.int64)))
        _check_int24_precondition(self.demand, self.capacity)


# ---------------------------------------------------------------------------
# LP support snapshot (x* restricted to support edges >= threshold)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LPSupport:
    n_nodes: int
    threshold: float
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    dense: np.ndarray
    degree: np.ndarray
    edge_u: np.ndarray
    edge_v: np.ndarray
    edge_x: np.ndarray

    @classmethod
    def from_dense(cls, dense: np.ndarray, *, threshold: float) -> "LPSupport":
        dense = np.ascontiguousarray(dense, dtype=np.float32)
        n = int(dense.shape[0])
        degree = dense.sum(axis=1)
        upper = np.triu(dense, k=1)
        edge_u, edge_v = np.nonzero(upper)
        edge_x = upper[edge_u, edge_v].astype(np.float64, copy=False)
        indptr = np.zeros(n + 1, dtype=np.int64)
        nz_r, nz_c = np.nonzero(dense)
        counts = np.bincount(nz_r, minlength=n)
        indptr[1:] = np.cumsum(counts)
        indices = nz_c.astype(np.int32, copy=False)
        data = dense[nz_r, nz_c].astype(np.float32, copy=False)
        return cls(
            n_nodes=n,
            threshold=float(threshold),
            indptr=freeze_array(indptr),
            indices=freeze_array(indices),
            data=freeze_array(data),
            dense=freeze_array(dense),
            degree=freeze_array(degree),
            edge_u=freeze_array(edge_u.astype(np.int64, copy=False)),
            edge_v=freeze_array(edge_v.astype(np.int64, copy=False)),
            edge_x=freeze_array(edge_x),
        )


# ---------------------------------------------------------------------------
# Old-cut bank snapshot (cross-iteration state, read-only inside families)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OldCutBankSnapshot:
    n_nodes: int
    mask: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray

    @property
    def size(self) -> int:
        return int(self.mask.shape[0])

    @classmethod
    def empty(cls, n_nodes: int) -> "OldCutBankSnapshot":
        return cls(
            n_nodes=int(n_nodes),
            mask=freeze_array(np.zeros((0, n_mask_words(n_nodes)), dtype=np.uint64)),
            indptr=freeze_array(np.zeros(1, dtype=np.int64)),
            indices=freeze_array(np.zeros(0, dtype=np.int32)),
        )


# ---------------------------------------------------------------------------
# CandidateContext -- the single immutable input of all three families
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CandidateContext:
    probs: np.ndarray
    lp: LPSupport
    demand: np.ndarray
    capacity: int
    n_nodes: int
    vehicle_count: int
    depot: int = 0
    instance: str = ""
    iteration: int = 0
    seed: int = 0
    tts_seed: int = 0
    bank: OldCutBankSnapshot = field(default_factory=lambda: OldCutBankSnapshot.empty(1))

    def __post_init__(self) -> None:
        object.__setattr__(self, "probs", freeze_array(np.asarray(self.probs, dtype=np.float32)))
        object.__setattr__(self, "demand", freeze_array(np.asarray(self.demand, dtype=np.int64)))
        if self.capacity > 0:
            _check_int24_precondition(self.demand, self.capacity)

    @property
    def mask_words(self) -> int:
        return n_mask_words(self.n_nodes)

    def rng(self, vehicle: int, source: str, *, base_seed: int) -> np.random.Generator:
        return derive_source_rng(
            base_seed,
            instance=self.instance,
            iteration=self.iteration,
            vehicle=int(vehicle),
            source=source,
        )


# ---------------------------------------------------------------------------
# Candidate batches (columnar CSR + uint64 bitmask; replaces RawCandidate list)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CandidateBatch:
    n_nodes: int
    indptr: np.ndarray
    indices: np.ndarray
    mask: np.ndarray
    family: np.ndarray
    group_id: np.ndarray
    variant: Tuple[str, ...]
    proposal_score: np.ndarray

    @property
    def size(self) -> int:
        return int(self.mask.shape[0])

    def row(self, i: int) -> np.ndarray:
        return self.indices[self.indptr[i] : self.indptr[i + 1]]

    def key_bytes(self, i: int) -> bytes:
        return mask_key_bytes(self.mask[i])

    @classmethod
    def empty(cls, n_nodes: int) -> "CandidateBatch":
        return cls(
            n_nodes=int(n_nodes),
            indptr=freeze_array(np.zeros(1, dtype=np.int64)),
            indices=freeze_array(np.zeros(0, dtype=np.int32)),
            mask=freeze_array(np.zeros((0, n_mask_words(n_nodes)), dtype=np.uint64)),
            family=freeze_array(np.zeros(0, dtype=np.int8)),
            group_id=freeze_array(np.zeros(0, dtype=np.int32)),
            variant=(),
            proposal_score=freeze_array(np.zeros(0, dtype=np.float64)),
        )

    @classmethod
    def from_selections(
        cls,
        selections: np.ndarray,
        *,
        family: int,
        group_ids: Sequence[int],
        variants: Sequence[str],
        proposal_scores: Optional[Sequence[float]] = None,
    ) -> "CandidateBatch":
        sel = np.asarray(selections).astype(bool, copy=False)
        if sel.ndim != 2:
            raise ValueError("selections must be (B, n_nodes)")
        if sel[:, 0].any():
            raise ValueError("depot bit must be clear in all selections")
        keep = sel.any(axis=1)
        sel = sel[keep]
        gids = np.asarray(group_ids, dtype=np.int32)[keep]
        var = tuple(v for v, k in zip(variants, keep) if k)
        if proposal_scores is None:
            scores = np.full(sel.shape[0], np.nan, dtype=np.float64)
        else:
            scores = np.asarray(proposal_scores, dtype=np.float64)[keep]
        n = int(sel.shape[1])
        rows, cols = np.nonzero(sel)
        counts = np.bincount(rows, minlength=sel.shape[0])
        indptr = np.zeros(sel.shape[0] + 1, dtype=np.int64)
        indptr[1:] = np.cumsum(counts)
        return cls(
            n_nodes=n,
            indptr=freeze_array(indptr),
            indices=freeze_array(cols.astype(np.int32, copy=False)),
            mask=freeze_array(pack_masks(sel)),
            family=freeze_array(np.full(sel.shape[0], family, dtype=np.int8)),
            group_id=freeze_array(gids),
            variant=var,
            proposal_score=freeze_array(scores),
        )

    @classmethod
    def concat(cls, batches: Sequence["CandidateBatch"]) -> "CandidateBatch":
        batches = [b for b in batches if b.size > 0]
        if not batches:
            raise ValueError("concat of all-empty batches needs an explicit n_nodes; use empty()")
        n = batches[0].n_nodes
        if any(b.n_nodes != n for b in batches):
            raise ValueError("n_nodes mismatch in concat")
        indptrs = [batches[0].indptr]
        offset = int(batches[0].indptr[-1])
        for b in batches[1:]:
            indptrs.append(b.indptr[1:] + offset)
            offset += int(b.indptr[-1])
        return cls(
            n_nodes=n,
            indptr=freeze_array(np.concatenate(indptrs)),
            indices=freeze_array(np.concatenate([b.indices for b in batches])),
            mask=freeze_array(np.vstack([b.mask for b in batches])),
            family=freeze_array(np.concatenate([b.family for b in batches])),
            group_id=freeze_array(np.concatenate([b.group_id for b in batches])),
            variant=tuple(v for b in batches for v in b.variant),
            proposal_score=freeze_array(np.concatenate([b.proposal_score for b in batches])),
        )


@dataclass(frozen=True, slots=True)
class DedupResult:
    unique: CandidateBatch
    origin_indptr: np.ndarray
    origin_raw: np.ndarray
    raw_count: int

    @property
    def duplicates(self) -> int:
        return self.raw_count - self.unique.size


# ---------------------------------------------------------------------------
# Scoring / selection / materialization
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScoredBatch:
    rhs: np.ndarray
    boundary: np.ndarray
    violation: np.ndarray
    row_nnz: np.ndarray
    coef_l2: np.ndarray


@dataclass(frozen=True, slots=True)
class SelectionBudget:
    max_rows: Optional[int] = None
    max_nnz: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_rows is not None and self.max_rows < 0:
            raise ValueError("max_rows must be nonnegative")
        if self.max_nnz is not None and self.max_nnz < 0:
            raise ValueError("max_nnz must be nonnegative")


@dataclass(frozen=True, slots=True)
class SelectionResult:
    selected: np.ndarray
    nonviolated: int


@dataclass(frozen=True, slots=True)
class MaterializedCuts:
    sets: Tuple[Tuple[int, ...], ...]
    rhs_values: Tuple[float, ...]
    set_sizes: Tuple[int, ...]
    cut_values: Tuple[float, ...]
    pool_rows: np.ndarray


# ---------------------------------------------------------------------------
# Diagnostics (JSON schema identical to the reference rcc_pipeline block)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FamilyDiag:
    raw: int = 0
    within_family_unique: int = 0
    contributed_global_unique: int = 0
    selected_with_origin: int = 0
    cpu_elapsed: float = 0.0

    def as_dict(self) -> Dict[str, object]:
        return {
            "raw": self.raw,
            "within_family_unique": self.within_family_unique,
            "contributed_global_unique": self.contributed_global_unique,
            "selected_with_origin": self.selected_with_origin,
            "cpu_elapsed": float(self.cpu_elapsed),
        }


@dataclass(slots=True)
class PipelineDiagnostics:
    raw: int = 0
    exact_unique: int = 0
    blocked: int = 0
    nonviolated: int = 0
    selector_pool: int = 0
    selected: int = 0
    materialized: int = 0
    added_to_lp: int = 0
    families: Dict[str, FamilyDiag] = field(default_factory=dict)
    selector_name: str = "passthrough"
    budget: SelectionBudget = field(default_factory=SelectionBudget)
    timings: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "global": {
                "raw": self.raw,
                "exact_unique": self.exact_unique,
                "duplicates": self.raw - self.exact_unique,
                "blocked": self.blocked,
                "nonviolated": self.nonviolated,
                "selector_pool": self.selector_pool,
                "selected": self.selected,
                "materialized": self.materialized,
                "added_to_lp": self.added_to_lp,
            },
            "families": {name: diag.as_dict() for name, diag in self.families.items()},
            "selector": {
                "name": self.selector_name,
                "max_rows": self.budget.max_rows,
                "max_nnz": self.budget.max_nnz,
            },
            "timings": {key: float(value) for key, value in self.timings.items()},
        }
