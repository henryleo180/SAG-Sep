from __future__ import annotations

import gc
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sagsep import features
from sagsep.config import SagSepConfig
from sagsep.families.onion import OnionConfig as FamilyOnionConfig
from sagsep.families.onion import generate_onion
from sagsep.families.sweep import SweepConfig as FamilySweepConfig
from sagsep.families.sweep import generate_sweep
from sagsep.families.tts import TTSConfig as FamilyTTSConfig
from sagsep.families.tts import generate_tts
from sagsep.gurobi_c import (
    GRB_OPTIMAL,
    GRB_SUBOPTIMAL,
    GurobiEnv,
    GurobiModel,
)
from sagsep.model import PolicyModel
from sagsep.spine import OldCutBank, run_spine
from sagsep.types import CandidateBatch, Instance, mask_key_for_ext_nodes, n_mask_words

__all__ = [
    "IterationStats",
    "RootResult",
    "LearnedSeparator",
    "run_instance",
    "malloc_trim",
    "build_root_lp",
    "build_cut_row",
    "add_cut_rows",
]


# ---------------------------------------------------------------------------
# glibc malloc_trim (per-instance arena release)
# ---------------------------------------------------------------------------

_libc = None
_libc_probed = False


def malloc_trim() -> None:
    global _libc, _libc_probed
    if not _libc_probed:
        _libc_probed = True
        try:
            import ctypes

            _libc = ctypes.CDLL("libc.so.6")
        except Exception:
            _libc = None
    if _libc is not None:
        try:
            _libc.malloc_trim(0)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Result containers (field names / asdict order match the reference implementation)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class IterationStats:
    iteration: int
    objective: float
    cuts_added: int
    max_violation: float
    policy_info: Sequence[float]
    solve_time: float
    cut_generation_time: float
    constraints: int
    violations: List[float] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    diagnostics: Dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class RootResult:
    instance_name: str
    vehicles: int
    objective: float
    runtime: float
    iterations: List[IterationStats] = field(default_factory=list)
    iteration_count: int = 0
    total_cuts: int = 0
    last_cuts_added: int = 0
    last_max_violation: float = 0.0
    optimal: Optional[float] = None
    time_limit: Optional[float] = None
    time_limit_hit: bool = False
    timing_totals: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        iterations = []
        for item in self.iterations:
            payload = asdict(item)
            if not payload.get("diagnostics"):
                payload.pop("diagnostics", None)
            iterations.append(payload)
        return {
            "instance": self.instance_name,
            "vehicles": self.vehicles,
            "lower_bound": self.objective,
            "runtime": self.runtime,
            "iterations": iterations,
            "iteration_count": self.iteration_count,
            "total_cuts": self.total_cuts,
            "last_cuts_added": self.last_cuts_added,
            "last_max_violation": self.last_max_violation,
            "optimal": self.optimal,
            "time_limit": self.time_limit,
            "time_limit_hit": self.time_limit_hit,
            "timing_totals": self.timing_totals,
        }


# ---------------------------------------------------------------------------
# config bridging: SagSepConfig leaf configs -> family-native configs
# ---------------------------------------------------------------------------


def _family_sweep_config(cfg: SagSepConfig) -> FamilySweepConfig:
    return FamilySweepConfig(
        threshold=cfg.sweep.threshold,
        min_size=cfg.sweep.min_size,
        min_violation=cfg.sweep.min_violation,
    )


def _family_onion_config(cfg: SagSepConfig) -> FamilyOnionConfig:
    return FamilyOnionConfig(
        mode=cfg.onion.mode,
        min_violation=cfg.onion.min_violation,
        cutpoint=cfg.onion.cutpoint,
    )


def _family_tts_config(cfg: SagSepConfig) -> FamilyTTSConfig:
    t = cfg.tts
    return FamilyTTSConfig(
        max_prefix=t.max_prefix,
        min_size=t.min_size,
        lp_grow_seeds=t.lp_grow_seeds,
        lp_grow_native_rules=t.lp_grow_native_rules,
        compress_mode=t.compress_mode,
        fcapfix_rounds=t.fcapfix_rounds,
        fcapfix_seed_limit=t.fcapfix_seed_limit,
        fcapfix_seed_mode=t.fcapfix_seed_mode,
        old_repair_bank_size=t.old_repair_bank_size,
        old_repair_top_b=t.old_repair_top_b,
        adddrop_seed_sources=tuple(t.adddrop_seed_sources),
        adddrop_seed_old_bank=t.adddrop_seed_old_bank,
        adddrop_seed_limit=t.adddrop_seed_limit,
        score_mode=t.score_mode,
        min_violation=t.min_violation,
        top_b=t.top_b,
        top_l=t.top_l,
        max_flips=t.max_flips,
        min_gain=t.min_gain,
        ls_neighborhood=t.ls_neighborhood,
        ls_swap=t.ls_swap,
        ls_swap_max_pairs=t.ls_swap_max_pairs,
        max_size_frac=t.max_size_frac,
        seed=t.seed,
        rng_mode=t.rng_mode,
        flow_enabled=t.flow_enabled,
        source_quotas=tuple(t.source_quotas),
        per_vehicle_quota=t.per_vehicle_quota,
        global_quota=t.global_quota,
        near_duplicate_hamming=t.near_duplicate_hamming,
        diversity_weight=t.diversity_weight,
        vehicle_tighten_start=t.vehicle_tighten_start,
        vehicle_tighten_power=t.vehicle_tighten_power,
        vehicle_tighten_min_scale=t.vehicle_tighten_min_scale,
        max_replicas=t.max_replicas,
    )


# ---------------------------------------------------------------------------
# LearnedSeparator -- LP support -> features -> forward -> families -> spine
# ---------------------------------------------------------------------------


class LearnedSeparator:
    def __init__(
        self,
        model: PolicyModel,
        cfg: SagSepConfig,
        *,
        n_nodes: int,
        load_time: float = 0.0,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.sweep_cfg = _family_sweep_config(cfg)
        self.onion_cfg = _family_onion_config(cfg)
        self.tts_cfg = _family_tts_config(cfg)
        self.bank = OldCutBank(n_nodes, bank_size=cfg.tts.old_repair_bank_size)
        self._load_time = float(load_time)

    def propose(
        self,
        instance: Instance,
        *,
        iteration: int,
        blocked: "frozenset[bytes]",
        support_u: np.ndarray,
        support_v: np.ndarray,
        support_x: np.ndarray,
        vehicle_count: int,
    ) -> Tuple[
        "object",
        "object",
        Tuple[dict, ...],
        Dict[str, float],
        List[float],
    ]:
        total_start = perf_counter()

        prep_start = perf_counter()
        lp = features.build_lp_support(
            support_u,
            support_v,
            support_x,
            n_nodes=instance.n_nodes,
            threshold=self.cfg.lp.lp_support_edge_threshold,
        )
        prepare_lp_time = perf_counter() - prep_start

        graph_start = perf_counter()
        feats = features.build_graph_features(
            instance,
            lp,
            vehicle_count=vehicle_count,
            support_edge_u=support_u,
            support_edge_v=support_v,
            support_edge_x=support_x,
            cross_depot=self.cfg.features.partial_connect_cross_depot,
        )
        graph_time = perf_counter() - graph_start

        forward_start = perf_counter()
        probs = self.model.forward_probs(feats)
        forward_time = perf_counter() - forward_start

        ctx_start = perf_counter()
        ctx = features.build_candidate_context(
            instance,
            lp,
            probs,
            iteration=iteration,
            seed=self.cfg.mbp_seed,
            tts_seed=self.cfg.tts.seed,
            bank=self.bank.snapshot(),
        )
        prepare_time = prepare_lp_time + (perf_counter() - ctx_start)

        decode_start = perf_counter()
        sweep_start = perf_counter()
        sweep_batch = generate_sweep(ctx, self.sweep_cfg)
        sweep_time = perf_counter() - sweep_start
        onion_start = perf_counter()
        onion_batch = generate_onion(ctx, self.onion_cfg)
        onion_time = perf_counter() - onion_start
        decode_wall = perf_counter() - decode_start
        tts_start = perf_counter()
        if self.cfg.tts.enabled:
            tts_batch, tts_diags = generate_tts(ctx, self.tts_cfg, collect_diagnostics=True)
        else:
            tts_batch, tts_diags = CandidateBatch.empty(instance.n_nodes), ()
        tts_time = perf_counter() - tts_start

        cuts, diag = run_spine(
            ctx,
            sweep_batch,
            onion_batch,
            tts_batch,
            blocked=blocked,
            budget=self.cfg.budget,
            vehicle_count=int(ctx.vehicle_count),
            family_elapsed={"sweep": sweep_time, "onion": onion_time},
            compute_cut_values=self.cfg.compute_cut_values,
        )
        self.bank.remember(cuts)

        total_time = perf_counter() - total_start
        unified_total = prepare_time + decode_wall + sum(
            diag.timings.get(k, 0.0) for k in ("dedup", "scoring", "selector", "materialize")
        )
        mean_set = float(np.mean(cuts.set_sizes)) if cuts.set_sizes else 0.0

        timings = {
            "total": float(total_time),
            "prepare_total": float(graph_time),
            "graph_build": float(graph_time),
            "batch_materialize": 0.0,
            "vehicle_unpack": 0.0,
            "forward": float(forward_time),
            "prob_export": 0.0,
            "threshold": float(decode_wall),
            "subset_materialize": float(diag.timings.get("materialize", 0.0)),
            "rhs": 0.0,
            "cut_value": 0.0,
            "unified_prepare": float(prepare_time),
            "unified_sweep": float(sweep_time),
            "unified_onion": float(onion_time),
            "unified_decode_wall": float(decode_wall),
            "unified_dedup": float(diag.timings.get("dedup", 0.0)),
            "unified_scoring": float(diag.timings.get("scoring", 0.0)),
            "unified_selector": float(diag.timings.get("selector", 0.0)),
            "unified_build": float(diag.timings.get("materialize", 0.0)),
            "postprocess_total": float(unified_total),
            "unified_tts": float(tts_time),
        }
        info = [
            float(total_time),
            float(self._load_time),
            float(graph_time),
            float(forward_time),
            float(mean_set),
        ]
        self._load_time = 0.0
        return cuts, diag, tts_diags, timings, info


# ---------------------------------------------------------------------------
# LP model construction (the reference model build, vectorized)
# ---------------------------------------------------------------------------


def _build_lp(
    model_lp: GurobiModel,
    instance: Instance,
    weight: np.ndarray,
    vehicles: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = instance.n_nodes
    iu, ju = np.triu_indices(n, k=1)
    iu = iu.astype(np.int64)
    ju = ju.astype(np.int64)
    n_edges = int(iu.size)
    obj = np.ascontiguousarray(weight[iu, ju], dtype=np.float64)
    ub = np.where((iu == 0) | (ju == 0), 2.0, 1.0)
    lb = np.zeros(n_edges, dtype=np.float64)
    model_lp.add_vars(n_edges, lb=lb, ub=ub, obj=obj)
    model_lp.set_minimize()
    model_lp.update()

    var_index = np.full((n, n), -1, dtype=np.int32)
    var_ids = np.arange(n_edges, dtype=np.int32)
    var_index[iu, ju] = var_ids
    var_index[ju, iu] = var_ids

    ones = np.ones(n, dtype=np.float64)
    for node in range(1, n):
        others = np.concatenate(
            (np.arange(0, node, dtype=np.int64), np.arange(node + 1, n, dtype=np.int64))
        )
        indices = np.ascontiguousarray(var_index[node, others], dtype=np.int32)
        model_lp.add_constr(indices, ones[: indices.size], "=", 2.0, f"deg_{node + 1}")
    depot_indices = np.ascontiguousarray(var_index[0, 1:n], dtype=np.int32)
    model_lp.add_constr(
        depot_indices, ones[: depot_indices.size], "=", 2.0 * float(vehicles), "deg_depot"
    )
    model_lp.update()
    return iu, ju, var_index


# ---------------------------------------------------------------------------
# cut row construction (the reference cut-row builder + its inner/between var-index helpers)
# ---------------------------------------------------------------------------


def _build_cut_row(
    nodes_ext: Sequence[int],
    rhs: float,
    var_index: np.ndarray,
    n_customers: int,
    upper_tri_cache: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, float]:
    nodes0 = np.asarray(nodes_ext, dtype=np.int64) - 1
    n = var_index.shape[0]

    def inner(sub: np.ndarray) -> np.ndarray:
        size = int(sub.size)
        if size < 2:
            return np.empty(0, dtype=np.int32)
        cached = upper_tri_cache.get(size)
        if cached is None:
            cached = np.triu_indices(size, k=1)
            upper_tri_cache[size] = cached
        block = var_index[np.ix_(sub, sub)]
        cols = block[cached[0], cached[1]]
        return cols[cols >= 0].astype(np.int32, copy=False)

    if nodes0.size > n_customers / 2:
        in_subset = np.zeros(n, dtype=bool)
        in_subset[nodes0] = True
        complement = np.flatnonzero(~in_subset)[1:]
        if complement.size == 0:
            cols = inner(nodes0)
            return cols, np.ones(cols.size, dtype=np.float64), float(nodes0.size) - float(rhs)
        cols_list = [inner(complement)]
        coeffs_list = [np.ones(cols_list[0].size, dtype=np.float64)]
        depot_comp = var_index[0, complement]
        depot_comp = depot_comp[depot_comp >= 0].astype(np.int32, copy=False)
        if depot_comp.size:
            cols_list.append(depot_comp)
            coeffs_list.append(np.full(depot_comp.size, 0.5, dtype=np.float64))
        depot_sub = var_index[0, nodes0]
        depot_sub = depot_sub[depot_sub >= 0].astype(np.int32, copy=False)
        if depot_sub.size:
            cols_list.append(depot_sub)
            coeffs_list.append(np.full(depot_sub.size, -0.5, dtype=np.float64))
        cols = np.concatenate(cols_list)
        coeffs = np.concatenate(coeffs_list)
        return cols, coeffs, float(complement.size) - float(rhs)

    cols = inner(nodes0)
    return cols, np.ones(cols.size, dtype=np.float64), float(nodes0.size) - float(rhs)


def _add_rows_csr(
    model_lp: GurobiModel,
    rows: Sequence[Tuple[np.ndarray, np.ndarray, float]],
) -> int:
    nr = len(rows)
    if nr == 0:
        return 0
    sizes = np.fromiter((r[0].size for r in rows), dtype=np.int64, count=nr)
    indptr = np.zeros(nr + 1, dtype=np.int64)
    np.cumsum(sizes, out=indptr[1:])
    nnz = int(indptr[-1])
    cols = np.empty(nnz, dtype=np.int32)
    vals = np.empty(nnz, dtype=np.float64)
    rhs = np.empty(nr, dtype=np.float64)
    for i, (c, v, b) in enumerate(rows):
        cols[indptr[i] : indptr[i + 1]] = c
        vals[indptr[i] : indptr[i + 1]] = v
        rhs[i] = b
    model_lp.add_constrs_csr(indptr, cols, vals, b"<", rhs)
    return nnz


# ---------------------------------------------------------------------------
# Public LP-construction wrappers
# ---------------------------------------------------------------------------


def build_root_lp(
    model_lp: GurobiModel,
    instance: Instance,
    weight: np.ndarray,
    vehicles: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _build_lp(model_lp, instance, weight, vehicles)


def build_cut_row(
    nodes_ext: Sequence[int],
    rhs: float,
    var_index: np.ndarray,
    n_customers: int,
    upper_tri_cache: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, float]:
    return _build_cut_row(nodes_ext, rhs, var_index, n_customers, upper_tri_cache)


def add_cut_rows(
    model_lp: GurobiModel,
    rows: Sequence[Tuple[np.ndarray, np.ndarray, float]],
) -> int:
    return _add_rows_csr(model_lp, rows)


# ---------------------------------------------------------------------------
# run_instance -- the root loop
# ---------------------------------------------------------------------------


def run_instance(
    instance: Instance,
    cfg: SagSepConfig,
    model: Optional[PolicyModel],
    *,
    load_time: float = 0.0,
) -> RootResult:
    if model is None:
        raise ValueError("run_instance requires a loaded PolicyModel")

    n = int(instance.n_nodes)
    n_customers = n - 1
    words = n_mask_words(n)
    vehicles = int(instance.vehicles)
    replica_count = features.vehicle_replica_count(instance)
    time_limit = cfg.lp.time_limit
    if time_limit is not None and time_limit <= 0.0:
        time_limit = None
    max_iter = cfg.loop.max_iter
    min_violation_default = float(cfg.loop.min_violation)

    separator = LearnedSeparator(
        model, cfg, n_nodes=n, load_time=load_time
    )

    env = GurobiEnv()
    model_lp = GurobiModel("cvrp_root_relaxation", env=env)
    try:
        model_lp.set_param_int("OutputFlag", 0)
        model_lp.set_param_int("Method", int(cfg.lp.gurobi_method))
        if cfg.lp.gurobi_threads and int(cfg.lp.gurobi_threads) > 0:
            model_lp.set_param_int("Threads", int(cfg.lp.gurobi_threads))

        weight = features.distance_matrix(instance)
        edge_i, edge_j, var_index = _build_lp(model_lp, instance, weight, vehicles)
        base = model_lp.num_constrs
        edge_ext = np.stack([edge_i + 1, edge_j + 1], axis=1)

        dense_buf = np.zeros((n, n), dtype=np.float64)
        degree_buf = np.zeros(n, dtype=np.float64)
        upper_tri_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        seen_cuts: set[bytes] = set()

        def violation_from_dense(subset, rhs) -> float:
            if not len(subset):
                return 0.0
            nodes0 = np.asarray(subset, dtype=np.int64) - 1
            inside_sum = float(dense_buf[np.ix_(nodes0, nodes0)].sum())
            boundary = float(degree_buf[nodes0].sum()) - inside_sum
            return 2.0 * float(rhs) - boundary

        start_time = perf_counter()
        iteration_logs: List[IterationStats] = []
        iteration = 0
        iteration_count = 0
        total_cuts = 0
        last_cuts_added = 0
        last_max_violation = 0.0
        final_objective = float("inf")
        time_limit_hit = False
        can_generate = max_iter is None or max_iter > 0

        while True:
            iteration += 1
            iteration_start = perf_counter()
            solve_start = perf_counter()
            model_lp.optimize()
            solve_time = perf_counter() - solve_start

            status = model_lp.status
            if status not in {GRB_OPTIMAL, GRB_SUBOPTIMAL}:
                raise RuntimeError(f"Gurobi terminated with status {status}")

            objective = model_lp.obj_val

            solution_extract_start = perf_counter()
            values = model_lp.get_solution()
            dense_buf[:] = 0.0
            dense_buf[edge_i, edge_j] = values
            dense_buf[edge_j, edge_i] = values
            degree_buf[:] = 0.0
            np.add.at(degree_buf, edge_i, values)
            np.add.at(degree_buf, edge_j, values)
            solution_extract_time = perf_counter() - solution_extract_start

            policy_time = 0.0
            policy_info: Sequence[float] = ()
            sets: List[List[int]] = []
            rhs_values: List[float] = []
            policy_proposed_count = 0
            support_count = 0
            support_extract_time = 0.0
            policy_timings: Dict[str, float] = {}
            iteration_diagnostics: Dict[str, object] = {}

            if can_generate:
                support_extract_start = perf_counter()
                support_mask = values > 1e-8
                support_u = edge_i[support_mask]
                support_v = edge_j[support_mask]
                support_x = np.ascontiguousarray(values[support_mask], dtype=np.float64)
                support_count = int(support_u.size)
                support_extract_time = perf_counter() - support_extract_start

                blocked = frozenset(seen_cuts)

                policy_start = perf_counter()
                cuts, diag, tts_diags, policy_timings, policy_info = separator.propose(
                    instance,
                    iteration=iteration,
                    blocked=blocked,
                    support_u=support_u,
                    support_v=support_v,
                    support_x=support_x,
                    vehicle_count=replica_count,
                )
                policy_time = perf_counter() - policy_start

                sets = [list(s) for s in cuts.sets]
                rhs_values = [float(r) for r in cuts.rhs_values]
                policy_proposed_count = len(sets)

                rcc_pipeline = diag.as_dict()
                iteration_diagnostics = {"rcc_pipeline": rcc_pipeline}
                if tts_diags:
                    iteration_diagnostics["tts"] = {"vehicles": list(tts_diags)}

            iteration_violations: List[float] = []
            raw_max_violation = 0.0
            cuts_added = 0
            policy_added_to_lp = 0
            cuts_removed = 0
            max_added_violation = 0.0
            candidate_count = 0
            candidate_prepare_time = 0.0
            candidate_violation_time = 0.0
            candidate_limit_time = 0.0
            constraint_build_time = 0.0
            constraint_add_time = 0.0
            batch_nnz = 0
            matrix_add_used = 0

            candidates: List[Tuple[List[int], float, float, Optional[bytes], int]] = []
            for emit_rank, (subset, rhs) in enumerate(zip(sets, rhs_values)):
                candidate_prepare_start = perf_counter()
                nodes = sorted({int(node) for node in subset if int(node) != 1})
                if not nodes:
                    candidate_prepare_time += perf_counter() - candidate_prepare_start
                    continue
                cut_key = mask_key_for_ext_nodes(nodes, words)
                duplicate = cut_key in seen_cuts
                candidate_prepare_time += perf_counter() - candidate_prepare_start
                candidate_violation_start = perf_counter()
                raw_violation = violation_from_dense(nodes, rhs)
                candidate_violation_time += perf_counter() - candidate_violation_start
                if not duplicate:
                    iteration_violations.append(float(raw_violation))
                violation = max(0.0, raw_violation)
                if duplicate:
                    continue
                if violation <= 0.0:
                    continue
                raw_max_violation = max(raw_max_violation, violation)
                candidates.append((nodes, float(rhs), violation, cut_key, emit_rank))

            candidate_count = len(candidates)
            candidate_limit_start = perf_counter()
            violation_threshold = min_violation_default
            candidate_limit_time += perf_counter() - candidate_limit_start

            selected_candidates = [
                item for item in candidates if item[2] > violation_threshold
            ]

            if selected_candidates:
                prepared_rows: List[Tuple[np.ndarray, np.ndarray, float]] = []
                constraint_build_start = perf_counter()
                for nodes, rhs, _violation, _key, _rank in selected_candidates:
                    prepared_rows.append(
                        _build_cut_row(nodes, rhs, var_index, n_customers, upper_tri_cache)
                    )
                constraint_build_time += perf_counter() - constraint_build_start
                constraint_add_start = perf_counter()
                batch_nnz = _add_rows_csr(model_lp, prepared_rows)
                constraint_add_time += perf_counter() - constraint_add_start
                matrix_add_used = 1
                for nodes, _rhs, violation, cut_key, _rank in selected_candidates:
                    seen_cuts.add(cut_key)
                    cuts_added += 1
                    max_added_violation = max(max_added_violation, violation)
                policy_added_to_lp = sum(
                    rank < policy_proposed_count
                    for _n, _r, _v, _k, rank in selected_candidates
                )
            constraint_count = base + int(total_cuts) + int(cuts_added)

            logged_max_violation = max(raw_max_violation, max_added_violation)
            iteration_total_time = perf_counter() - iteration_start
            counts: Dict[str, int] = {
                "support_edges": int(support_count),
                "proposed_cuts": len(sets),
                "policy_proposed": int(policy_proposed_count),
                "policy_added": int(policy_added_to_lp),
                "candidate_cuts": int(candidate_count),
                "cuts_added": int(cuts_added),
                "cuts_removed": int(cuts_removed),
                "batch_nnz": int(batch_nnz),
                "matrix_add_used": int(matrix_add_used),
                "constraints": int(constraint_count),
            }

            if iteration_diagnostics:
                pipeline_diag = iteration_diagnostics.get("rcc_pipeline")
                if isinstance(pipeline_diag, dict):
                    pipeline_global = pipeline_diag.get("global")
                    if isinstance(pipeline_global, dict):
                        pipeline_global["added_to_lp"] = int(policy_added_to_lp)

            timings: Dict[str, float] = {
                "lp_solve": solve_time,
                "solution_extract": solution_extract_time,
                "support_extract": support_extract_time,
                "policy_call": policy_time,
                "cvrpsep_wait": 0.0,
                "iteration_hook": 0.0,
                "candidate_prepare": candidate_prepare_time,
                "candidate_violation": candidate_violation_time,
                "candidate_limit": candidate_limit_time,
                "constraint_build": constraint_build_time,
                "constraint_add": constraint_add_time,
                "model_update": 0.0,
                "iteration_total": iteration_total_time,
            }
            for key, value in policy_timings.items():
                timings[f"policy_{key}"] = float(value)
            timings = {k: float(v) for k, v in timings.items() if float(v) > 0.0}

            iteration_count = iteration
            total_cuts += cuts_added
            last_cuts_added = cuts_added
            last_max_violation = logged_max_violation
            final_objective = objective

            iteration_logs.append(
                IterationStats(
                    iteration=iteration,
                    objective=objective,
                    cuts_added=cuts_added,
                    max_violation=logged_max_violation,
                    violations=iteration_violations,
                    policy_info=list(policy_info),
                    solve_time=solve_time,
                    cut_generation_time=policy_time,
                    constraints=constraint_count,
                    counts=counts,
                    timings=timings,
                    diagnostics=iteration_diagnostics,
                )
            )

            # ---- termination: closure, then the two optional caps ----
            elapsed = perf_counter() - start_time
            should_stop = False
            if cuts_added == 0:
                should_stop = True
            elif max_iter is not None and iteration >= max_iter:
                should_stop = True
            if time_limit is not None and elapsed >= time_limit:
                time_limit_hit = True
                should_stop = True
            if should_stop:
                break

        total_time = perf_counter() - start_time
        timing_totals: Dict[str, float] = {}
        for item in iteration_logs:
            for key, value in item.timings.items():
                timing_totals[key] = timing_totals.get(key, 0.0) + float(value)

        return RootResult(
            instance_name=instance.name,
            vehicles=vehicles,
            objective=final_objective,
            runtime=total_time,
            iterations=iteration_logs,
            iteration_count=iteration_count,
            total_cuts=total_cuts,
            last_cuts_added=last_cuts_added,
            last_max_violation=last_max_violation,
            optimal=instance.optimal,
            time_limit=time_limit,
            time_limit_hit=time_limit_hit,
            timing_totals=timing_totals,
        )
    finally:
        model_lp.dispose()
        env.dispose()
        gc.collect()
        malloc_trim()
