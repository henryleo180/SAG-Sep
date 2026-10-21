from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sagsep_bc_common.gurobi_capi import (
    GRB_CB_MIPNODE,
    GRB_CB_MIPNODE_NODCNT,
    GRB_CB_MIPNODE_STATUS,
    GRB_CB_MIPSOL,
    GRB_OPTIMAL,
    GRB_TIME_LIMIT,
    STATUS_NAMES,
    CallbackContext,
    GurobiEnv,
)

from .connectivity import ConnectivityCut, separate_connectivity
from .formulation import (
    add_rows_csr,
    build_tree_model,
    canonical_subset,
    capacity_rhs,
    cutset_violation_from_dense,
    detect_violations,
    inner_violation_from_dense,
)
from .instance import CmstInstance
from .padding import PaddedSupport, build_padded_support, plain_support
from .separators import SeparationRequest, Separator, cut_mask_key

__all__ = ["BCOptions", "BCResult", "CallbackStats", "Candidate", "BranchAndCutSolver"]


@dataclass
class BCOptions:
    time_limit: Optional[float] = 300.0
    mip_gap: float = 1e-4
    threads: int = 1
    seed: int = 0
    root_rounds: int = 100
    cuts_per_round: int = 100
    min_violation: float = 1e-6
    usercut_max_nodes: int = 15
    padded_support: bool = False
    connectivity_only: bool = False
    tree_commit: str = "offered"
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.time_limit is not None and not (self.time_limit > 0.0):
            raise ValueError(f"time_limit must be positive or None, got {self.time_limit}")
        if not 0.0 <= self.mip_gap < 1.0:
            raise ValueError(f"mip_gap must be in [0, 1), got {self.mip_gap}")
        if self.threads < 1:
            raise ValueError(f"threads must be at least 1, got {self.threads}")
        if self.root_rounds < 0:
            raise ValueError(f"root_rounds must be non-negative, got {self.root_rounds}")
        if self.cuts_per_round <= 0:
            raise ValueError(f"cuts_per_round must be positive, got {self.cuts_per_round}")
        if self.min_violation < 0.0:
            raise ValueError(f"min_violation must be non-negative, got {self.min_violation}")
        if self.usercut_max_nodes < 0:
            raise ValueError(
                f"usercut_max_nodes must be non-negative, got {self.usercut_max_nodes}"
            )
        if self.tree_commit not in ("root", "offered"):
            raise ValueError(
                f"tree_commit must be 'root' or 'offered', got {self.tree_commit!r}"
            )


@dataclass(frozen=True, slots=True)
class Candidate:
    nodes: List[int]
    rhs: int
    violation: float
    inner_violation: float
    cutset_violation: float
    add_inner: bool
    add_cutset: bool
    proposal_index: int

    @property
    def rows(self) -> int:
        return int(self.add_inner) + int(self.add_cutset)


def _lp_termination(status: int) -> str:
    if status == GRB_TIME_LIMIT:
        return "time_limit"
    name = STATUS_NAMES.get(int(status), f"STATUS_{int(status)}")
    return f"lp_{name.lower()}"


def _finite_or_none(value: Optional[float]) -> Optional[float]:
    if value is None or not math.isfinite(value) or abs(value) >= 1e99:
        return None
    return float(value)


@dataclass
class CallbackStats:
    mipnode_calls: int = 0
    mipnode_cuts_added: int = 0
    mipsol_calls: int = 0
    lazy_cuts_added: int = 0
    lazy_connectivity_cuts: int = 0


@dataclass
class BCResult:
    instance_name: str
    n_customers: int
    capacity: int
    root_degree: int
    separator: str
    separator_detail: str
    status: int
    status_name: str
    objective: Optional[float]
    best_bound: Optional[float]
    gap: Optional[float]
    runtime: float
    setup_time: float
    node_count: float
    sol_count: int
    root_bound: Optional[float]
    root_termination: str
    root_bound_final: bool
    root_rounds: int
    root_cuts: int
    generator_inner_rows: int
    generator_cutset_rows: int
    connectivity_cuts: int
    root_time: float
    padded_support: bool
    connectivity_only: bool
    padding_high_degree_max: int
    padding_max_clip: float
    padding_max_identity_gap: float
    callback: CallbackStats = field(default_factory=CallbackStats)
    optimal_value: Optional[float] = None

    @property
    def matches_optimum(self) -> Optional[bool]:
        if self.optimal_value is None or self.objective is None:
            return None
        if self.status_name != "OPTIMAL":
            return None
        return abs(self.objective - self.optimal_value) < 0.5

    def as_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["matches_optimum"] = self.matches_optimum
        return payload


class BranchAndCutSolver:
    def __init__(
        self,
        instance: CmstInstance,
        separator: Separator,
        options: BCOptions,
        *,
        optimal_value: Optional[float] = None,
        setup_start: Optional[float] = None,
    ) -> None:
        self.instance = instance
        self.separator = separator
        self.options = options
        self.optimal_value = optimal_value
        self._setup_start = perf_counter() if setup_start is None else float(setup_start)
        self.stats = CallbackStats()

        self._env = GurobiEnv()
        self._tree = build_tree_model(self._env, instance)
        self._model = self._tree.model
        self._model.set_param_int("OutputFlag", 1 if options.verbose else 0)
        self._model.set_param_int("Threads", max(1, options.threads))
        self._model.set_param_int("Seed", int(options.seed))

        self._root_inner_masks: set[bytes] = set()
        self._root_cutset_masks: set[bytes] = set()
        self._root_connectivity_masks: set[bytes] = set()
        self._blocked_masks: set[bytes] = set()
        self._blocked: frozenset[bytes] = frozenset()
        self._root_inner_rows = 0
        self._root_cutset_rows = 0
        self._round = 0
        self._tree_connectivity_cuts = 0
        self._padding: Optional[PaddedSupport] = None
        self._padding_gap = 0.0
        self._padding_high_degree_max = 0
        self._padding_max_clip = 0.0
        self._padding_max_identity_gap = 0.0

    def _separate_connectivity(self, values: np.ndarray) -> List[ConnectivityCut]:
        return separate_connectivity(
            values,
            self._tree.edge_i,
            self._tree.edge_j,
            self.instance,
            min_violation=self.options.min_violation,
            blocked=self._root_connectivity_masks,
        )

    def _add_root_connectivity(
        self, cuts: Sequence[ConnectivityCut]
    ) -> List[ConnectivityCut]:
        accepted: List[ConnectivityCut] = []
        for cut in cuts:
            cols, coeffs, bound = self._tree.cutset_row(cut.nodes, cut.rhs)
            if cols.size == 0:
                continue
            name = f"connectivity_{len(self._root_connectivity_masks) + len(accepted)}"
            self._model.add_constr(cols, coeffs, b">", bound, name)
            accepted.append(cut)
        if accepted:
            self._model.update()
            for cut in accepted:
                self._root_connectivity_masks.add(cut.key)
        return accepted

    def _separate(self, values: np.ndarray) -> List["Candidate"]:
        self._round += 1
        self._padding = None
        self._padding_gap = 0.0
        dense = self._tree.dense_support(values)
        if self.options.padded_support:
            padding = build_padded_support(dense)
            self._padding = padding
            support_u = padding.support_u
            support_v = padding.support_v
            support_x = padding.support_x
            if padding.high_degree > self._padding_high_degree_max:
                self._padding_high_degree_max = padding.high_degree
            if padding.max_clip > self._padding_max_clip:
                self._padding_max_clip = padding.max_clip
        else:
            support_u, support_v, support_x = plain_support(dense)
        if support_u.size == 0:
            return []
        subsets, rhs_values = self.separator.propose(
            SeparationRequest(
                support_u=support_u,
                support_v=support_v,
                support_x=support_x,
                iteration=self._round,
                blocked=self._blocked,
            )
        )
        if len(subsets) != len(rhs_values):
            raise ValueError(
                f"{self.separator.name} returned {len(subsets)} subsets but "
                f"{len(rhs_values)} right-hand sides"
            )
        if not subsets:
            return []

        n_nodes = self._tree.n_nodes
        minimum = self.options.min_violation
        eligible: List[Tuple[List[int], int, float, float, bool, bool, int]] = []
        seen: set[bytes] = set()
        for proposal_index, subset in enumerate(subsets):
            nodes = canonical_subset(subset, n_nodes)
            rhs = capacity_rhs(nodes, self.instance)
            if rhs <= 1:
                continue
            key = cut_mask_key(nodes, n_nodes)
            if key in seen:
                continue
            inner = inner_violation_from_dense(nodes, rhs, dense)
            cutset = cutset_violation_from_dense(nodes, rhs, dense)
            add_inner = inner > minimum and key not in self._root_inner_masks
            add_cutset = (
                cutset > minimum
                and key not in self._root_cutset_masks
                and key not in self._root_connectivity_masks
            )
            if not (add_inner or add_cutset):
                continue
            seen.add(key)
            eligible.append(
                (nodes, rhs, inner, cutset, add_inner, add_cutset, proposal_index)
            )

        ranked: List[Tuple[float, int, int]] = []
        for index, entry in enumerate(eligible):
            if entry[4]:
                ranked.append((entry[2], index, 0))
            if entry[5]:
                ranked.append((entry[3], index, 1))
        ranked.sort(key=lambda row: (-row[0], eligible[row[1]][6], row[2]))
        del ranked[self.options.cuts_per_round:]
        chosen_forms: Dict[int, Tuple[bool, bool]] = {}
        for _violation, index, form in ranked:
            inner_taken, cutset_taken = chosen_forms.get(index, (False, False))
            chosen_forms[index] = (
                inner_taken or form == 0, cutset_taken or form == 1
            )

        candidates: List[Candidate] = []
        for index, entry in enumerate(eligible):
            forms = chosen_forms.get(index)
            if forms is None:
                continue
            add_inner, add_cutset = forms
            nodes, rhs, inner, cutset, _ei, _ec, proposal_index = entry
            violation = max(
                inner if add_inner else -math.inf,
                cutset if add_cutset else -math.inf,
            )
            candidates.append(
                Candidate(
                    nodes, rhs, violation, inner, cutset,
                    add_inner, add_cutset, proposal_index,
                )
            )

        candidates.sort(key=lambda candidate: candidate.violation, reverse=True)
        self._measure_padding_identity(candidates)
        return candidates

    def _measure_padding_identity(self, candidates: Sequence["Candidate"]) -> None:
        padding = self._padding
        if padding is None:
            return
        worst = 0.0
        for candidate in candidates:
            gap = abs(
                padding.identity_gap(
                    candidate.nodes, candidate.rhs, candidate.inner_violation
                )
            )
            if gap > worst:
                worst = gap
        self._padding_gap = worst
        if worst > self._padding_max_identity_gap:
            self._padding_max_identity_gap = worst

    def _padding_line(self, round_index: int) -> str:
        padding = self._padding
        assert padding is not None
        return (
            f"[pad] round {round_index}: c>2 customers={padding.high_degree}, "
            f"max clip={padding.max_clip:.6g}, "
            f"max |v_cvrp/2 - v_in|={self._padding_gap:.6g}"
        )

    def _commit(self, accepted: Sequence["Candidate"], retain: bool) -> None:
        self.separator.commit(
            [candidate.proposal_index for candidate in accepted if candidate.add_inner],
            retain,
        )

    def _remember_root_rows(self, candidate: "Candidate") -> None:
        key = cut_mask_key(candidate.nodes, self._tree.n_nodes)
        if candidate.add_inner:
            self._root_inner_masks.add(key)
            self._blocked_masks.add(key)
        if candidate.add_cutset:
            self._root_cutset_masks.add(key)

    def _run_root_loop(
        self, start_time: float
    ) -> Tuple[Optional[float], str, bool, int, int, int]:
        options = self.options
        self._tree.relax_to_continuous()

        root_bound: Optional[float] = None
        rounds_run = 0
        cuts_total = 0
        connectivity_total = 0
        cuts_pending = False
        termination = "round_limit"
        for _ in range(options.root_rounds):
            if not self._apply_time_budget(start_time):
                termination = "time_limit"
                break
            self._model.optimize()
            status = self._model.status
            if status != GRB_OPTIMAL:
                if options.verbose:
                    print(f"[root] LP stopped with status {status}", flush=True)
                termination = _lp_termination(status)
                break
            rounds_run += 1
            root_bound = self._model.get_dbl_attr("ObjVal")
            cuts_pending = False
            values = self._model.get_solution(0, self._tree.n_edge_vars)

            if self._out_of_time(start_time):
                termination = "time_limit"
                break

            exact = self._separate_connectivity(values)
            exact_added = self._add_root_connectivity(exact)
            connectivity_total += len(exact_added)

            candidates: List[Candidate] = []
            if not options.connectivity_only:
                candidates = self._separate(values)
                if options.verbose and self._padding is not None:
                    print(self._padding_line(rounds_run), flush=True)
            round_inner = 0
            round_cutset = 0
            if candidates:
                rows = []
                for candidate in candidates:
                    if candidate.add_inner:
                        cols, coeffs, bound = self._tree.inner_row(
                            candidate.nodes, candidate.rhs
                        )
                        rows.append((cols, coeffs, bound, b"<"))
                        round_inner += 1
                    if candidate.add_cutset:
                        cols, coeffs, bound = self._tree.cutset_row(
                            candidate.nodes, candidate.rhs
                        )
                        rows.append((cols, coeffs, bound, b">"))
                        round_cutset += 1
                add_rows_csr(self._model, rows)
                self._model.update()
                for candidate in candidates:
                    self._remember_root_rows(candidate)
                self._blocked = frozenset(self._blocked_masks)
                self._commit(candidates, True)
                self._root_inner_rows += round_inner
                self._root_cutset_rows += round_cutset
                cuts_total += round_inner + round_cutset
            elif not options.connectivity_only:
                self._commit((), True)

            if not exact_added and not candidates:
                termination = "closure"
                if options.verbose:
                    print(
                        f"[root] round {rounds_run}: bound {root_bound:.4f}, "
                        "exact pass and separator found nothing — root converged",
                        flush=True,
                    )
                break

            cuts_pending = True
            if options.verbose:
                message = (
                    f"[root] round {rounds_run}: bound {root_bound:.4f}, "
                    f"+{len(exact_added)} connectivity cuts, "
                    f"+{round_inner + round_cutset} heuristic rows "
                    f"({round_inner} inner, {round_cutset} cutset)"
                )
                if candidates:
                    message += f", max heuristic violation {candidates[0].violation:.6f}"
                elif exact_added:
                    message += (
                        f", max connectivity violation "
                        f"{max(cut.violation for cut in exact_added):.6f}"
                    )
                print(message, flush=True)

        if cuts_pending and self._apply_time_budget(start_time):
            self._model.optimize()
            if self._model.status == GRB_OPTIMAL:
                root_bound = self._model.get_dbl_attr("ObjVal")
                cuts_pending = False
                if options.verbose:
                    print(f"[root] final batch priced: bound {root_bound:.4f}", flush=True)
        return (root_bound, termination, not cuts_pending, rounds_run,
                cuts_total, connectivity_total)

    def _remaining_time(self, start_time: float) -> Optional[float]:
        limit = self.options.time_limit
        return None if limit is None else limit - (perf_counter() - start_time)

    def _apply_time_budget(self, start_time: float) -> bool:
        remaining = self._remaining_time(start_time)
        if remaining is None:
            return True
        if remaining <= 0.0:
            return False
        self._model.set_param_dbl("TimeLimit", remaining)
        return True

    def _out_of_time(self, start_time: float) -> bool:
        remaining = self._remaining_time(start_time)
        return remaining is not None and remaining <= 0.0

    def _callback(self, ctx: CallbackContext) -> None:
        if ctx.where == GRB_CB_MIPNODE:
            self._on_mipnode(ctx)
        elif ctx.where == GRB_CB_MIPSOL:
            self._on_mipsol(ctx)

    def _on_mipnode(self, ctx: CallbackContext) -> None:
        if self.options.usercut_max_nodes <= 0:
            return
        if ctx.get_int(GRB_CB_MIPNODE_STATUS) != GRB_OPTIMAL:
            return
        node_count = int(ctx.get_double(GRB_CB_MIPNODE_NODCNT))
        if node_count > self.options.usercut_max_nodes:
            return

        self.stats.mipnode_calls += 1
        values = ctx.get_node_relaxation()[: self._tree.n_edge_vars]
        if not np.all(np.isfinite(values)):
            return

        exact = self._separate_connectivity(values)
        for cut in exact:
            cols, coeffs, bound = self._tree.cutset_row(cut.nodes, cut.rhs)
            if cols.size == 0:
                continue
            ctx.add_cut(cols, coeffs, b">", bound)
            self._tree_connectivity_cuts += 1

        if np.allclose(values, np.rint(values), atol=1e-6):
            return

        if self.options.connectivity_only:
            return

        candidates = self._separate(values)
        accepted: List[Candidate] = []
        for candidate in candidates:
            added = False
            if candidate.add_inner:
                cols, coeffs, bound = self._tree.inner_row(
                    candidate.nodes, candidate.rhs
                )
                if cols.size:
                    ctx.add_cut(cols, coeffs, b"<", bound)
                    self.stats.mipnode_cuts_added += 1
                    added = True
            if candidate.add_cutset:
                cols, coeffs, bound = self._tree.cutset_row(
                    candidate.nodes, candidate.rhs
                )
                if cols.size:
                    ctx.add_cut(cols, coeffs, b">", bound)
                    self.stats.mipnode_cuts_added += 1
                    added = True
            if added:
                accepted.append(candidate)
        retain = self.options.tree_commit == "offered"
        if retain:
            for candidate in accepted:
                if candidate.add_inner:
                    self._blocked_masks.add(
                        cut_mask_key(candidate.nodes, self._tree.n_nodes)
                    )
            self._blocked = frozenset(self._blocked_masks)
        self._commit(accepted, retain)

    def _on_mipsol(self, ctx: CallbackContext) -> None:
        self.stats.mipsol_calls += 1
        values = ctx.get_incumbent()[: self._tree.n_edge_vars]
        for row in detect_violations(
            values, self._tree.edge_i, self._tree.edge_j, self.instance
        ):
            if row.inner:
                cols, coeffs, bound = self._tree.inner_row(row.nodes, row.rhs)
                sense = b"<"
            else:
                cols, coeffs, bound = self._tree.cutset_row(row.nodes, row.rhs)
                sense = b">"
                self.stats.lazy_connectivity_cuts += 1
            if cols.size == 0:
                continue
            ctx.add_lazy(cols, coeffs, sense, bound)
            self.stats.lazy_cuts_added += 1

    def solve(self) -> BCResult:
        options = self.options
        start_time = perf_counter()
        self._setup_time = start_time - self._setup_start

        (
            root_bound,
            root_termination,
            root_bound_final,
            root_rounds,
            root_cuts,
            root_connectivity_cuts,
        ) = self._run_root_loop(start_time)
        root_time = perf_counter() - start_time

        if options.verbose:
            print(
                f"[root] done: bound={root_bound}, rounds={root_rounds}, "
                f"cuts={root_cuts}, connectivity_cuts={root_connectivity_cuts}, "
                f"{root_time:.2f}s",
                flush=True,
            )

        remaining = self._remaining_time(start_time)
        if remaining is not None and remaining <= 0.0:
            return self._build_result(
                start_time,
                root_bound,
                root_termination,
                root_bound_final,
                root_rounds,
                root_cuts,
                root_connectivity_cuts,
                root_time,
                mip_ran=False,
            )

        self._tree.restore_integer_types()
        self._model.set_param_dbl("MIPGap", options.mip_gap)
        self._model.set_param_int("LazyConstraints", 1)
        if options.usercut_max_nodes > 0:
            self._model.set_param_int("PreCrush", 1)
        self._model.set_callback(self._callback)

        remaining = self._remaining_time(start_time)
        if remaining is not None:
            if remaining <= 0.0:
                return self._build_result(
                    start_time,
                    root_bound,
                    root_termination,
                    root_bound_final,
                    root_rounds,
                    root_cuts,
                    root_connectivity_cuts,
                    root_time,
                    mip_ran=False,
                )
            self._model.set_param_dbl("TimeLimit", remaining)

        if options.verbose:
            print("[mip] branch and bound …", flush=True)
        self._model.optimize()

        error = self._model.callback_error
        if error is not None:
            raise RuntimeError("separation callback failed during the MIP solve") from error

        return self._build_result(
            start_time,
            root_bound,
            root_termination,
            root_bound_final,
            root_rounds,
            root_cuts,
            root_connectivity_cuts,
            root_time,
            mip_ran=True,
        )

    def _build_result(
        self,
        start_time: float,
        root_bound: Optional[float],
        root_termination: str,
        root_bound_final: bool,
        root_rounds: int,
        root_cuts: int,
        root_connectivity_cuts: int,
        root_time: float,
        *,
        mip_ran: bool,
    ) -> BCResult:
        model = self._model
        if mip_ran:
            status = model.status
            best_bound = model.try_get_dbl_attr("ObjBound")
            node_count = model.try_get_dbl_attr("NodeCount") or 0.0
            try:
                sol_count = model.get_int_attr("SolCount")
            except RuntimeError:
                sol_count = 0
            objective = model.try_get_dbl_attr("ObjVal") if sol_count > 0 else None
            gap = model.try_get_dbl_attr("MIPGap") if sol_count > 0 else None
        else:
            status, objective, gap = GRB_TIME_LIMIT, None, None
            best_bound, node_count, sol_count = root_bound, 0.0, 0
        if best_bound is None:
            best_bound = root_bound
        objective = _finite_or_none(objective)
        best_bound = _finite_or_none(best_bound)
        gap = _finite_or_none(gap)
        root_bound = _finite_or_none(root_bound)

        return BCResult(
            instance_name=self.instance.name,
            n_customers=self._tree.n_customers,
            capacity=int(self.instance.capacity),
            root_degree=self._tree.root_degree,
            separator=self.separator.name,
            separator_detail=self.separator.describe(),
            status=int(status),
            status_name=STATUS_NAMES.get(int(status), f"STATUS_{int(status)}"),
            objective=objective,
            best_bound=best_bound,
            gap=gap,
            runtime=perf_counter() - start_time,
            setup_time=self._setup_time,
            node_count=float(node_count),
            sol_count=int(sol_count),
            root_bound=root_bound,
            root_termination=root_termination,
            root_bound_final=root_bound_final,
            root_rounds=root_rounds,
            root_cuts=root_cuts,
            generator_inner_rows=self._root_inner_rows,
            generator_cutset_rows=self._root_cutset_rows,
            connectivity_cuts=root_connectivity_cuts
            + self._tree_connectivity_cuts,
            root_time=root_time,
            padded_support=bool(self.options.padded_support),
            connectivity_only=bool(self.options.connectivity_only),
            padding_high_degree_max=self._padding_high_degree_max,
            padding_max_clip=self._padding_max_clip,
            padding_max_identity_gap=self._padding_max_identity_gap,
            callback=self.stats,
            optimal_value=self.optimal_value,
        )

    def close(self) -> None:
        self.separator.close()
        self._model.dispose()
        self._env.dispose()

    def __enter__(self) -> "BranchAndCutSolver":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
