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

from .formulation import (
    add_rows_csr,
    build_directed_model,
    canonical_subset,
    violation_from_dense,
)
from .instance import DirectedInstance
from .separators import SeparationRequest, Separator, cut_mask_key

__all__ = ["BCOptions", "BCResult", "CallbackStats", "Candidate", "BranchAndCutSolver"]


def _lp_termination(status: int) -> str:
    if status == GRB_TIME_LIMIT:
        return "time_limit"
    name = STATUS_NAMES.get(int(status), f"STATUS_{int(status)}")
    return f"lp_{name.lower()}"


def _finite(value: Optional[float]) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


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
    tree_commit: str = "offered"
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.time_limit is not None and self.time_limit <= 0.0:
            raise ValueError("time_limit must be positive, or None for no limit")
        if self.mip_gap < 0.0:
            raise ValueError(f"mip_gap must be non-negative, got {self.mip_gap}")
        if self.threads < 1:
            raise ValueError(f"threads must be at least 1, got {self.threads}")
        if self.root_rounds < 0:
            raise ValueError(f"root_rounds must be non-negative, got {self.root_rounds}")
        if self.cuts_per_round < 1:
            raise ValueError(
                f"cuts_per_round must be at least 1, got {self.cuts_per_round}"
            )
        if self.min_violation < 0.0:
            raise ValueError(
                f"min_violation must be non-negative, got {self.min_violation}"
            )
        if self.usercut_max_nodes < 0:
            raise ValueError(
                f"usercut_max_nodes must be non-negative, got {self.usercut_max_nodes}"
            )
        if self.tree_commit not in ("root", "offered"):
            raise ValueError(
                f"tree_commit must be 'root' or 'offered', got {self.tree_commit!r}"
            )


@dataclass(frozen=True)
class Candidate:
    nodes: List[int]
    rhs: float
    violation: float
    proposal_index: int


@dataclass
class CallbackStats:
    mipnode_calls: int = 0
    mipnode_cuts_added: int = 0
    mipsol_calls: int = 0
    lazy_cuts_added: int = 0


@dataclass
class BCResult:
    instance_name: str
    n_customers: int
    vehicles: int
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
    root_time: float
    callback: CallbackStats = field(default_factory=CallbackStats)
    optimal_value: Optional[float] = None
    optimal_tolerance: float = 0.5

    @property
    def matches_optimum(self) -> Optional[bool]:
        if self.optimal_value is None or self.objective is None:
            return None
        if self.status_name != "OPTIMAL":
            return None
        return abs(self.objective - self.optimal_value) <= self.optimal_tolerance

    def as_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["matches_optimum"] = self.matches_optimum
        return payload


class BranchAndCutSolver:
    def __init__(
        self,
        instance: DirectedInstance,
        vehicles: int,
        separator: Separator,
        options: BCOptions,
        *,
        open_routes: bool,
        optimal_value: Optional[float] = None,
        optimal_tolerance: float = 0.5,
        setup_start: Optional[float] = None,
    ) -> None:
        self.instance = instance
        self.vehicles = int(vehicles)
        self.separator = separator
        self.options = options
        self.optimal_value = optimal_value
        self.optimal_tolerance = float(optimal_tolerance)
        self._setup_start = perf_counter() if setup_start is None else float(setup_start)
        self.stats = CallbackStats()

        self._env = GurobiEnv()
        self._formulation = build_directed_model(
            self._env, instance, self.vehicles, open_routes=open_routes
        )
        self._model = self._formulation.model
        self._model.set_param_int("OutputFlag", 1 if options.verbose else 0)
        self._model.set_param_int("Threads", max(1, options.threads))
        self._model.set_param_int("Seed", int(options.seed))

        self._root_row_masks: set[bytes] = set()
        self._blocked_masks: set[bytes] = set()
        self._blocked: frozenset[bytes] = frozenset()
        self._round = 0

    def _separate(self, values: np.ndarray) -> List[Candidate]:
        self._round += 1
        y = self._formulation.project(values)
        support_u, support_v, support_y = self._formulation.support(y)
        if support_u.size == 0:
            return []
        subsets, rhs_values = self.separator.propose(
            SeparationRequest(
                support_u=support_u,
                support_v=support_v,
                support_x=support_y,
                iteration=self._round,
                blocked=self._blocked,
            )
        )
        if len(subsets) != len(rhs_values):
            raise ValueError(
                f"separator returned {len(subsets)} subsets and {len(rhs_values)} "
                "right-hand sides; the two lists describe the same cuts and must "
                "have the same length"
            )
        if not subsets:
            return []

        dense, degree = self._formulation.dense_support(y)
        n_sep = self._formulation.n_sep_nodes
        candidates: List[Candidate] = []
        seen: set[bytes] = set()
        for proposal_index, subset in enumerate(subsets):
            nodes = canonical_subset(subset, n_sep)
            key = cut_mask_key(nodes, n_sep)
            if key in self._root_row_masks or key in seen:
                continue
            seen.add(key)
            rhs = self._formulation.rcc_rhs(nodes)
            violation = violation_from_dense(nodes, rhs, dense, degree)
            if violation <= self.options.min_violation:
                continue
            candidates.append(Candidate(nodes, rhs, violation, proposal_index))

        candidates.sort(key=lambda candidate: candidate.violation, reverse=True)
        return candidates[: self.options.cuts_per_round]

    def _commit(self, accepted: Sequence[Candidate], retain: bool) -> None:
        self.separator.commit(
            [candidate.proposal_index for candidate in accepted], retain
        )

    def _cut_key(self, nodes: List[int]) -> bytes:
        return cut_mask_key(nodes, self._formulation.n_sep_nodes)

    def _remember_root_row(self, nodes: List[int]) -> None:
        key = self._cut_key(nodes)
        self._root_row_masks.add(key)
        self._blocked_masks.add(key)

    def _solve_root_lp(self, start_time: float) -> Tuple[bool, str]:
        if not self._apply_time_budget(start_time):
            return False, "time_limit"
        self._model.optimize()
        status = self._model.status
        if status != GRB_OPTIMAL:
            if self.options.verbose:
                print(f"[root] LP stopped with status {status}", flush=True)
            return False, _lp_termination(status)
        return True, "closure"

    def _run_root_loop(
        self, start_time: float
    ) -> Tuple[Optional[float], str, bool, int, int]:
        options = self.options
        self._formulation.relax_to_continuous()

        root_bound: Optional[float] = None
        rounds_run = 0
        cuts_total = 0
        cuts_pending = False
        termination = "round_limit"
        for _ in range(options.root_rounds):
            solved, lp_termination = self._solve_root_lp(start_time)
            if not solved:
                termination = lp_termination
                break
            rounds_run += 1
            root_bound = self._model.get_dbl_attr("ObjVal")
            cuts_pending = False
            values = self._model.get_solution(0, self._formulation.n_arc_vars)

            if self._out_of_time(start_time):
                termination = "time_limit"
                break
            candidates = self._separate(values)
            if not candidates:
                termination = "closure"
                self._commit((), True)
                if options.verbose:
                    print(
                        f"[root] round {rounds_run}: bound {root_bound:.4f}, "
                        "separator found nothing — root converged",
                        flush=True,
                    )
                break

            rows = [
                self._formulation.cut_row(candidate.nodes, candidate.rhs)
                for candidate in candidates
            ]
            add_rows_csr(self._model, rows)
            self._model.update()
            cuts_pending = True
            for candidate in candidates:
                self._remember_root_row(candidate.nodes)
            self._blocked = frozenset(self._blocked_masks)
            self._commit(candidates, True)
            cuts_total += len(candidates)
            if options.verbose:
                print(
                    f"[root] round {rounds_run}: bound {root_bound:.4f}, "
                    f"+{len(candidates)} cuts, "
                    f"max violation {candidates[0].violation:.6f}",
                    flush=True,
                )

        if cuts_pending:
            if self._apply_time_budget(start_time):
                self._model.optimize()
                if self._model.status == GRB_OPTIMAL:
                    root_bound = self._model.get_dbl_attr("ObjVal")
                    cuts_pending = False
                    if options.verbose:
                        print(f"[root] final re-solve: bound {root_bound:.4f}", flush=True)
                elif options.verbose:
                    print("[root] final re-solve did not reach optimality; the bound "
                          "reported is the last certified one", flush=True)
            elif options.verbose:
                print("[root] no budget left for the final re-solve; the bound "
                      "reported is the last certified one", flush=True)
        return root_bound, termination, not cuts_pending, rounds_run, cuts_total

    def _remaining(self, start_time: float) -> Optional[float]:
        limit = self.options.time_limit
        if limit is None:
            return None
        return limit - (perf_counter() - start_time)

    def _out_of_time(self, start_time: float) -> bool:
        remaining = self._remaining(start_time)
        return remaining is not None and remaining <= 0.0

    def _apply_time_budget(self, start_time: float) -> bool:
        remaining = self._remaining(start_time)
        if remaining is None:
            return True
        if remaining <= 0.0:
            return False
        self._model.set_param_dbl("TimeLimit", remaining)
        return True

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
        values = ctx.get_node_relaxation()[: self._formulation.n_arc_vars]
        if not np.all(np.isfinite(values)):
            return
        if np.allclose(values, np.rint(values), rtol=0.0, atol=1e-6):
            return

        candidates = self._separate(values)
        accepted: List[Candidate] = []
        for candidate in candidates:
            cols, coeffs, bound = self._formulation.cut_row(candidate.nodes, candidate.rhs)
            if cols.size == 0:
                continue
            ctx.add_cut(cols, coeffs, b">", bound)
            accepted.append(candidate)
            self.stats.mipnode_cuts_added += 1
        retain = self.options.tree_commit == "offered"
        if retain and accepted:
            for candidate in accepted:
                self._blocked_masks.add(self._cut_key(candidate.nodes))
            self._blocked = frozenset(self._blocked_masks)
        self._commit(accepted, retain)

    def _on_mipsol(self, ctx: CallbackContext) -> None:
        self.stats.mipsol_calls += 1
        values = ctx.get_incumbent()[: self._formulation.n_arc_vars]
        for nodes, rhs in self._formulation.detect_violations(values):
            cols, coeffs, bound = self._formulation.cut_row(nodes, rhs)
            if cols.size == 0:
                continue
            ctx.add_lazy(cols, coeffs, b">", bound)
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
        ) = self._run_root_loop(start_time)
        root_time = perf_counter() - start_time

        if options.verbose:
            print(
                f"[root] done: bound={root_bound}, rounds={root_rounds}, "
                f"cuts={root_cuts}, {root_time:.2f}s",
                flush=True,
            )

        remaining = self._remaining(start_time)
        if remaining is not None and remaining <= 0.0:
            return self._build_result(
                start_time, root_bound, root_termination, root_bound_final,
                root_rounds, root_cuts, root_time,
                mip_ran=False,
            )

        self._formulation.restore_integer_types()
        self._model.set_param_dbl("MIPGap", options.mip_gap)
        self._model.set_param_int("LazyConstraints", 1)
        if options.usercut_max_nodes > 0:
            self._model.set_param_int("PreCrush", 1)
        self._model.set_callback(self._callback)

        remaining = self._remaining(start_time)
        if remaining is not None:
            if remaining <= 0.0:
                return self._build_result(
                    start_time, root_bound, root_termination, root_bound_final,
                    root_rounds, root_cuts, root_time, mip_ran=False,
                )
            self._model.set_param_dbl("TimeLimit", remaining)

        if options.verbose:
            print("[mip] branch and bound …", flush=True)
        self._model.optimize()

        error = self._model.callback_error
        if error is not None:
            raise RuntimeError("separation callback failed during the MIP solve") from error

        return self._build_result(
            start_time, root_bound, root_termination, root_bound_final,
            root_rounds, root_cuts, root_time, mip_ran=True
        )

    def _build_result(
        self,
        start_time: float,
        root_bound: Optional[float],
        root_termination: str,
        root_bound_final: bool,
        root_rounds: int,
        root_cuts: int,
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
        objective = _finite(objective)
        best_bound = _finite(best_bound)
        gap = _finite(gap)
        root_bound = _finite(root_bound)

        return BCResult(
            instance_name=self.instance.name,
            n_customers=self._formulation.n_customers,
            vehicles=self.vehicles,
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
            root_time=root_time,
            callback=self.stats,
            optimal_value=self.optimal_value,
            optimal_tolerance=self.optimal_tolerance,
        )

    def close(self) -> None:
        self.separator.close()
        self._model.dispose()
        self._env.dispose()

    def __enter__(self) -> "BranchAndCutSolver":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
