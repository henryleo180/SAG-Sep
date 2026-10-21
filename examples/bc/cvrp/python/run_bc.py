#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Dict, List, NamedTuple, Optional

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "common" / "python"))
sys.path.insert(0, str(_HERE))

from sagsep.io_vrp import load_instance
from sagsep.types import Instance

from sagsep_bc_common.protocol import validate_instance

from sagsep_bc.separators import CvrpsepCutSeparator, SagSepTTSSeparator, Separator
from sagsep_bc.solver import BCOptions, BCResult, BranchAndCutSolver

SEPARATOR_CHOICES = ("cvrpsep", "sagseptts")
_PROVED_STATUS = "proved"


class ManifestEntry(NamedTuple):
    vehicles: Optional[int]
    optimal: Optional[float]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Branch-and-cut for the two-index CVRP with RCC separation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--separator", choices=SEPARATOR_CHOICES, required=True,
        help="Cut source: Lysgaard's CVRPSEP heuristics, or SAG-Sep's learned separator",
    )
    parser.add_argument(
        "--instance", nargs="+", required=True,
        help="Instance names (resolved under --instances-root) or paths to .vrp files",
    )
    parser.add_argument(
        "--instances-root", default=os.environ.get("SAGSEP_INSTANCES_ROOT", ""),
        help="Directory holding the .vrp files (default: $SAGSEP_INSTANCES_ROOT)",
    )
    parser.add_argument(
        "--manifest", default=None,
        help="CSV with instance,n,vehicles,optimal columns "
             "(default: manifest.csv next to the instances, if present)",
    )
    parser.add_argument(
        "--vehicles", type=int, default=None,
        help="Fleet size K, applied to every instance listed; overrides the manifest. "
             "Required when the manifest does not give one",
    )
    parser.add_argument(
        "--optimal", type=float, default=None,
        help="Known optimum of a single instance, recorded in its result; "
             "overrides the manifest",
    )
    parser.add_argument(
        "--run-dir", default=os.environ.get("SAGSEP_RUN_DIR", ""),
        help="sagseptts only: checkpoint directory or .ckpt file "
             "(default: $SAGSEP_RUN_DIR)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="sagseptts only: torch device for the policy network",
    )
    parser.add_argument(
        "--size-preset", default=None,
        help="sagseptts only: SAG-Sep preset tag (n50 … n1000); default: by instance size",
    )
    parser.add_argument(
        "--cvrpsep-lib", default=None,
        help="cvrpsep only: path to libcvrpsep.so "
             "(default: examples/third_party/cvrpsep/libcvrpsep.so)",
    )
    parser.add_argument("--time-limit", type=float, default=300.0,
                        help="Wall-clock budget per instance in seconds (0 = none)")
    parser.add_argument("--mip-gap", type=float, default=1e-4, help="Relative MIP gap")
    parser.add_argument("--threads", type=int, default=1, help="Gurobi threads")
    parser.add_argument("--seed", type=int, default=0, help="Gurobi seed")
    parser.add_argument("--root-rounds", type=int, default=100,
                        help="Maximum root cutting-plane rounds (0 skips the root phase)")
    parser.add_argument("--cuts-per-round", type=int, default=100,
                        help="Cap on cuts added per separation round")
    parser.add_argument("--min-violation", type=float, default=1e-6,
                        help="Smallest cut violation worth adding (the driver's rule, "
                             "applied to both cut sources after exact rescoring)")
    parser.add_argument("--usercut-max-nodes", type=int, default=15,
                        help="Separate MIPNODE user cuts while the node count is at most this")
    parser.add_argument("--tree-commit", choices=("root", "offered"), default="offered",
                        help="What a MIPNODE batch tells the separators: offered = the "
                             "rows the driver offered through GRBcbcut, which is CVRPSEP's "
                             "own convention; root = nothing, so each records only the "
                             "permanent root rows. Applied to both cut sources")
    parser.add_argument("--out", type=Path, default=None,
                        help="Directory for per-instance result JSON files")
    parser.add_argument("--quiet", dest="verbose", action="store_false",
                        help="Suppress the Gurobi log and the per-round root trace")
    args = parser.parse_args(argv)
    if args.optimal is not None and len(args.instance) > 1:
        parser.error("--optimal applies to a single instance; use the manifest for a batch")
    if args.time_limit < 0.0:
        parser.error("--time-limit must be non-negative (0 means no limit)")
    if args.vehicles is not None and args.vehicles < 1:
        parser.error("--vehicles must be at least 1")
    if args.root_rounds < 0:
        parser.error("--root-rounds must be non-negative")
    if args.cuts_per_round < 1:
        parser.error("--cuts-per-round must be at least 1")
    if args.min_violation < 0.0:
        parser.error("--min-violation must be non-negative")
    if args.usercut_max_nodes < 0:
        parser.error("--usercut-max-nodes must be non-negative")
    if args.mip_gap < 0.0:
        parser.error("--mip-gap must be non-negative")
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    return args


def resolve_instance_path(name_or_path: str, instances_root: str) -> Path:
    candidate = Path(name_or_path)
    if candidate.suffix == ".vrp":
        return candidate
    if not str(instances_root).strip():
        raise ValueError(
            f"cannot resolve instance {name_or_path!r}: pass a path to a .vrp file, "
            "set --instances-root, or set SAGSEP_INSTANCES_ROOT"
        )
    return Path(instances_root) / f"{name_or_path}.vrp"


def read_manifest(path: Path) -> Dict[str, ManifestEntry]:
    entries: Dict[str, ManifestEntry] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("instance") or "").strip()
            if not name:
                continue
            status = (row.get("status") or "").strip()
            proved = not status or status == _PROVED_STATUS
            entries[name] = ManifestEntry(
                vehicles=_optional_int(row.get("vehicles")),
                optimal=_optional_float(row.get("optimal")) if proved else None,
            )
    return entries


def _optional_int(text: Optional[str]) -> Optional[int]:
    text = (text or "").strip()
    return int(text) if text else None


def _optional_float(text: Optional[str]) -> Optional[float]:
    text = (text or "").strip()
    return float(text) if text else None


def build_separator(args: argparse.Namespace, instance: Instance) -> Separator:
    if args.separator == "cvrpsep":
        return CvrpsepCutSeparator(
            instance,
            max_cuts=args.cuts_per_round,
            min_violation=args.min_violation,
            tree_commit=args.tree_commit,
            library_path=args.cvrpsep_lib,
        )
    if not str(args.run_dir).strip():
        raise ValueError(
            "--separator sagseptts needs a policy checkpoint: pass --run-dir or set "
            "SAGSEP_RUN_DIR (a run directory, or the released .ckpt file)"
        )
    return SagSepTTSSeparator(
        instance,
        run_dir=args.run_dir,
        device=args.device,
        size_preset=args.size_preset,
        threads=args.threads,
        tree_commit=args.tree_commit,
    )


def format_summary(result: BCResult) -> str:
    objective = "-" if result.objective is None else f"{result.objective:.1f}"
    bound = "-" if result.best_bound is None else f"{result.best_bound:.1f}"
    root = "-" if result.root_bound is None else f"{result.root_bound:.1f}"
    match = {True: "opt-match", False: "OPT-MISMATCH", None: "opt-unknown"}[
        result.matches_optimum
    ]
    return (
        f"{result.instance_name}  sep={result.separator}  {result.status_name}  "
        f"obj={objective} ({match})  bound={bound}  root={root}  "
        f"nodes={result.node_count:.0f}  lazy={result.callback.lazy_cuts_added}  "
        f"usercuts={result.callback.mipnode_cuts_added}  "
        f"rootcuts={result.root_cuts}  {result.runtime:.2f}s"
    )


def run_instance(args: argparse.Namespace, name_or_path: str,
                 manifest: Dict[str, ManifestEntry]) -> BCResult:
    vrp_path = resolve_instance_path(name_or_path, args.instances_root)
    entry = manifest.get(vrp_path.stem, ManifestEntry(None, None))
    optimal = args.optimal if args.optimal is not None else entry.optimal

    instance = load_instance(vrp_path, vehicles=0, optimal=optimal)
    vehicles = args.vehicles or entry.vehicles
    if not vehicles:
        raise ValueError(
            f"no fleet size for {vrp_path.stem}: pass --vehicles, or give the "
            "instance a 'vehicles' column in manifest.csv. The fleet is fixed by "
            "an equality in this model, so it cannot be inferred"
        )
    validate_instance(
        instance.name, instance.capacity, instance.demand,
        learned=args.separator == "sagseptts",
        what="run_bc.py",
    )
    instance = replace(instance, vehicles=vehicles)

    options = BCOptions(
        time_limit=args.time_limit if args.time_limit > 0 else None,
        mip_gap=args.mip_gap,
        threads=args.threads,
        seed=args.seed,
        root_rounds=args.root_rounds,
        cuts_per_round=args.cuts_per_round,
        min_violation=args.min_violation,
        usercut_max_nodes=args.usercut_max_nodes,
        tree_commit=args.tree_commit,
        verbose=args.verbose,
    )
    setup_start = perf_counter()
    separator = build_separator(args, instance)
    with BranchAndCutSolver(
        instance, vehicles, separator, options, optimal_value=optimal,
        setup_start=setup_start,
    ) as solver:
        result = solver.solve()

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        out_file = args.out / f"bc_{vrp_path.stem}_{args.separator}.json"
        with out_file.open("w", encoding="utf-8") as handle:
            json.dump(result.as_dict(), handle, indent=2)
    return result


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    manifest: Dict[str, ManifestEntry] = {}
    manifest_path = Path(args.manifest) if args.manifest else None
    if manifest_path is None and str(args.instances_root).strip():
        default_manifest = Path(args.instances_root) / "manifest.csv"
        manifest_path = default_manifest if default_manifest.is_file() else None
    if manifest_path is not None:
        manifest = read_manifest(manifest_path)

    mismatches = 0
    for name_or_path in args.instance:
        result = run_instance(args, name_or_path, manifest)
        print(format_summary(result), flush=True)
        if result.matches_optimum is False:
            mismatches += 1
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
