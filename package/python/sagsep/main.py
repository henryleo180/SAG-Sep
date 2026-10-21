from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import List, Optional, Sequence

import numpy as np

from sagsep.config import (
    DEFAULT_INSTANCES_ROOT,
    InstanceSpec,
    PRESET_SIZES,
    RECIPES,
    from_toml,
    preset,
)
from sagsep.io_vrp import load_instance
from sagsep.loop import malloc_trim, run_instance
from sagsep.model import load_policy_model
from sagsep.types import Instance

__all__ = ["main", "parse_args"]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sagsep",
        description="SAG-Sep root relaxation (frozen separation recipe).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Reference TOML config (frozen key universe, thin reader)",
    )
    group.add_argument(
        "--size-preset",
        type=str,
        default=None,
        choices=list(PRESET_SIZES),
        help="Frozen per-size production preset",
    )
    parser.add_argument(
        "--instance",
        type=str,
        default=None,
        help="Instance name (resolved under instances_root) or path to a .vrp file; "
        "with --config it selects a subset of the configured instances",
    )
    parser.add_argument(
        "--recipe",
        type=str,
        default=None,
        choices=list(RECIPES),
        help="With --size-preset: 'full' (default) = Sweep + Onion + TTS, the released "
        "recipe; 'os' = Onion-Sweep (Sweep + Onion, TTS off). With --config "
        "the TOML's [tts] enabled decides",
    )
    parser.add_argument("--vehicles", type=int, default=None, help="Vehicle count K (depot degree)")
    parser.add_argument("--optimal", type=float, default=None, help="Known optimal value (JSON field)")
    parser.add_argument("--device", type=str, default=None, help="Override model device (e.g. cpu, cuda:0)")
    parser.add_argument("--max-iter", type=int, default=None, help="Override maximum cut rounds")
    parser.add_argument(
        "--instances-root",
        type=str,
        default=None,
        help="Override the instances root directory",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output directory for result JSONs")
    return parser.parse_args(argv)


def _infer_vehicle_count(instance: Instance) -> int:
    total_demand = float(sum(float(d) for d in instance.demand.tolist()))
    capacity = max(float(instance.capacity), 1e-6)
    return max(1, int(math.ceil(total_demand / capacity)))


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _resolve_vrp_path(name_or_path: str, instances_root: str) -> Path:
    candidate = Path(name_or_path)
    if candidate.suffix == ".vrp":
        return candidate
    if not str(instances_root).strip():
        raise ValueError(
            f"cannot resolve instance {name_or_path!r}: no instances root -- set "
            "SAGSEP_INSTANCES_ROOT, pass --instances-root, or put instances_root "
            "in the TOML (a bare instance name needs a root to resolve against)"
        )
    return Path(instances_root) / f"{name_or_path}.vrp"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    instance_path_override: Optional[str] = None
    if args.config is not None:
        if args.recipe is not None:
            raise ValueError(
                "--recipe applies to --size-preset only; with --config set [tts] enabled "
                "in the TOML (false = the Onion-Sweep recipe)"
            )
        cfg = from_toml(args.config)
        specs: List[InstanceSpec] = list(cfg.instances)
        if args.instance is not None:
            wanted = {token.strip() for token in args.instance.split(",") if token.strip()}
            specs = [spec for spec in specs if spec.name in wanted]
            missing = wanted - {spec.name for spec in specs}
            if missing:
                raise ValueError(f"--instance name(s) not in config: {sorted(missing)}")
        if not specs:
            raise ValueError("no instances selected")
        out_dir = args.out if args.out is not None else Path(cfg.output_dir)
        if str(out_dir) in ("", "."):
            raise ValueError("no output directory: pass --out or set output_dir in the TOML")
    else:
        cfg = preset(args.size_preset, recipe=args.recipe or "full")
        if args.instance is None:
            raise ValueError("--size-preset mode requires --instance")
        specs = [
            InstanceSpec(
                name=Path(args.instance).stem if args.instance.endswith(".vrp") else args.instance,
                vehicles=args.vehicles,
                optimal=args.optimal,
            )
        ]
        if args.instance.endswith(".vrp"):
            instance_path_override = args.instance
        if args.out is None:
            raise ValueError("--size-preset mode requires --out")
        out_dir = args.out

    instances_root = args.instances_root or cfg.instances_root or DEFAULT_INSTANCES_ROOT
    if args.device is not None:
        cfg = replace(cfg, model=replace(cfg.model, device=args.device))
    if args.max_iter is not None:
        cfg = replace(cfg, loop=replace(cfg.loop, max_iter=int(args.max_iter)))

    recipe = "full (sweep+onion+tts)" if cfg.tts.enabled else "os (sweep+onion, tts off)"
    print(f"[sagsep] arm=learned size={cfg.size or '(toml)'} recipe={recipe}")
    print(f"[sagsep] model.run_dir = {cfg.model.run_dir}")
    print(f"[sagsep] device = {cfg.model.device}")
    print(f"[sagsep] instances_root = {instances_root}")
    print(f"[sagsep] output_dir = {out_dir}")
    print(
        f"[sagsep] gurobi Method={cfg.lp.gurobi_method} Threads={cfg.lp.gurobi_threads} "
        f"time_limit={cfg.lp.time_limit}"
    )

    if not str(cfg.model.run_dir).strip():
        raise ValueError(
            "no model run_dir: set SAGSEP_RUN_DIR or put policy_run_dir in the TOML "
            "(the checkpoint directory has no built-in default)"
        )

    load_start = perf_counter()
    model = load_policy_model(
        cfg.model.run_dir,
        device=cfg.model.device,
        partial_connect_cross_depot=cfg.features.partial_connect_cross_depot,
    )
    load_time = perf_counter() - load_start
    print(f"[sagsep] checkpoint loaded in {load_time:.2f}s: {model.checkpoint_path}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, spec in enumerate(specs):
        vrp_path = _resolve_vrp_path(instance_path_override or spec.name, instances_root)
        vehicles = spec.vehicles
        if vehicles is not None and vehicles <= 0:
            raise ValueError(
                f"instance {spec.name!r}: vehicles must be positive when given, "
                f"got {vehicles} (omit it to infer ceil(d(V)/Q))"
            )
        instance = load_instance(
            vrp_path,
            vehicles=vehicles if vehicles is not None else 0,
            optimal=spec.optimal,
        )
        if vehicles is None:
            vehicles = _infer_vehicle_count(instance)
            instance = replace(instance, vehicles=vehicles)
        run_cfg = cfg
        if spec.max_iter is not None:
            run_cfg = replace(cfg, loop=replace(cfg.loop, max_iter=int(spec.max_iter)))

        result = None
        try:
            result = run_instance(
                instance,
                run_cfg,
                model,
                load_time=(load_time if idx == 0 else 0.0),
            )
            output_file = out_dir / f"root_relax_{spec.name}_learned_fb.json"
            with output_file.open("w", encoding="utf-8") as handle:
                json.dump(result.as_dict(), handle, indent=2, default=_json_default)

            print("------------------------------------------------------------")
            print(f"Instance : {spec.name}")
            print(f"Vehicles : {vehicles}")
            if spec.optimal is not None:
                print(f"Optimal  : {spec.optimal:.2f}")
            print(f"Lower bd : {result.objective:.2f}")
            print(f"Runtime  : {result.runtime:.2f} s")
            if result.time_limit_hit and result.time_limit is not None:
                print(f"Note     : stopped after hitting the {result.time_limit:.2f} s time limit")
            print(
                "Cut iters: "
                f"{result.iteration_count} (total cuts {result.total_cuts}, "
                f"last added {result.last_cuts_added} cuts, "
                f"max violation {result.last_max_violation:.4f})"
            )
            print(f"Stored at: {output_file}")
        finally:
            if result is not None:
                del result
            gc.collect()
            malloc_trim()
    return 0


if __name__ == "__main__":
    sys.exit(main())
