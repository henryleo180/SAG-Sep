from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from sagsep import features, gurobi_c
from sagsep.types import Instance
from sagsep_train.oracle import ExactSeparator, k_label
from sagsep_train.rootloop import RootLoopResult, StateRecord, run_root_loop

__all__ = [
    "LabeledInstance",
    "label_instance",
    "labeled_to_shard_data",
    "write_labeled",
    "label_set",
    "GUROBI_METHOD",
]

GUROBI_METHOD: int = 1


@dataclass
class LabeledInstance:
    name: str
    source_sha256: str
    instance: Instance
    k_fleet: int
    k_label: int
    result: RootLoopResult
    states: List[StateRecord] = field(default_factory=list)
    solver_info: dict = field(default_factory=dict)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_instance(
    vrp_path: str | Path,
    *,
    vehicles: Optional[int],
    max_iter: Optional[int],
    threads: int = 1,
    min_violation: float = 1e-6,
    tol: float = 1e-6,
    time_limit: Optional[float] = None,
) -> LabeledInstance:
    path = Path(vrp_path)
    source_sha256 = _sha256_file(path)

    probe = features.load_instance(path, vehicles=1)
    labels = k_label(probe.demand, int(probe.capacity))
    replicas = features.vehicle_replica_count(probe)
    if labels != replicas:
        raise ValueError(
            f"{path.name}: k_label={labels} disagrees with vehicle_replica_count={replicas}"
        )

    k_fleet = int(vehicles) if vehicles is not None else labels
    if k_fleet < 1:
        raise ValueError(f"vehicles must be >= 1, got {vehicles}")
    instance = replace(probe, vehicles=k_fleet)

    separator = ExactSeparator.from_instance(instance, threads=int(threads), tol=float(tol))
    try:
        result = run_root_loop(
            instance,
            separator=separator,
            vehicles=k_fleet,
            max_iter=max_iter,
            min_violation=float(min_violation),
            gurobi_method=GUROBI_METHOD,
            gurobi_threads=int(threads),
            time_limit=time_limit,
        )
    finally:
        separator.close()

    solver_info = {
        "gurobi_version": list(gurobi_c.version()),
        "gurobi_method": int(GUROBI_METHOD),
        "gurobi_threads": int(threads),
        "min_violation": float(min_violation),
        "oracle_tol": float(tol),
        "max_iter": None if max_iter is None else int(max_iter),
        "time_limit": None if time_limit is None else float(time_limit),
        "support_rule": "x_gt_0_float64",
    }
    return LabeledInstance(
        name=path.stem,
        source_sha256=source_sha256,
        instance=instance,
        k_fleet=k_fleet,
        k_label=labels,
        result=result,
        states=result.states,
        solver_info=solver_info,
    )


def labeled_to_shard_data(labeled: LabeledInstance):
    from sagsep_train.data import format as shard_format

    if not labeled.states:
        raise ValueError(
            f"{labeled.name}: trajectory has no state "
            f"(termination={labeled.result.termination!r}); drop the instance"
        )

    builder = shard_format.ShardBuilder()
    row = builder.add_instance(
        shard_format.InstanceRecord(
            id=labeled.name,
            source_sha256=labeled.source_sha256,
            n_nodes=int(labeled.instance.n_nodes),
            capacity=int(labeled.instance.capacity),
            k_fleet=int(labeled.k_fleet),
            k_label=int(labeled.k_label),
            termination=labeled.result.termination,
            coords=np.asarray(labeled.instance.coords, dtype=np.float64),
            demand=np.asarray(labeled.instance.demand, dtype=np.int64),
        )
    )
    last = len(labeled.states) - 1
    for index, state in enumerate(labeled.states):
        payload = state.payload
        if not payload:
            raise ValueError(
                f"{labeled.name}: state {state.iteration} carries no oracle payload; "
                "label with ExactSeparator"
            )
        builder.add_state(
            row,
            shard_format.StateRecord(
                iteration=int(state.iteration),
                lp_value=float(state.lp_value),
                cuts_added=int(state.cuts_added),
                terminal=index == last,
                support_u=state.support_u,
                support_v=state.support_v,
                support_x=state.support_x,
                s_hat=payload["s_hat"],
                objective=payload["objective"],
                rhs=payload["rhs"],
                slot_violated=payload["slot_violated"],
            ),
        )
    return builder.build()


def _write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle_fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def write_labeled(labeled: LabeledInstance, out_dir: str | Path) -> Path:
    from sagsep_train.data import format as shard_format

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    data = labeled_to_shard_data(labeled)

    result = labeled.result
    sidecar = {
        "name": labeled.name,
        "source_sha256": labeled.source_sha256,
        "n_nodes": int(labeled.instance.n_nodes),
        "capacity": int(labeled.instance.capacity),
        "k_fleet": int(labeled.k_fleet),
        "k_label": int(labeled.k_label),
        "termination": result.termination,
        "iterations": int(result.iterations),
        "total_cuts": int(result.total_cuts),
        "base_lower_bound": float(result.base_lower_bound),
        "lower_bound": float(result.lower_bound),
        "runtime_s": float(result.runtime_s),
        "n_states": len(labeled.states),
        "solver_info": labeled.solver_info,
    }
    _write_json_atomic(out_path / f"{labeled.name}.labels.json", sidecar)

    npz_path = out_path / f"{labeled.name}.labels.npz"
    shard_format.write_shard(npz_path, data)
    return npz_path


def _parse_shard(shard: str) -> tuple[int, int]:
    parts = str(shard).split("/")
    if len(parts) != 2:
        raise ValueError(f"shard must look like 'i/N', got {shard!r}")
    index, count = int(parts[0]), int(parts[1])
    if count < 1 or not 1 <= index <= count:
        raise ValueError(f"shard must satisfy 1 <= i <= N with N >= 1, got {shard!r}")
    return index, count


def _read_vehicles(manifest: Path) -> Dict[str, Optional[int]]:
    table: Dict[str, Optional[int]] = {}
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "instance" not in reader.fieldnames:
            raise ValueError(f"{manifest}: manifest needs an 'instance' column")
        if "vehicles" not in reader.fieldnames:
            raise ValueError(f"{manifest}: manifest needs a 'vehicles' column")
        for row in reader:
            raw = (row.get("vehicles") or "").strip()
            table[str(row["instance"]).strip()] = int(raw) if raw else None
    return table


def label_set(
    instances_dir: str | Path,
    out_dir: str | Path,
    *,
    shard: str = "1/1",
    vehicles_from: Optional[Path] = None,
    **kwargs,
) -> List[Path]:
    index, count = _parse_shard(shard)
    src = Path(instances_dir)
    dst = Path(out_dir)
    dst.mkdir(parents=True, exist_ok=True)

    table: Optional[Dict[str, Optional[int]]] = None
    if vehicles_from is not None:
        table = _read_vehicles(Path(vehicles_from))

    kwargs.setdefault("max_iter", None)
    names = sorted(p.name for p in src.glob("*.vrp"))
    written: List[Path] = []
    for position, name in enumerate(names):
        if position % count != index - 1:
            continue
        stem = Path(name).stem
        npz_path = dst / f"{stem}.labels.npz"
        if npz_path.exists():
            written.append(npz_path)
            continue

        vehicles: Optional[int] = None
        if table is not None:
            if stem not in table:
                raise ValueError(f"{vehicles_from}: no row for instance {stem!r}")
            vehicles = table[stem]

        labeled = label_instance(src / name, vehicles=vehicles, **kwargs)
        if not labeled.states:
            continue
        written.append(write_labeled(labeled, dst))
    return written
