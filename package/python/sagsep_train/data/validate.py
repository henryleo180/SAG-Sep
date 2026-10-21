from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sagsep_train.data.format import (
    FEATURE_CONTRACT,
    SHARD_AXES,
    SHARD_DTYPES,
    SHARD_KEYS,
    SHARD_SCHEMA_VERSION,
    ShardData,
    ceil_div,
    is_sha256_hex,
    read_shard_counts,
    sha256_file,
)
from sagsep_train.data.manifest import (
    MANIFEST_FILENAME,
    SHARD_GLOB,
    read_manifest,
)

__all__ = [
    "ValidationError",
    "validate_shard",
    "assert_valid_shard",
    "validate_dataset",
    "assert_valid_dataset",
]


class ValidationError(Exception):
    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        count = len(self.problems)
        joined = "\n  ".join(self.problems)
        super().__init__(f"{count} problem(s):\n  {joined}")


def _load_arrays(target: Any, problems: list[str]) -> dict[str, np.ndarray] | None:
    if isinstance(target, ShardData):
        return target.to_arrays()
    if isinstance(target, Mapping):
        return {str(key): np.asarray(value) for key, value in target.items()}
    path = Path(target)
    if not path.is_file():
        problems.append(f"shard: file is missing: {path}")
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    except Exception as exc:
        problems.append(f"shard: cannot read {path}: {exc}")
        return None


def _scalar(arrays: Mapping[str, np.ndarray], key: str) -> Any:
    return np.asarray(arrays[key]).reshape(()).item()


def _check_structure(arrays: Mapping[str, np.ndarray]) -> list[str]:
    problems: list[str] = []
    missing = [key for key in SHARD_KEYS if key not in arrays]
    for key in missing:
        problems.append(f"{key}: key is missing from the shard")
    unexpected = sorted(set(arrays) - set(SHARD_KEYS))
    for key in unexpected:
        problems.append(f"{key}: unexpected key for schema version {SHARD_SCHEMA_VERSION}")
    if missing:
        return problems

    for key in SHARD_KEYS:
        expected = SHARD_DTYPES[key]
        found = np.asarray(arrays[key]).dtype
        if expected == "U":
            if found.kind != "U":
                problems.append(f"{key}: dtype must be unicode (U), got {found}")
        elif found != np.dtype(expected):
            problems.append(f"{key}: dtype must be {expected}, got {found}")

    def _len(key: str) -> int | None:
        array = np.asarray(arrays[key])
        if array.ndim != 1:
            problems.append(f"{key}: must be 1-d, got shape {array.shape}")
            return None
        return int(array.shape[0])

    n_inst = _len("inst_id")
    n_states = _len("st_inst_row")
    n_nodes = _len("inst_demand")
    n_edges = _len("sup_u")
    n_label = _len("lab_s_hat")
    n_slots = _len("m_objective")
    if None in (n_inst, n_states, n_nodes, n_edges, n_label, n_slots):
        return problems

    lengths = {
        "I": n_inst,
        "I1": n_inst + 1,
        "N": n_nodes,
        "N2": n_nodes,
        "S": n_states,
        "S1": n_states + 1,
        "E": n_edges,
        "L": n_label,
        "M": n_slots,
    }
    for key in SHARD_KEYS:
        axis = SHARD_AXES[key]
        array = np.asarray(arrays[key])
        if axis == "scalar":
            if array.shape != ():
                problems.append(f"{key}: must be a scalar, got shape {array.shape}")
        elif axis == "N2":
            if array.shape != (lengths["N2"], 2):
                problems.append(f"{key}: must have shape ({lengths['N2']}, 2), got {array.shape}")
        else:
            if array.shape != (lengths[axis],):
                problems.append(f"{key}: must have shape ({lengths[axis]},), got {array.shape}")

    if problems:
        return problems

    version = _scalar(arrays, "schema_version")
    if int(version) != SHARD_SCHEMA_VERSION:
        problems.append(
            f"schema_version: must be {SHARD_SCHEMA_VERSION}, got {int(version)}"
        )
    contract = str(_scalar(arrays, "feature_contract"))
    if contract != FEATURE_CONTRACT:
        problems.append(
            f"feature_contract: must be {FEATURE_CONTRACT!r}, got {contract!r}"
        )
    return problems


def _check_offsets(
    array: np.ndarray, key: str, total: int, problems: list[str]
) -> bool:
    ok = True
    if int(array[0]) != 0:
        problems.append(f"{key}: must start at 0, got {int(array[0])}")
        ok = False
    steps = np.diff(array)
    bad = np.flatnonzero(steps < 0)
    if bad.size:
        first = int(bad[0])
        problems.append(
            f"{key}: must be non-decreasing, decreases at index {first} "
            f"({int(array[first])} -> {int(array[first + 1])})"
        )
        ok = False
    if int(array[-1]) != total:
        problems.append(
            f"{key}: must end at the array length {total}, got {int(array[-1])}"
        )
        ok = False
    return ok


def _check_instances(arrays: Mapping[str, np.ndarray], problems: list[str]) -> None:
    ids = arrays["inst_id"]
    shas = arrays["inst_source_sha256"]
    n_nodes = arrays["inst_n_nodes"]
    capacity = arrays["inst_capacity"]
    k_fleet = arrays["inst_k_fleet"]
    k_label = arrays["inst_k_label"]
    termination = arrays["inst_termination"]
    offsets = arrays["inst_node_offsets"]
    coords = arrays["inst_coords"]
    demand = arrays["inst_demand"]
    n_inst = int(ids.shape[0])

    if n_inst == 0:
        problems.append("inst_id: a shard must hold at least one instance")
        return

    empty = [i for i in range(n_inst) if not str(ids[i])]
    if empty:
        problems.append(f"inst_id: empty instance id at row(s) {empty[:5]}")
    seen: dict[str, int] = {}
    for i in range(n_inst):
        name = str(ids[i])
        if name in seen:
            problems.append(f"inst_id: duplicate instance id {name!r} at rows {seen[name]} and {i}")
        else:
            seen[name] = i
    bad_sha = [i for i in range(n_inst) if not is_sha256_hex(str(shas[i]))]
    if bad_sha:
        problems.append(
            f"inst_source_sha256: not a 64-character lowercase hex digest at row(s) {bad_sha[:5]}"
        )

    for key, array, low in (
        ("inst_n_nodes", n_nodes, 2),
        ("inst_capacity", capacity, 1),
        ("inst_k_fleet", k_fleet, 1),
        ("inst_k_label", k_label, 1),
    ):
        bad = np.flatnonzero(np.asarray(array) < low)
        if bad.size:
            problems.append(
                f"{key}: must be >= {low}, got {int(np.asarray(array)[bad[0]])} at row {int(bad[0])}"
            )

    bad_term = np.flatnonzero((termination < 0) | (termination > 4))
    if bad_term.size:
        problems.append(
            f"inst_termination: code must be 0..4, got {int(termination[bad_term[0]])} "
            f"at row {int(bad_term[0])}"
        )

    if not _check_offsets(offsets, "inst_node_offsets", int(demand.shape[0]), problems):
        return
    block = np.diff(offsets)
    mismatch = np.flatnonzero(block != n_nodes.astype(np.int64))
    if mismatch.size:
        row = int(mismatch[0])
        problems.append(
            f"inst_node_offsets: block size {int(block[row])} at row {row} does not match "
            f"inst_n_nodes {int(n_nodes[row])}"
        )
        return

    if coords.size and not np.all(np.isfinite(coords)):
        problems.append("inst_coords: contains non-finite values")

    for i in range(n_inst):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        rows = demand[lo:hi]
        tag = f"instance {i} ({str(ids[i])!r})"
        if rows.size and int(rows[0]) != 0:
            problems.append(f"inst_demand: depot demand must be 0 for {tag}, got {int(rows[0])}")
        if rows.size and int(rows.min()) < 0:
            problems.append(f"inst_demand: demand must be non-negative for {tag}, got min {int(rows.min())}")
        total = int(rows.sum())
        if total <= 0:
            problems.append(f"inst_demand: total demand must be positive for {tag}, got {total}")
            continue
        cap = int(capacity[i])
        if cap <= 0:
            continue
        expected = ceil_div(total, cap)
        if int(k_label[i]) != expected:
            problems.append(
                f"inst_k_label: {tag} has k_label {int(k_label[i])} but ceil(D / Q) = "
                f"ceil({total} / {cap}) = {expected}"
            )


def _check_states(arrays: Mapping[str, np.ndarray], problems: list[str]) -> None:
    ids = arrays["inst_id"]
    n_inst = int(ids.shape[0])
    inst_n_nodes = arrays["inst_n_nodes"].astype(np.int64)
    inst_k_label = arrays["inst_k_label"].astype(np.int64)
    inst_row = arrays["st_inst_row"]
    iteration = arrays["st_iteration"]
    lp_value = arrays["st_lp_value"]
    cuts_added = arrays["st_cuts_added"]
    terminal = arrays["st_terminal"]
    n_states = int(inst_row.shape[0])

    if n_states == 0:
        problems.append("st_inst_row: a shard must hold at least one state")
        return

    out_of_range = np.flatnonzero((inst_row < 0) | (inst_row >= n_inst))
    if out_of_range.size:
        state = int(out_of_range[0])
        problems.append(
            f"st_inst_row: instance row {int(inst_row[state])} at state {state} is out of range "
            f"(0 .. {n_inst - 1})"
        )
        return

    steps = np.diff(inst_row.astype(np.int64))
    bad = np.flatnonzero(steps < 0)
    if bad.size:
        state = int(bad[0]) + 1
        problems.append(
            f"st_inst_row: states must be grouped by instance and in instance order, "
            f"row drops from {int(inst_row[state - 1])} to {int(inst_row[state])} at state {state}"
        )
        return

    counts = np.bincount(inst_row.astype(np.int64), minlength=n_inst)
    without = np.flatnonzero(counts == 0)
    if without.size:
        row = int(without[0])
        problems.append(f"st_inst_row: instance {row} ({str(ids[row])!r}) has no state")
        return

    if lp_value.size and not np.all(np.isfinite(lp_value)):
        problems.append("st_lp_value: contains non-finite values")
    negative = np.flatnonzero(cuts_added < 0)
    if negative.size:
        problems.append(
            f"st_cuts_added: must be non-negative, got {int(cuts_added[negative[0]])} "
            f"at state {int(negative[0])}"
        )
    not_flag = np.flatnonzero(terminal > 1)
    if not_flag.size:
        problems.append(
            f"st_terminal: must be 0/1, got {int(terminal[not_flag[0]])} at state {int(not_flag[0])}"
        )

    starts = np.concatenate(([0], np.cumsum(counts)))
    for i in range(n_inst):
        lo, hi = int(starts[i]), int(starts[i + 1])
        tag = f"instance {i} ({str(ids[i])!r})"
        expected = np.arange(1, hi - lo + 1, dtype=iteration.dtype)
        block = iteration[lo:hi]
        if not np.array_equal(block, expected):
            offset = int(np.flatnonzero(block != expected)[0])
            problems.append(
                f"st_iteration: {tag} must run 1..{hi - lo}; expected {int(expected[offset])} "
                f"at state {lo + offset}, got {int(block[offset])}"
            )
        flags = terminal[lo:hi]
        n_terminal = int(np.count_nonzero(flags))
        if n_terminal == 0:
            problems.append(f"st_terminal: {tag} has no terminal state, expected exactly 1")
        elif n_terminal > 1:
            problems.append(
                f"st_terminal: {tag} has {n_terminal} terminal states, expected exactly 1"
            )
        elif not flags[-1]:
            problems.append(
                f"st_terminal: {tag} marks state {lo + int(np.flatnonzero(flags)[0])} terminal, "
                f"but the terminal state must be the last one ({hi - 1})"
            )

    state_n_nodes = inst_n_nodes[inst_row.astype(np.int64)]
    state_k_label = inst_k_label[inst_row.astype(np.int64)]

    _check_support(arrays, state_n_nodes, problems)
    _check_labels(arrays, state_n_nodes, state_k_label, problems)


def _check_support(
    arrays: Mapping[str, np.ndarray], state_n_nodes: np.ndarray, problems: list[str]
) -> None:
    offsets = arrays["sup_offsets"]
    u = arrays["sup_u"]
    v = arrays["sup_v"]
    x = arrays["sup_x"]
    if not _check_offsets(offsets, "sup_offsets", int(u.shape[0]), problems):
        return
    sizes = np.diff(offsets)
    n_states = int(sizes.shape[0])
    edge_state = np.repeat(np.arange(n_states, dtype=np.int64), sizes)
    edge_n = np.repeat(state_n_nodes, sizes)

    if u.size:
        bad = np.flatnonzero(u < 0)
        if bad.size:
            edge = int(bad[0])
            problems.append(
                f"sup_u: must be >= 0, got {int(u[edge])} at edge {edge} (state {int(edge_state[edge])})"
            )
        bad = np.flatnonzero(u.astype(np.int64) >= v.astype(np.int64))
        if bad.size:
            edge = int(bad[0])
            problems.append(
                f"sup_u: support requires u < v, got u={int(u[edge])} >= v={int(v[edge])} "
                f"at edge {edge} (state {int(edge_state[edge])}); {bad.size} edge(s) affected"
            )
        bad = np.flatnonzero(v.astype(np.int64) >= edge_n)
        if bad.size:
            edge = int(bad[0])
            problems.append(
                f"sup_v: must be < n_nodes, got v={int(v[edge])} with n={int(edge_n[edge])} "
                f"at edge {edge} (state {int(edge_state[edge])})"
            )
        if not np.all(np.isfinite(x)):
            problems.append("sup_x: contains non-finite values")
        bad = np.flatnonzero(x <= 0.0)
        if bad.size:
            edge = int(bad[0])
            problems.append(
                f"sup_x: support requires x > 0, got {float(x[edge])} at edge {edge} "
                f"(state {int(edge_state[edge])}); {bad.size} edge(s) affected"
            )
        if u.size > 1:
            key = u.astype(np.int64) * edge_n + v.astype(np.int64)
            same_state = edge_state[1:] == edge_state[:-1]
            bad = np.flatnonzero(same_state & (np.diff(key) <= 0))
            if bad.size:
                edge = int(bad[0]) + 1
                problems.append(
                    f"sup_u/sup_v: support edges must be strictly increasing in lexicographic "
                    f"(u, v) order, violated at edge {edge} (state {int(edge_state[edge])})"
                )


def _check_labels(
    arrays: Mapping[str, np.ndarray],
    state_n_nodes: np.ndarray,
    state_k_label: np.ndarray,
    problems: list[str],
) -> None:
    lab_offsets = arrays["lab_offsets"]
    s_hat = arrays["lab_s_hat"]
    m_offsets = arrays["m_offsets"]
    objective = arrays["m_objective"]
    rhs = arrays["m_rhs"]
    violated = arrays["m_slot_violated"]

    labels_ok = _check_offsets(lab_offsets, "lab_offsets", int(s_hat.shape[0]), problems)
    slots_ok = _check_offsets(m_offsets, "m_offsets", int(objective.shape[0]), problems)

    if labels_ok:
        sizes = np.diff(lab_offsets)
        expected = state_n_nodes * state_k_label
        bad = np.flatnonzero(sizes != expected)
        if bad.size:
            state = int(bad[0])
            problems.append(
                f"lab_offsets: state {state} holds {int(sizes[state])} label element(s), "
                f"expected n * k_label = {int(state_n_nodes[state])} * {int(state_k_label[state])} "
                f"= {int(expected[state])}"
            )
        else:
            bad_values = np.flatnonzero(s_hat > 1)
            if bad_values.size:
                problems.append(
                    f"lab_s_hat: values must be 0/1, got {int(s_hat[bad_values[0]])} at element "
                    f"{int(bad_values[0])}; {bad_values.size} element(s) affected"
                )
            for state in range(int(sizes.shape[0])):
                lo = int(lab_offsets[state])
                width = int(state_k_label[state])
                depot = s_hat[lo : lo + width]
                if depot.size and int(depot.max()) != 0:
                    problems.append(
                        f"lab_s_hat: the depot row (node 0) must be all zeros, "
                        f"state {state} has {int(np.count_nonzero(depot))} non-zero entries"
                    )
                    break

    if slots_ok:
        sizes = np.diff(m_offsets)
        bad = np.flatnonzero(sizes != state_k_label)
        if bad.size:
            state = int(bad[0])
            problems.append(
                f"m_offsets: state {state} holds {int(sizes[state])} per-M entries, "
                f"expected k_label = {int(state_k_label[state])}"
            )
        if objective.size and not np.all(np.isfinite(objective)):
            problems.append("m_objective: contains non-finite values")
        bad = np.flatnonzero(rhs < 0)
        if bad.size:
            problems.append(
                f"m_rhs: must be non-negative, got {int(rhs[bad[0]])} at slot {int(bad[0])}"
            )
        bad = np.flatnonzero(violated > 1)
        if bad.size:
            problems.append(
                f"m_slot_violated: must be 0/1, got {int(violated[bad[0]])} at slot {int(bad[0])}"
            )


def validate_shard(target: Any) -> list[str]:
    problems: list[str] = []
    arrays = _load_arrays(target, problems)
    if arrays is None:
        return problems
    structural = _check_structure(arrays)
    if structural:
        return problems + structural
    _check_instances(arrays, problems)
    _check_states(arrays, problems)
    return problems


def assert_valid_shard(target: Any) -> None:
    problems = validate_shard(target)
    if problems:
        raise ValidationError(problems)


def validate_dataset(
    dataset_dir: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str] | None = None,
) -> list[str]:
    directory = Path(dataset_dir)
    problems: list[str] = []
    if not directory.is_dir():
        return [f"dataset: not a directory: {directory}"]

    path = Path(manifest_path) if manifest_path is not None else directory / MANIFEST_FILENAME
    if not path.is_file():
        return [f"manifest: file is missing: {path}"]
    try:
        manifest = read_manifest(path)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return [f"manifest: {exc}"]

    listed: list[str] = []
    owner: dict[str, str] = {}
    for index, entry in enumerate(manifest.shards):
        tag = f"shards[{index}] {entry.file!r}"
        if entry.file in listed:
            problems.append(f"{tag}: listed more than once")
        listed.append(entry.file)
        shard_path = directory / entry.file
        if not shard_path.is_file():
            problems.append(f"{tag}: file is missing")
            continue
        size = shard_path.stat().st_size
        if size != entry.bytes:
            problems.append(f"{tag}: bytes mismatch (manifest {entry.bytes}, file {size})")
        digest = sha256_file(shard_path)
        if digest != entry.sha256:
            problems.append(f"{tag}: sha256 mismatch (manifest {entry.sha256}, file {digest})")

        shard_problems: list[str] = []
        arrays = _load_arrays(shard_path, shard_problems)
        if arrays is None:
            problems.extend(f"{tag}: {problem}" for problem in shard_problems)
            continue
        shard_problems = validate_shard(arrays)
        problems.extend(f"{tag}: {problem}" for problem in shard_problems)
        if shard_problems:
            continue

        n_instances, n_states = read_shard_counts(shard_path)
        if n_instances != entry.n_instances:
            problems.append(
                f"{tag}: n_instances mismatch (manifest {entry.n_instances}, file {n_instances})"
            )
        if n_states != entry.n_states:
            problems.append(
                f"{tag}: n_states mismatch (manifest {entry.n_states}, file {n_states})"
            )
        for name in (str(value) for value in arrays["inst_id"]):
            if name in owner:
                problems.append(
                    f"{tag}: instance id {name!r} also appears in {owner[name]!r}"
                )
            else:
                owner[name] = entry.file

    on_disk = sorted(item.name for item in directory.glob(SHARD_GLOB))
    unlisted = [name for name in on_disk if name not in set(listed)]
    if unlisted:
        problems.append(f"manifest: shard file(s) present but not listed: {', '.join(unlisted)}")

    total_instances = sum(entry.n_instances for entry in manifest.shards)
    total_states = sum(entry.n_states for entry in manifest.shards)
    if manifest.n_instances != total_instances:
        problems.append(
            f"counts.instances mismatch (manifest {manifest.n_instances}, "
            f"sum over shards {total_instances})"
        )
    if manifest.n_states != total_states:
        problems.append(
            f"counts.states mismatch (manifest {manifest.n_states}, "
            f"sum over shards {total_states})"
        )
    return problems


def assert_valid_dataset(
    dataset_dir: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str] | None = None,
) -> None:
    problems = validate_dataset(dataset_dir, manifest_path)
    if problems:
        raise ValidationError(problems)
