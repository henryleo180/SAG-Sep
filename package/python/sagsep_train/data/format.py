from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = [
    "SHARD_SCHEMA_VERSION",
    "FEATURE_CONTRACT",
    "SUPPORT_RULE",
    "TERMINATION_CODES",
    "TERMINATION_NAMES",
    "SHARD_KEYS",
    "SHARD_DTYPES",
    "SHARD_AXES",
    "InstanceRecord",
    "StateRecord",
    "StateView",
    "ShardBuilder",
    "ShardData",
    "write_shard",
    "read_shard",
    "read_shard_counts",
    "merge_shards",
    "sha256_file",
]

SHARD_SCHEMA_VERSION = 1

FEATURE_CONTRACT = "partial_2hop_max_min_v1"

SUPPORT_RULE = "x_gt_0_float64"

TERMINATION_CODES: dict[str, int] = {
    "closed": 0,
    "iteration_cap": 1,
    "stalled_duplicate": 2,
    "time_limit": 3,
    "error": 4,
}

TERMINATION_NAMES: dict[int, str] = {code: name for name, code in TERMINATION_CODES.items()}

SHARD_KEYS: tuple[str, ...] = (
    "schema_version",
    "feature_contract",
    "inst_id",
    "inst_source_sha256",
    "inst_n_nodes",
    "inst_capacity",
    "inst_k_fleet",
    "inst_k_label",
    "inst_termination",
    "inst_node_offsets",
    "inst_coords",
    "inst_demand",
    "st_inst_row",
    "st_iteration",
    "st_lp_value",
    "st_cuts_added",
    "st_terminal",
    "sup_offsets",
    "sup_u",
    "sup_v",
    "sup_x",
    "lab_offsets",
    "lab_s_hat",
    "m_offsets",
    "m_objective",
    "m_rhs",
    "m_slot_violated",
)

SHARD_DTYPES: dict[str, str] = {
    "schema_version": "int32",
    "feature_contract": "U",
    "inst_id": "U",
    "inst_source_sha256": "U",
    "inst_n_nodes": "int32",
    "inst_capacity": "int64",
    "inst_k_fleet": "int32",
    "inst_k_label": "int32",
    "inst_termination": "int8",
    "inst_node_offsets": "int64",
    "inst_coords": "float64",
    "inst_demand": "int64",
    "st_inst_row": "int32",
    "st_iteration": "int32",
    "st_lp_value": "float64",
    "st_cuts_added": "int32",
    "st_terminal": "uint8",
    "sup_offsets": "int64",
    "sup_u": "int32",
    "sup_v": "int32",
    "sup_x": "float64",
    "lab_offsets": "int64",
    "lab_s_hat": "uint8",
    "m_offsets": "int64",
    "m_objective": "float64",
    "m_rhs": "int32",
    "m_slot_violated": "uint8",
}

SHARD_AXES: dict[str, str] = {
    "schema_version": "scalar",
    "feature_contract": "scalar",
    "inst_id": "I",
    "inst_source_sha256": "I",
    "inst_n_nodes": "I",
    "inst_capacity": "I",
    "inst_k_fleet": "I",
    "inst_k_label": "I",
    "inst_termination": "I",
    "inst_node_offsets": "I1",
    "inst_coords": "N2",
    "inst_demand": "N",
    "st_inst_row": "S",
    "st_iteration": "S",
    "st_lp_value": "S",
    "st_cuts_added": "S",
    "st_terminal": "S",
    "sup_offsets": "S1",
    "sup_u": "E",
    "sup_v": "E",
    "sup_x": "E",
    "lab_offsets": "S1",
    "lab_s_hat": "L",
    "m_offsets": "S1",
    "m_objective": "M",
    "m_rhs": "M",
    "m_slot_violated": "M",
}

_HEX_DIGITS = frozenset("0123456789abcdef")
_SHA256_LENGTH = 64


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256_hex(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and set(value) <= _HEX_DIGITS
    )


def ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError(f"ceil_div: denominator must be positive, got {denominator}")
    return -(-int(numerator) // int(denominator))


# --------------------------------------------------------------------------
# argument normalisation helpers
# --------------------------------------------------------------------------


def _as_array(values, dtype: str, name: str) -> np.ndarray:
    target = np.dtype(dtype)
    array = np.asarray(values)
    if target.kind in "iu":
        if array.dtype.kind not in "iub":
            raise ValueError(
                f"{name}: expected integer values for dtype {target.name}, got dtype {array.dtype}"
            )
        if array.size:
            info = np.iinfo(target)
            low = int(array.min())
            high = int(array.max())
            if low < info.min or high > info.max:
                raise ValueError(
                    f"{name}: values [{low}, {high}] do not fit in {target.name}"
                )
    elif target.kind == "f":
        if array.dtype.kind not in "iufb":
            raise ValueError(
                f"{name}: expected numeric values for dtype {target.name}, got dtype {array.dtype}"
            )
    return np.array(array, dtype=target, copy=True, order="C")


def _check_shape(array: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if array.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {array.shape}")


def _check_finite(array: np.ndarray, name: str) -> None:
    if array.size and not np.all(np.isfinite(array)):
        raise ValueError(f"{name}: contains non-finite values")


def _concat(parts: list[np.ndarray], dtype: str, width: int | None = None) -> np.ndarray:
    target = np.dtype(dtype)
    if parts:
        return np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=target)
    if width is None:
        return np.zeros(0, dtype=target)
    return np.zeros((0, width), dtype=target)


def _offsets(sizes: list[int]) -> np.ndarray:
    out = np.zeros(len(sizes) + 1, dtype=np.int64)
    if sizes:
        np.cumsum(np.asarray(sizes, dtype=np.int64), out=out[1:])
    return out


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InstanceRecord:
    id: str
    source_sha256: str
    n_nodes: int
    capacity: int
    k_fleet: int
    k_label: int
    termination: str
    coords: np.ndarray
    demand: np.ndarray


@dataclass(frozen=True)
class StateRecord:
    iteration: int
    lp_value: float
    cuts_added: int
    terminal: bool
    support_u: np.ndarray
    support_v: np.ndarray
    support_x: np.ndarray
    s_hat: np.ndarray
    objective: np.ndarray
    rhs: np.ndarray
    slot_violated: np.ndarray


@dataclass(frozen=True)
class StateView:
    instance: InstanceRecord
    inst_row: int
    iteration: int
    lp_value: float
    cuts_added: int
    terminal: bool
    support_u: np.ndarray
    support_v: np.ndarray
    support_x: np.ndarray
    s_hat: np.ndarray
    objective: np.ndarray
    rhs: np.ndarray
    slot_violated: np.ndarray


# --------------------------------------------------------------------------
# shard data
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardData:
    inst_id: np.ndarray
    inst_source_sha256: np.ndarray
    inst_n_nodes: np.ndarray
    inst_capacity: np.ndarray
    inst_k_fleet: np.ndarray
    inst_k_label: np.ndarray
    inst_termination: np.ndarray
    inst_node_offsets: np.ndarray
    inst_coords: np.ndarray
    inst_demand: np.ndarray
    st_inst_row: np.ndarray
    st_iteration: np.ndarray
    st_lp_value: np.ndarray
    st_cuts_added: np.ndarray
    st_terminal: np.ndarray
    sup_offsets: np.ndarray
    sup_u: np.ndarray
    sup_v: np.ndarray
    sup_x: np.ndarray
    lab_offsets: np.ndarray
    lab_s_hat: np.ndarray
    m_offsets: np.ndarray
    m_objective: np.ndarray
    m_rhs: np.ndarray
    m_slot_violated: np.ndarray
    schema_version: int = SHARD_SCHEMA_VERSION
    feature_contract: str = FEATURE_CONTRACT

    @property
    def n_instances(self) -> int:
        return int(self.inst_id.shape[0])

    @property
    def n_states(self) -> int:
        return int(self.st_inst_row.shape[0])

    def instance(self, i: int) -> InstanceRecord:
        i = int(i)
        lo = int(self.inst_node_offsets[i])
        hi = int(self.inst_node_offsets[i + 1])
        code = int(self.inst_termination[i])
        return InstanceRecord(
            id=str(self.inst_id[i]),
            source_sha256=str(self.inst_source_sha256[i]),
            n_nodes=int(self.inst_n_nodes[i]),
            capacity=int(self.inst_capacity[i]),
            k_fleet=int(self.inst_k_fleet[i]),
            k_label=int(self.inst_k_label[i]),
            termination=TERMINATION_NAMES.get(code, f"unknown({code})"),
            coords=self.inst_coords[lo:hi],
            demand=self.inst_demand[lo:hi],
        )

    def state_rows(self, i: int) -> range:
        i = int(i)
        rows = np.flatnonzero(self.st_inst_row == i)
        if rows.size == 0:
            return range(0, 0)
        return range(int(rows[0]), int(rows[-1]) + 1)

    def state(self, j: int) -> StateView:
        j = int(j)
        row = int(self.st_inst_row[j])
        inst = self.instance(row)
        s_lo, s_hi = int(self.sup_offsets[j]), int(self.sup_offsets[j + 1])
        l_lo, l_hi = int(self.lab_offsets[j]), int(self.lab_offsets[j + 1])
        m_lo, m_hi = int(self.m_offsets[j]), int(self.m_offsets[j + 1])
        s_hat = self.lab_s_hat[l_lo:l_hi]
        if s_hat.size == inst.n_nodes * inst.k_label:
            s_hat = s_hat.reshape(inst.n_nodes, inst.k_label)
        return StateView(
            instance=inst,
            inst_row=row,
            iteration=int(self.st_iteration[j]),
            lp_value=float(self.st_lp_value[j]),
            cuts_added=int(self.st_cuts_added[j]),
            terminal=bool(self.st_terminal[j]),
            support_u=self.sup_u[s_lo:s_hi],
            support_v=self.sup_v[s_lo:s_hi],
            support_x=self.sup_x[s_lo:s_hi],
            s_hat=s_hat,
            objective=self.m_objective[m_lo:m_hi],
            rhs=self.m_rhs[m_lo:m_hi],
            slot_violated=self.m_slot_violated[m_lo:m_hi],
        )

    def state_record(self, j: int) -> StateRecord:
        view = self.state(j)
        return StateRecord(
            iteration=view.iteration,
            lp_value=view.lp_value,
            cuts_added=view.cuts_added,
            terminal=view.terminal,
            support_u=view.support_u,
            support_v=view.support_v,
            support_x=view.support_x,
            s_hat=view.s_hat,
            objective=view.objective,
            rhs=view.rhs,
            slot_violated=view.slot_violated,
        )

    def to_arrays(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {
            "schema_version": np.asarray(self.schema_version, dtype=np.int32),
            "feature_contract": np.asarray(self.feature_contract, dtype=np.str_),
        }
        for key in SHARD_KEYS:
            if key not in out:
                out[key] = getattr(self, key)
        return {key: out[key] for key in SHARD_KEYS}

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> "ShardData":
        missing = [key for key in SHARD_KEYS if key not in arrays]
        if missing:
            raise ValueError(f"shard is missing key(s): {', '.join(missing)}")
        fields = {key: arrays[key] for key in SHARD_KEYS if key not in ("schema_version", "feature_contract")}
        return cls(
            schema_version=int(np.asarray(arrays["schema_version"]).reshape(()).item()),
            feature_contract=str(np.asarray(arrays["feature_contract"]).reshape(()).item()),
            **fields,
        )


# --------------------------------------------------------------------------
# builder
# --------------------------------------------------------------------------


@dataclass
class _PendingInstance:
    record: InstanceRecord
    states: list[StateRecord] = field(default_factory=list)


class ShardBuilder:
    def __init__(self) -> None:
        self._instances: list[_PendingInstance] = []
        self._ids: set[str] = set()

    def add_instance(self, rec: InstanceRecord) -> int:
        if not isinstance(rec, InstanceRecord):
            raise TypeError(f"add_instance: expected InstanceRecord, got {type(rec).__name__}")
        name = str(rec.id)
        if not name:
            raise ValueError("add_instance: id must be a non-empty string")
        if name in self._ids:
            raise ValueError(f"add_instance: duplicate instance id {name!r}")
        sha = str(rec.source_sha256)
        if not is_sha256_hex(sha):
            raise ValueError(
                f"add_instance[{name}]: source_sha256 must be 64 lowercase hex characters, got {sha!r}"
            )

        n_nodes = int(rec.n_nodes)
        if n_nodes < 2:
            raise ValueError(f"add_instance[{name}]: n_nodes must be >= 2 (depot + customers), got {n_nodes}")
        capacity = int(rec.capacity)
        if capacity <= 0:
            raise ValueError(f"add_instance[{name}]: capacity must be positive, got {capacity}")
        k_fleet = int(rec.k_fleet)
        if k_fleet < 1:
            raise ValueError(f"add_instance[{name}]: k_fleet must be >= 1, got {k_fleet}")
        termination = str(rec.termination)
        if termination not in TERMINATION_CODES:
            raise ValueError(
                f"add_instance[{name}]: termination must be one of "
                f"{sorted(TERMINATION_CODES)}, got {termination!r}"
            )

        coords = _as_array(rec.coords, "float64", f"add_instance[{name}].coords")
        _check_shape(coords, (n_nodes, 2), f"add_instance[{name}].coords")
        _check_finite(coords, f"add_instance[{name}].coords")

        demand = _as_array(rec.demand, "int64", f"add_instance[{name}].demand")
        _check_shape(demand, (n_nodes,), f"add_instance[{name}].demand")
        if int(demand[0]) != 0:
            raise ValueError(f"add_instance[{name}]: depot demand must be 0, got {int(demand[0])}")
        if int(demand.min()) < 0:
            raise ValueError(f"add_instance[{name}]: demand must be non-negative, got min {int(demand.min())}")
        total_demand = int(demand.sum())
        if total_demand <= 0:
            raise ValueError(f"add_instance[{name}]: total customer demand must be positive, got {total_demand}")

        k_label = int(rec.k_label)
        expected = ceil_div(total_demand, capacity)
        if k_label != expected:
            raise ValueError(
                f"add_instance[{name}]: k_label must equal ceil(D / Q) = ceil({total_demand} / {capacity}) "
                f"= {expected}, got {k_label}"
            )

        stored = InstanceRecord(
            id=name,
            source_sha256=sha,
            n_nodes=n_nodes,
            capacity=capacity,
            k_fleet=k_fleet,
            k_label=k_label,
            termination=termination,
            coords=coords,
            demand=demand,
        )
        self._instances.append(_PendingInstance(record=stored))
        self._ids.add(name)
        return len(self._instances) - 1

    def add_state(self, inst_row: int, st: StateRecord) -> None:
        if not isinstance(st, StateRecord):
            raise TypeError(f"add_state: expected StateRecord, got {type(st).__name__}")
        row = int(inst_row)
        if not 0 <= row < len(self._instances):
            raise ValueError(
                f"add_state: inst_row {row} out of range (0 .. {len(self._instances) - 1})"
            )
        pending = self._instances[row]
        rec = pending.record
        tag = f"add_state[{rec.id}]"
        n_nodes = rec.n_nodes
        k_label = rec.k_label

        if pending.states and pending.states[-1].terminal:
            raise ValueError(f"{tag}: instance already has a terminal state; no state may follow it")
        expected_iter = len(pending.states) + 1
        iteration = int(st.iteration)
        if iteration != expected_iter:
            raise ValueError(
                f"{tag}: iteration must be {expected_iter} (1-based, contiguous), got {iteration}"
            )

        lp_value = float(st.lp_value)
        if not np.isfinite(lp_value):
            raise ValueError(f"{tag}: lp_value must be finite, got {lp_value}")
        cuts_added = int(st.cuts_added)
        if cuts_added < 0:
            raise ValueError(f"{tag}: cuts_added must be non-negative, got {cuts_added}")

        support_u = _as_array(st.support_u, "int32", f"{tag}.support_u")
        support_v = _as_array(st.support_v, "int32", f"{tag}.support_v")
        support_x = _as_array(st.support_x, "float64", f"{tag}.support_x")
        if support_u.ndim != 1 or support_v.ndim != 1 or support_x.ndim != 1:
            raise ValueError(f"{tag}: support arrays must be 1-d")
        n_edges = support_u.shape[0]
        if support_v.shape[0] != n_edges or support_x.shape[0] != n_edges:
            raise ValueError(
                f"{tag}: support arrays must have equal length, got "
                f"u={support_u.shape[0]}, v={support_v.shape[0]}, x={support_x.shape[0]}"
            )
        if n_edges:
            if int(support_u.min()) < 0:
                raise ValueError(f"{tag}: support_u must be >= 0, got min {int(support_u.min())}")
            if int(support_v.max()) >= n_nodes:
                raise ValueError(
                    f"{tag}: support_v must be < n_nodes ({n_nodes}), got max {int(support_v.max())}"
                )
            bad = int(np.count_nonzero(support_u >= support_v))
            if bad:
                raise ValueError(f"{tag}: support requires u < v, violated by {bad} edge(s)")
            key = support_u.astype(np.int64) * n_nodes + support_v.astype(np.int64)
            if np.any(np.diff(key) <= 0):
                raise ValueError(
                    f"{tag}: support edges must be strictly increasing in lexicographic (u, v) order"
                )
            _check_finite(support_x, f"{tag}.support_x")
            if float(support_x.min()) <= 0.0:
                raise ValueError(
                    f"{tag}: support requires x > 0, got min {float(support_x.min())}"
                )

        s_hat = _as_array(st.s_hat, "uint8", f"{tag}.s_hat")
        _check_shape(s_hat, (n_nodes, k_label), f"{tag}.s_hat")
        if s_hat.size and int(s_hat.max()) > 1:
            raise ValueError(f"{tag}: s_hat must be 0/1, got max {int(s_hat.max())}")
        if int(s_hat[0].max(initial=0)) != 0:
            raise ValueError(f"{tag}: s_hat depot row (node 0) must be all zeros")

        objective = _as_array(st.objective, "float64", f"{tag}.objective")
        _check_shape(objective, (k_label,), f"{tag}.objective")
        _check_finite(objective, f"{tag}.objective")
        rhs = _as_array(st.rhs, "int32", f"{tag}.rhs")
        _check_shape(rhs, (k_label,), f"{tag}.rhs")
        if k_label and int(rhs.min()) < 0:
            raise ValueError(f"{tag}: rhs must be non-negative, got min {int(rhs.min())}")
        slot_violated = _as_array(st.slot_violated, "uint8", f"{tag}.slot_violated")
        _check_shape(slot_violated, (k_label,), f"{tag}.slot_violated")
        if k_label and int(slot_violated.max()) > 1:
            raise ValueError(
                f"{tag}: slot_violated must be 0/1, got max {int(slot_violated.max())}"
            )

        pending.states.append(
            StateRecord(
                iteration=iteration,
                lp_value=lp_value,
                cuts_added=cuts_added,
                terminal=bool(st.terminal),
                support_u=support_u,
                support_v=support_v,
                support_x=support_x,
                s_hat=s_hat,
                objective=objective,
                rhs=rhs,
                slot_violated=slot_violated,
            )
        )

    @property
    def n_instances(self) -> int:
        return len(self._instances)

    @property
    def n_states(self) -> int:
        return sum(len(pending.states) for pending in self._instances)

    def build(self) -> ShardData:
        if not self._instances:
            raise ValueError("build: a shard must hold at least one instance")
        for row, pending in enumerate(self._instances):
            if not pending.states:
                raise ValueError(
                    f"build: instance {row} ({pending.record.id!r}) has no state; drop it instead"
                )
            if not pending.states[-1].terminal:
                raise ValueError(
                    f"build: instance {row} ({pending.record.id!r}) has no terminal state; "
                    "the last state of a trajectory must set terminal=True"
                )

        records = [pending.record for pending in self._instances]
        node_sizes = [rec.n_nodes for rec in records]

        st_inst_row: list[int] = []
        st_iteration: list[int] = []
        st_lp_value: list[float] = []
        st_cuts_added: list[int] = []
        st_terminal: list[int] = []
        sup_sizes: list[int] = []
        lab_sizes: list[int] = []
        m_sizes: list[int] = []
        sup_u_parts: list[np.ndarray] = []
        sup_v_parts: list[np.ndarray] = []
        sup_x_parts: list[np.ndarray] = []
        lab_parts: list[np.ndarray] = []
        obj_parts: list[np.ndarray] = []
        rhs_parts: list[np.ndarray] = []
        viol_parts: list[np.ndarray] = []

        for row, pending in enumerate(self._instances):
            for state in pending.states:
                st_inst_row.append(row)
                st_iteration.append(state.iteration)
                st_lp_value.append(state.lp_value)
                st_cuts_added.append(state.cuts_added)
                st_terminal.append(1 if state.terminal else 0)
                sup_sizes.append(int(state.support_u.shape[0]))
                lab_sizes.append(int(state.s_hat.size))
                m_sizes.append(int(state.objective.shape[0]))
                sup_u_parts.append(state.support_u)
                sup_v_parts.append(state.support_v)
                sup_x_parts.append(state.support_x)
                lab_parts.append(state.s_hat.reshape(-1))
                obj_parts.append(state.objective)
                rhs_parts.append(state.rhs)
                viol_parts.append(state.slot_violated)

        return ShardData(
            inst_id=np.asarray([rec.id for rec in records], dtype=np.str_),
            inst_source_sha256=np.asarray([rec.source_sha256 for rec in records], dtype=np.str_),
            inst_n_nodes=np.asarray([rec.n_nodes for rec in records], dtype=np.int32),
            inst_capacity=np.asarray([rec.capacity for rec in records], dtype=np.int64),
            inst_k_fleet=np.asarray([rec.k_fleet for rec in records], dtype=np.int32),
            inst_k_label=np.asarray([rec.k_label for rec in records], dtype=np.int32),
            inst_termination=np.asarray(
                [TERMINATION_CODES[rec.termination] for rec in records], dtype=np.int8
            ),
            inst_node_offsets=_offsets(node_sizes),
            inst_coords=_concat([rec.coords for rec in records], "float64", width=2),
            inst_demand=_concat([rec.demand for rec in records], "int64"),
            st_inst_row=np.asarray(st_inst_row, dtype=np.int32),
            st_iteration=np.asarray(st_iteration, dtype=np.int32),
            st_lp_value=np.asarray(st_lp_value, dtype=np.float64),
            st_cuts_added=np.asarray(st_cuts_added, dtype=np.int32),
            st_terminal=np.asarray(st_terminal, dtype=np.uint8),
            sup_offsets=_offsets(sup_sizes),
            sup_u=_concat(sup_u_parts, "int32"),
            sup_v=_concat(sup_v_parts, "int32"),
            sup_x=_concat(sup_x_parts, "float64"),
            lab_offsets=_offsets(lab_sizes),
            lab_s_hat=_concat(lab_parts, "uint8"),
            m_offsets=_offsets(m_sizes),
            m_objective=_concat(obj_parts, "float64"),
            m_rhs=_concat(rhs_parts, "int32"),
            m_slot_violated=_concat(viol_parts, "uint8"),
        )


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------


def write_shard(path: str | os.PathLike[str], data: ShardData) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = data.to_arrays()
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=target.name + ".", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _check_header(arrays: dict[str, np.ndarray], path: Path) -> None:
    for key in ("schema_version", "feature_contract"):
        if key not in arrays:
            raise ValueError(f"{path}: not a SAG-Sep shard, key {key!r} is missing")
    version = int(np.asarray(arrays["schema_version"]).reshape(()).item())
    if version != SHARD_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: shard schema_version {version} is not supported "
            f"(this build reads version {SHARD_SCHEMA_VERSION})"
        )
    contract = str(np.asarray(arrays["feature_contract"]).reshape(()).item())
    if contract != FEATURE_CONTRACT:
        raise ValueError(
            f"{path}: shard feature_contract {contract!r} does not match "
            f"{FEATURE_CONTRACT!r}; the features would change meaning"
        )


def read_shard(path: str | os.PathLike[str]) -> ShardData:
    src = Path(path)
    arrays = _load_npz(src)
    _check_header(arrays, src)
    missing = [key for key in SHARD_KEYS if key not in arrays]
    if missing:
        raise ValueError(f"{src}: shard is missing key(s): {', '.join(missing)}")
    return ShardData.from_arrays(arrays)


def read_shard_counts(path: str | os.PathLike[str]) -> tuple[int, int]:
    src = Path(path)
    with np.load(src, allow_pickle=False) as archive:
        header = {key: archive[key] for key in ("schema_version", "feature_contract") if key in archive.files}
        _check_header(header, src)
        for key in ("inst_node_offsets", "sup_offsets"):
            if key not in archive.files:
                raise ValueError(f"{src}: shard is missing key(s): {key}")
        n_instances = int(archive["inst_node_offsets"].shape[0]) - 1
        n_states = int(archive["sup_offsets"].shape[0]) - 1
    return n_instances, n_states


def merge_shards(
    inputs: Sequence[str | os.PathLike[str]],
    out_dir: str | os.PathLike[str],
    *,
    shard_size: int,
    prefix: str = "shard",
) -> list[Path]:
    if int(shard_size) < 1:
        raise ValueError(f"merge_shards: shard_size must be >= 1, got {shard_size}")
    shard_size = int(shard_size)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    builder = ShardBuilder()
    pending = 0

    def flush() -> None:
        nonlocal builder, pending
        out_path = destination / f"{prefix}_{len(written) + 1:04d}.npz"
        write_shard(out_path, builder.build())
        written.append(out_path)
        builder = ShardBuilder()
        pending = 0

    for source in inputs:
        data = read_shard(source)
        for i in range(data.n_instances):
            row = builder.add_instance(data.instance(i))
            for j in data.state_rows(i):
                builder.add_state(row, data.state_record(j))
            pending += 1
            if pending == shard_size:
                flush()
    if pending:
        flush()
    return written
