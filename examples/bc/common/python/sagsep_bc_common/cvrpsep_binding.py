from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from sagsep.types import Instance

__all__ = ["CvrpsepSeparator", "find_library", "DEFAULT_LIBRARY_PATH"]

DEFAULT_LIBRARY_PATH = (
    Path(__file__).resolve().parents[4] / "third_party" / "cvrpsep" / "libcvrpsep.so"
)

LIBRARY_ENV_VAR = "SAGSEP_BC_CVRPSEP_LIB"


class _CnstrRecord(ctypes.Structure):
    _fields_ = [
        ("CType", ctypes.c_int),
        ("Key", ctypes.c_int),
        ("IntListSize", ctypes.c_int),
        ("IntList", ctypes.POINTER(ctypes.c_int)),
        ("ExtListSize", ctypes.c_int),
        ("ExtList", ctypes.POINTER(ctypes.c_int)),
        ("CListSize", ctypes.c_int),
        ("CList", ctypes.POINTER(ctypes.c_int)),
        ("CoeffList", ctypes.POINTER(ctypes.c_double)),
        ("A", ctypes.c_int),
        ("B", ctypes.c_int),
        ("L", ctypes.c_int),
        ("RHS", ctypes.c_double),
        ("BranchLevel", ctypes.c_int),
        ("GlobalNr", ctypes.c_int),
    ]


class _CnstrMgrRecord(ctypes.Structure):
    _fields_ = [
        ("CPL", ctypes.POINTER(ctypes.POINTER(_CnstrRecord))),
        ("Dim", ctypes.c_int),
        ("Size", ctypes.c_int),
    ]


_CnstrMgrPtr = ctypes.POINTER(_CnstrMgrRecord)

_lib: Optional[ctypes.CDLL] = None
_lib_path: Optional[Path] = None


def find_library(explicit: Optional[str | Path] = None) -> Path:
    for candidate in (explicit, os.environ.get(LIBRARY_ENV_VAR), DEFAULT_LIBRARY_PATH):
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"libcvrpsep.so not found (looked at {DEFAULT_LIBRARY_PATH}). Build it with "
        f"'make -C {DEFAULT_LIBRARY_PATH.parent}' or set {LIBRARY_ENV_VAR}."
    )


def _load_library(path: Path) -> ctypes.CDLL:
    global _lib, _lib_path
    if _lib is not None and _lib_path == path:
        return _lib
    lib = ctypes.CDLL(str(path))
    lib.cvrpsep_capi_create_cmgr.argtypes = [ctypes.c_int]
    lib.cvrpsep_capi_create_cmgr.restype = _CnstrMgrPtr
    lib.cvrpsep_capi_free_cmgr.argtypes = [_CnstrMgrPtr]
    lib.cvrpsep_capi_free_cmgr.restype = None
    lib.cvrpsep_capi_move_cnstr.argtypes = [_CnstrMgrPtr, _CnstrMgrPtr, ctypes.c_int, ctypes.c_int]
    lib.cvrpsep_capi_move_cnstr.restype = None
    lib.cvrpsep_capi_separate_capacity_cuts.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        _CnstrMgrPtr,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_char),
        ctypes.POINTER(ctypes.c_double),
        _CnstrMgrPtr,
    ]
    lib.cvrpsep_capi_separate_capacity_cuts.restype = None
    _lib, _lib_path = lib, path
    return lib


class CvrpsepSeparator:
    def __init__(
        self,
        instance: Instance,
        *,
        max_cuts: int = 100,
        integrality_tolerance: float = 1e-6,
        min_violation: float = 1e-4,
        library_path: Optional[str | Path] = None,
    ) -> None:
        if max_cuts <= 0:
            raise ValueError(f"max_cuts must be positive, got {max_cuts}")
        self._lib = _load_library(find_library(library_path))
        self._n_nodes = int(instance.n_nodes)
        self._n_customers = self._n_nodes - 1
        self._capacity = float(instance.capacity)
        self._max_cuts = int(max_cuts)
        self._integrality_tolerance = float(integrality_tolerance)
        self._min_violation = float(min_violation)
        self._demand = np.ascontiguousarray(instance.demand, dtype=np.float64)
        self._old_cuts = self._lib.cvrpsep_capi_create_cmgr(self._max_cuts)
        if not self._old_cuts:
            raise RuntimeError("CVRPSEP could not allocate its constraint manager")
        self._pending: Optional[_CnstrMgrPtr] = None
        self._pending_records: List[int] = []

    @property
    def pool_size(self) -> int:
        return 0 if not self._old_cuts else int(self._old_cuts.contents.Size)

    def separate(
        self,
        support_u: np.ndarray,
        support_v: np.ndarray,
        support_x: np.ndarray,
    ) -> Tuple[List[List[int]], List[float], float]:
        if not self._old_cuts:
            raise RuntimeError("separator is closed")
        self._discard_pending()
        n_edges = int(support_u.size)
        if n_edges == 0:
            return [], [], 0.0

        depot_id = self._n_nodes
        tail = np.zeros(n_edges + 1, dtype=np.int32)
        head = np.zeros(n_edges + 1, dtype=np.int32)
        value = np.zeros(n_edges + 1, dtype=np.float64)
        tail[1:] = np.where(support_u == 0, depot_id, support_u)
        head[1:] = np.where(support_v == 0, depot_id, support_v)
        value[1:] = support_x

        new_cuts = self._lib.cvrpsep_capi_create_cmgr(self._max_cuts)
        if not new_cuts:
            raise RuntimeError("CVRPSEP could not allocate its cut manager")
        integer_and_feasible = ctypes.c_char(0)
        max_violation = ctypes.c_double(0.0)
        try:
            self._lib.cvrpsep_capi_separate_capacity_cuts(
                self._n_customers,
                self._demand.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                self._capacity,
                n_edges,
                tail.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                head.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                value.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                self._old_cuts,
                self._max_cuts,
                self._integrality_tolerance,
                self._min_violation,
                ctypes.byref(integer_and_feasible),
                ctypes.byref(max_violation),
                new_cuts,
            )
            subsets, rhs_values, records = self._collect(new_cuts)
        except BaseException:
            self._lib.cvrpsep_capi_free_cmgr(new_cuts)
            raise
        if subsets:
            self._pending = new_cuts
            self._pending_records = records
        else:
            self._lib.cvrpsep_capi_free_cmgr(new_cuts)
        return subsets, rhs_values, float(max_violation.value)

    def commit(self, accepted_indices: Iterable[int]) -> None:
        pending = self._pending
        if pending is None:
            return
        try:
            for index in accepted_indices:
                record_index = self._pending_records[index]
                if not pending.contents.CPL[record_index]:
                    continue
                self._lib.cvrpsep_capi_move_cnstr(
                    pending, self._old_cuts, record_index, 0
                )
        finally:
            self._pending = None
            self._pending_records = []
            self._lib.cvrpsep_capi_free_cmgr(pending)

    def _discard_pending(self) -> None:
        if self._pending is not None:
            self._lib.cvrpsep_capi_free_cmgr(self._pending)
            self._pending = None
            self._pending_records = []

    def _collect(
        self, cuts: _CnstrMgrPtr
    ) -> Tuple[List[List[int]], List[float], List[int]]:
        subsets: List[List[int]] = []
        rhs_values: List[float] = []
        records: List[int] = []
        for cut_idx in range(int(cuts.contents.Size)):
            record = cuts.contents.CPL[cut_idx]
            if not record:
                continue
            entry = record.contents
            nodes: List[int] = []
            seen: set[int] = set()
            for list_idx in range(1, int(entry.IntListSize) + 1):
                token = int(entry.IntList[list_idx])
                if token <= 0 or token >= self._n_nodes:
                    raise ValueError(
                        f"cvrpsep record {cut_idx} contains node token {token}, "
                        f"which is neither a customer id in [1, {self._n_nodes}) "
                        f"nor anything this driver can use (the depot is "
                        f"{self._n_nodes})"
                    )
                if token in seen:
                    raise ValueError(
                        f"cvrpsep record {cut_idx} lists customer {token} twice"
                    )
                seen.add(token)
                nodes.append(token + 1)
            if not nodes:
                raise ValueError(
                    f"cvrpsep record {cut_idx} proposes an empty customer set"
                )
            subsets.append(sorted(nodes))
            rhs_values.append(max(1.0, float(len(nodes)) - float(entry.RHS)))
            records.append(cut_idx)
        return subsets, rhs_values, records

    def close(self) -> None:
        if getattr(self, "_pending", None) is not None:
            self._discard_pending()
        if getattr(self, "_old_cuts", None):
            self._lib.cvrpsep_capi_free_cmgr(self._old_cuts)
            self._old_cuts = None

    def __enter__(self) -> "CvrpsepSeparator":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
