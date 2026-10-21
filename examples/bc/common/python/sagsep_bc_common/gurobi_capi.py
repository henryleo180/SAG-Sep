from __future__ import annotations

import ctypes
import ctypes.util
import os
import traceback
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

__all__ = [
    "GurobiEnv",
    "BranchCutModel",
    "CallbackContext",
    "GRB_OPTIMAL",
    "GRB_SUBOPTIMAL",
    "GRB_TIME_LIMIT",
    "GRB_CB_MIPNODE",
    "GRB_CB_MIPSOL",
    "GRB_CB_MIPNODE_STATUS",
    "GRB_CB_MIPNODE_NODCNT",
    "STATUS_NAMES",
]

# --- model / status constants (gurobi_c.h) ---------------------------------

GRB_MINIMIZE = 1

GRB_OPTIMAL = 2
GRB_INFEASIBLE = 3
GRB_INF_OR_UNBD = 4
GRB_UNBOUNDED = 5
GRB_CUTOFF = 6
GRB_ITERATION_LIMIT = 7
GRB_NODE_LIMIT = 8
GRB_TIME_LIMIT = 9
GRB_SOLUTION_LIMIT = 10
GRB_INTERRUPTED = 11
GRB_NUMERIC = 12
GRB_SUBOPTIMAL = 13
GRB_USER_OBJ_LIMIT = 15
GRB_WORK_LIMIT = 16
GRB_MEM_LIMIT = 17

STATUS_NAMES = {
    GRB_OPTIMAL: "OPTIMAL",
    GRB_INFEASIBLE: "INFEASIBLE",
    GRB_INF_OR_UNBD: "INF_OR_UNBD",
    GRB_UNBOUNDED: "UNBOUNDED",
    GRB_CUTOFF: "CUTOFF",
    GRB_ITERATION_LIMIT: "ITERATION_LIMIT",
    GRB_NODE_LIMIT: "NODE_LIMIT",
    GRB_TIME_LIMIT: "TIME_LIMIT",
    GRB_SOLUTION_LIMIT: "SOLUTION_LIMIT",
    GRB_INTERRUPTED: "INTERRUPTED",
    GRB_NUMERIC: "NUMERIC",
    GRB_SUBOPTIMAL: "SUBOPTIMAL",
    GRB_USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
    GRB_WORK_LIMIT: "WORK_LIMIT",
    GRB_MEM_LIMIT: "MEM_LIMIT",
}

# --- callback constants (gurobi_c.h) ---------------------------------------

GRB_CB_MIPSOL = 4
GRB_CB_MIPNODE = 5

GRB_CB_MIPSOL_SOL = 4001
GRB_CB_MIPSOL_OBJBND = 4004
GRB_CB_MIPSOL_NODCNT = 4005

GRB_CB_MIPNODE_STATUS = 5001
GRB_CB_MIPNODE_REL = 5002
GRB_CB_MIPNODE_OBJBND = 5004
GRB_CB_MIPNODE_NODCNT = 5005

_CB_FUNCTYPE = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
)

_lib: Optional[ctypes.CDLL] = None


def _find_lib() -> ctypes.CDLL:
    candidates = []
    home = os.environ.get("GUROBI_HOME", "")
    if home:
        candidates.append(Path(home) / "lib")
    candidates.extend(
        Path(entry) for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":") if entry
    )
    for lib_dir in candidates:
        if not lib_dir.is_dir():
            continue
        for path in sorted(lib_dir.glob("libgurobi[0-9]*.so"), reverse=True):
            if "c++" in path.name or "_light" in path.name:
                continue
            try:
                return ctypes.CDLL(str(path))
            except OSError:
                continue
    name = ctypes.util.find_library("gurobi")
    if name:
        return ctypes.CDLL(name)
    raise RuntimeError(
        "Cannot locate the Gurobi shared library. Set GUROBI_HOME (so that "
        "$GUROBI_HOME/lib/libgurobi*.so exists) or add its directory to "
        "LD_LIBRARY_PATH."
    )


def _declare(lib: ctypes.CDLL) -> None:
    VP = ctypes.c_void_p
    IP = ctypes.POINTER(ctypes.c_int)
    DP = ctypes.POINTER(ctypes.c_double)
    CP = ctypes.c_char_p
    BP = ctypes.POINTER(ctypes.c_char)

    def sig(fn, argtypes, restype=ctypes.c_int) -> None:
        fn.argtypes = argtypes
        fn.restype = restype

    sig(lib.GRBloadenv, [ctypes.POINTER(VP), CP])
    sig(lib.GRBfreeenv, [VP], None)
    sig(lib.GRBnewmodel, [VP, ctypes.POINTER(VP), CP, ctypes.c_int, VP, VP, VP, VP, VP])
    sig(lib.GRBfreemodel, [VP])
    sig(lib.GRBupdatemodel, [VP])
    sig(lib.GRBoptimize, [VP])
    sig(lib.GRBgetenv, [VP], VP)
    sig(lib.GRBsetintparam, [VP, CP, ctypes.c_int])
    sig(lib.GRBsetdblparam, [VP, CP, ctypes.c_double])
    sig(lib.GRBsetintattr, [VP, CP, ctypes.c_int])
    sig(lib.GRBgetintattr, [VP, CP, IP])
    sig(lib.GRBgetdblattr, [VP, CP, DP])
    sig(lib.GRBgetdblattrarray, [VP, CP, ctypes.c_int, ctypes.c_int, DP])
    sig(lib.GRBsetcharattrarray, [VP, CP, ctypes.c_int, ctypes.c_int, BP])
    sig(lib.GRBaddvars, [VP, ctypes.c_int, ctypes.c_int, IP, IP, DP, DP, DP, DP, BP, VP])
    sig(lib.GRBaddconstr, [VP, ctypes.c_int, IP, DP, ctypes.c_char, ctypes.c_double, CP])
    sig(lib.GRBaddconstrs, [VP, ctypes.c_int, ctypes.c_int, IP, IP, DP, BP, DP, VP])
    sig(lib.GRBsetcallbackfunc, [VP, _CB_FUNCTYPE, VP])
    sig(lib.GRBcbget, [VP, ctypes.c_int, ctypes.c_int, VP])
    sig(lib.GRBcbcut, [VP, ctypes.c_int, IP, DP, ctypes.c_char, ctypes.c_double])
    sig(lib.GRBcblazy, [VP, ctypes.c_int, IP, DP, ctypes.c_char, ctypes.c_double])
    sig(lib.GRBversion, [IP, IP, IP], None)


def _get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        lib = _find_lib()
        _declare(lib)
        _lib = lib
    return _lib


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"Gurobi C API error {rc} in {what}")


def _dptr(arr: Optional[np.ndarray]):
    return None if arr is None else arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _iptr(arr: Optional[np.ndarray]):
    return None if arr is None else arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


def version() -> tuple[int, int, int]:
    lib = _get_lib()
    major, minor, tech = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    lib.GRBversion(ctypes.byref(major), ctypes.byref(minor), ctypes.byref(tech))
    return int(major.value), int(minor.value), int(tech.value)


class GurobiEnv:
    def __init__(self) -> None:
        self._lib = _get_lib()
        self._ptr: Optional[ctypes.c_void_p] = ctypes.c_void_p()
        _check(self._lib.GRBloadenv(ctypes.byref(self._ptr), None), "GRBloadenv")

    @property
    def ptr(self) -> ctypes.c_void_p:
        if self._ptr is None:
            raise RuntimeError("Gurobi environment already disposed")
        return self._ptr

    def dispose(self) -> None:
        if self._ptr is not None:
            self._lib.GRBfreeenv(self._ptr)
            self._ptr = None

    def __enter__(self) -> "GurobiEnv":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.dispose()

    def __del__(self) -> None:
        self.dispose()


class CallbackContext:
    __slots__ = ("_lib", "_cbdata", "where", "_n_vars")

    def __init__(
        self,
        lib: ctypes.CDLL,
        cbdata: ctypes.c_void_p,
        where: int,
        n_vars: int,
    ) -> None:
        self._lib = lib
        self._cbdata = cbdata
        self.where = where
        self._n_vars = n_vars

    def get_node_relaxation(self) -> np.ndarray:
        buf = np.empty(self._n_vars, dtype=np.float64)
        _check(
            self._lib.GRBcbget(self._cbdata, self.where, GRB_CB_MIPNODE_REL, _dptr(buf)),
            "cbget MIPNODE_REL",
        )
        return buf

    def get_incumbent(self) -> np.ndarray:
        buf = np.empty(self._n_vars, dtype=np.float64)
        _check(
            self._lib.GRBcbget(self._cbdata, self.where, GRB_CB_MIPSOL_SOL, _dptr(buf)),
            "cbget MIPSOL_SOL",
        )
        return buf

    def get_int(self, what: int) -> int:
        value = ctypes.c_int()
        _check(
            self._lib.GRBcbget(self._cbdata, self.where, what, ctypes.byref(value)),
            f"cbget int {what}",
        )
        return value.value

    def get_double(self, what: int) -> float:
        value = ctypes.c_double()
        _check(
            self._lib.GRBcbget(self._cbdata, self.where, what, ctypes.byref(value)),
            f"cbget double {what}",
        )
        return value.value

    def add_cut(
        self,
        indices: Sequence[int] | np.ndarray,
        coeffs: Sequence[float] | np.ndarray,
        sense: bytes,
        rhs: float,
    ) -> None:
        cols = np.ascontiguousarray(indices, dtype=np.int32)
        vals = np.ascontiguousarray(coeffs, dtype=np.float64)
        if cols.size == 0:
            return
        _check(
            self._lib.GRBcbcut(self._cbdata, int(cols.size), _iptr(cols), _dptr(vals),
                               sense, float(rhs)),
            "cbcut",
        )

    def add_lazy(
        self,
        indices: Sequence[int] | np.ndarray,
        coeffs: Sequence[float] | np.ndarray,
        sense: bytes,
        rhs: float,
    ) -> None:
        cols = np.ascontiguousarray(indices, dtype=np.int32)
        vals = np.ascontiguousarray(coeffs, dtype=np.float64)
        if cols.size == 0:
            return
        _check(
            self._lib.GRBcblazy(self._cbdata, int(cols.size), _iptr(cols), _dptr(vals),
                                sense, float(rhs)),
            "cblazy",
        )


class BranchCutModel:
    def __init__(self, name: str, *, env: GurobiEnv) -> None:
        lib = _get_lib()
        ptr = ctypes.c_void_p()
        _check(
            lib.GRBnewmodel(env.ptr, ctypes.byref(ptr), name.encode(), 0,
                            None, None, None, None, None),
            "GRBnewmodel",
        )
        self._lib = lib
        self._ptr: Optional[ctypes.c_void_p] = ptr
        self._n_vars = 0
        self._callback_ref: Optional[ctypes.CFUNCTYPE] = None
        self._callback_error: Optional[BaseException] = None

    def set_param_int(self, name: str, value: int) -> None:
        env = self._lib.GRBgetenv(self._ptr)
        _check(self._lib.GRBsetintparam(env, name.encode(), int(value)), f"param {name}")

    def set_param_dbl(self, name: str, value: float) -> None:
        env = self._lib.GRBgetenv(self._ptr)
        _check(self._lib.GRBsetdblparam(env, name.encode(), float(value)), f"param {name}")

    @property
    def n_vars(self) -> int:
        return self._n_vars

    def add_vars(
        self,
        count: int,
        *,
        lb: np.ndarray,
        ub: np.ndarray,
        obj: np.ndarray,
        vtypes: np.ndarray,
    ) -> None:
        lb_a = np.ascontiguousarray(lb, dtype=np.float64)
        ub_a = np.ascontiguousarray(ub, dtype=np.float64)
        obj_a = np.ascontiguousarray(obj, dtype=np.float64)
        vt_a = np.ascontiguousarray(vtypes, dtype=np.uint8)
        _check(
            self._lib.GRBaddvars(
                self._ptr, int(count), 0, None, None, None,
                _dptr(obj_a), _dptr(lb_a), _dptr(ub_a),
                vt_a.ctypes.data_as(ctypes.POINTER(ctypes.c_char)), None,
            ),
            "GRBaddvars",
        )
        self._n_vars += int(count)

    def add_constr(
        self,
        indices: Sequence[int] | np.ndarray,
        coeffs: Sequence[float] | np.ndarray,
        sense: bytes,
        rhs: float,
        name: Optional[str] = None,
    ) -> None:
        cols = np.ascontiguousarray(indices, dtype=np.int32)
        vals = np.ascontiguousarray(coeffs, dtype=np.float64)
        if cols.size == 0:
            return
        _check(
            self._lib.GRBaddconstr(
                self._ptr, int(cols.size), _iptr(cols), _dptr(vals),
                sense, float(rhs), name.encode() if name else None,
            ),
            "GRBaddconstr",
        )

    def add_constrs_csr(
        self,
        indptr: np.ndarray,
        col_indices: np.ndarray,
        values: np.ndarray,
        sense: bytes,
        rhs: np.ndarray,
    ) -> None:
        n_rows = int(len(rhs))
        if n_rows == 0:
            return
        nnz = int(indptr[n_rows])
        begins = np.ascontiguousarray(indptr[:n_rows], dtype=np.int32)
        cols = np.ascontiguousarray(col_indices[:nnz], dtype=np.int32)
        vals = np.ascontiguousarray(values[:nnz], dtype=np.float64)
        rhs_a = np.ascontiguousarray(rhs, dtype=np.float64)
        senses = (ctypes.c_char * n_rows)(*(sense * n_rows))
        _check(
            self._lib.GRBaddconstrs(
                self._ptr, n_rows, nnz, _iptr(begins), _iptr(cols), _dptr(vals),
                senses, _dptr(rhs_a), None,
            ),
            "GRBaddconstrs",
        )

    def set_vtypes(self, first: int, vtypes: np.ndarray) -> None:
        vt_a = np.ascontiguousarray(vtypes, dtype=np.uint8)
        _check(
            self._lib.GRBsetcharattrarray(
                self._ptr, b"VType", int(first), int(vt_a.size),
                vt_a.ctypes.data_as(ctypes.POINTER(ctypes.c_char)),
            ),
            "set VType",
        )

    def set_minimize(self) -> None:
        _check(
            self._lib.GRBsetintattr(self._ptr, b"ModelSense", GRB_MINIMIZE),
            "ModelSense",
        )

    def update(self) -> None:
        _check(self._lib.GRBupdatemodel(self._ptr), "GRBupdatemodel")

    def set_callback(self, callback: Callable[[CallbackContext], None]) -> None:
        lib = self._lib
        n_vars = self._n_vars

        def _raw(_model_ptr, cbdata, where, _usrdata):
            try:
                callback(CallbackContext(lib, cbdata, where, n_vars))
            except BaseException as exc:
                if self._callback_error is None:
                    self._callback_error = exc
                traceback.print_exc()
            return 0

        self._callback_ref = _CB_FUNCTYPE(_raw)
        _check(
            self._lib.GRBsetcallbackfunc(self._ptr, self._callback_ref, None),
            "GRBsetcallbackfunc",
        )

    @property
    def callback_error(self) -> Optional[BaseException]:
        return self._callback_error

    def optimize(self) -> None:
        _check(self._lib.GRBoptimize(self._ptr), "GRBoptimize")

    def get_int_attr(self, name: str) -> int:
        value = ctypes.c_int()
        _check(
            self._lib.GRBgetintattr(self._ptr, name.encode(), ctypes.byref(value)),
            f"get {name}",
        )
        return value.value

    def get_dbl_attr(self, name: str) -> float:
        value = ctypes.c_double()
        _check(
            self._lib.GRBgetdblattr(self._ptr, name.encode(), ctypes.byref(value)),
            f"get {name}",
        )
        return value.value

    def try_get_dbl_attr(self, name: str) -> Optional[float]:
        try:
            return self.get_dbl_attr(name)
        except RuntimeError:
            return None

    @property
    def status(self) -> int:
        return self.get_int_attr("Status")

    def get_solution(self, first: int = 0, count: Optional[int] = None) -> np.ndarray:
        if count is None:
            count = self._n_vars - first
        buf = np.empty(int(count), dtype=np.float64)
        _check(
            self._lib.GRBgetdblattrarray(self._ptr, b"X", int(first), int(count), _dptr(buf)),
            "get X",
        )
        return buf

    def dispose(self) -> None:
        if self._ptr is not None:
            self._lib.GRBfreemodel(self._ptr)
            self._ptr = None
        self._callback_ref = None

    def __del__(self) -> None:
        self.dispose()
