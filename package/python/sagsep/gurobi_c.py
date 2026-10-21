from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

GRB_MINIMIZE = 1
GRB_OPTIMAL = 2
GRB_SUBOPTIMAL = 13

_lib: Optional[ctypes.CDLL] = None


def _find_lib() -> ctypes.CDLL:
    home = os.environ.get("GUROBI_HOME", "")
    if home:
        lib_dir = Path(home) / "lib"
        for p in sorted(lib_dir.glob("libgurobi[0-9]*.so"), reverse=True):
            if "c++" not in p.name and "_light" not in p.name:
                try:
                    return ctypes.CDLL(str(p))
                except OSError:
                    continue
    for d in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if not d:
            continue
        lib_dir = Path(d)
        if not lib_dir.is_dir():
            continue
        for p in sorted(lib_dir.glob("libgurobi[0-9]*.so"), reverse=True):
            if "c++" not in p.name and "_light" not in p.name:
                try:
                    return ctypes.CDLL(str(p))
                except OSError:
                    continue
    name = ctypes.util.find_library("gurobi")
    if name:
        return ctypes.CDLL(name)
    raise RuntimeError("Cannot locate Gurobi shared library. Set GUROBI_HOME or LD_LIBRARY_PATH.")


def _get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = _find_lib()
        _setup(_lib)
    return _lib


def _setup(lib: ctypes.CDLL) -> None:
    VP = ctypes.c_void_p
    IP = ctypes.POINTER(ctypes.c_int)
    DP = ctypes.POINTER(ctypes.c_double)
    CP = ctypes.c_char_p
    BP = ctypes.POINTER(ctypes.c_char)

    def s(fn, a, r=ctypes.c_int):
        fn.argtypes = a
        fn.restype = r

    s(lib.GRBloadenv, [ctypes.POINTER(VP), CP])
    s(lib.GRBfreeenv, [VP], None)
    s(lib.GRBnewmodel, [VP, ctypes.POINTER(VP), CP, ctypes.c_int, VP, VP, VP, VP, VP])
    s(lib.GRBfreemodel, [VP])
    s(lib.GRBupdatemodel, [VP])
    s(lib.GRBoptimize, [VP])
    s(lib.GRBgetenv, [VP], VP)
    s(lib.GRBsetintparam, [VP, CP, ctypes.c_int])
    s(lib.GRBsetintattr, [VP, CP, ctypes.c_int])
    s(lib.GRBgetintattr, [VP, CP, IP])
    s(lib.GRBgetdblattr, [VP, CP, DP])
    s(lib.GRBgetdblattrarray, [VP, CP, ctypes.c_int, ctypes.c_int, DP])
    s(lib.GRBaddvars, [VP, ctypes.c_int, ctypes.c_int, IP, IP, DP, DP, DP, DP, BP, VP])
    s(lib.GRBaddconstr, [VP, ctypes.c_int, IP, DP, ctypes.c_char, ctypes.c_double, CP])
    s(lib.GRBaddconstrs, [VP, ctypes.c_int, ctypes.c_int, IP, IP, DP, BP, DP, VP])
    s(lib.GRBdelconstrs, [VP, ctypes.c_int, IP])
    s(lib.GRBversion, [IP, IP, IP], None)


def _check(rc: int, msg: str = "") -> None:
    if rc != 0:
        raise RuntimeError(f"Gurobi C API error {rc}" + (f": {msg}" if msg else ""))


def _dptr(arr: Optional[np.ndarray]):
    if arr is None:
        return None
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _iptr(arr: Optional[np.ndarray]):
    if arr is None:
        return None
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


def version() -> tuple[int, int, int]:
    lib = _get_lib()
    major = ctypes.c_int()
    minor = ctypes.c_int()
    tech = ctypes.c_int()
    lib.GRBversion(ctypes.byref(major), ctypes.byref(minor), ctypes.byref(tech))
    return int(major.value), int(minor.value), int(tech.value)


class GurobiEnv:
    def __init__(self) -> None:
        self._lib = _get_lib()
        self._ptr = ctypes.c_void_p()
        _check(self._lib.GRBloadenv(ctypes.byref(self._ptr), None), "GRBloadenv")

    @property
    def ptr(self) -> ctypes.c_void_p:
        return self._ptr

    def dispose(self) -> None:
        if self._ptr is not None:
            self._lib.GRBfreeenv(self._ptr)
            self._ptr = None

    def __del__(self) -> None:
        self.dispose()

    def __enter__(self) -> "GurobiEnv":
        return self

    def __exit__(self, *args) -> None:
        self.dispose()


class GurobiModel:

    def __init__(self, name: str = "", *, env: GurobiEnv) -> None:
        lib = _get_lib()
        model = ctypes.c_void_p()
        _check(
            lib.GRBnewmodel(
                env.ptr, ctypes.byref(model), name.encode(), 0,
                None, None, None, None, None,
            ),
            "GRBnewmodel",
        )
        self._lib = lib
        self._ptr = model
        self._n = 0

    def set_param_int(self, name: str, value: int) -> None:
        env = self._lib.GRBgetenv(self._ptr)
        _check(self._lib.GRBsetintparam(env, name.encode(), int(value)), name)

    def add_vars(
        self,
        count: int,
        *,
        lb: Optional[np.ndarray] = None,
        ub: Optional[np.ndarray] = None,
        obj: Optional[np.ndarray] = None,
        vtype: str = "C",
    ) -> int:
        start = self._n
        lb_a = np.ascontiguousarray(lb, dtype=np.float64) if lb is not None else None
        ub_a = np.ascontiguousarray(ub, dtype=np.float64) if ub is not None else None
        obj_a = np.ascontiguousarray(obj, dtype=np.float64) if obj is not None else None
        vt_a = np.full(count, ord(vtype), dtype=np.uint8)
        _check(
            self._lib.GRBaddvars(
                self._ptr, count, 0, None, None, None,
                _dptr(obj_a), _dptr(lb_a), _dptr(ub_a),
                vt_a.ctypes.data_as(ctypes.POINTER(ctypes.c_char)),
                None,
            ),
            "GRBaddvars",
        )
        self._n += count
        return start

    _sense_cache: dict[str, bytes] = {}

    def add_constr(
        self,
        indices: Sequence[int] | np.ndarray,
        coeffs: Sequence[float] | np.ndarray,
        sense: str | bytes,
        rhs: float,
        name: Optional[str | bytes] = None,
    ) -> None:
        nz = len(indices)
        if nz == 0:
            return
        ci = np.ascontiguousarray(indices, dtype=np.int32)
        cv = np.ascontiguousarray(coeffs, dtype=np.float64)
        if isinstance(sense, bytes):
            sense_b = sense
        else:
            cache = GurobiModel._sense_cache
            sense_b = cache.get(sense)
            if sense_b is None:
                sense_b = sense.encode()
                cache[sense] = sense_b
        name_b = name.encode() if isinstance(name, str) else name
        _check(
            self._lib.GRBaddconstr(
                self._ptr, nz, _iptr(ci), _dptr(cv),
                sense_b, float(rhs), name_b,
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
        nr = len(rhs)
        if nr == 0:
            return
        nnz = int(indptr[nr])
        cb = np.ascontiguousarray(indptr[:nr], dtype=np.int32)
        ci = np.ascontiguousarray(col_indices[:nnz], dtype=np.int32)
        cv = np.ascontiguousarray(values[:nnz], dtype=np.float64)
        rh = np.ascontiguousarray(rhs, dtype=np.float64)
        sa = (ctypes.c_char * nr)(*(sense * nr))
        _check(
            self._lib.GRBaddconstrs(
                self._ptr, nr, nnz,
                _iptr(cb), _iptr(ci), _dptr(cv), sa, _dptr(rh), None,
            ),
            "GRBaddconstrs",
        )

    def del_constrs(self, indices: Sequence[int] | np.ndarray) -> None:
        n = len(indices)
        if n == 0:
            return
        arr = np.ascontiguousarray(indices, dtype=np.int32)
        _check(
            self._lib.GRBdelconstrs(self._ptr, int(n), _iptr(arr)),
            "GRBdelconstrs",
        )

    def get_constr_dbl_attr(self, attr: str | bytes, first: int, count: int) -> np.ndarray:
        if count <= 0:
            return np.empty(0, dtype=np.float64)
        attr_b = attr.encode() if isinstance(attr, str) else attr
        buf = np.empty(int(count), dtype=np.float64)
        _check(
            self._lib.GRBgetdblattrarray(
                self._ptr, attr_b, int(first), int(count), _dptr(buf),
            ),
            f"get {attr_b!r}",
        )
        return buf

    def update(self) -> None:
        _check(self._lib.GRBupdatemodel(self._ptr), "update")

    def optimize(self) -> None:
        _check(self._lib.GRBoptimize(self._ptr), "optimize")

    def set_minimize(self) -> None:
        _check(self._lib.GRBsetintattr(self._ptr, b"ModelSense", GRB_MINIMIZE), "ModelSense")

    @property
    def status(self) -> int:
        return self.get_int_attr("Status")

    @property
    def obj_val(self) -> float:
        return self.get_dbl_attr("ObjVal")

    @property
    def num_constrs(self) -> int:
        return self.get_int_attr("NumConstrs")

    def get_int_attr(self, attr: str | bytes) -> int:
        attr_b = attr.encode() if isinstance(attr, str) else attr
        v = ctypes.c_int()
        _check(self._lib.GRBgetintattr(self._ptr, attr_b, ctypes.byref(v)), f"get {attr_b!r}")
        return v.value

    def get_dbl_attr(self, attr: str | bytes) -> float:
        attr_b = attr.encode() if isinstance(attr, str) else attr
        v = ctypes.c_double()
        _check(self._lib.GRBgetdblattr(self._ptr, attr_b, ctypes.byref(v)), f"get {attr_b!r}")
        return v.value

    def get_solution(self, first: int = 0, count: Optional[int] = None) -> np.ndarray:
        if count is None:
            count = self._n - first
        buf = self._get_dbl_buffer(count)
        _check(self._lib.GRBgetdblattrarray(self._ptr, b"X", first, count, _dptr(buf)), "get X")
        return buf

    def _get_dbl_buffer(self, size: int) -> np.ndarray:
        buf = getattr(self, "_dbl_buf", None)
        if buf is None or buf.shape[0] < size:
            buf = np.empty(size, dtype=np.float64)
            self._dbl_buf = buf
        return buf[:size]

    def dispose(self) -> None:
        if self._ptr is not None:
            self._lib.GRBfreemodel(self._ptr)
            self._ptr = None

    def __del__(self) -> None:
        self.dispose()


__all__ = ["GurobiEnv", "GurobiModel", "GRB_MINIMIZE", "GRB_OPTIMAL", "GRB_SUBOPTIMAL",
           "version"]
