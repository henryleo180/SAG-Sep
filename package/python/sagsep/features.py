from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from sagsep.types import (
    CandidateContext,
    Instance,
    LPSupport,
    OldCutBankSnapshot,
    freeze_array,
)

__all__ = [
    "load_instance",
    "distance_matrix",
    "build_lp_support",
    "vehicle_replica_count",
    "GraphFeatures",
    "build_graph_features",
    "build_candidate_context",
]


# ---------------------------------------------------------------------------
# Frozen eval data config (the released checkpoint's config_snapshot.yaml, data section)
# ---------------------------------------------------------------------------

_PARTIAL_CONNECT: int = 2

_HOP_ORIGINAL: float = 1.0
_HOP_SHORTCUT: float = 2.0
_HOP_DEPOT_FILL: float = 0.0


# ---------------------------------------------------------------------------
# Instance loading (TSPLIB .vrp, EUC_2D only)
# ---------------------------------------------------------------------------


def load_instance(
    vrp_path: str | Path,
    *,
    vehicles: int,
    optimal: Optional[float] = None,
) -> Instance:
    path = Path(vrp_path)
    if not path.exists():
        raise FileNotFoundError(f"VRP file not found: {path}")

    name = path.stem
    dimension: Optional[int] = None
    capacity: Optional[float] = None
    edge_weight_type = "EUC_2D"

    coords: dict[int, Tuple[float, float]] = {}
    demand: dict[int, float] = {}
    depot = 1

    section: Optional[str] = None
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            upper = line.upper()
            if upper.startswith("NODE_COORD_SECTION"):
                section = "coords"
                continue
            if upper.startswith("DEMAND_SECTION"):
                section = "demand"
                continue
            if upper.startswith("DEPOT_SECTION"):
                section = "depot"
                continue
            if upper.startswith("EOF"):
                break

            if section == "coords":
                parts = line.split()
                if len(parts) < 3:
                    continue
                idx = int(parts[0])
                coords[idx] = (float(parts[1]), float(parts[2]))
            elif section == "demand":
                parts = line.split()
                if len(parts) < 2:
                    continue
                idx = int(parts[0])
                demand[idx] = float(parts[1])
            elif section == "depot":
                if line.startswith("-1"):
                    section = None
                    continue
                depot = int(line)
            else:
                if ":" in line:
                    key, value = [item.strip() for item in line.split(":", 1)]
                    key_upper = key.upper()
                    if key_upper == "NAME":
                        name = value
                    elif key_upper == "DIMENSION":
                        dimension = int(value)
                    elif key_upper == "CAPACITY":
                        capacity = float(value)
                    elif key_upper == "EDGE_WEIGHT_TYPE":
                        edge_weight_type = value

    if dimension is None or capacity is None:
        raise ValueError(f"Failed to parse dimension/capacity from {path}")

    if edge_weight_type.strip().upper() != "EUC_2D":
        raise ValueError(
            f"SAG-Sep only supports EDGE_WEIGHT_TYPE=EUC_2D, "
            f"got {edge_weight_type!r} in {path}"
        )
    if depot != 1:
        raise ValueError(
            f"SAG-Sep expects depot at external node 1, got {depot} in {path}"
        )

    coords_arr = np.zeros((dimension, 2), dtype=np.float64)
    demand_arr = np.zeros(dimension, dtype=np.int64)
    for idx in range(1, dimension + 1):
        if idx not in coords:
            raise ValueError(f"Missing coordinates for node {idx} in {path}")
        if idx not in demand:
            raise ValueError(f"Missing demand for node {idx} in {path}")
        coords_arr[idx - 1] = coords[idx]
        raw_demand = demand[idx]
        if not float(raw_demand).is_integer():
            raise ValueError(
                f"Non-integer demand {raw_demand} for node {idx} in {path} "
                f"(demands are carried as int64)"
            )
        demand_arr[idx - 1] = int(raw_demand)

    if not float(capacity).is_integer():
        raise ValueError(
            f"Non-integer capacity {capacity} in {path} (capacity is carried as int64)"
        )
    if demand_arr[0] != 0:
        raise ValueError(f"Depot demand must be 0, got {demand_arr[0]} in {path}")

    return Instance(
        name=name,
        n_nodes=int(dimension),
        depot=0,
        coords=coords_arr,
        demand=demand_arr,
        capacity=int(capacity),
        vehicles=int(vehicles),
        optimal=None if optimal is None else float(optimal),
    )


def distance_matrix(instance: Instance) -> np.ndarray:
    n = instance.n_nodes
    coords = instance.coords
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        xi = float(coords[i, 0])
        yi = float(coords[i, 1])
        for j in range(i + 1, n):
            dist = math.hypot(xi - float(coords[j, 0]), yi - float(coords[j, 1]))
            rounded = float(int(dist + 0.5))
            mat[i, j] = rounded
            mat[j, i] = rounded
    return mat


# ---------------------------------------------------------------------------
# LP support snapshot
# ---------------------------------------------------------------------------


def build_lp_support(
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_x: np.ndarray,
    *,
    n_nodes: int,
    threshold: float,
) -> LPSupport:
    u = np.asarray(edge_u, dtype=np.int64).reshape(-1)
    v = np.asarray(edge_v, dtype=np.int64).reshape(-1)
    val = np.asarray(edge_x, dtype=np.float32).reshape(-1)
    if u.shape != v.shape or u.shape != val.shape:
        raise ValueError("edge_u/edge_v/edge_x must have matching shapes")

    n = int(n_nodes)
    adj = np.zeros((n, n), dtype=np.float32)
    if u.size:
        keep = val >= float(threshold)
        if keep.any():
            ku = u[keep]
            kv = v[keep]
            kval = val[keep]
            valid = (ku != kv) & (ku >= 0) & (kv >= 0) & (ku < n) & (kv < n)
            ku = ku[valid]
            kv = kv[valid]
            kval = kval[valid]
            adj[ku, kv] = kval
            adj[kv, ku] = kval
    return LPSupport.from_dense(adj, threshold=float(threshold))


# ---------------------------------------------------------------------------
# Vehicle replica count
# ---------------------------------------------------------------------------


def vehicle_replica_count(instance: Instance) -> int:
    demand32 = instance.demand.astype(np.float32)
    total = np.float32(demand32.sum(dtype=np.float32))
    cap = np.float32(float(instance.capacity))
    cap = np.float32(max(cap, np.float32(1e-6)))
    k = int(np.ceil(total / cap))
    return max(k, 1)


# ---------------------------------------------------------------------------
# GraphFeatures (frozen model input; graph construction + eval collator chain)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphFeatures:
    n_nodes: int
    vehicle_count: int
    edge_index: np.ndarray
    edge_attr: np.ndarray
    node_attr: np.ndarray

    def __post_init__(self) -> None:
        ei = np.asarray(self.edge_index, dtype=np.int64)
        ea = np.asarray(self.edge_attr, dtype=np.float32)
        na = np.asarray(self.node_attr, dtype=np.float32)
        if ei.ndim != 2 or ei.shape[0] != 2:
            raise ValueError("edge_index must be (2, E)")
        if ei.shape[1] == 0:
            raise ValueError("graph must contain at least one edge")
        if ea.ndim != 2 or ea.shape[0] != ei.shape[1]:
            raise ValueError("edge_attr must be (E, edge_dim)")
        if na.ndim != 3 or na.shape[0] != self.vehicle_count or na.shape[1] != self.n_nodes:
            raise ValueError("node_attr must be (K, n_nodes, node_dim)")
        object.__setattr__(self, "edge_index", freeze_array(ei))
        object.__setattr__(self, "edge_attr", freeze_array(ea))
        object.__setattr__(self, "node_attr", freeze_array(na))

    @property
    def edge_count(self) -> int:
        return int(self.edge_index.shape[1])


def _two_hop_shortcuts(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    dense: np.ndarray,
    cross_depot: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    deg = np.diff(indptr).astype(np.int64)
    blk = deg * deg
    if not cross_depot and blk.size:
        blk[0] = 0
    total = int(blk.sum())
    if total == 0:
        empty_i = np.zeros(0, dtype=np.int64)
        return empty_i, empty_i.copy(), np.zeros(0, dtype=np.float32)

    n = int(dense.shape[0])
    centers = np.arange(n, dtype=np.int64)
    blk_starts = np.cumsum(blk) - blk
    pos = np.arange(total, dtype=np.int64) - np.repeat(blk_starts, blk)
    deg_b = np.repeat(deg, blk)
    i_rank = pos // deg_b
    j_rank = pos - i_rank * deg_b
    row_start = np.repeat(indptr[:-1].astype(np.int64), blk)
    s_nodes = indices[row_start + i_rank].astype(np.int64)
    c_nodes = indices[row_start + j_rank].astype(np.int64)
    b_nodes = np.repeat(centers, blk)
    w_sb = data[row_start + i_rank]
    w_bc = data[row_start + j_rank]
    vals = np.minimum(w_sb, w_bc)

    keep = (s_nodes != c_nodes) & (dense[s_nodes, c_nodes] == 0)
    if not cross_depot:
        keep &= (s_nodes == 0) | (c_nodes != 0)
    if not keep.any():
        empty_i = np.zeros(0, dtype=np.int64)
        return empty_i, empty_i.copy(), np.zeros(0, dtype=np.float32)
    s_nodes = s_nodes[keep]
    b_nodes = b_nodes[keep]
    c_nodes = c_nodes[keep]
    vals = vals[keep]

    order = np.lexsort((c_nodes, b_nodes, s_nodes))
    s_nodes = s_nodes[order]
    c_nodes = c_nodes[order]
    vals = vals[order]

    gid = s_nodes * np.int64(n) + c_nodes
    uniq, first_idx, inv = np.unique(gid, return_index=True, return_inverse=True)
    gmax = np.full(uniq.size, -np.inf, dtype=np.float32)
    np.maximum.at(gmax, inv, vals)

    emit = np.sort(first_idx)
    out_s = s_nodes[emit]
    out_c = c_nodes[emit]
    out_v = gmax[inv[emit]]
    return out_s, out_c, out_v


def build_graph_features(
    instance: Instance,
    lp: LPSupport,
    *,
    vehicle_count: int,
    support_edge_u: np.ndarray,
    support_edge_v: np.ndarray,
    support_edge_x: np.ndarray,
    cross_depot: bool = True,
) -> GraphFeatures:
    n = int(instance.n_nodes)
    if lp.n_nodes != n:
        raise ValueError(f"lp.n_nodes={lp.n_nodes} != instance.n_nodes={n}")
    k = int(vehicle_count)
    if k < 1:
        raise ValueError("vehicle_count must be >= 1")

    if lp.edge_u.size > 0:
        keep64 = lp.edge_x >= float(lp.threshold)
    else:
        keep64 = np.zeros(0, dtype=bool)

    if keep64.size and bool(keep64.any()):
        if bool(keep64.all()):
            pair_u = lp.edge_u
            pair_v = lp.edge_v
            pair_w = lp.edge_x.astype(np.float32)
            adj_dense = lp.dense
            csr_indptr = lp.indptr
            csr_indices = lp.indices
            csr_data = lp.data
        else:
            pair_u = lp.edge_u[keep64]
            pair_v = lp.edge_v[keep64]
            pair_w = lp.edge_x[keep64].astype(np.float32)
            adj_dense = np.zeros((n, n), dtype=np.float32)
            adj_dense[pair_u, pair_v] = pair_w
            adj_dense[pair_v, pair_u] = pair_w
            nz_r, nz_c = np.nonzero(adj_dense)
            counts = np.bincount(nz_r, minlength=n)
            csr_indptr = np.zeros(n + 1, dtype=np.int64)
            csr_indptr[1:] = np.cumsum(counts)
            csr_indices = nz_c.astype(np.int32)
            csr_data = adj_dense[nz_r, nz_c]
    else:
        raw_u = np.asarray(support_edge_u, dtype=np.int64).reshape(-1)
        raw_v = np.asarray(support_edge_v, dtype=np.int64).reshape(-1)
        raw_w = np.asarray(support_edge_x, dtype=np.float32).reshape(-1)
        no_loop = raw_u != raw_v
        raw_u, raw_v, raw_w = raw_u[no_loop], raw_v[no_loop], raw_w[no_loop]
        if raw_u.size == 0:
            raise ValueError("Graph must contain at least one edge")
        best = int(np.argmax(raw_w))
        u_idx, v_idx, weight = int(raw_u[best]), int(raw_v[best]), raw_w[best]
        print(
            "[eval-policy] "
            f"lp_support_edge_threshold={lp.threshold} removed all edges; "
            f"keeping max-weight edge ({u_idx + 1}, {v_idx + 1})={float(weight):.6f} as fallback."
        )
        pair_u = np.array([u_idx], dtype=np.int64)
        pair_v = np.array([v_idx], dtype=np.int64)
        pair_w = np.array([weight], dtype=np.float32)
        adj_dense = np.zeros((n, n), dtype=np.float32)
        adj_dense[u_idx, v_idx] = weight
        adj_dense[v_idx, u_idx] = weight
        nz_r, nz_c = np.nonzero(adj_dense)
        counts = np.bincount(nz_r, minlength=n)
        csr_indptr = np.zeros(n + 1, dtype=np.int64)
        csr_indptr[1:] = np.cumsum(counts)
        csr_indices = nz_c.astype(np.int32)
        csr_data = adj_dense[nz_r, nz_c]

    m_pairs = int(pair_u.size)
    sup_src = np.empty(2 * m_pairs, dtype=np.int64)
    sup_dst = np.empty(2 * m_pairs, dtype=np.int64)
    sup_src[0::2] = pair_u
    sup_src[1::2] = pair_v
    sup_dst[0::2] = pair_v
    sup_dst[1::2] = pair_u
    sup_w = np.repeat(pair_w, 2)

    sc_src, sc_dst, sc_w = _two_hop_shortcuts(
        csr_indptr, csr_indices, csr_data, adj_dense, cross_depot
    )

    to_depot = adj_dense[:, 0] > 0
    from_depot = adj_dense[0, :] > 0
    to_depot = to_depot.copy()
    from_depot = from_depot.copy()
    if sc_src.size:
        to_depot[sc_src[sc_dst == 0]] = True
        from_depot[sc_dst[sc_src == 0]] = True
    cand = np.arange(1, n, dtype=np.int64)
    dep_src_all = np.empty(2 * (n - 1), dtype=np.int64)
    dep_dst_all = np.empty(2 * (n - 1), dtype=np.int64)
    dep_mask = np.empty(2 * (n - 1), dtype=bool)
    dep_src_all[0::2] = cand
    dep_dst_all[0::2] = 0
    dep_mask[0::2] = ~to_depot[cand]
    dep_src_all[1::2] = 0
    dep_dst_all[1::2] = cand
    dep_mask[1::2] = ~from_depot[cand]
    dep_src = dep_src_all[dep_mask]
    dep_dst = dep_dst_all[dep_mask]

    src = np.concatenate([sup_src, sc_src, dep_src])
    dst = np.concatenate([sup_dst, sc_dst, dep_dst])
    edge_index = np.vstack([src, dst])
    e_total = src.size
    edge_attr = np.zeros((e_total, 2), dtype=np.float32)
    edge_attr[: sup_w.size, 0] = sup_w
    edge_attr[: sup_w.size, 1] = np.float32(_HOP_ORIGINAL)
    edge_attr[sup_w.size : sup_w.size + sc_w.size, 0] = sc_w
    edge_attr[sup_w.size : sup_w.size + sc_w.size, 1] = np.float32(_HOP_SHORTCUT)
    edge_attr[sup_w.size + sc_w.size :, 1] = np.float32(_HOP_DEPOT_FILL)

    demand32 = instance.demand.astype(np.float32)
    cap32 = np.float32(float(instance.capacity))
    cap32 = np.float32(max(cap32, np.float32(1e-6)))
    node_demand_frac = demand32 / cap32

    scale_span = float(max(k - 1, 1))
    scaled_m = (np.arange(k, dtype=np.float64) / scale_span).astype(np.float32)

    node_attr = np.empty((k, n, 2), dtype=np.float32)
    node_attr[:, :, 0] = node_demand_frac[None, :]
    node_attr[:, :, 1] = scaled_m[:, None]

    return GraphFeatures(
        n_nodes=n,
        vehicle_count=k,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_attr=node_attr,
    )


# ---------------------------------------------------------------------------
# CandidateContext assembly
# ---------------------------------------------------------------------------


def build_candidate_context(
    instance: Instance,
    lp: LPSupport,
    probs: np.ndarray,
    *,
    iteration: int,
    seed: int,
    tts_seed: int,
    bank: OldCutBankSnapshot,
) -> CandidateContext:
    probs_arr = np.ascontiguousarray(probs, dtype=np.float32)
    if probs_arr.ndim != 2 or probs_arr.shape[1] != instance.n_nodes:
        raise ValueError(
            f"probs must be (K, n_nodes={instance.n_nodes}), got {probs_arr.shape}"
        )
    return CandidateContext(
        probs=probs_arr,
        lp=lp,
        demand=instance.demand,
        capacity=int(instance.capacity),
        n_nodes=int(instance.n_nodes),
        vehicle_count=int(probs_arr.shape[0]),
        depot=0,
        instance=str(instance.name),
        iteration=int(iteration),
        seed=int(seed),
        tts_seed=int(tts_seed),
        bank=bank,
    )
