from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = ["score_subsets", "local_search_1flip"]


def score_subsets(
    subsets_mat: np.ndarray,
    adj_view: np.ndarray,
    deg: np.ndarray,
    demand_view: np.ndarray,
    capacity: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sum_deg_S = subsets_mat @ deg
    intra = ((subsets_mat @ adj_view) * subsets_mat).sum(axis=1)
    lhs = sum_deg_S - intra
    d_S = subsets_mat @ demand_view
    rhs = 2.0 * np.ceil(d_S / max(capacity, 1e-9)).astype(np.float32)
    viol = rhs - lhs
    return (
        viol.astype(np.float32, copy=False),
        lhs.astype(np.float32, copy=False),
        rhs.astype(np.float32, copy=False),
        d_S.astype(np.float32, copy=False),
    )


def local_search_1flip(
    sel: np.ndarray,
    adj_view: np.ndarray,
    deg: np.ndarray,
    demand_view: np.ndarray,
    capacity: float,
    *,
    max_flips: int,
    min_gain: float,
    max_size_frac: float,
    swap_max_pairs: int = 250_000,
    min_keep: int = 2,
    diagnostics: Optional[Dict[str, object]] = None,
) -> Tuple[np.ndarray, float]:
    n = int(sel.shape[0])
    sel = sel.copy()
    sel[0] = 0.0
    deg_in_S = adj_view @ sel
    d_S = float((demand_view * sel).sum())
    sum_deg = float((deg * sel).sum())
    intra_d = float((deg_in_S * sel).sum())
    lhs = sum_deg - intra_d
    Q = max(float(capacity), 1e-9)
    rhs = 2.0 * math.ceil(d_S / Q)
    viol = rhs - lhs
    initial_viol = float(viol)
    steps = 0
    max_size = int(math.floor(max_size_frac * (n - 1)))
    cust_idx = np.arange(1, n, dtype=np.int64)
    for _ in range(int(max_flips)):
        best_gain = 0.0
        best_v = -1
        best_is_add = False
        best_swap_u = -1
        in_S = sel[cust_idx] > 0.5
        if int(in_S.sum()) > int(min_keep):
            v_in = cust_idx[in_S]
            if v_in.size > 0:
                d_v = demand_view[v_in]
                deg_v = deg[v_in]
                dis_v = deg_in_S[v_in]
                new_d = d_S - d_v
                new_rhs = 2.0 * np.ceil(new_d / Q)
                delta_lhs = -deg_v + 2.0 * dis_v
                new_lhs = lhs + delta_lhs
                new_viol = new_rhs - new_lhs
                gain = new_viol - viol
                if gain.size > 0:
                    g_idx = int(np.argmax(gain))
                    if float(gain[g_idx]) > best_gain:
                        best_gain = float(gain[g_idx])
                        best_v = int(v_in[g_idx])
                        best_is_add = False
        if int(in_S.sum()) < max_size:
            v_out = cust_idx[~in_S]
            if v_out.size > 0:
                d_v = demand_view[v_out]
                deg_v = deg[v_out]
                dis_v = deg_in_S[v_out]
                new_d = d_S + d_v
                new_rhs = 2.0 * np.ceil(new_d / Q)
                delta_lhs = deg_v - 2.0 * dis_v
                new_lhs = lhs + delta_lhs
                new_viol = new_rhs - new_lhs
                gain = new_viol - viol
                if gain.size > 0:
                    g_idx = int(np.argmax(gain))
                    if float(gain[g_idx]) > best_gain:
                        best_gain = float(gain[g_idx])
                        best_v = int(v_out[g_idx])
                        best_is_add = True
        u_in = cust_idx[in_S]
        v_out = cust_idx[~in_S]
        if (
            u_in.size > 0
            and v_out.size > 0
            and u_in.size * v_out.size <= int(swap_max_pairs)
        ):
            new_d = d_S - demand_view[u_in][:, None] + demand_view[v_out][None, :]
            new_rhs = 2.0 * np.ceil(new_d / Q)
            rem_term = -deg[u_in] + 2.0 * deg_in_S[u_in]
            add_term = deg[v_out] - 2.0 * deg_in_S[v_out]
            cross = 2.0 * adj_view[np.ix_(u_in, v_out)]
            new_lhs = lhs + rem_term[:, None] + add_term[None, :] + cross
            new_viol = new_rhs - new_lhs
            gain = new_viol - viol
            flat = int(np.argmax(gain))
            gi, gj = divmod(flat, v_out.size)
            if float(gain[gi, gj]) > best_gain:
                best_gain = float(gain[gi, gj])
                best_swap_u = int(u_in[gi])
                best_v = int(v_out[gj])
                best_is_add = True
        if best_v < 0 or best_gain <= float(min_gain):
            break
        steps += 1
        if best_swap_u >= 0:
            sel[best_swap_u] = 0.0
            d_S -= float(demand_view[best_swap_u])
            sum_deg -= float(deg[best_swap_u])
            intra_d -= 2.0 * float(deg_in_S[best_swap_u])
            deg_in_S = deg_in_S - adj_view[best_swap_u]
        old_dis_v = float(deg_in_S[best_v])
        if best_is_add:
            sel[best_v] = 1.0
            d_S += float(demand_view[best_v])
            sum_deg += float(deg[best_v])
            intra_d += 2.0 * old_dis_v
            deg_in_S = deg_in_S + adj_view[best_v]
        else:
            sel[best_v] = 0.0
            d_S -= float(demand_view[best_v])
            sum_deg -= float(deg[best_v])
            intra_d -= 2.0 * old_dis_v
            deg_in_S = deg_in_S - adj_view[best_v]
        d_S = max(d_S, 0.0)
        rhs = 2.0 * math.ceil(d_S / Q)
        lhs = sum_deg - intra_d
        viol = rhs - lhs
    sel[0] = 0.0
    if diagnostics is not None:
        diagnostics.update(
            {
                "initial_violation": initial_viol,
                "best_violation": float(viol),
                "gain": float(viol) - initial_viol,
                "steps": int(steps),
                "hit_max_flips": bool(max_flips > 0 and steps >= int(max_flips)),
            }
        )
    return sel.astype(np.float32, copy=False), float(viol)
