from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch

from sagsep import features as sagsep_features
from sagsep.model import GraphFeatures, PolicyModel

__all__ = [
    "INPUT_NAMES",
    "OUTPUT_NAME",
    "PARITY_TOL",
    "builtin_cases",
    "compare",
    "compare_onnx_files",
    "instance_features",
    "ort_probs",
    "ort_session",
    "random_features",
    "torch_probs",
]

INPUT_NAMES: Tuple[str, str, str] = ("edge_index", "node_attr", "edge_attr")
OUTPUT_NAME = "probs"

PARITY_TOL = 1e-5


# ---------------------------------------------------------------------------
# runtime session
# ---------------------------------------------------------------------------


def ort_session(path: str | Path) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def ort_probs(session: Any, feats: GraphFeatures) -> np.ndarray:
    feeds = {
        "edge_index": np.ascontiguousarray(feats.edge_index, dtype=np.int64),
        "node_attr": np.ascontiguousarray(feats.node_attr, dtype=np.float32),
        "edge_attr": np.ascontiguousarray(feats.edge_attr, dtype=np.float32),
    }
    return session.run([OUTPUT_NAME], feeds)[0]


def torch_probs(net: Any, feats: GraphFeatures) -> np.ndarray:
    return PolicyModel.from_network(net, "cpu").forward_probs(feats)


# ---------------------------------------------------------------------------
# builtin cases
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def random_features(
    seed: int, *, n_nodes: int, vehicles: int, density: float = 0.45
) -> GraphFeatures:
    rng = np.random.default_rng(seed)
    src: List[int] = []
    dst: List[int] = []
    val: List[float] = []
    hop: List[float] = []
    for u in range(n_nodes):
        for v in range(u + 1, n_nodes):
            if rng.random() < density:
                x = float(rng.uniform(0.05, 1.0))
                h = float(rng.integers(0, 2))
                src.extend([u, v])
                dst.extend([v, u])
                val.extend([x, x])
                hop.extend([h, h])
    if not src:
        src, dst, val, hop = [0, 1], [1, 0], [0.5, 0.5], [1.0, 1.0]
    edge_index = np.asarray([src, dst], dtype=np.int64)
    edge_attr = np.stack(
        [np.asarray(val, dtype=np.float32), np.asarray(hop, dtype=np.float32)], axis=1
    )
    demand_over_cap = rng.uniform(0.02, 0.4, size=n_nodes).astype(np.float32)
    demand_over_cap[0] = 0.0
    scale_span = float(max(vehicles - 1, 1))
    scaled_m = (np.arange(vehicles, dtype=np.float64) / scale_span).astype(np.float32)
    node_attr = np.empty((vehicles, n_nodes, 2), dtype=np.float32)
    for k in range(vehicles):
        node_attr[k, :, 0] = demand_over_cap
        node_attr[k, :, 1] = scaled_m[k]
    return GraphFeatures(
        n_nodes=n_nodes,
        vehicle_count=vehicles,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_attr=node_attr,
    )


def instance_features(
    vrp_path: str | Path, seed: int, *, cross_depot: bool = True
) -> GraphFeatures:
    instance = sagsep_features.load_instance(vrp_path, vehicles=0)
    n = int(instance.n_nodes)
    rng = np.random.default_rng(seed)
    order = np.arange(1, n)
    rng.shuffle(order)
    dense = np.zeros((n, n), dtype=np.float64)
    tour = [0, *order.tolist(), 0]
    for a, b in zip(tour[:-1], tour[1:]):
        w = 1.0 if rng.random() < 0.7 else float(rng.uniform(0.3, 0.9))
        dense[a, b] = dense[b, a] = w
    for _ in range(2 * n):
        a, b = rng.integers(0, n, size=2)
        if a != b and dense[a, b] == 0.0:
            w = float(rng.choice([0.5, 0.25, 0.125, 2e-6, 5e-7]))
            dense[a, b] = dense[b, a] = w
    iu, ju = np.triu_indices(n, k=1)
    x = dense[iu, ju]
    keep = x > 1e-8
    u, v, x = iu[keep].astype(np.int64), ju[keep].astype(np.int64), x[keep]
    lp = sagsep_features.build_lp_support(u, v, x, n_nodes=n, threshold=1e-6)
    return sagsep_features.build_graph_features(
        instance,
        lp,
        vehicle_count=sagsep_features.vehicle_replica_count(instance),
        support_edge_u=u,
        support_edge_v=v,
        support_edge_x=x,
        cross_depot=cross_depot,
    )


def builtin_cases(repo_root: Optional[str | Path] = None) -> Dict[str, GraphFeatures]:
    root = Path(repo_root) if repo_root is not None else _repo_root()
    cases: Dict[str, GraphFeatures] = {
        "rand_n12_k3": random_features(1, n_nodes=12, vehicles=3, density=0.45),
        "rand_n40_k8": random_features(7, n_nodes=40, vehicles=8, density=0.30),
    }
    instances = root / "instances" / "random"
    n50 = instances / "random-24-X-n50.vrp"
    n100 = instances / "random-26-X-n100.vrp"
    if n50.exists():
        cases["inst_n50_cd1"] = instance_features(n50, 23, cross_depot=True)
        cases["inst_n50_cd0"] = instance_features(n50, 23, cross_depot=False)
    if n100.exists():
        cases["inst_n100_cd1"] = instance_features(n100, 29, cross_depot=True)
    return cases


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def _case_shape(feats: GraphFeatures) -> Tuple[int, int]:
    return int(feats.vehicle_count), int(feats.n_nodes)


def compare(
    net: Any,
    session: Any,
    cases: Mapping[str, GraphFeatures],
    *,
    tol: float = PARITY_TOL,
) -> Dict[str, Any]:
    if not cases:
        raise ValueError("compare needs at least one case")
    net64 = copy.deepcopy(net).to(torch.float64)

    rows: Dict[str, Any] = {}
    worst = 0.0
    all_ok = True
    for name, feats in cases.items():
        ref32 = torch_probs(net, feats).astype(np.float64)
        ref64 = torch_probs(net64, feats).astype(np.float64)
        got = ort_probs(session, feats)
        if tuple(got.shape) != tuple(ref32.shape):
            raise RuntimeError(f"case {name}: runtime shape {got.shape} != torch {ref32.shape}")
        got = got.astype(np.float64)
        a = float(np.abs(got - ref32).max())
        b = float(np.abs(ref32 - ref64).max())
        c = float(np.abs(got - ref64).max())
        if a < tol:
            rule, ok = "a<tol", True
        elif a <= 2.0 * b:
            rule, ok = "a<=2b(fp32-noise)", True
        else:
            rule, ok = "FAIL", False
        all_ok = all_ok and ok
        worst = max(worst, a)
        k, n = _case_shape(feats)
        rows[name] = {
            "shape": [k, n],
            "n_edges": int(feats.edge_index.shape[1]),
            "ort_vs_torch32": a,
            "torch32_vs_torch64": b,
            "ort_vs_torch64": c,
            "rule": rule,
            "ok": ok,
        }
    return {
        "tol": float(tol),
        "cases": rows,
        "max_abs_diff": worst,
        "ok": bool(all_ok),
        "verdict": "PASS" if all_ok else "FAIL",
    }


def compare_onnx_files(
    path_a: str | Path, path_b: str | Path, cases: Mapping[str, GraphFeatures]
) -> Dict[str, Any]:
    if not cases:
        raise ValueError("compare_onnx_files needs at least one case")
    session_a = ort_session(path_a)
    session_b = ort_session(path_b)
    rows: Dict[str, Any] = {}
    worst = 0.0
    all_bitwise = True
    for name, feats in cases.items():
        out_a = ort_probs(session_a, feats)
        out_b = ort_probs(session_b, feats)
        if tuple(out_a.shape) != tuple(out_b.shape):
            raise RuntimeError(f"case {name}: shapes differ {out_a.shape} vs {out_b.shape}")
        bitwise = out_a.tobytes() == out_b.tobytes()
        diff = float(np.abs(out_a.astype(np.float64) - out_b.astype(np.float64)).max())
        all_bitwise = all_bitwise and bitwise
        worst = max(worst, diff)
        k, n = _case_shape(feats)
        rows[name] = {"shape": [k, n], "bitwise": bitwise, "max_abs_diff": diff}
    return {
        "cases": rows,
        "bitwise": bool(all_bitwise),
        "max_abs_diff": worst,
        "n_cases": len(rows),
    }
