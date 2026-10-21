from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from sagsep import features as F
from sagsep.model import GraphFeatures


def _random_graph(seed: int, *, n_nodes: int, vehicles: int, density: float) -> GraphFeatures:
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


def _instance_graph(vrp_path: Path, seed: int, *, cross_depot: bool) -> GraphFeatures:
    instance = F.load_instance(vrp_path, vehicles=0)
    n = instance.n_nodes
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
    lp = F.build_lp_support(u, v, x, n_nodes=n, threshold=1e-6)
    k = F.vehicle_replica_count(instance)
    return F.build_graph_features(
        instance, lp, vehicle_count=k, support_edge_u=u, support_edge_v=v,
        support_edge_x=x, cross_depot=cross_depot,
    )


def golden_cases(repo_root: Path) -> Dict[str, GraphFeatures]:
    cases: Dict[str, GraphFeatures] = {
        "rand_n12_k3": _random_graph(1, n_nodes=12, vehicles=3, density=0.45),
        "rand_n15_k4": _random_graph(5, n_nodes=15, vehicles=4, density=0.45),
        "rand_n40_k8": _random_graph(7, n_nodes=40, vehicles=8, density=0.30),
        "rand_n60_k1": _random_graph(11, n_nodes=60, vehicles=1, density=0.20),
        "sparse_n25_k6": _random_graph(13, n_nodes=25, vehicles=6, density=0.06),
        "single_edge_n10_k2": _random_graph(17, n_nodes=10, vehicles=2, density=0.0),
    }
    inst = repo_root / "instances" / "random"
    cases["inst_n50_cd1"] = _instance_graph(inst / "random-24-X-n50.vrp", 23, cross_depot=True)
    cases["inst_n50_cd0"] = _instance_graph(inst / "random-24-X-n50.vrp", 23, cross_depot=False)
    cases["inst_n100_cd1"] = _instance_graph(inst / "random-26-X-n100.vrp", 29, cross_depot=True)
    return cases

EDGE_CHUNKS: Tuple[int, ...] = (0, 7, 64)
CHUNK_ENVS: Tuple[Tuple[str, str], ...] = (("auto", "auto"), ("5", "2"), ("0", "0"))
