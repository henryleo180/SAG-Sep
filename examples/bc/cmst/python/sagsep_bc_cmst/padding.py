from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

__all__ = [
    "PADDED_SUPPORT_THRESHOLD",
    "TARGET_DEGREE",
    "PaddedSupport",
    "build_padded_support",
    "plain_support",
]

PADDED_SUPPORT_THRESHOLD = 1e-8

TARGET_DEGREE = 2.0


@dataclass(frozen=True, slots=True)
class PaddedSupport:
    support_u: np.ndarray
    support_v: np.ndarray
    support_x: np.ndarray
    pair_degree: np.ndarray
    high_degree: int
    max_clip: float

    def cut_value(self, nodes_ext: Sequence[int]) -> float:
        if len(nodes_ext) == 0:
            return 0.0
        inside = np.zeros(self.pair_degree.size, dtype=bool)
        inside[np.asarray(nodes_ext, dtype=np.int64) - 1] = True
        crossing = inside[self.support_u] != inside[self.support_v]
        return float(self.support_x[crossing].sum())

    def identity_gap(self, nodes_ext: Sequence[int], rhs: int, inner_violation: float) -> float:
        cvrp_violation = 2.0 * float(rhs) - self.cut_value(nodes_ext)
        return 0.5 * cvrp_violation - float(inner_violation)


def build_padded_support(
    dense: np.ndarray,
    *,
    threshold: float = PADDED_SUPPORT_THRESHOLD,
) -> PaddedSupport:
    dense = np.asarray(dense, dtype=np.float64)
    if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
        raise ValueError(f"dense support must be square, got shape {dense.shape}")
    n_nodes = int(dense.shape[0])
    if n_nodes < 2:
        raise ValueError(f"instance must have at least one customer, got {n_nodes} nodes")

    pair_degree = np.zeros(n_nodes, dtype=np.float64)
    pair_degree[1:] = dense[1:, 1:].sum(axis=1)

    depot_value = TARGET_DEGREE - pair_degree[1:]
    clipped = depot_value < 0.0
    high_degree = int(np.count_nonzero(clipped))
    max_clip = float(-depot_value[clipped].min()) if high_degree else 0.0
    depot_value = np.where(clipped, 0.0, depot_value)

    edge_i, edge_j = (idx.astype(np.int64) for idx in np.triu_indices(n_nodes, k=1))
    values = dense[edge_i, edge_j]
    is_depot = edge_i == 0
    values[is_depot] = depot_value
    keep = values > float(threshold)

    return PaddedSupport(
        support_u=np.ascontiguousarray(edge_i[keep]),
        support_v=np.ascontiguousarray(edge_j[keep]),
        support_x=np.ascontiguousarray(values[keep]),
        pair_degree=pair_degree,
        high_degree=high_degree,
        max_clip=max_clip,
    )


def plain_support(
    dense: np.ndarray,
    *,
    threshold: float = PADDED_SUPPORT_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dense = np.asarray(dense, dtype=np.float64)
    n_nodes = int(dense.shape[0])
    edge_i, edge_j = (idx.astype(np.int64) for idx in np.triu_indices(n_nodes, k=1))
    values = dense[edge_i, edge_j]
    keep = values > float(threshold)
    return (
        np.ascontiguousarray(edge_i[keep]),
        np.ascontiguousarray(edge_j[keep]),
        np.ascontiguousarray(values[keep]),
    )
