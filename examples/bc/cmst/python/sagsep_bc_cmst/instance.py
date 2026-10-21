from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sagsep_bc_common.protocol import total_demand

__all__ = ["CmstInstance", "load_instance", "min_root_degree"]


@dataclass(frozen=True)
class CmstInstance:
    name: str
    n_nodes: int
    capacity: int
    demand: np.ndarray
    weight: np.ndarray
    capacities: Tuple[int, ...]
    optimal: Optional[float] = None

    @property
    def n_customers(self) -> int:
        return int(self.n_nodes) - 1

    @property
    def total_demand(self) -> int:
        return total_demand(self.demand)


def min_root_degree(demand: np.ndarray, capacity: int) -> int:
    total = total_demand(demand)
    quotient = int(capacity)
    return max(1, -(-total // quotient))


def _split_header(lines: Sequence[str]) -> Tuple[dict, int]:
    header: dict = {}
    for index, line in enumerate(lines):
        if ":" not in line:
            return header, index
        key, value = line.split(":", 1)
        header[key.strip().upper()] = value.strip()
    return header, len(lines)


def load_instance(
    path: str | Path,
    *,
    capacity: Optional[int] = None,
    optimal: Optional[float] = None,
) -> CmstInstance:
    path = Path(path)
    lines = [line.strip() for line in path.read_text(encoding="ascii").splitlines()]
    lines = [line for line in lines if line]
    header, cursor = _split_header(lines)

    for key in ("DIMENSION", "CAPACITIES"):
        if key not in header:
            raise ValueError(f"{path}: no {key} header")
    if header.get("TYPE", "CMST").upper() != "CMST":
        raise ValueError(f"{path}: TYPE must be CMST, got {header['TYPE']!r}")
    if header.get("EDGE_WEIGHT_TYPE", "EXPLICIT").upper() != "EXPLICIT":
        raise ValueError(f"{path}: only EDGE_WEIGHT_TYPE=EXPLICIT is supported")
    if header.get("EDGE_WEIGHT_FORMAT", "FULL_MATRIX").upper() != "FULL_MATRIX":
        raise ValueError(f"{path}: only EDGE_WEIGHT_FORMAT=FULL_MATRIX is supported")
    if int(header.get("ROOT", "1")) != 1:
        raise ValueError(f"{path}: the root must be external node 1")

    n_nodes = int(header["DIMENSION"])
    if n_nodes < 2:
        raise ValueError(f"{path}: DIMENSION = {n_nodes} (need the root and a customer)")
    capacities = tuple(int(token) for token in header["CAPACITIES"].split())
    if not capacities:
        raise ValueError(f"{path}: CAPACITIES is empty")
    if capacity is None:
        if len(capacities) > 1:
            raise ValueError(
                f"{path} is posed with capacities {list(capacities)}; pass --capacity"
            )
        capacity = capacities[0]
    elif int(capacity) not in capacities:
        raise ValueError(
            f"{path}: capacity {capacity} is not one of {list(capacities)}"
        )
    capacity = int(capacity)
    if capacity <= 0:
        raise ValueError(f"{path}: capacity must be positive, got {capacity}")

    if lines[cursor] != "DEMAND_SECTION":
        raise ValueError(f"{path}: expected DEMAND_SECTION, got {lines[cursor]!r}")
    cursor += 1
    demand = np.zeros(n_nodes, dtype=np.int64)
    for node in range(n_nodes):
        parts = lines[cursor + node].split()
        if len(parts) != 2 or int(parts[0]) != node + 1:
            raise ValueError(f"{path}: bad demand row {lines[cursor + node]!r}")
        demand[node] = int(parts[1])
    cursor += n_nodes

    if lines[cursor] != "EDGE_WEIGHT_SECTION":
        raise ValueError(f"{path}: expected EDGE_WEIGHT_SECTION, got {lines[cursor]!r}")
    cursor += 1
    values: List[int] = []
    while cursor < len(lines) and lines[cursor] != "EOF" and len(values) < n_nodes * n_nodes:
        values.extend(int(token) for token in lines[cursor].split())
        cursor += 1
    if len(values) != n_nodes * n_nodes:
        raise ValueError(
            f"{path}: expected {n_nodes * n_nodes} costs, got {len(values)}"
        )
    weight = np.asarray(values, dtype=np.float64).reshape(n_nodes, n_nodes)

    if demand[0] != 0:
        raise ValueError(f"{path}: the root's demand must be 0, got {demand[0]}")
    if np.any(demand[1:] <= 0):
        raise ValueError(f"{path}: customer demands must be positive")
    if np.any(demand[1:] > capacity):
        offender = int(np.argmax(demand[1:] > capacity)) + 1
        raise ValueError(
            f"{path}: customer {offender + 1} has demand {int(demand[offender])} "
            f"> capacity {capacity}; the instance is infeasible"
        )
    if not np.array_equal(weight, weight.T):
        raise ValueError(f"{path}: the cost matrix is not symmetric")
    if np.any(np.diagonal(weight) != 0.0):
        raise ValueError(f"{path}: the cost matrix has a non-zero diagonal")

    return CmstInstance(
        name=header.get("NAME", path.stem),
        n_nodes=n_nodes,
        capacity=capacity,
        demand=demand,
        weight=weight,
        capacities=capacities,
        optimal=optimal,
    )
