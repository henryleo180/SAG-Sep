from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sagsep.types import Instance

__all__ = [
    "DISTANCE_CONVENTIONS",
    "DirectedInstance",
    "load_directed_instance",
]

DISTANCE_CONVENTIONS = ("exact", "round")


@dataclass(frozen=True)
class DirectedInstance:
    name: str
    n_nodes: int
    demand: np.ndarray
    capacity: int
    cost: np.ndarray
    vehicles: Optional[int] = None
    coords: Optional[np.ndarray] = None
    distance_convention: str = "explicit"

    @property
    def n_customers(self) -> int:
        return int(self.n_nodes) - 1

    def separator_instance(self) -> Instance:
        coords = self.coords
        if coords is None:
            coords = np.zeros((int(self.n_nodes), 2), dtype=np.float64)
        return Instance(
            name=self.name,
            n_nodes=int(self.n_nodes),
            depot=0,
            coords=coords,
            demand=self.demand,
            capacity=int(self.capacity),
            vehicles=int(self.vehicles or 0),
        )


def load_directed_instance(
    path: str | Path, *, distance: str = "exact"
) -> DirectedInstance:
    if distance not in DISTANCE_CONVENTIONS:
        raise ValueError(
            f"--distance must be one of {DISTANCE_CONVENTIONS}, got {distance!r}"
        )
    path = Path(path)
    header, coords, demand, depot_ids, matrix = _parse(path)

    dimension = _required_int(header, "DIMENSION", path)
    if dimension < 2:
        raise ValueError(f"{path}: DIMENSION = {dimension} (need a depot and a customer)")
    capacity = _required_int(header, "CAPACITY", path)
    if capacity <= 0:
        raise ValueError(f"{path}: CAPACITY = {capacity} (must be positive)")
    weight_type = header.get("EDGE_WEIGHT_TYPE", "EUC_2D").upper()

    depot = depot_ids[0] if depot_ids else 1
    if len(depot_ids) > 1:
        raise ValueError(f"{path}: {len(depot_ids)} depots; these examples model one")
    if not 1 <= depot <= dimension:
        raise ValueError(f"{path}: depot {depot} is outside 1..{dimension}")
    order = [depot] + [node for node in range(1, dimension + 1) if node != depot]

    demands = np.empty(dimension, dtype=np.int64)
    for internal, external in enumerate(order):
        if external not in demand:
            raise ValueError(f"{path}: missing demand for node {external}")
        value = demand[external]
        if not math.isfinite(value) or value != math.floor(value):
            raise ValueError(
                f"{path}: demand {value} for node {external} is not a finite integer "
                "(demands are carried as int64)"
            )
        if value < 0:
            raise ValueError(f"{path}: negative demand {value} for node {external}")
        demands[internal] = int(value)
    if demands[0] != 0:
        raise ValueError(f"{path}: depot demand must be 0, got {demands[0]}")
    if int(demands[1:].max(initial=0)) > capacity:
        raise ValueError(f"{path}: a customer demands more than one vehicle carries")

    coordinates: Optional[np.ndarray] = None
    if weight_type == "EXPLICIT":
        weight_format = header.get("EDGE_WEIGHT_FORMAT", "").upper()
        if weight_format != "FULL_MATRIX":
            raise ValueError(
                f"{path}: EDGE_WEIGHT_FORMAT must be FULL_MATRIX for an explicit "
                f"matrix, got {weight_format or '(absent)'}; a triangular format "
                "cannot hold an asymmetric cost matrix"
            )
        if len(matrix) != dimension * dimension:
            raise ValueError(
                f"{path}: EDGE_WEIGHT_SECTION has {len(matrix)} entries, "
                f"expected {dimension * dimension}"
            )
        full = np.asarray(matrix, dtype=np.float64).reshape(dimension, dimension)
        if not np.all(np.isfinite(full)):
            raise ValueError(f"{path}: EDGE_WEIGHT_SECTION holds a non-finite cost")
        index = np.asarray(order, dtype=np.int64) - 1
        cost = np.ascontiguousarray(full[np.ix_(index, index)])
        convention = "explicit"
    elif weight_type == "EUC_2D":
        coordinates = np.empty((dimension, 2), dtype=np.float64)
        for internal, external in enumerate(order):
            if external not in coords:
                raise ValueError(f"{path}: missing coordinates for node {external}")
            coordinates[internal] = coords[external]
        if not np.all(np.isfinite(coordinates)):
            raise ValueError(f"{path}: NODE_COORD_SECTION holds a non-finite coordinate")
        cost = _euclidean_cost(coordinates, distance)
        convention = distance
    else:
        raise ValueError(
            f"{path}: EDGE_WEIGHT_TYPE must be EXPLICIT or EUC_2D, got {weight_type}"
        )

    fleet = header.get("VEHICLES")
    vehicles = int(fleet) if fleet else None
    if vehicles is not None and not 1 <= vehicles <= dimension - 1:
        raise ValueError(
            f"{path}: VEHICLES = {vehicles}; a fixed fleet must be between 1 and the "
            f"customer count {dimension - 1}, since every route visits a customer"
        )
    return DirectedInstance(
        name=header.get("NAME", path.stem),
        n_nodes=dimension,
        demand=demands,
        capacity=capacity,
        cost=cost,
        vehicles=vehicles,
        coords=coordinates,
        distance_convention=convention,
    )


def _euclidean_cost(coords: np.ndarray, distance: str) -> np.ndarray:
    n = int(coords.shape[0])
    cost = np.zeros((n, n), dtype=np.float64)
    exact = distance == "exact"
    for i in range(n):
        xi = float(coords[i, 0])
        yi = float(coords[i, 1])
        for j in range(i + 1, n):
            raw = math.hypot(xi - float(coords[j, 0]), yi - float(coords[j, 1]))
            value = raw if exact else float(int(raw + 0.5))
            cost[i, j] = value
            cost[j, i] = value
    return cost


# ---------------------------------------------------------------------------
# The file itself
# ---------------------------------------------------------------------------

_SECTIONS = {
    "NODE_COORD_SECTION": "coords",
    "DEMAND_SECTION": "demand",
    "DEPOT_SECTION": "depot",
    "EDGE_WEIGHT_SECTION": "weights",
}


def _parse(path: Path) -> Tuple[
    Dict[str, str],
    Dict[int, Tuple[float, float]],
    Dict[int, float],
    List[int],
    List[float],
]:
    header: Dict[str, str] = {}
    coords: Dict[int, Tuple[float, float]] = {}
    demand: Dict[int, float] = {}
    depot_ids: List[int] = []
    weights: List[float] = []
    section: Optional[str] = None

    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            keyword = line.split(":", 1)[0].strip().upper()
            if keyword in _SECTIONS:
                section = _SECTIONS[keyword]
                continue
            if keyword == "EOF":
                break
            if section == "coords":
                parts = line.split()
                if len(parts) >= 3:
                    coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
            elif section == "demand":
                parts = line.split()
                if len(parts) >= 2:
                    demand[int(parts[0])] = float(parts[1])
            elif section == "depot":
                for token in line.split():
                    node = int(token)
                    if node < 0:
                        section = None
                        break
                    depot_ids.append(node)
            elif section == "weights":
                weights.extend(float(token) for token in line.split())
            elif ":" in line:
                key, value = line.split(":", 1)
                header[key.strip().upper()] = value.strip()
    return header, coords, demand, depot_ids, weights


def _required_int(header: Dict[str, str], key: str, path: Path) -> int:
    if key not in header:
        raise ValueError(f"{path}: no {key} header")
    return int(header[key])
