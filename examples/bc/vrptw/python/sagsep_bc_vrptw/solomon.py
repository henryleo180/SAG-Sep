from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "DISTANCE_CONVENTIONS",
    "DISTANCE_ALIASES",
    "SolomonInstance",
    "PreprocessedArcs",
    "load_solomon",
    "travel_matrix",
    "preprocess_arcs",
    "check_customer_degrees",
]

DISTANCE_CONVENTIONS = ("trunc1", "exact")

DISTANCE_ALIASES = {
    "trunc1": "trunc1",
    "truncated_1dp": "trunc1",
    "exact": "exact",
    "euclidean": "exact",
    "full_precision": "exact",
}

FEASIBILITY_EPSILON = 1e-9


@dataclass(frozen=True)
class SolomonInstance:
    name: str
    n_customers: int
    capacity: int
    available_vehicles: int
    coords: np.ndarray
    demand: np.ndarray
    ready: np.ndarray
    due: np.ndarray
    service: np.ndarray

    @property
    def horizon(self) -> float:
        return float(self.due[0])


@dataclass(frozen=True)
class PreprocessedArcs:
    n_model_nodes: int
    tail: np.ndarray
    head: np.ndarray
    cost: np.ndarray
    time: np.ndarray
    big_m: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    service: np.ndarray
    n_arcs_before: int
    removed_time_window: int
    removed_pair_capacity: int

    @property
    def n_arcs(self) -> int:
        return int(self.tail.size)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _finite_number(token: str) -> Optional[float]:
    try:
        value = float(token)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _integral(value: float, path: Path, what: str) -> int:
    if value != math.floor(value):
        raise ValueError(f"{path}: {what} must be an integer, got {value}")
    return int(value)


def load_solomon(path: str | Path) -> SolomonInstance:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        lines = handle.read().splitlines()

    name = ""
    capacity: Optional[int] = None
    available = 0
    records: List[Tuple[int, float, float, float, float, float, float]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not name:
            name = stripped
            continue
        fields = stripped.split()
        values = [_finite_number(field) for field in fields]
        if any(value is None for value in values):
            continue
        if len(values) == 2 and capacity is None:
            available = _integral(values[0], path, "the NUMBER field")
            capacity = _integral(values[1], path, "CAPACITY")
        elif len(values) >= 7:
            node = _integral(values[0], path, "a node id")
            records.append((node, *values[1:7]))
    if not name:
        raise ValueError(f"{path}: the file is empty")
    if capacity is None:
        raise ValueError(f"{path}: no 'NUMBER CAPACITY' record found")
    if not records:
        raise ValueError(f"{path}: no customer records found")
    if capacity <= 0:
        raise ValueError(f"{path}: CAPACITY = {capacity} (must be positive)")

    records.sort(key=lambda record: record[0])
    ids = [record[0] for record in records]
    if ids != list(range(len(records))):
        raise ValueError(
            f"{path}: node ids must be 0..{len(records) - 1} without gaps or "
            f"repeats, got {ids[0]}..{ids[-1]} over {len(records)} rows"
        )

    n_nodes = len(records)
    if n_nodes < 2:
        raise ValueError(f"{path}: need the depot plus at least one customer")
    coords = np.zeros((n_nodes, 2), dtype=np.float64)
    demand = np.zeros(n_nodes, dtype=np.int64)
    ready = np.zeros(n_nodes, dtype=np.float64)
    due = np.zeros(n_nodes, dtype=np.float64)
    service = np.zeros(n_nodes, dtype=np.float64)
    for index, record in enumerate(records):
        _, x, y, q, a, b, s = record
        quantity = _integral(q, path, f"the demand of node {index}")
        if index == 0 and quantity != 0:
            raise ValueError(f"{path}: the depot's demand must be 0, got {quantity}")
        if index > 0 and quantity <= 0:
            raise ValueError(
                f"{path}: customer {index} has demand {quantity}; this example "
                "requires strictly positive customer demand (see the module "
                "docstring: it is what makes the MIPSOL check complete)"
            )
        if quantity > capacity:
            raise ValueError(
                f"{path}: node {index} has demand {quantity} > capacity "
                f"{capacity}; no feasible route exists"
            )
        if s < 0.0:
            raise ValueError(f"{path}: node {index} has negative service time {s}")
        if a > b:
            raise ValueError(f"{path}: node {index} has an empty time window")
        coords[index] = (x, y)
        demand[index] = quantity
        ready[index] = a
        due[index] = b
        service[index] = s

    return SolomonInstance(
        name=name,
        n_customers=n_nodes - 1,
        capacity=int(capacity),
        available_vehicles=int(available),
        coords=coords,
        demand=demand,
        ready=ready,
        due=due,
        service=service,
    )


def travel_matrix(instance: SolomonInstance, convention: str) -> np.ndarray:
    if convention not in DISTANCE_CONVENTIONS:
        raise ValueError(
            f"unknown distance convention {convention!r}; "
            f"valid: {', '.join(DISTANCE_CONVENTIONS)}"
        )
    coords = instance.coords
    n = int(coords.shape[0])
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        xi = float(coords[i, 0])
        yi = float(coords[i, 1])
        for j in range(i + 1, n):
            distance = math.hypot(xi - float(coords[j, 0]), yi - float(coords[j, 1]))
            if convention == "trunc1":
                distance = math.floor(distance * 10.0 + 1e-6) / 10.0
            matrix[i, j] = distance
            matrix[j, i] = distance
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{instance.name}: the travel matrix is not finite")
    return matrix


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def _dijkstra(weight: np.ndarray, source: int) -> np.ndarray:
    n = int(weight.shape[0])
    dist = np.full(n, np.inf, dtype=np.float64)
    dist[source] = 0.0
    settled = np.zeros(n, dtype=bool)
    for _ in range(n):
        candidates = np.where(settled, np.inf, dist)
        node = int(np.argmin(candidates))
        if not math.isfinite(float(candidates[node])):
            break
        settled[node] = True
        np.minimum(dist, dist[node] + weight[node], out=dist)
    return dist


def preprocess_arcs(
    instance: SolomonInstance, travel: np.ndarray
) -> PreprocessedArcs:
    n = instance.n_customers
    sink = n + 1
    n_model_nodes = n + 2

    service = np.empty(n_model_nodes, dtype=np.float64)
    ready = np.empty(n_model_nodes, dtype=np.float64)
    due = np.empty(n_model_nodes, dtype=np.float64)
    ready[: n + 1] = instance.ready
    due[: n + 1] = instance.due
    service[: n + 1] = instance.service
    ready[sink] = instance.ready[0]
    due[sink] = instance.due[0]
    service[sink] = 0.0

    demand = instance.demand
    capacity = int(instance.capacity)

    def physical(node: int) -> int:
        return 0 if node in (0, sink) else node

    candidate = np.zeros((n_model_nodes, n_model_nodes), dtype=bool)
    n_before = 0
    removed_pair = 0
    for tail in range(0, n + 1):
        for head in range(1, n + 2):
            if tail == head:
                continue
            if tail == 0 and head == sink:
                continue
            n_before += 1
            if (
                tail != 0
                and head != sink
                and int(demand[tail]) > capacity - int(demand[head])
            ):
                removed_pair += 1
                continue
            candidate[tail, head] = True

    weight = np.full((n_model_nodes, n_model_nodes), np.inf, dtype=np.float64)
    rows, cols = np.nonzero(candidate)
    for tail, head in zip(rows.tolist(), cols.tolist()):
        weight[tail, head] = service[tail] + float(
            travel[physical(tail), physical(head)]
        )
    from_source = _dijkstra(weight, 0)
    to_sink = _dijkstra(np.ascontiguousarray(weight.T), sink)

    lower = ready.copy()
    upper = due.copy()
    for node in range(1, n + 1):
        if not math.isfinite(from_source[node]) or not math.isfinite(to_sink[node]):
            raise ValueError(
                f"{instance.name}: customer {node} cannot be reached from the "
                "depot and returned from within the surviving arcs"
            )
        lower[node] = max(lower[node], ready[0] + from_source[node])
        upper[node] = min(upper[node], due[sink] - to_sink[node])
    upper[0] = lower[0]
    if np.any(lower > upper + FEASIBILITY_EPSILON):
        bad = int(np.flatnonzero(lower > upper + FEASIBILITY_EPSILON)[0])
        raise ValueError(
            f"{instance.name}: node {bad} cannot be served at all "
            f"(tightened window [{lower[bad]}, {upper[bad]}] is empty)"
        )

    tails: List[int] = []
    heads: List[int] = []
    times: List[float] = []
    removed_time = 0
    for tail in range(0, n + 1):
        for head in range(1, n + 2):
            if not candidate[tail, head]:
                continue
            travel_time = float(travel[physical(tail), physical(head)])
            if lower[tail] + service[tail] + travel_time > upper[head] + FEASIBILITY_EPSILON:
                removed_time += 1
                continue
            tails.append(tail)
            heads.append(head)
            times.append(travel_time)

    if not tails:
        raise ValueError(f"{instance.name}: preprocessing removed every arc")

    tail_arr = np.asarray(tails, dtype=np.int64)
    head_arr = np.asarray(heads, dtype=np.int64)
    time_arr = np.asarray(times, dtype=np.float64)
    big_m = upper[tail_arr] + service[tail_arr] + time_arr - lower[head_arr]
    np.maximum(big_m, 0.0, out=big_m)

    return PreprocessedArcs(
        n_model_nodes=n_model_nodes,
        tail=tail_arr,
        head=head_arr,
        cost=time_arr.copy(),
        time=time_arr,
        big_m=big_m,
        lower=lower,
        upper=upper,
        service=service,
        n_arcs_before=n_before,
        removed_time_window=removed_time,
        removed_pair_capacity=removed_pair,
    )


def check_customer_degrees(arcs: PreprocessedArcs, n_customers: int) -> None:
    out_degree = np.bincount(arcs.tail, minlength=arcs.n_model_nodes)
    in_degree = np.bincount(arcs.head, minlength=arcs.n_model_nodes)
    for node in range(1, n_customers + 1):
        if out_degree[node] == 0 or in_degree[node] == 0:
            raise ValueError(
                f"preprocessing left customer {node} without an incoming or "
                "outgoing arc; the instance is infeasible as modelled"
            )
