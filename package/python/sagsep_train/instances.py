from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

__all__ = [
    "XmlSpec",
    "GeneratedInstance",
    "COORD_MAX",
    "CLUSTER_DECAY",
    "CLUSTER_SEED_RANGE",
    "ROUTE_SIZE_INTERVALS",
    "DEPOT_POS_VALUES",
    "CUSTOMER_POS_VALUES",
    "DEMAND_TYPE_VALUES",
    "ROUTE_SIZE_VALUES",
    "spec_code",
    "all_specs",
    "set_seeds",
    "generate_instance",
    "write_vrp",
    "generate_set",
    "MANIFEST_COLUMNS",
]

COORD_MAX: int = 1000

CLUSTER_DECAY: float = 40.0

CLUSTER_SEED_RANGE: tuple[int, int] = (3, 8)

ROUTE_SIZE_INTERVALS: dict[int, tuple[float, float]] = {
    1: (3.0, 5.0),
    2: (5.0, 8.0),
    3: (8.0, 12.0),
    4: (12.0, 16.0),
    5: (16.0, 25.0),
    6: (25.0, 50.0),
}

DEPOT_POS_VALUES: tuple[int, ...] = (1, 2, 3)
CUSTOMER_POS_VALUES: tuple[int, ...] = (1, 2, 3)
DEMAND_TYPE_VALUES: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
ROUTE_SIZE_VALUES: tuple[int, ...] = (1, 2, 3, 4, 5, 6)

_DEMAND_INTERVALS: dict[int, tuple[int, int]] = {
    1: (1, 1),
    2: (1, 10),
    3: (5, 10),
    4: (1, 100),
    5: (50, 100),
}

_QUADRANT_INTERVALS: tuple[tuple[int, int], tuple[int, int]] = ((51, 100), (1, 50))

_MIXED_INTERVALS: tuple[tuple[int, int], tuple[int, int]] = ((1, 10), (50, 100))
_MIXED_SMALL_FRACTION: tuple[float, float] = (0.70, 0.95)

_MAX_PLACEMENT_ATTEMPTS: int = 1_000_000

MANIFEST_COLUMNS: tuple[str, ...] = (
    "instance",
    "n",
    "vehicles",
    "optimal",
    "spec",
    "seed",
    "sha256",
)

_GENERATOR_NAME = "sagsep-train uchoa-style generator"


@dataclass(frozen=True)
class XmlSpec:
    n_customers: int
    depot_pos: int
    customer_pos: int
    demand_type: int
    route_size: int

    def __post_init__(self) -> None:
        if int(self.n_customers) < 1:
            raise ValueError(f"n_customers must be >= 1, got {self.n_customers}")
        if int(self.depot_pos) not in DEPOT_POS_VALUES:
            raise ValueError(f"depot_pos must be in {DEPOT_POS_VALUES}, got {self.depot_pos}")
        if int(self.customer_pos) not in CUSTOMER_POS_VALUES:
            raise ValueError(
                f"customer_pos must be in {CUSTOMER_POS_VALUES}, got {self.customer_pos}"
            )
        if int(self.demand_type) not in DEMAND_TYPE_VALUES:
            raise ValueError(
                f"demand_type must be in {DEMAND_TYPE_VALUES}, got {self.demand_type}"
            )
        if int(self.route_size) not in ROUTE_SIZE_VALUES:
            raise ValueError(f"route_size must be in {ROUTE_SIZE_VALUES}, got {self.route_size}")


@dataclass
class GeneratedInstance:
    name: str
    coords: np.ndarray
    demand: np.ndarray
    capacity: int
    k_fleet: int
    spec: XmlSpec
    seed: int

    @property
    def n_customers(self) -> int:
        return int(self.demand.shape[0]) - 1

    @property
    def total_demand(self) -> int:
        return int(self.demand.sum())


def spec_code(spec: XmlSpec) -> str:
    return f"{int(spec.depot_pos)}{int(spec.customer_pos)}{int(spec.demand_type)}{int(spec.route_size)}"


def all_specs(n_customers: int) -> List[XmlSpec]:
    return [
        XmlSpec(n_customers, depot, cust, demand, route)
        for depot in DEPOT_POS_VALUES
        for cust in CUSTOMER_POS_VALUES
        for demand in DEMAND_TYPE_VALUES
        for route in ROUTE_SIZE_VALUES
    ]


def set_seeds(seed: int, count: int) -> List[int]:
    if int(count) < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    children = np.random.SeedSequence(int(seed)).spawn(int(count))
    seeds: List[int] = []
    for child in children:
        words = child.generate_state(2, dtype=np.uint32)
        seeds.append(int(words[0]) | (int(words[1]) << 32))
    return seeds


# ---------------------------------------------------------------------------
# coordinates
# ---------------------------------------------------------------------------


def _draw_point(rng: np.random.Generator) -> tuple[int, int]:
    point = rng.integers(0, COORD_MAX + 1, size=2)
    return int(point[0]), int(point[1])


def _draw_depot(rng: np.random.Generator, depot_pos: int) -> tuple[int, int]:
    if depot_pos == 1:
        return _draw_point(rng)
    if depot_pos == 2:
        return COORD_MAX // 2, COORD_MAX // 2
    return 0, 0


def _accept_probability(point: tuple[int, int], seeds: np.ndarray) -> float:
    dx = seeds[:, 0] - float(point[0])
    dy = seeds[:, 1] - float(point[1])
    dist = np.hypot(dx, dy)
    return min(1.0, float(np.exp(-dist / CLUSTER_DECAY).sum()))


def _draw_coords(rng: np.random.Generator, spec: XmlSpec) -> np.ndarray:
    n = int(spec.n_customers)
    depot = _draw_depot(rng, int(spec.depot_pos))

    cluster_seeds: Optional[np.ndarray] = None
    if int(spec.customer_pos) in (2, 3):
        n_seeds = int(rng.integers(CLUSTER_SEED_RANGE[0], CLUSTER_SEED_RANGE[1] + 1))
        cluster_seeds = rng.integers(0, COORD_MAX + 1, size=(n_seeds, 2)).astype(np.float64)

    if int(spec.customer_pos) == 1:
        n_uniform = n
    elif int(spec.customer_pos) == 2:
        n_uniform = 0
    else:
        n_uniform = (n + 1) // 2

    occupied = {depot}
    coords = np.zeros((n + 1, 2), dtype=np.int64)
    coords[0] = depot
    for index in range(n):
        clustered = index >= n_uniform
        attempts = 0
        while True:
            attempts += 1
            if attempts > _MAX_PLACEMENT_ATTEMPTS:
                raise RuntimeError(
                    f"could not place customer {index + 1} of {n} after "
                    f"{_MAX_PLACEMENT_ATTEMPTS} proposals (spec {spec_code(spec)})"
                )
            point = _draw_point(rng)
            if clustered:
                assert cluster_seeds is not None
                if float(rng.random()) >= _accept_probability(point, cluster_seeds):
                    continue
            if point in occupied:
                continue
            occupied.add(point)
            coords[index + 1] = point
            break
    return coords


# ---------------------------------------------------------------------------
# demands, capacity, fleet
# ---------------------------------------------------------------------------


def _draw_demands(rng: np.random.Generator, spec: XmlSpec, coords: np.ndarray) -> np.ndarray:
    n = int(spec.n_customers)
    demand_type = int(spec.demand_type)
    demand = np.zeros(n + 1, dtype=np.int64)

    if demand_type in _DEMAND_INTERVALS:
        lo, hi = _DEMAND_INTERVALS[demand_type]
        demand[1:] = rng.integers(lo, hi + 1, size=n)
        return demand

    if demand_type == 6:
        (even_lo, even_hi), (odd_lo, odd_hi) = _QUADRANT_INTERVALS
        even_values = rng.integers(even_lo, even_hi + 1, size=n)
        odd_values = rng.integers(odd_lo, odd_hi + 1, size=n)
        xs = coords[1:, 0]
        ys = coords[1:, 1]
        half = COORD_MAX // 2
        even_quadrant = ((xs < half) & (ys < half)) | ((xs >= half) & (ys >= half))
        demand[1:] = np.where(even_quadrant, even_values, odd_values)
        return demand

    (small_lo, small_hi), (large_lo, large_hi) = _MIXED_INTERVALS
    fraction = float(rng.uniform(_MIXED_SMALL_FRACTION[0], _MIXED_SMALL_FRACTION[1]))
    n_small = int(round(fraction * n))
    order = rng.permutation(n)
    small_values = rng.integers(small_lo, small_hi + 1, size=n)
    large_values = rng.integers(large_lo, large_hi + 1, size=n)
    is_small = np.zeros(n, dtype=bool)
    is_small[order[:n_small]] = True
    demand[1:] = np.where(is_small, small_values, large_values)
    return demand


def _draw_capacity(rng: np.random.Generator, spec: XmlSpec, demand: np.ndarray) -> int:
    lo, hi = ROUTE_SIZE_INTERVALS[int(spec.route_size)]
    route_size = float(rng.uniform(lo, hi))
    total = int(demand.sum())
    n = int(spec.n_customers)
    capacity = int(math.ceil(route_size * total / n))
    return max(capacity, int(demand.max()), 1)


# ---------------------------------------------------------------------------
# public generation
# ---------------------------------------------------------------------------


def generate_instance(spec: XmlSpec, *, seed: int, name: str) -> GeneratedInstance:
    if not name:
        raise ValueError("name must be a non-empty string")
    rng = np.random.default_rng(int(seed))
    coords = _draw_coords(rng, spec)
    demand = _draw_demands(rng, spec, coords)
    capacity = _draw_capacity(rng, spec, demand)
    total = int(demand.sum())
    k_fleet = -(-total // capacity)
    return GeneratedInstance(
        name=str(name),
        coords=coords,
        demand=demand,
        capacity=int(capacity),
        k_fleet=int(k_fleet),
        spec=spec,
        seed=int(seed),
    )


def write_vrp(path: str | Path, inst: GeneratedInstance) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    n_nodes = int(inst.demand.shape[0])
    lines: List[str] = [
        f"NAME : {inst.name}",
        f"COMMENT : {_GENERATOR_NAME}, spec {spec_code(inst.spec)}, seed {int(inst.seed)}",
        "TYPE : CVRP",
        f"DIMENSION : {n_nodes}",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        f"CAPACITY : {int(inst.capacity)}",
        "NODE_COORD_SECTION",
    ]
    for node in range(n_nodes):
        lines.append(f"{node + 1} {int(inst.coords[node, 0])} {int(inst.coords[node, 1])}")
    lines.append("DEMAND_SECTION")
    for node in range(n_nodes):
        lines.append(f"{node + 1} {int(inst.demand[node])}")
    lines.append("DEPOT_SECTION")
    lines.append(" 1")
    lines.append(" -1")
    lines.append("EOF")
    text = "\n".join(lines) + "\n"

    tmp = target.with_name(target.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _resolve_specs(
    n_customers: int, count: int, specs: Optional[Sequence[XmlSpec]], rng: np.random.Generator
) -> List[XmlSpec]:
    if specs is None:
        drawn: List[XmlSpec] = []
        for _ in range(count):
            drawn.append(
                XmlSpec(
                    n_customers=n_customers,
                    depot_pos=int(rng.integers(1, len(DEPOT_POS_VALUES) + 1)),
                    customer_pos=int(rng.integers(1, len(CUSTOMER_POS_VALUES) + 1)),
                    demand_type=int(rng.integers(1, len(DEMAND_TYPE_VALUES) + 1)),
                    route_size=int(rng.integers(1, len(ROUTE_SIZE_VALUES) + 1)),
                )
            )
        return drawn

    given = list(specs)
    if not given:
        raise ValueError("specs must be None or a non-empty list")
    chosen = [given[i % len(given)] for i in range(count)]
    return [
        spec if int(spec.n_customers) == int(n_customers) else replace(spec, n_customers=int(n_customers))
        for spec in chosen
    ]


def generate_set(
    out_dir: str | Path,
    *,
    n_customers: int,
    count: int,
    seed: int,
    specs: Optional[Sequence[XmlSpec]] = None,
) -> List[dict]:
    if int(count) < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    spec_rng = np.random.default_rng(int(seed))
    chosen = _resolve_specs(int(n_customers), int(count), specs, spec_rng)
    seeds = set_seeds(int(seed), int(count))

    rows: List[dict] = []
    for index, (spec, instance_seed) in enumerate(zip(chosen, seeds)):
        name = f"uchoa-n{int(n_customers)}-{spec_code(spec)}-{index:03d}"
        inst = generate_instance(spec, seed=instance_seed, name=name)
        vrp_path = out_path / f"{name}.vrp"
        write_vrp(vrp_path, inst)
        digest = hashlib.sha256(vrp_path.read_bytes()).hexdigest()
        rows.append(
            {
                "instance": name,
                "n": int(n_customers),
                "vehicles": int(inst.k_fleet),
                "optimal": "",
                "spec": spec_code(spec),
                "seed": int(instance_seed),
                "sha256": digest,
            }
        )

    manifest = out_path / "manifest.csv"
    tmp = manifest.with_name(manifest.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        tmp.replace(manifest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return rows
