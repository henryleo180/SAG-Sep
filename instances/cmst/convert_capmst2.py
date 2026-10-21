#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

SEPARATOR = re.compile(r"^\s*-{10,}\s*$")
WEIGHT_BLOCK = re.compile(r"^priz(?P<size>\d+)r\.dat$", re.IGNORECASE)
COST_BLOCK = re.compile(r"^cm(?P<size>\d+)r(?P<replica>\d+)\.dat$", re.IGNORECASE)

CAPACITIES = (200, 400, 800)


def split_blocks(text: str) -> List[Tuple[str, List[int]]]:
    blocks: List[Tuple[str, List[int]]] = []
    current: List[str] = []

    def flush() -> None:
        if not current:
            return
        name = current[0].strip()
        blocks.append((name, [int(token) for token in " ".join(current[1:]).split()]))
        current.clear()

    for line in text.splitlines():
        if SEPARATOR.match(line):
            flush()
        elif line.strip():
            current.append(line)
    flush()
    return blocks


def parse(text: str) -> Tuple[Dict[int, List[int]], Dict[str, Tuple[int, List[int]]]]:
    weights: Dict[int, List[int]] = {}
    matrices: Dict[str, Tuple[int, List[int]]] = {}
    for name, values in split_blocks(text):
        weight = WEIGHT_BLOCK.match(name)
        if weight:
            size = int(weight.group("size"))
            if len(values) != size - 1:
                raise ValueError(f"{name}: expected {size - 1} weights, got {len(values)}")
            weights[size] = values
            continue
        cost = COST_BLOCK.match(name)
        if cost is None:
            raise ValueError(f"unrecognised block {name!r}")
        if not values or values[0] != int(cost.group("size")):
            raise ValueError(f"{name}: payload size does not match the block name")
        size = values[0]
        if len(values) != 1 + size * size:
            raise ValueError(f"{name}: expected {1 + size * size} integers, got {len(values)}")
        matrices[name[: -len(".dat")].lower()] = (size, values[1:])
    missing = sorted({size for size, _ in matrices.values()} - set(weights))
    if missing:
        raise ValueError(f"no weight block for sizes {missing}")
    return weights, matrices


def write_instance(path: Path, name: str, size: int, costs: List[int], weights: List[int]) -> None:
    order = [size - 1] + list(range(size - 1))
    demand = [0] + list(weights)
    for i in range(size):
        if costs[order[i] * size + order[i]] != 0:
            raise ValueError(f"{name}: non-zero diagonal at vertex {order[i] + 1}")
        for j in range(i + 1, size):
            if costs[order[i] * size + order[j]] != costs[order[j] * size + order[i]]:
                raise ValueError(f"{name}: cost matrix is not symmetric")

    lines = [
        f"NAME: {name}",
        "TYPE: CMST",
        "COMMENT: OR-Library capmst2; root moved from vertex N to node 1",
        f"DIMENSION: {size}",
        "ROOT: 1",
        "EDGE_WEIGHT_TYPE: EXPLICIT",
        "EDGE_WEIGHT_FORMAT: FULL_MATRIX",
        "CAPACITIES: " + " ".join(str(q) for q in CAPACITIES),
        "DEMAND_SECTION",
    ]
    lines += [f"{node + 1} {demand[node]}" for node in range(size)]
    lines.append("EDGE_WEIGHT_SECTION")
    for i in range(size):
        row = costs[order[i] * size :][: size]
        lines.append(" ".join(str(row[order[j]]) for j in range(size)))
    lines.append("EOF")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert OR-Library's ``capmst2.txt`` into one ``.cmst`` file per instance.")
    parser.add_argument("capmst2", type=Path, help="path to OR-Library capmst2.txt")
    parser.add_argument("--out", type=Path, default=Path("."), help="output directory")
    args = parser.parse_args()

    weights, matrices = parse(args.capmst2.read_text(encoding="ascii"))
    args.out.mkdir(parents=True, exist_ok=True)
    for name in sorted(matrices, key=lambda key: (matrices[key][0], key)):
        size, costs = matrices[name]
        write_instance(args.out / f"{name}.cmst", name, size, costs, weights[size])
        print(f"{name}: {size - 1} customers, total demand {sum(weights[size])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
