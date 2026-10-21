#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "common" / "python"))

from sagsep_bc_directed.driver import ProblemSpec, main

OVRP = ProblemSpec(
    key="ovrp",
    prog="run_bc.py",
    summary="Branch-and-cut for the open VRP with RCC separation.",
    open_routes=True,
    distance_flag=True,
)


if __name__ == "__main__":
    sys.exit(main(OVRP))
