from __future__ import annotations

from typing import Iterable, Optional, Sequence

__all__ = ["LEARNED_DEMAND_MAX", "pin_cpu_threads", "total_demand", "validate_instance"]

LEARNED_DEMAND_MAX = 1 << 24


def total_demand(demands: Iterable[object]) -> int:
    return sum(int(value) for value in demands)


def validate_instance(
    name: str,
    capacity: object,
    demands: Sequence[object],
    *,
    learned: bool,
    what: str,
    min_customer_demand: int = 0,
) -> int:
    capacity_int = int(capacity)
    if capacity_int <= 0:
        raise ValueError(f"{what}: instance {name} has capacity {capacity_int} "
                         "(must be positive)")
    if len(demands) < 2:
        raise ValueError(f"{what}: instance {name} has {len(demands)} nodes; a depot "
                         "and at least one customer are required")
    if int(demands[0]) != 0:
        raise ValueError(f"{what}: instance {name} has depot demand "
                         f"{int(demands[0])} (must be 0)")
    total = 0
    for index in range(1, len(demands)):
        demand = int(demands[index])
        if demand < min_customer_demand:
            raise ValueError(
                f"{what}: instance {name} has demand {demand} at customer {index} "
                f"(must be at least {min_customer_demand})"
            )
        if demand > capacity_int:
            raise ValueError(
                f"{what}: instance {name} has demand {demand} at customer {index}, "
                f"above the capacity {capacity_int} — no vehicle can serve it, so "
                "the instance is infeasible and its singleton capacity row has no "
                "nonzeros"
            )
        if learned and demand >= LEARNED_DEMAND_MAX:
            raise ValueError(
                f"{what}: instance {name} has demand {demand} at customer {index}; "
                "the learned separator's C API requires every demand below 2**24"
            )
        total += demand
    if learned and total >= LEARNED_DEMAND_MAX:
        raise ValueError(
            f"{what}: instance {name} has total demand {total}; the learned "
            "separator's C API requires it below 2**24"
        )
    return total


def pin_cpu_threads(threads: int) -> Optional[str]:
    try:
        import torch
    except ImportError:
        return None
    count = max(1, int(threads))
    torch.set_num_threads(count)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return f"torch_threads={torch.get_num_threads()}, interop={torch.get_num_interop_threads()}"
