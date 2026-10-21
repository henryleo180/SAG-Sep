from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

__all__ = ["TTSConfig"]


@dataclass(frozen=True, slots=True)
class TTSConfig:
    max_prefix: int = 90
    min_size: int = 2
    lp_grow_seeds: int = 16
    lp_grow_native_rules: bool = True
    shrink_eps: float = 0.01
    compress_mode: str = "full"
    fcapfix_rounds: int = 3
    fcapfix_seed_limit: int = 32
    fcapfix_seed_mode: str = "model"
    old_repair_bank_size: int = 100000
    old_repair_top_b: int = 16
    adddrop_seed_sources: Tuple[str, ...] = ()
    adddrop_seed_old_bank: bool = True
    adddrop_seed_limit: int = 0
    score_mode: str = "efficacy"
    min_violation: float = 1e-4
    top_b: int = 120
    top_l: int = 24
    max_flips: int = 40
    min_gain: float = 1e-6
    ls_neighborhood: str = "both"
    ls_swap: bool = True
    ls_swap_max_pairs: int = 250_000
    max_size_frac: float = 1.0
    seed: int = 19
    rng_mode: str = "derived"
    flow_enabled: bool = True
    source_quotas: Tuple[Tuple[str, int], ...] = (
        ("fcapfix", 4),
        ("old_repair", 40),
        ("lp_grow", 80),
        ("adddrop", 40),
    )
    per_vehicle_quota: int = 200
    global_quota: int = 200
    near_duplicate_hamming: int = 3
    diversity_weight: float = 0.15
    vehicle_tighten_start: int = 8
    vehicle_tighten_power: float = 0.75
    vehicle_tighten_min_scale: float = 0.4
    # ---- replica budget ----------------------------------------------------------------
    max_replicas: int = 0

    def __post_init__(self) -> None:
        if self.rng_mode != "derived":
            raise ValueError("only rng_mode='derived' is supported (legacy path deleted)")
        if self.compress_mode != "full":
            raise ValueError("only compress_mode='full' is supported (one_edge path deleted)")
        if not self.lp_grow_native_rules:
            raise ValueError("only lp_grow_native_rules=True is supported")
        if self.adddrop_seed_sources != ():
            raise ValueError("adddrop_seed_sources must be () (cascade seeding deleted)")
        if self.adddrop_seed_limit != 0:
            raise ValueError("adddrop_seed_limit must be 0 (seed truncation deleted)")
        if self.score_mode != "efficacy":
            raise ValueError("only score_mode='efficacy' is supported (violation path deleted)")
        if self.ls_neighborhood != "both":
            raise ValueError(
                "only ls_neighborhood='both' is supported (add/remove-only paths deleted)"
            )
        if not self.ls_swap:
            raise ValueError("only ls_swap=True is supported (no-swap path deleted)")
        if not self.flow_enabled:
            raise ValueError("only flow_enabled=True is supported (flow bypass deleted)")
        if self.fcapfix_seed_mode != "model":
            raise ValueError(
                "only fcapfix_seed_mode='model' is supported (demand seeding deleted)"
            )
        if self.max_replicas < 0:
            raise ValueError(
                f"max_replicas must be >= 0 (0 = all K), got {self.max_replicas}"
            )
