from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from .types import SelectionBudget

__all__ = [
    "ModelConfig",
    "FeaturesConfig",
    "LPConfig",
    "LoopConfig",
    "SweepConfig",
    "OnionConfig",
    "TTSConfig",
    "InstanceSpec",
    "SagSepConfig",
    "PRESETS",
    "PRESET_SIZES",
    "RECIPES",
    "preset",
    "from_toml",
]


# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------

SOTA_RUN_DIR = os.environ.get("SAGSEP_RUN_DIR", "")
DEFAULT_INSTANCES_ROOT = os.environ.get("SAGSEP_INSTANCES_ROOT", "")

OXPS_QUOTAS: Tuple[Tuple[str, int], ...] = (
    ("fcapfix", 4),
    ("old_repair", 40),
    ("lp_grow", 80),
    ("adddrop", 40),
)


# ---------------------------------------------------------------------------
# Leaf config dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelConfig:
    run_dir: str = SOTA_RUN_DIR
    device: str = "cuda:0"


@dataclass(frozen=True, slots=True)
class FeaturesConfig:
    partial_connect_cross_depot: bool = True


@dataclass(frozen=True, slots=True)
class LPConfig:
    gurobi_method: int = 1
    gurobi_threads: int = 1
    lp_support_edge_threshold: float = 1e-6
    time_limit: Optional[float] = 3600.0


@dataclass(frozen=True, slots=True)
class LoopConfig:
    max_iter: Optional[int] = None
    min_violation: float = 1e-6
    cut_violation_ratio: float = 0.0
    dup_check: bool = True


@dataclass(frozen=True, slots=True)
class SweepConfig:
    threshold: float = 0.5
    min_size: int = 2
    min_violation: float = 1e-6


@dataclass(frozen=True, slots=True)
class OnionConfig:
    mode: str = "mean"
    min_violation: float = 1e-4
    threshold: float = 0.5
    cutpoint: str = "band_first"
    m_max_cap: int = 0
    skip_below_M: int = 0


@dataclass(frozen=True, slots=True)
class TTSConfig:
    enabled: bool = True
    max_prefix: int = 90
    min_size: int = 2
    lp_grow_seeds: int = 16
    lp_grow_native_rules: bool = True
    compress_mode: str = "full"
    fcapfix_rounds: int = 3
    fcapfix_seed_limit: int = 32
    fcapfix_seed_mode: str = "model"
    old_repair_bank_size: int = 100000
    old_repair_top_b: int = 16
    adddrop_seed_old_bank: bool = True
    adddrop_seed_sources: Tuple[str, ...] = ()
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
    source_quotas: Tuple[Tuple[str, int], ...] = OXPS_QUOTAS
    per_vehicle_quota: int = 200
    global_quota: int = 200
    near_duplicate_hamming: int = 3
    diversity_weight: float = 0.15
    vehicle_tighten_start: int = 8
    vehicle_tighten_power: float = 0.75
    vehicle_tighten_min_scale: float = 0.4
    max_replicas: int = 0


@dataclass(frozen=True, slots=True)
class InstanceSpec:
    name: str
    vehicles: Optional[int] = None
    optimal: Optional[float] = None
    max_iter: Optional[int] = None


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SagSepConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    lp: LPConfig = field(default_factory=LPConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    onion: OnionConfig = field(default_factory=OnionConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    budget: SelectionBudget = field(default_factory=SelectionBudget)
    compute_cut_values: bool = False
    mbp_seed: int = 0
    instances_root: str = DEFAULT_INSTANCES_ROOT
    output_dir: str = ""
    instances: Tuple[InstanceSpec, ...] = ()
    size: str = ""


# ---------------------------------------------------------------------------
# PRESETS -- the frozen per-size production configurations
# ---------------------------------------------------------------------------

_PER_SIZE: Dict[str, int] = {
    "n50": 90,
    "n75": 90,
    "n100": 90,
    "n200": 180,
    "n300": 270,
    "n400": 360,
    "n500": 360,
    "n750": 360,
    "n1000": 360,
}

PRESET_SIZES: Tuple[str, ...] = tuple(_PER_SIZE)


def _build_preset(size: str) -> SagSepConfig:
    max_prefix = _PER_SIZE[size]
    return SagSepConfig(
        model=ModelConfig(),
        features=FeaturesConfig(),
        lp=LPConfig(),
        loop=LoopConfig(),
        sweep=SweepConfig(),
        onion=OnionConfig(),
        tts=TTSConfig(max_prefix=max_prefix, max_replicas=1),
        budget=SelectionBudget(),
        compute_cut_values=False,
        mbp_seed=0,
        size=size,
    )


PRESETS: Dict[str, SagSepConfig] = {size: _build_preset(size) for size in _PER_SIZE}

RECIPES: Tuple[str, ...] = ("full", "os")


def preset(size: str, recipe: str = "full") -> SagSepConfig:
    try:
        cfg = PRESETS[size]
    except KeyError:
        raise ValueError(
            f"unknown size preset {size!r}; valid: {', '.join(PRESET_SIZES)}"
        ) from None
    if recipe == "full":
        return cfg
    if recipe == "os":
        return replace(cfg, tts=replace(cfg.tts, enabled=False))
    raise ValueError(f"unknown recipe {recipe!r}; valid: {', '.join(RECIPES)}")


# ---------------------------------------------------------------------------
# Thin TOML reader (reference configuration files only)
# ---------------------------------------------------------------------------

_MISSING = object()

_TOP_FROZEN: Dict[str, object] = {
    "decode_pipeline": "unified",
    "method": "learned",
    "policy_decode_mode": "sweep_viol",
    "root_mode": "off",
    "mbp_delta": 0.0,
    "mbp_strategies": [],
    "mbp_temperature": 1.0,
    "mbp_gate": "always",
    "mbp_kbest_max_iters": 0,
    "mbp_onion_mode": "mean",
    "mbp_onion_cutpoint": "band_first",
    "policy_emit_base_subsets": True,
    "cut_violation_ratio": 0.0,
    "dup_check": True,
    "mbp_onion_m_max": 0,
    "mbp_onion_skip_below_M": 0,
}

_TTS_FROZEN: Dict[str, object] = {
    "samples_per_tau": 0,
    "threshold_eps": [],
    "pl_enabled": False,
    "jitter_enabled": False,
    "shrink_one_edges": False,
    "sweep_topk": 0,
    "lp_grow_enabled": True,
    "fcapfix_enabled": True,
    "old_repair_enabled": True,
    "adddrop_enabled": True,
}

_TOP_REQUIRED_EXPLICIT: Tuple[str, ...] = (
    "policy_decode_mode",
    "policy_sweep_coexist",
    "policy_device",
    "mbp_onion_mode",
)
_TTS_REQUIRED_EXPLICIT: Tuple[str, ...] = (
    "enabled",
    "samples_per_tau",
    "threshold_eps",
    "pl_enabled",
    "jitter_enabled",
    "max_prefix",
    "lp_grow_seeds",
    "lp_grow_native_rules",
    "compress_mode",
    "fcapfix_enabled",
    "fcapfix_seed_limit",
    "fcapfix_seed_mode",
    "old_repair_enabled",
    "old_repair_bank_size",
    "adddrop_enabled",
    "adddrop_seed_old_bank",
    "adddrop_seed_limit",
    "adddrop_seed_sources",
    "score_mode",
    "top_b",
    "top_l",
    "max_flips",
    "ls_swap",
    "max_size_frac",
    "seed",
    "rng_mode",
    "flow_enabled",
    "source_quotas",
    "per_vehicle_quota",
    "global_quota",
    "near_duplicate_hamming",
    "diversity_weight",
    "vehicle_tighten_start",
    "vehicle_tighten_power",
    "vehicle_tighten_min_scale",
)

_FEATURES_KEYS = {
    "partial_connect_cross_depot",
}


def _require_explicit(scope: str, present: set, required: Sequence[str]) -> None:
    missing = [key for key in required if key not in present]
    if missing:
        raise ValueError(
            f"missing required key(s) {sorted(missing)} in {scope}: their SAG-Sep "
            "frozen value differs from the reference implementation's loader "
            "default, so leaving them implicit would silently run the two as "
            "different pipelines -- write them out explicitly"
        )



def _check_frozen(scope: str, key: str, value: object, expected: object) -> None:
    actual = value
    if isinstance(expected, str):
        actual = str(value).strip().lower()
    elif isinstance(expected, bool):
        actual = bool(value)
    elif isinstance(expected, float):
        actual = float(value)
    elif isinstance(expected, int):
        actual = int(value)
    elif isinstance(expected, list):
        actual = list(value)
    if actual != expected:
        raise ValueError(
            f"{scope}{key} = {value!r} deviates from the SAG-Sep frozen recipe "
            f"(the frozen value is {expected!r}); this reader refuses "
            "unsupported semantics instead of silently dropping them"
        )


def _parse_time_limit(value: object) -> Optional[float]:
    if isinstance(value, bool):
        raise ValueError(
            f"time_limit = {value!r} (bool) is almost certainly a typo; it would "
            "coerce to a 1.0/0.0 second limit -- write an explicit number instead"
        )
    time_limit: Optional[float] = None
    if isinstance(value, (int, float)):
        time_limit = float(value)
    elif isinstance(value, str):
        token = value.strip().lower()
        if token not in {"", "none", "null"}:
            time_limit = float(value)
    if time_limit is not None and time_limit <= 0.0:
        time_limit = None
    return time_limit


def _parse_gurobi_threads(value: object) -> int:
    if value in (None, ""):
        return 0
    threads = int(value)
    return threads if threads > 0 else 0


def _parse_source_quotas(value: object) -> Tuple[Tuple[str, int], ...]:
    if value is None:
        return ()
    if isinstance(value, dict):
        return tuple((str(key), int(val)) for key, val in value.items())
    parsed: List[Tuple[str, int]] = []
    for entry in value:
        source, limit = str(entry).replace("=", ":", 1).split(":", 1)
        parsed.append((source.strip(), int(limit)))
    return tuple(parsed)


def _parse_tts_table(tts_raw: Dict[str, object]) -> TTSConfig:
    data = dict(tts_raw)
    enabled = bool(data.get("enabled", True))
    if enabled:
        _require_explicit("[tts]", set(data), _TTS_REQUIRED_EXPLICIT)
    data.pop("enabled", None)
    for key, expected in _TTS_FROZEN.items():
        if key in data:
            _check_frozen("tts.", key, data.pop(key), expected)
    data.pop("taus", None)

    kwargs: Dict[str, object] = {}

    def take_int(key: str) -> None:
        if key in data:
            kwargs[key] = int(data.pop(key))

    def take_float(key: str) -> None:
        if key in data:
            kwargs[key] = float(data.pop(key))

    def take_bool(key: str) -> None:
        if key in data:
            kwargs[key] = bool(data.pop(key))

    def take_mode(key: str) -> None:
        if key in data:
            kwargs[key] = str(data.pop(key)).strip().lower()

    for key in (
        "max_prefix",
        "min_size",
        "lp_grow_seeds",
        "fcapfix_rounds",
        "fcapfix_seed_limit",
        "old_repair_bank_size",
        "old_repair_top_b",
        "adddrop_seed_limit",
        "top_b",
        "top_l",
        "max_flips",
        "ls_swap_max_pairs",
        "seed",
        "per_vehicle_quota",
        "global_quota",
        "near_duplicate_hamming",
        "vehicle_tighten_start",
        "max_replicas",
    ):
        take_int(key)
    for key in (
        "min_violation",
        "min_gain",
        "max_size_frac",
        "diversity_weight",
        "vehicle_tighten_power",
        "vehicle_tighten_min_scale",
    ):
        take_float(key)
    for key in (
        "lp_grow_native_rules",
        "adddrop_seed_old_bank",
        "ls_swap",
        "flow_enabled",
    ):
        take_bool(key)
    for key in (
        "compress_mode",
        "fcapfix_seed_mode",
        "score_mode",
        "ls_neighborhood",
        "rng_mode",
    ):
        take_mode(key)
    if "adddrop_seed_sources" in data:
        raw_sources = data.pop("adddrop_seed_sources")
        if isinstance(raw_sources, str):
            kwargs["adddrop_seed_sources"] = (raw_sources.strip(),)
        else:
            kwargs["adddrop_seed_sources"] = tuple(
                str(item).strip() for item in raw_sources
            )
    if "source_quotas" in data:
        kwargs["source_quotas"] = _parse_source_quotas(data.pop("source_quotas"))

    if data:
        raise ValueError(f"unknown [tts] key(s): {sorted(data)}")

    cfg = TTSConfig(enabled=enabled, **kwargs)

    if cfg.rng_mode != "derived":
        raise ValueError(f"tts.rng_mode must be 'derived' in SAG-Sep, got {cfg.rng_mode!r}")
    if cfg.compress_mode != "full":
        raise ValueError(f"tts.compress_mode must be 'full' in SAG-Sep, got {cfg.compress_mode!r}")
    if cfg.score_mode != "efficacy":
        raise ValueError(f"tts.score_mode must be 'efficacy' in SAG-Sep, got {cfg.score_mode!r}")
    if cfg.ls_neighborhood != "both":
        raise ValueError(f"tts.ls_neighborhood must be 'both' in SAG-Sep, got {cfg.ls_neighborhood!r}")
    if cfg.fcapfix_seed_mode != "model":
        raise ValueError(
            f"tts.fcapfix_seed_mode must be 'model' in SAG-Sep, got {cfg.fcapfix_seed_mode!r}"
        )
    if not cfg.ls_swap:
        raise ValueError("tts.ls_swap must be true in SAG-Sep (no-swap local search deleted)")
    if not cfg.flow_enabled:
        raise ValueError("tts.flow_enabled must be true in SAG-Sep (flow bypass deleted)")
    if cfg.old_repair_bank_size <= 0:
        raise ValueError(
            f"tts.old_repair_bank_size must be > 0 in SAG-Sep (the empty-bank branch "
            f"is deleted), got {cfg.old_repair_bank_size}"
        )
    if cfg.old_repair_top_b < 0:
        raise ValueError(f"tts.old_repair_top_b must be >= 0, got {cfg.old_repair_top_b}")
    if cfg.adddrop_seed_limit != 0:
        raise ValueError(
            f"tts.adddrop_seed_limit must be 0 (full pool) in SAG-Sep, got "
            f"{cfg.adddrop_seed_limit}"
        )
    if cfg.max_replicas < 0:
        raise ValueError(
            f"tts.max_replicas must be >= 0 (0 = all K), got {cfg.max_replicas}"
        )
    if not 0.0 < cfg.max_size_frac <= 1.0:
        raise ValueError(f"tts.max_size_frac must be in (0,1], got {cfg.max_size_frac}")
    if not 0.0 < cfg.vehicle_tighten_min_scale <= 1.0:
        raise ValueError(
            f"tts.vehicle_tighten_min_scale must be in (0,1], got {cfg.vehicle_tighten_min_scale}"
        )
    return cfg


def _parse_instances(raw_instances: object) -> Tuple[InstanceSpec, ...]:
    if not isinstance(raw_instances, list) or not raw_instances:
        raise ValueError("Missing or invalid 'instances' list in config")
    out: List[InstanceSpec] = []
    for entry in raw_instances:
        if not isinstance(entry, dict) or "name" not in entry:
            raise ValueError("Each instance must define 'name'")
        unknown = set(entry) - {"name", "vehicles", "optimal", "max_iter"}
        if unknown:
            raise ValueError(f"unknown instance key(s): {sorted(unknown)}")
        vehicles_raw = entry.get("vehicles")
        out.append(
            InstanceSpec(
                name=str(entry["name"]),
                vehicles=int(vehicles_raw) if vehicles_raw not in (None, "") else None,
                optimal=float(entry["optimal"]) if "optimal" in entry else None,
                max_iter=int(entry["max_iter"]) if "max_iter" in entry else None,
            )
        )
    return tuple(out)


def from_toml(path: Union[str, Path]) -> SagSepConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    data: Dict[str, object] = dict(raw)

    _require_explicit("the top level", set(data), _TOP_REQUIRED_EXPLICIT)
    for key, expected in _TOP_FROZEN.items():
        if key in data:
            _check_frozen("", key, data.pop(key), expected)
    sweep_coexist = bool(data.pop("policy_sweep_coexist"))

    if "mbp_m" in data:
        mbp_m = int(data.pop("mbp_m"))
        if mbp_m not in (0, 1):
            raise ValueError(f"mbp_m must be 0 or 1 in the frozen recipe, got {mbp_m}")

    def take(key: str, default: object = _MISSING) -> object:
        if key in data:
            return data.pop(key)
        return default

    # --- model ---
    run_dir = take("policy_run_dir", SOTA_RUN_DIR)
    model_cfg = ModelConfig(
        run_dir=str(run_dir),
        device=str(take("policy_device", "cuda:0")),
    )

    # --- features ---
    features_cfg = FeaturesConfig()
    features_raw = take("features", None)
    if features_raw is not None:
        if not isinstance(features_raw, dict):
            raise ValueError("[features] must be a table")
        unknown_features = set(features_raw) - _FEATURES_KEYS
        if unknown_features:
            raise ValueError(f"unknown [features] key(s): {sorted(unknown_features)}")
        if "partial_connect_cross_depot" in features_raw:
            cross_depot = features_raw["partial_connect_cross_depot"]
            if not isinstance(cross_depot, bool):
                raise ValueError(
                    "[features] partial_connect_cross_depot must be a bool, got "
                    f"{cross_depot!r}"
                )
            features_cfg = FeaturesConfig(partial_connect_cross_depot=cross_depot)

    # --- lp ---
    gurobi_method = int(take("gurobi_method", 1))
    if gurobi_method not in {-1, 0, 1, 2, 3, 4, 5, 6}:
        raise ValueError("gurobi_method must be one of -1, 0, 1, 2, 3, 4, 5, or 6")
    threads_raw = take("gurobi_threads", None)
    if threads_raw is None:
        threads_raw = take("gurobi_thread", None)
    else:
        take("gurobi_thread", None)
    lp_cfg = LPConfig(
        gurobi_method=gurobi_method,
        gurobi_threads=_parse_gurobi_threads(threads_raw),
        lp_support_edge_threshold=float(take("lp_support_edge_threshold", 1e-6)),
        time_limit=_parse_time_limit(take("time_limit", None)),
    )

    # --- loop ---
    max_iter_raw = take("max_iter", None)
    loop_cfg = LoopConfig(
        max_iter=int(max_iter_raw) if max_iter_raw is not None else None,
        min_violation=1e-6,
    )

    # --- sweep / onion (flat top-level keys, not a nested table) ---
    policy_threshold = float(take("policy_threshold", 0.5))
    sweep_cfg = SweepConfig(threshold=policy_threshold)
    onion_cfg = OnionConfig(
        mode="mean",
        min_violation=float(take("mbp_onion_min_violation", 1e-4)),
        threshold=policy_threshold,
        cutpoint="band_first",
    )

    mbp_seed = int(take("mbp_seed", 0))

    # --- tts ---
    tts_raw = take("tts", None)
    if not isinstance(tts_raw, dict):
        raise ValueError("reference configs must define an inline [tts] table")
    tts_cfg = _parse_tts_table(tts_raw)
    if not sweep_coexist and tts_cfg.enabled:
        raise ValueError(
            "policy_sweep_coexist = false deviates from the SAG-Sep frozen recipe while the "
            "test-time cut-refinement layer is on (the frozen value is True); it is accepted only "
            "together with [tts] enabled = false, where it is inert"
        )

    # --- cvrpsep tail (entirely out of scope: not implemented, no interface reserved) ---
    cvrpsep_raw = take("cvrpsep", None)
    if isinstance(cvrpsep_raw, dict) and int(
        cvrpsep_raw.get("rounds", cvrpsep_raw.get("max_rounds", 0))
    ) > 0 and int(
        cvrpsep_raw.get("cuts_per_round", cvrpsep_raw.get("max_cuts", 0))
    ) > 0:
        raise ValueError(
            "[cvrpsep] requests the CVRPSEP tail phase, which SAG-Sep does not "
            "implement (it is outside the frozen scope). Another implementation "
            "reading this file would append tail iterations and diverge "
            "silently -- remove the [cvrpsep] table to run the learned-only "
            "recipe here, or run that benchmark with a solver that implements it"
        )


    # --- cut manager ---
    if take("cut_manager", None) is not None:
        raise ValueError("cut_manager is not part of the released recipe")

    # --- instances / paths ---
    instances = _parse_instances(take("instances", None))
    instances_root = str(take("instances_root", DEFAULT_INSTANCES_ROOT))
    output_dir = str(take("output_dir", ""))

    if data:
        raise ValueError(f"unknown config key(s): {sorted(data)}")

    return SagSepConfig(
        model=model_cfg,
        features=features_cfg,
        lp=lp_cfg,
        loop=loop_cfg,
        sweep=sweep_cfg,
        onion=onion_cfg,
        tts=tts_cfg,
        budget=SelectionBudget(),
        compute_cut_values=False,
        mbp_seed=mbp_seed,
        instances_root=instances_root,
        output_dir=output_dir,
        instances=instances,
        size="",
    )
