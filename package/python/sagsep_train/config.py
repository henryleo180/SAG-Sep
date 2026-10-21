from __future__ import annotations

import copy
import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

__all__ = [
    "RECIPE_SCHEMA_VERSION",
    "MODEL_CONFIG_KEYS",
    "DENIED_KEYS",
    "SUPPORT_RULES",
    "SHUFFLE_MODES",
    "OPTIMIZERS",
    "SCHEDULERS",
    "LOSSES",
    "RecipeError",
    "TransformerParams",
    "PolicyParams",
    "ModelSection",
    "PartialConnectEdgeValue",
    "DataSection",
    "OptimizerSection",
    "SchedulerSection",
    "TrainerSection",
    "LoggingSection",
    "TrainingRecipe",
    "MigrationNote",
    "MigrationReport",
    "load_recipe",
    "parse_recipe",
    "apply_overrides",
    "recipe_to_dict",
    "write_snapshot",
    "from_legacy_snapshot",
]

RECIPE_SCHEMA_VERSION = 1

MODEL_CONFIG_KEYS: tuple[str, ...] = (
    "node_dim",
    "edge_dim",
    "latent_dim",
    "n_layer",
    "backbone_type",
    "transformer_params",
    "transformer_params.num_heads",
    "transformer_params.dropout",
    "transformer_params.use_edge_features",
    "policy_params",
    "policy_params.hidden",
    "policy_params.out_act",
    "debug",
)

DENIED_KEYS: frozenset[str] = frozenset(
    {
        "pre_dataset_filter",
        "head_sampling",
        "tailoff",
        "train_graph_keep_prob",
        "expansion_sampling",
        "lambda_band",
        "label_mode",
        "mfree_mode",
        "positive_weight",
        "instance_equal_weight",
        "symmetric_loss",
        "amp",
        "precision",
        "compile_model",
        "fork_from",
        "contract_config",
        "manual_splits",
        "sharded",
        "train_fraction",
        "val_fraction",
        "test_fraction",
        "most_violate_mode",
        "fcm15",
        "onion_only",
        "eval_devices",
        "eval_microbatch_size",
        "m_value_lower",
        "m_value_upper",
        "m_value_method",
        "m_value_size_cap",
        "m_value_sqrt_coef",
        "m_value_linear_coef",
    }
)

SUPPORT_RULES: tuple[str, ...] = ("x_gt_0", "x_ge_1e-6")

SHUFFLE_MODES: tuple[str, ...] = ("global", "shard")

OPTIMIZERS: tuple[str, ...] = ("Adam", "AdamW")
SCHEDULERS: tuple[str, ...] = ("CosineAnnealingWarmRestarts", "none")
LOSSES: tuple[str, ...] = ("BCEWithLogitsLoss",)

_FROZEN_FEATURES: dict[str, Any] = {
    "depot_complete": True,
    "partial_connect": 2,
    "m_value_closed_interval": True,
    "partial_connect_edge_value.method": "max_min",
    "partial_connect_edge_value.coef": 1.0,
}


class RecipeError(ValueError):
    pass


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


@dataclass
class TransformerParams:
    num_heads: int = 8
    dropout: float = 0.1
    use_edge_features: bool = True


@dataclass
class PolicyParams:
    hidden: list[int] = field(default_factory=lambda: [128, 64])
    out_act: str = "Identity"
    hidden_act: str | None = None
    bias: bool | None = None


@dataclass
class ModelSection:
    node_dim: int = 2
    edge_dim: int = 2
    latent_dim: int = 256
    n_layer: int = 6
    backbone_type: str = "transformer"
    transformer_params: TransformerParams = field(default_factory=TransformerParams)
    policy_params: PolicyParams = field(default_factory=PolicyParams)
    debug: bool = False

    def model_config_dict(self) -> dict[str, Any]:
        policy: dict[str, Any] = {
            "hidden": list(self.policy_params.hidden),
            "out_act": self.policy_params.out_act,
        }
        if self.policy_params.hidden_act is not None:
            policy["hidden_act"] = self.policy_params.hidden_act
        if self.policy_params.bias is not None:
            policy["bias"] = self.policy_params.bias
        return {
            "node_dim": self.node_dim,
            "edge_dim": self.edge_dim,
            "latent_dim": self.latent_dim,
            "n_layer": self.n_layer,
            "backbone_type": self.backbone_type,
            "transformer_params": {
                "num_heads": self.transformer_params.num_heads,
                "dropout": self.transformer_params.dropout,
                "use_edge_features": self.transformer_params.use_edge_features,
            },
            "policy_params": policy,
            "debug": self.debug,
        }


@dataclass
class PartialConnectEdgeValue:
    method: str = "max_min"
    coef: float = 1.0


@dataclass
class DataSection:
    dataset_dir: str = ""
    manifest: str = "dataset-manifest.json"
    train_shards: list[str] = field(default_factory=list)
    val_shards: list[str] = field(default_factory=list)
    val_limit: int | None = None
    batch_size: int = 16
    eval_batch_size: int = 16
    shuffle_mode: str = "global"
    num_workers: int = 0
    drop_last: bool = False
    pin_memory: bool = False
    support_rule: str = "x_gt_0"
    depot_complete: bool = True
    partial_connect: int = 2
    partial_connect_edge_value: PartialConnectEdgeValue = field(
        default_factory=PartialConnectEdgeValue
    )
    partial_connect_cross_depot: bool = True
    m_value_closed_interval: bool = True


@dataclass
class OptimizerSection:
    name: str = "Adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8


@dataclass
class SchedulerSection:
    name: str = "CosineAnnealingWarmRestarts"
    T_0: int = 200
    T_mult: int = 1
    eta_min: float = 1e-5


@dataclass
class TrainerSection:
    seed: int = 42
    device: str = "cuda:0"
    max_epochs: int = 500
    max_steps: int | None = None
    log_every: int = 100
    eval_every: int = 500
    checkpoint_every: int = 1000
    loss: str = "BCEWithLogitsLoss"
    grad_clip_norm: float | None = 0.8
    resume_from: str | None = None
    deterministic: bool = False


@dataclass
class LoggingSection:
    jsonl: bool = True
    wandb: bool = False
    project: str = ""
    run_name: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class TrainingRecipe:
    recipe_schema_version: int = RECIPE_SCHEMA_VERSION
    model: ModelSection = field(default_factory=ModelSection)
    data: DataSection = field(default_factory=DataSection)
    optimizer: OptimizerSection = field(default_factory=OptimizerSection)
    scheduler: SchedulerSection = field(default_factory=SchedulerSection)
    trainer: TrainerSection = field(default_factory=TrainerSection)
    logging: LoggingSection = field(default_factory=LoggingSection)
    metadata: dict[str, Any] = field(default_factory=dict)

    def model_config_dict(self) -> dict[str, Any]:
        return self.model.model_config_dict()


# ---------------------------------------------------------------------------
# leaf parsers
# ---------------------------------------------------------------------------


def _fail(path: str, message: str) -> None:
    raise RecipeError(f"{path}: {message}")


def _type_name(value: Any) -> str:
    return type(value).__name__


def _parse_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(path, f"expected a boolean, got {_type_name(value)}")
    return bool(value)


def _parse_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, f"expected an integer, got {_type_name(value)}")
    return int(value)


def _parse_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, f"expected a number, got {_type_name(value)}")
    return float(value)


def _parse_str(value: Any, path: str) -> str:
    if not isinstance(value, str):
        _fail(path, f"expected a string, got {_type_name(value)}")
    return value


def _optional(parser: Callable[[Any, str], Any]) -> Callable[[Any, str], Any]:
    def parse(value: Any, path: str) -> Any:
        if value is None:
            return None
        return parser(value, path)

    return parse


def _choice(parser: Callable[[Any, str], Any], allowed: Sequence[Any]) -> Callable[[Any, str], Any]:
    def parse(value: Any, path: str) -> Any:
        parsed = parser(value, path)
        if parsed not in allowed:
            _fail(path, f"must be one of {list(allowed)}, got {parsed!r}")
        return parsed

    return parse


def _list_of(
    parser: Callable[[Any, str], Any], *, length: int | None = None
) -> Callable[[Any, str], list[Any]]:
    def parse(value: Any, path: str) -> list[Any]:
        if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
            _fail(path, f"expected a list, got {_type_name(value)}")
        if length is not None and len(value) != length:
            _fail(path, f"expected {length} entries, got {len(value)}")
        return [parser(item, f"{path}[{i}]") for i, item in enumerate(value)]

    return parse


def _parse_shard_name(value: Any, path: str) -> str:
    name = _parse_str(value, path)
    if not name.strip():
        _fail(path, "shard name must not be empty")
    if "/" in name or "\\" in name or name in (".", ".."):
        _fail(path, f"shard name must be a bare file name, got {name!r}")
    return name


def _frozen(path: str, value: Any) -> Any:
    expected = _FROZEN_FEATURES[path]
    if value != expected or isinstance(value, bool) != isinstance(expected, bool):
        raise RecipeError(
            f"data.{path} = {value!r}: only {expected!r} is implemented in this release "
            "(the released feature builder computes exactly one graph construction)"
        )
    return value


# ---------------------------------------------------------------------------
# section parsers
# ---------------------------------------------------------------------------


def _as_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, f"expected a table/mapping, got {_type_name(value)}")
    out: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            _fail(path, f"keys must be strings, got {_type_name(key)}")
        out[key] = item
    return out


def _section(
    payload: Any,
    path: str,
    fields: Mapping[str, Callable[[Any, str], Any]],
) -> dict[str, Any]:
    mapping = _as_mapping(payload, path)
    unknown = [key for key in mapping if key not in fields]
    if unknown:
        dotted = ", ".join(f"{path}.{key}" if path else key for key in sorted(unknown))
        raise RecipeError(f"unknown key(s): {dotted}")
    out: dict[str, Any] = {}
    for key, parser in fields.items():
        if key in mapping:
            out[key] = parser(mapping[key], f"{path}.{key}" if path else key)
    return out


def _parse_transformer_params(value: Any, path: str) -> TransformerParams:
    return TransformerParams(
        **_section(
            value,
            path,
            {
                "num_heads": _parse_int,
                "dropout": _parse_float,
                "use_edge_features": _parse_bool,
            },
        )
    )


def _parse_policy_params(value: Any, path: str) -> PolicyParams:
    return PolicyParams(
        **_section(
            value,
            path,
            {
                "hidden": _list_of(_parse_int),
                "out_act": _parse_str,
                "hidden_act": _optional(_parse_str),
                "bias": _optional(_parse_bool),
            },
        )
    )


def _parse_model(value: Any, path: str = "model") -> ModelSection:
    return ModelSection(
        **_section(
            value,
            path,
            {
                "node_dim": _parse_int,
                "edge_dim": _parse_int,
                "latent_dim": _parse_int,
                "n_layer": _parse_int,
                "backbone_type": _parse_str,
                "transformer_params": _parse_transformer_params,
                "policy_params": _parse_policy_params,
                "debug": _parse_bool,
            },
        )
    )


def _parse_edge_value(value: Any, path: str) -> PartialConnectEdgeValue:
    parsed = _section(value, path, {"method": _parse_str, "coef": _parse_float})
    if "method" in parsed:
        _frozen("partial_connect_edge_value.method", parsed["method"])
    if "coef" in parsed:
        _frozen("partial_connect_edge_value.coef", parsed["coef"])
    return PartialConnectEdgeValue(**parsed)


def _parse_data(value: Any, path: str = "data") -> DataSection:
    parsed = _section(
        value,
        path,
        {
            "dataset_dir": _parse_str,
            "manifest": _parse_str,
            "train_shards": _list_of(_parse_shard_name),
            "val_shards": _list_of(_parse_shard_name),
            "val_limit": _optional(_parse_int),
            "batch_size": _parse_int,
            "eval_batch_size": _parse_int,
            "shuffle_mode": _choice(_parse_str, SHUFFLE_MODES),
            "num_workers": _parse_int,
            "drop_last": _parse_bool,
            "pin_memory": _parse_bool,
            "support_rule": _choice(_parse_str, SUPPORT_RULES),
            "depot_complete": _parse_bool,
            "partial_connect": _parse_int,
            "partial_connect_edge_value": _parse_edge_value,
            "partial_connect_cross_depot": _parse_bool,
            "m_value_closed_interval": _parse_bool,
        },
    )
    if "dataset_dir" not in parsed or not parsed["dataset_dir"].strip():
        raise RecipeError("data.dataset_dir: required, and must not be empty")
    for key in ("depot_complete", "partial_connect", "m_value_closed_interval"):
        if key in parsed:
            _frozen(key, parsed[key])
    for key in ("batch_size", "eval_batch_size"):
        if key in parsed and parsed[key] < 1:
            _fail(f"{path}.{key}", f"must be >= 1, got {parsed[key]}")
    if "num_workers" in parsed and parsed["num_workers"] < 0:
        _fail(f"{path}.num_workers", f"must be >= 0, got {parsed['num_workers']}")
    if parsed.get("val_limit") is not None and parsed["val_limit"] < 1:
        _fail(f"{path}.val_limit", f"must be >= 1 or null, got {parsed['val_limit']}")
    return DataSection(**parsed)


def _parse_optimizer(value: Any, path: str = "optimizer") -> OptimizerSection:
    return OptimizerSection(
        **_section(
            value,
            path,
            {
                "name": _choice(_parse_str, OPTIMIZERS),
                "lr": _parse_float,
                "weight_decay": _parse_float,
                "betas": _list_of(_parse_float, length=2),
                "eps": _parse_float,
            },
        )
    )


def _parse_scheduler(value: Any, path: str = "scheduler") -> SchedulerSection:
    return SchedulerSection(
        **_section(
            value,
            path,
            {
                "name": _choice(_parse_str, SCHEDULERS),
                "T_0": _parse_int,
                "T_mult": _parse_int,
                "eta_min": _parse_float,
            },
        )
    )


def _parse_trainer(value: Any, path: str = "trainer") -> TrainerSection:
    return TrainerSection(
        **_section(
            value,
            path,
            {
                "seed": _parse_int,
                "device": _parse_str,
                "max_epochs": _parse_int,
                "max_steps": _optional(_parse_int),
                "log_every": _parse_int,
                "eval_every": _parse_int,
                "checkpoint_every": _parse_int,
                "loss": _choice(_parse_str, LOSSES),
                "grad_clip_norm": _optional(_parse_float),
                "resume_from": _optional(_parse_str),
                "deterministic": _parse_bool,
            },
        )
    )


def _parse_logging(value: Any, path: str = "logging") -> LoggingSection:
    return LoggingSection(
        **_section(
            value,
            path,
            {
                "jsonl": _parse_bool,
                "wandb": _parse_bool,
                "project": _parse_str,
                "run_name": _parse_str,
                "tags": _list_of(_parse_str),
            },
        )
    )


def _parse_metadata(value: Any, path: str = "metadata") -> dict[str, Any]:
    return copy.deepcopy(_as_mapping(value, path))


_RECIPE_SECTIONS: dict[str, Callable[[Any, str], Any]] = {
    "model": _parse_model,
    "data": _parse_data,
    "optimizer": _parse_optimizer,
    "scheduler": _parse_scheduler,
    "trainer": _parse_trainer,
    "logging": _parse_logging,
    "metadata": _parse_metadata,
}


# ---------------------------------------------------------------------------
# deny-list scan
# ---------------------------------------------------------------------------


def _scan_denied(payload: Any, prefix: str = "") -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            dotted = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(key, str) and key in DENIED_KEYS:
                raise RecipeError(f"{dotted}: unsupported by this release")
            if prefix == "" and key == "metadata":
                continue
            _scan_denied(value, dotted)
    elif isinstance(payload, (list, tuple)):
        for i, item in enumerate(payload):
            _scan_denied(item, f"{prefix}[{i}]")


# ---------------------------------------------------------------------------
# parsing entry points
# ---------------------------------------------------------------------------


def _is_annotation(value: Any) -> bool:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return True
    if isinstance(value, (list, tuple)):
        return all(
            isinstance(item, (str, int, float, bool)) or item is None for item in value
        )
    return False


def parse_recipe(payload: Mapping[str, Any]) -> TrainingRecipe:
    mapping = _as_mapping(payload, "")
    _scan_denied(mapping)

    version = mapping.get("recipe_schema_version", RECIPE_SCHEMA_VERSION)
    version = _parse_int(version, "recipe_schema_version")
    if version != RECIPE_SCHEMA_VERSION:
        raise RecipeError(
            f"recipe_schema_version: this build reads version {RECIPE_SCHEMA_VERSION}, got {version}"
        )

    unknown_sections = [
        key
        for key, value in mapping.items()
        if key not in _RECIPE_SECTIONS
        and key != "recipe_schema_version"
        and not _is_annotation(value)
    ]
    if unknown_sections:
        raise RecipeError(f"unknown key(s): {', '.join(sorted(unknown_sections))}")

    sections: dict[str, Any] = {"recipe_schema_version": version}
    for key, parser in _RECIPE_SECTIONS.items():
        if key in mapping:
            sections[key] = parser(mapping[key], key)
    if "data" not in sections:
        raise RecipeError("data.dataset_dir: required, and must not be empty")
    return TrainingRecipe(**sections)


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib as toml_reader
    except ModuleNotFoundError:
        try:
            import tomli as toml_reader
        except ModuleNotFoundError as exc:
            raise RecipeError(
                f"{path}: reading TOML needs python 3.11+ (tomllib) or the tomli package"
            ) from exc
    with open(path, "rb") as handle:
        return toml_reader.load(handle)


def _load_yaml(path: Path) -> Any:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_recipe(path: str | os.PathLike[str]) -> TrainingRecipe:
    src = Path(path)
    suffix = src.suffix.lower()
    if suffix == ".toml":
        payload = _load_toml(src)
    elif suffix in (".yaml", ".yml"):
        payload = _load_yaml(src)
    else:
        raise RecipeError(f"{src}: unsupported recipe format {suffix!r} (use .toml, .yaml or .yml)")
    if not isinstance(payload, Mapping):
        raise RecipeError(f"{src}: a recipe file must hold a mapping, got {_type_name(payload)}")

    try:
        recipe = parse_recipe(payload)
    except RecipeError as exc:
        raise RecipeError(f"{src}: {exc}") from None

    base = src.resolve().parent
    if not Path(recipe.data.dataset_dir).is_absolute():
        recipe.data.dataset_dir = str(base / recipe.data.dataset_dir)
    return recipe


# ---------------------------------------------------------------------------
# overrides
# ---------------------------------------------------------------------------


def _parse_override_value(text: str, dotted: str) -> Any:
    stripped = text.strip()
    if stripped == "null":
        return None
    try:
        import tomllib as toml_reader
    except ModuleNotFoundError:
        try:
            import tomli as toml_reader
        except ModuleNotFoundError as exc:
            raise RecipeError(
                f"{dotted}: parsing an override needs python 3.11+ (tomllib) or tomli"
            ) from exc
    try:
        return toml_reader.loads(f"value = {stripped}")["value"]
    except Exception:
        return text


def _set_dotted(payload: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node: Any = payload
    for depth, part in enumerate(parts[:-1]):
        if not isinstance(node, dict) or part not in node:
            raise RecipeError(f"unknown override path: {'.'.join(parts[: depth + 1])}")
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        raise RecipeError(f"unknown override path: {dotted}")
    node[parts[-1]] = value


def apply_overrides(recipe: TrainingRecipe, overrides: Iterable[str]) -> TrainingRecipe:
    payload = recipe_to_dict(recipe)
    for item in overrides:
        if not isinstance(item, str) or "=" not in item:
            raise RecipeError(f"override must look like key.path=value, got {item!r}")
        dotted, _, raw = item.partition("=")
        dotted = dotted.strip()
        if not dotted:
            raise RecipeError(f"override must look like key.path=value, got {item!r}")
        _set_dotted(payload, dotted, _parse_override_value(raw, dotted))
    return parse_recipe(payload)


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def recipe_to_dict(recipe: TrainingRecipe) -> dict[str, Any]:
    if not isinstance(recipe, TrainingRecipe):
        raise RecipeError(f"recipe_to_dict: expected a TrainingRecipe, got {_type_name(recipe)}")
    payload = _plain(recipe)
    payload["model"] = recipe.model.model_config_dict()
    return payload


def write_snapshot(
    recipe: TrainingRecipe,
    path: str | os.PathLike[str],
    *,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    payload = recipe_to_dict(recipe)
    if extra:
        for key, value in extra.items():
            if not isinstance(key, str):
                raise RecipeError(f"write_snapshot: extra keys must be strings, got {key!r}")
            if key in payload:
                raise RecipeError(
                    f"write_snapshot: extra key {key!r} collides with a recipe section"
                )
            if not _is_annotation(value):
                raise RecipeError(
                    f"write_snapshot: extra[{key!r}] must be a scalar or a list of scalars"
                )
            payload[key] = _plain(value)

    import yaml

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)
    return target


# ---------------------------------------------------------------------------
# legacy snapshot migration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MigrationNote:
    key: str
    action: str
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.action}: {self.key}" + (f" ({self.detail})" if self.detail else "")


@dataclass(frozen=True)
class MigrationReport:
    notes: tuple[MigrationNote, ...] = ()

    def _of(self, action: str) -> tuple[str, ...]:
        return tuple(note.key for note in self.notes if note.action == action)

    @property
    def migrated(self) -> tuple[str, ...]:
        return self._of("migrated")

    @property
    def dropped(self) -> tuple[str, ...]:
        return self._of("dropped")

    @property
    def defaulted(self) -> tuple[str, ...]:
        return self._of("defaulted")

    @property
    def legacy_keys(self) -> tuple[str, ...]:
        return self.migrated + self.dropped

    def as_lines(self) -> list[str]:
        return [str(note) for note in self.notes]


class _Migration:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.payload = payload
        self.notes: list[MigrationNote] = []

    def migrated(self, key: str, detail: str = "") -> None:
        self.notes.append(MigrationNote(key, "migrated", detail))

    def dropped(self, key: str, detail: str = "") -> None:
        self.notes.append(MigrationNote(key, "dropped", detail))

    def defaulted(self, key: str, detail: str = "") -> None:
        self.notes.append(MigrationNote(key, "defaulted", detail))

    def section(self, name: str) -> dict[str, Any]:
        value = self.payload.get(name)
        if value is None:
            return {}
        return _as_mapping(value, name)


def _legacy_reject(path: str, value: Any, expected: Any) -> None:
    raise RecipeError(
        f"{path} = {value!r} is not supported by this release "
        f"(only {expected!r} can be migrated)"
    )


def _legacy_neutral(
    mapping: Mapping[str, Any],
    path: str,
    key: str,
    expected: Any,
    mig: _Migration,
    *,
    detail: str = "",
) -> None:
    if key not in mapping:
        return
    value = mapping[key]
    ok = value == expected and isinstance(value, bool) == isinstance(expected, bool)
    if expected is None:
        ok = value is None
    if not ok:
        _legacy_reject(f"{path}.{key}", value, expected)
    mig.dropped(f"{path}.{key}", detail)


def _legacy_unknown(mapping: Mapping[str, Any], path: str, allowed: Iterable[str]) -> None:
    known = set(allowed)
    unknown = sorted(key for key in mapping if key not in known)
    if unknown:
        dotted = ", ".join(f"{path}.{key}" if path else key for key in unknown)
        raise RecipeError(f"legacy key(s) this release cannot migrate: {dotted}")


def _legacy_shard_names(selectors: Any, path: str) -> list[str]:
    if isinstance(selectors, (str, int)) and not isinstance(selectors, bool):
        selectors = [selectors]
    if not isinstance(selectors, (list, tuple)):
        _fail(path, f"expected a list of selectors, got {_type_name(selectors)}")

    names: list[str] = []
    seen: set[str] = set()

    def add(index: int) -> None:
        if index < 1:
            _fail(path, f"selector index must be >= 1, got {index}")
        name = f"shard_{index:04d}"
        if name not in seen:
            seen.add(name)
            names.append(name)

    for i, selector in enumerate(selectors):
        item_path = f"{path}[{i}]"
        if isinstance(selector, bool):
            _fail(item_path, "expected an integer index or an 'a-b' range")
        if isinstance(selector, int):
            add(selector)
            continue
        if not isinstance(selector, str):
            _fail(item_path, f"expected an integer index or an 'a-b' range, got {_type_name(selector)}")
        token = selector.strip()
        if token.isdigit():
            add(int(token))
            continue
        if token.count("-") == 1:
            lhs, _, rhs = token.partition("-")
            if lhs.strip().isdigit() and rhs.strip().isdigit():
                start, end = int(lhs), int(rhs)
                if end < start:
                    _fail(item_path, f"range must be ascending, got {token!r}")
                for index in range(start, end + 1):
                    add(index)
                continue
        _fail(
            item_path,
            f"selector {token!r} is not an index or an 'a-b' range; "
            "this release names shards explicitly",
        )
    return names


def _migrate_data(mig: _Migration) -> DataSection:
    data = mig.section("data")
    allowed = {
        "dataset_path",
        "train_fraction",
        "val_fraction",
        "test_fraction",
        "batch_size",
        "eval_batch_size",
        "shuffle",
        "num_workers",
        "drop_last",
        "pin_memory",
        "train_limit",
        "val_limit",
        "test_limit",
        "contract_config",
        "dataset_type",
        "sharded",
        "manual_splits",
        "flat_files",
        "flat_pattern",
        "depot_complete",
        "partial_connect",
        "partial_connect_hop_attr",
        "partial_connect_edge_value",
        "partial_connect_cross_depot",
        "most_violate_mode",
        "m_value_closed_interval",
    }
    _legacy_unknown(data, "data", allowed)

    out: dict[str, Any] = {}

    if "dataset_path" not in data or not str(data["dataset_path"]).strip():
        raise RecipeError("data.dataset_path: required by the migration, and must not be empty")
    out["dataset_dir"] = str(data["dataset_path"])
    mig.migrated("data.dataset_path", "-> data.dataset_dir, kept as given")

    for key, expected in (
        ("train_fraction", 1.0),
        ("val_fraction", 0.0),
        ("test_fraction", 0.0),
        ("train_limit", None),
        ("test_limit", None),
        ("contract_config", None),
        ("dataset_type", "flat"),
        ("sharded", None),
        ("manual_splits", None),
        ("partial_connect_hop_attr", "edge_hop_type"),
        ("most_violate_mode", False),
    ):
        _legacy_neutral(data, "data", key, expected, mig)
    if "flat_pattern" in data:
        mig.dropped("data.flat_pattern", "shards are named explicitly")

    flat_files = data.get("flat_files")
    if flat_files is not None:
        mapping = _as_mapping(flat_files, "data.flat_files")
        unknown = sorted(key for key in mapping if key.lower() not in ("train", "val"))
        if unknown:
            raise RecipeError(
                "data.flat_files: only the 'train' and 'val' splits can be migrated, "
                f"got {unknown}"
            )
        for split, target in (("train", "train_shards"), ("val", "val_shards")):
            selectors = mapping.get(split)
            if selectors is None:
                continue
            out[target] = _legacy_shard_names(selectors, f"data.flat_files.{split}")
        mig.migrated("data.flat_files", "-> data.train_shards / data.val_shards")

    if "shuffle" in data:
        if data["shuffle"] is not True:
            _legacy_reject("data.shuffle", data["shuffle"], True)
        out["shuffle_mode"] = "global"
        mig.migrated("data.shuffle", "-> data.shuffle_mode = 'global'")

    for key in ("batch_size", "eval_batch_size", "num_workers", "drop_last", "pin_memory", "val_limit"):
        if key in data:
            out[key] = data[key]
            mig.migrated(f"data.{key}")

    for key in (
        "depot_complete",
        "partial_connect",
        "partial_connect_edge_value",
        "partial_connect_cross_depot",
        "m_value_closed_interval",
    ):
        if key in data:
            out[key] = data[key]
            mig.migrated(f"data.{key}")

    mig.defaulted("data.manifest", "the dataset manifest is a new artefact")
    mig.defaulted("data.support_rule", "'x_gt_0' reproduces the legacy training graphs")

    return _parse_data(out)


def _migrate_trainer(mig: _Migration) -> TrainerSection:
    trainer = mig.section("trainer")
    allowed = {
        "seed",
        "device",
        "max_epochs",
        "log_every",
        "eval_every",
        "checkpoint_every",
        "checkpoint_dir",
        "run_id",
        "resume_from",
        "loss",
        "threshold",
        "grad_clip_norm",
        "amp",
        "positive_weight",
        "symmetric_loss",
        "instance_equal_weight",
        "eval_devices",
        "eval_microbatch_size",
    }
    _legacy_unknown(trainer, "trainer", allowed)

    out: dict[str, Any] = {}
    for key in ("seed", "device", "max_epochs", "log_every", "eval_every", "checkpoint_every", "loss"):
        if key in trainer:
            out[key] = trainer[key]
            mig.migrated(f"trainer.{key}")
    if "grad_clip_norm" in trainer:
        out["grad_clip_norm"] = trainer["grad_clip_norm"]
        mig.migrated("trainer.grad_clip_norm")
    if "resume_from" in trainer:
        if trainer["resume_from"] is not None:
            _legacy_reject("trainer.resume_from", trainer["resume_from"], None)
        out["resume_from"] = None
        mig.migrated("trainer.resume_from")

    for key, detail in (
        ("checkpoint_dir", "the run directory is a runtime argument"),
        ("run_id", "the run directory is a runtime argument"),
        ("threshold", "legacy trainer key; ignored"),
    ):
        if key in trainer:
            mig.dropped(f"trainer.{key}", detail)
    for key, expected in (
        ("amp", False),
        ("positive_weight", False),
        ("symmetric_loss", False),
        ("instance_equal_weight", False),
        ("eval_devices", None),
        ("eval_microbatch_size", None),
    ):
        _legacy_neutral(trainer, "trainer", key, expected, mig)

    mig.defaulted("trainer.max_steps")
    mig.defaulted("trainer.deterministic")
    return _parse_trainer(out)


def _migrate_logging(mig: _Migration) -> LoggingSection:
    logging_cfg = mig.section("logging")
    _legacy_unknown(
        logging_cfg,
        "logging",
        {"enable_wandb", "project", "entity", "group", "run_name", "mode", "tags"},
    )
    out: dict[str, Any] = {}
    if "enable_wandb" in logging_cfg:
        out["wandb"] = logging_cfg["enable_wandb"]
        mig.migrated("logging.enable_wandb", "-> logging.wandb")
    for key in ("project", "run_name"):
        if key in logging_cfg:
            out[key] = "" if logging_cfg[key] is None else logging_cfg[key]
            mig.migrated(f"logging.{key}")
    if "tags" in logging_cfg:
        out["tags"] = list(logging_cfg["tags"] or [])
        mig.migrated("logging.tags")
    for key in ("entity", "group", "mode"):
        if key in logging_cfg:
            mig.dropped(f"logging.{key}", "not part of the released logging surface")
    mig.defaulted("logging.jsonl", "the JSONL run log is a new artefact")
    return _parse_logging(out)


def from_legacy_snapshot(
    path_or_dict: str | os.PathLike[str] | Mapping[str, Any],
) -> tuple[TrainingRecipe, MigrationReport]:
    if isinstance(path_or_dict, Mapping):
        payload: Mapping[str, Any] = path_or_dict
    else:
        src = Path(path_or_dict)
        if src.suffix.lower() in (".yaml", ".yml"):
            loaded = _load_yaml(src)
        elif src.suffix.lower() == ".toml":
            loaded = _load_toml(src)
        else:
            raise RecipeError(f"{src}: unsupported snapshot format {src.suffix!r}")
        if not isinstance(loaded, Mapping):
            raise RecipeError(f"{src}: a snapshot must hold a mapping")
        payload = loaded

    mapping = _as_mapping(payload, "")
    _legacy_unknown(mapping, "", {"model", "data", "optimizer", "scheduler", "trainer", "logging"})

    mig = _Migration(mapping)

    model_raw = mapping.get("model")
    if model_raw is None:
        raise RecipeError("model: required by the migration")
    model = _parse_model(model_raw, "model")
    for key in _as_mapping(model_raw, "model"):
        mig.migrated(f"model.{key}", "verbatim")

    data = _migrate_data(mig)

    optimizer_raw = mig.section("optimizer")
    for key in optimizer_raw:
        mig.migrated(f"optimizer.{key}")
    optimizer = _parse_optimizer(optimizer_raw, "optimizer")

    scheduler_raw = dict(mig.section("scheduler"))
    if "gamma" in scheduler_raw:
        scheduler_raw.pop("gamma")
        mig.dropped("scheduler.gamma", "unused by CosineAnnealingWarmRestarts")
    for key in scheduler_raw:
        mig.migrated(f"scheduler.{key}")
    scheduler = _parse_scheduler(scheduler_raw, "scheduler")

    trainer = _migrate_trainer(mig)
    logging_section = _migrate_logging(mig)

    mig.defaulted("recipe_schema_version")
    mig.defaulted("metadata")

    recipe = TrainingRecipe(
        recipe_schema_version=RECIPE_SCHEMA_VERSION,
        model=model,
        data=data,
        optimizer=optimizer,
        scheduler=scheduler,
        trainer=trainer,
        logging=logging_section,
        metadata={},
    )
    return recipe, MigrationReport(tuple(mig.notes))
