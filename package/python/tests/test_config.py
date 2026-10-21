from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from sagsep_train.config import (
    DENIED_KEYS,
    MODEL_CONFIG_KEYS,
    RECIPE_SCHEMA_VERSION,
    MigrationReport,
    RecipeError,
    TrainingRecipe,
    apply_overrides,
    from_legacy_snapshot,
    load_recipe,
    parse_recipe,
    recipe_to_dict,
    write_snapshot,
)

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

RECIPE_TOML = """
recipe_schema_version = 1

[model]
node_dim = 2
edge_dim = 2
latent_dim = 256
n_layer = 6
backbone_type = "transformer"
debug = false

[model.transformer_params]
num_heads = 8
dropout = 0.1
use_edge_features = true

[model.policy_params]
hidden = [128, 64]
out_act = "Identity"

[data]
dataset_dir = "shards"
train_shards = ["shard_0001", "shard_0002"]
val_shards = ["shard_0003.npz"]
val_limit = 2048
batch_size = 16
eval_batch_size = 16
shuffle_mode = "global"
num_workers = 4
support_rule = "x_gt_0"

[optimizer]
name = "Adam"
lr = 0.001

[scheduler]
name = "CosineAnnealingWarmRestarts"
T_0 = 200

[trainer]
seed = 42
device = "cuda:0"
max_epochs = 500
grad_clip_norm = 0.8

[logging]
jsonl = true
tags = ["release"]

[metadata]
note = "acceptance recipe"
"""

LEGACY_SNAPSHOT: dict[str, Any] = {
    "model": {
        "node_dim": 2,
        "edge_dim": 2,
        "latent_dim": 256,
        "n_layer": 6,
        "backbone_type": "transformer",
        "transformer_params": {"num_heads": 8, "dropout": 0.1, "use_edge_features": True},
        "policy_params": {"hidden": [128, 64], "out_act": "Identity"},
        "debug": False,
    },
    "data": {
        "dataset_path": "data/xml100/pyg",
        "train_fraction": 1.0,
        "val_fraction": 0.0,
        "test_fraction": 0.0,
        "batch_size": 16,
        "eval_batch_size": 16,
        "shuffle": True,
        "num_workers": 12,
        "drop_last": False,
        "pin_memory": True,
        "train_limit": None,
        "val_limit": 2048,
        "test_limit": None,
        "contract_config": None,
        "dataset_type": "flat",
        "sharded": None,
        "manual_splits": None,
        "flat_files": {"train": ["1-8"], "val": [26]},
        "flat_pattern": "shard_*.pt",
        "depot_complete": True,
        "partial_connect": 2,
        "partial_connect_hop_attr": "edge_hop_type",
        "partial_connect_edge_value": {"method": "max_min", "coef": 1.0},
        "partial_connect_cross_depot": True,
        "most_violate_mode": False,
        "m_value_closed_interval": True,
    },
    "optimizer": {"name": "Adam", "lr": 0.001, "weight_decay": 0.0, "betas": [0.9, 0.999], "eps": 1.0e-08},
    "scheduler": {"name": "CosineAnnealingWarmRestarts", "T_0": 200, "T_mult": 1, "eta_min": 1.0e-05, "gamma": 0.5},
    "trainer": {
        "seed": 42,
        "device": "cuda:0",
        "max_epochs": 500,
        "log_every": 100,
        "eval_every": 500,
        "checkpoint_every": 1000,
        "checkpoint_dir": "checkpoints",
        "resume_from": None,
        "loss": "BCEWithLogitsLoss",
        "threshold": 0.5,
        "grad_clip_norm": 0.8,
        "amp": False,
        "positive_weight": False,
        "symmetric_loss": False,
        "instance_equal_weight": False,
        "eval_devices": None,
        "eval_microbatch_size": None,
    },
    "logging": {
        "enable_wandb": True,
        "project": "cvrp-separator",
        "entity": None,
        "group": None,
        "run_name": None,
        "mode": "online",
        "tags": ["partial-lp", "hop-feature", "max-min-shortcut"],
    },
}

ALLOWED_LEGACY_KEYS = {
    "model.node_dim",
    "model.edge_dim",
    "model.latent_dim",
    "model.n_layer",
    "model.backbone_type",
    "model.transformer_params",
    "model.policy_params",
    "model.debug",
    "data.dataset_path",
    "data.train_fraction",
    "data.val_fraction",
    "data.test_fraction",
    "data.train_limit",
    "data.test_limit",
    "data.contract_config",
    "data.dataset_type",
    "data.sharded",
    "data.manual_splits",
    "data.flat_pattern",
    "data.flat_files",
    "data.partial_connect_hop_attr",
    "data.most_violate_mode",
    "data.pin_memory",
    "data.num_workers",
    "data.shuffle",
    "data.eval_batch_size",
    "data.batch_size",
    "data.drop_last",
    "data.val_limit",
    "data.depot_complete",
    "data.partial_connect",
    "data.partial_connect_edge_value",
    "data.partial_connect_cross_depot",
    "data.m_value_closed_interval",
    "optimizer.name",
    "optimizer.lr",
    "optimizer.weight_decay",
    "optimizer.betas",
    "optimizer.eps",
    "scheduler.name",
    "scheduler.T_0",
    "scheduler.T_mult",
    "scheduler.eta_min",
    "scheduler.gamma",
    "trainer.seed",
    "trainer.device",
    "trainer.max_epochs",
    "trainer.log_every",
    "trainer.eval_every",
    "trainer.checkpoint_every",
    "trainer.loss",
    "trainer.grad_clip_norm",
    "trainer.checkpoint_dir",
    "trainer.resume_from",
    "trainer.run_id",
    "trainer.threshold",
    "trainer.amp",
    "trainer.positive_weight",
    "trainer.symmetric_loss",
    "trainer.instance_equal_weight",
    "trainer.eval_devices",
    "trainer.eval_microbatch_size",
    "logging.enable_wandb",
    "logging.project",
    "logging.entity",
    "logging.group",
    "logging.run_name",
    "logging.mode",
    "logging.tags",
}


@pytest.fixture()
def recipe_file(tmp_path: Path) -> Path:
    path = tmp_path / "recipe.toml"
    path.write_text(RECIPE_TOML, encoding="utf-8")
    return path


def _write_yaml(path: Path, payload: Any) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# round trips
# ---------------------------------------------------------------------------


def test_toml_round_trip(recipe_file: Path) -> None:
    recipe = load_recipe(recipe_file)
    assert recipe.recipe_schema_version == RECIPE_SCHEMA_VERSION
    assert recipe.model.latent_dim == 256
    assert recipe.data.train_shards == ["shard_0001", "shard_0002"]
    assert recipe.data.val_shards == ["shard_0003.npz"]
    assert recipe.data.val_limit == 2048
    assert recipe.data.num_workers == 4
    assert recipe.metadata == {"note": "acceptance recipe"}
    assert recipe.optimizer.betas == [0.9, 0.999]
    assert recipe.scheduler.eta_min == pytest.approx(1e-5)
    assert recipe.trainer.max_steps is None
    assert recipe.logging.wandb is False


def test_relative_dataset_dir_resolves_against_the_recipe_file(recipe_file: Path) -> None:
    recipe = load_recipe(recipe_file)
    assert Path(recipe.data.dataset_dir) == recipe_file.parent / "shards"
    absolute = recipe_file.parent / "elsewhere.toml"
    absolute.write_text(
        RECIPE_TOML.replace('dataset_dir = "shards"', f'dataset_dir = "{recipe_file.parent}"'),
        encoding="utf-8",
    )
    assert load_recipe(absolute).data.dataset_dir == str(recipe_file.parent)


def test_yaml_round_trip(tmp_path: Path, recipe_file: Path) -> None:
    recipe = load_recipe(recipe_file)
    as_yaml = _write_yaml(tmp_path / "recipe.yaml", recipe_to_dict(recipe))
    reloaded = load_recipe(as_yaml)
    assert recipe_to_dict(reloaded) == recipe_to_dict(recipe)


def test_recipe_to_dict_key_order_follows_the_dataclasses(recipe_file: Path) -> None:
    payload = recipe_to_dict(load_recipe(recipe_file))
    assert list(payload) == [
        "recipe_schema_version",
        "model",
        "data",
        "optimizer",
        "scheduler",
        "trainer",
        "logging",
        "metadata",
    ]
    assert list(payload["model"]) == [
        "node_dim",
        "edge_dim",
        "latent_dim",
        "n_layer",
        "backbone_type",
        "transformer_params",
        "policy_params",
        "debug",
    ]
    assert list(payload["trainer"])[-1] == "deterministic"


def test_unsupported_file_format(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(RecipeError, match="unsupported recipe format"):
        load_recipe(path)


# ---------------------------------------------------------------------------
# strictness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mutate", "dotted"),
    [
        (lambda p: p["trainer"].__setitem__("max_step", 10), "trainer.max_step"),
        (lambda p: p["data"].__setitem__("shuffle", True), "data.shuffle"),
        (lambda p: p["model"]["policy_params"].__setitem__("depth", 3), "model.policy_params.depth"),
        (lambda p: p.__setitem__("trianer", {"seed": 1}), "trianer"),
    ],
)
def test_unknown_key_is_refused_with_its_dotted_path(
    tmp_path: Path, recipe_file: Path, mutate, dotted: str
) -> None:
    payload = recipe_to_dict(load_recipe(recipe_file))
    mutate(payload)
    path = _write_yaml(tmp_path / "broken.yaml", payload)
    with pytest.raises(RecipeError) as excinfo:
        load_recipe(path)
    assert dotted in str(excinfo.value)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("trainer", "amp"),
        ("trainer", "positive_weight"),
        ("trainer", "precision"),
        ("data", "pre_dataset_filter"),
        ("data", "train_fraction"),
        ("data", "mfree_mode"),
    ],
)
def test_denied_keys_are_refused_by_name(recipe_file: Path, section: str, key: str) -> None:
    payload = recipe_to_dict(load_recipe(recipe_file))
    payload[section][key] = False
    with pytest.raises(RecipeError) as excinfo:
        parse_recipe(payload)
    assert f"{section}.{key}: unsupported by this release" in str(excinfo.value)


def test_deny_list_takes_precedence_over_the_unknown_key_error(recipe_file: Path) -> None:
    payload = recipe_to_dict(load_recipe(recipe_file))
    payload["trainer"]["amp"] = True
    payload["trainer"]["typo_key"] = 1
    with pytest.raises(RecipeError) as excinfo:
        parse_recipe(payload)
    message = str(excinfo.value)
    assert "trainer.amp: unsupported by this release" in message
    assert "typo_key" not in message


def test_deny_list_takes_precedence_over_an_unsupported_value(recipe_file: Path) -> None:
    payload = recipe_to_dict(load_recipe(recipe_file))
    payload["data"]["depot_complete"] = False
    payload["data"]["most_violate_mode"] = True
    with pytest.raises(RecipeError, match="data.most_violate_mode: unsupported by this release"):
        parse_recipe(payload)


def test_deny_list_ignores_the_metadata_section(recipe_file: Path) -> None:
    payload = recipe_to_dict(load_recipe(recipe_file))
    payload["metadata"]["precision"] = "captured in fp32"
    payload["metadata"]["nested"] = {"amp": "not a knob"}
    recipe = parse_recipe(payload)
    assert recipe.metadata["nested"] == {"amp": "not a knob"}


def test_deny_list_covers_the_documented_names() -> None:
    assert "pre_dataset_filter" in DENIED_KEYS
    assert "m_value_sqrt_coef" in DENIED_KEYS
    assert len(DENIED_KEYS) == 32


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("depot_complete", False),
        ("partial_connect", 3),
        ("m_value_closed_interval", False),
    ],
)
def test_unsupported_feature_values_raise(recipe_file: Path, key: str, value: Any) -> None:
    payload = recipe_to_dict(load_recipe(recipe_file))
    payload["data"][key] = value
    with pytest.raises(RecipeError, match="only .* is implemented in this release"):
        parse_recipe(payload)


@pytest.mark.parametrize(
    ("key", "value"), [("method", "max_smin"), ("coef", 2.0)]
)
def test_unsupported_shortcut_value_config_raises(recipe_file: Path, key: str, value: Any) -> None:
    payload = recipe_to_dict(load_recipe(recipe_file))
    payload["data"]["partial_connect_edge_value"][key] = value
    with pytest.raises(RecipeError, match="only .* is implemented in this release"):
        parse_recipe(payload)


def test_cross_depot_false_is_accepted(recipe_file: Path) -> None:
    payload = recipe_to_dict(load_recipe(recipe_file))
    payload["data"]["partial_connect_cross_depot"] = False
    assert parse_recipe(payload).data.partial_connect_cross_depot is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p["trainer"].__setitem__("seed", "42"), "expected an integer"),
        (lambda p: p["trainer"].__setitem__("seed", True), "expected an integer"),
        (lambda p: p["data"].__setitem__("pin_memory", 1), "expected a boolean"),
        (lambda p: p["optimizer"].__setitem__("lr", "0.001"), "expected a number"),
        (lambda p: p["optimizer"].__setitem__("betas", [0.9]), "expected 2 entries"),
        (lambda p: p["logging"].__setitem__("tags", "release"), "expected a list"),
        (lambda p: p["trainer"].__setitem__("loss", "MSELoss"), "must be one of"),
        (lambda p: p["data"].__setitem__("support_rule", "x_gt_1"), "must be one of"),
        (lambda p: p["data"].__setitem__("shuffle_mode", "random"), "must be one of"),
        (lambda p: p["data"].__setitem__("dataset_dir", ""), "required"),
        (lambda p: p["data"].__setitem__("train_shards", ["a/b"]), "bare file name"),
        (lambda p: p.__setitem__("recipe_schema_version", 2), "version 1"),
    ],
)
def test_leaf_type_and_value_checks(recipe_file: Path, mutate, message: str) -> None:
    payload = recipe_to_dict(load_recipe(recipe_file))
    mutate(payload)
    with pytest.raises(RecipeError, match=message):
        parse_recipe(payload)


def test_missing_data_section_is_refused() -> None:
    with pytest.raises(RecipeError, match="data.dataset_dir"):
        parse_recipe({"recipe_schema_version": 1})


# ---------------------------------------------------------------------------
# overrides
# ---------------------------------------------------------------------------


def test_apply_overrides(recipe_file: Path) -> None:
    recipe = load_recipe(recipe_file)
    updated = apply_overrides(
        recipe,
        [
            "trainer.max_steps=100",
            "data.dataset_dir=/shards/xml100",
            "data.batch_size=32",
            "optimizer.lr=5e-4",
            "optimizer.betas=[0.8, 0.95]",
            "logging.tags=['a', 'b']",
            "data.partial_connect_edge_value.coef=1.0",
        ],
    )
    assert updated.trainer.max_steps == 100
    assert updated.data.dataset_dir == "/shards/xml100"
    assert updated.data.batch_size == 32
    assert updated.optimizer.lr == pytest.approx(5e-4)
    assert updated.optimizer.betas == [0.8, 0.95]
    assert updated.logging.tags == ["a", "b"]
    assert recipe.trainer.max_steps is None
    assert recipe.data.batch_size == 16


def test_override_can_clear_an_optional(recipe_file: Path) -> None:
    recipe = apply_overrides(load_recipe(recipe_file), ["data.val_limit=null", "trainer.grad_clip_norm=null"])
    assert recipe.data.val_limit is None
    assert recipe.trainer.grad_clip_norm is None


@pytest.mark.parametrize(
    "override",
    ["trainer.nope=1", "nope.seed=1", "trainer.seed.nope=1", "trainer.max_steps", ""],
)
def test_unknown_override_path_raises(recipe_file: Path, override: str) -> None:
    with pytest.raises(RecipeError):
        apply_overrides(load_recipe(recipe_file), [override])


def test_override_is_type_checked(recipe_file: Path) -> None:
    with pytest.raises(RecipeError, match="expected an integer"):
        apply_overrides(load_recipe(recipe_file), ["trainer.seed=1.5"])


def test_override_cannot_smuggle_in_a_denied_feature(recipe_file: Path) -> None:
    with pytest.raises(RecipeError, match="only 2 is implemented"):
        apply_overrides(load_recipe(recipe_file), ["data.partial_connect=1"])


# ---------------------------------------------------------------------------
# snapshot
# ---------------------------------------------------------------------------


def test_snapshot_write_and_read_back(tmp_path: Path, recipe_file: Path) -> None:
    recipe = load_recipe(recipe_file)
    snapshot = write_snapshot(
        recipe,
        tmp_path / "run" / "config_snapshot.yaml",
        extra={
            "dataset_manifest_sha256": "0" * 64,
            "val_record_ids_file": "val_record_ids.json",
        },
    )
    raw = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
    assert raw["dataset_manifest_sha256"] == "0" * 64
    assert raw["val_record_ids_file"] == "val_record_ids.json"
    assert raw["model"] == recipe.model_config_dict()
    assert list(raw)[:3] == ["recipe_schema_version", "model", "data"]

    reloaded = load_recipe(snapshot)
    assert recipe_to_dict(reloaded) == recipe_to_dict(recipe)


def test_snapshot_extra_must_not_collide(tmp_path: Path, recipe_file: Path) -> None:
    recipe = load_recipe(recipe_file)
    with pytest.raises(RecipeError, match="collides"):
        write_snapshot(recipe, tmp_path / "snap.yaml", extra={"trainer": "oops"})


def test_a_mistyped_section_is_not_mistaken_for_an_annotation(
    tmp_path: Path, recipe_file: Path
) -> None:
    recipe = load_recipe(recipe_file)
    snapshot = write_snapshot(recipe, tmp_path / "config_snapshot.yaml")
    payload = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
    payload["trianer"] = {"seed": 1}
    _write_yaml(snapshot, payload)
    with pytest.raises(RecipeError, match="trianer"):
        load_recipe(snapshot)


# ---------------------------------------------------------------------------
# model config
# ---------------------------------------------------------------------------


def test_model_config_dict_is_the_released_thirteen_key_shape(recipe_file: Path) -> None:
    config = load_recipe(recipe_file).model_config_dict()
    assert config == {
        "node_dim": 2,
        "edge_dim": 2,
        "latent_dim": 256,
        "n_layer": 6,
        "backbone_type": "transformer",
        "transformer_params": {"num_heads": 8, "dropout": 0.1, "use_edge_features": True},
        "policy_params": {"hidden": [128, 64], "out_act": "Identity"},
        "debug": False,
    }
    leaves = [key for key in config if not isinstance(config[key], dict)]
    nested = [
        f"{group}.{key}"
        for group in ("transformer_params", "policy_params")
        for key in config[group]
    ]
    assert len(leaves) + 2 + len(nested) == len(MODEL_CONFIG_KEYS) == 13


def test_model_config_dict_is_accepted_by_the_released_builder(recipe_file: Path) -> None:
    from sagsep.model import build_policy_network

    net = build_policy_network(load_recipe(recipe_file).model_config_dict())
    keys = list(net.state_dict())
    assert keys, "the built network has no parameters"
    assert any(key.startswith("policy.") for key in keys)


def test_optional_policy_params_only_appear_when_set(recipe_file: Path) -> None:
    recipe = load_recipe(recipe_file)
    assert "hidden_act" not in recipe.model_config_dict()["policy_params"]
    recipe.model.policy_params.hidden_act = "ReLU"
    recipe.model.policy_params.bias = True
    policy = recipe.model_config_dict()["policy_params"]
    assert policy == {"hidden": [128, 64], "hidden_act": "ReLU", "out_act": "Identity", "bias": True}

    from sagsep.model import build_policy_network

    build_policy_network(recipe.model_config_dict())


# ---------------------------------------------------------------------------
# legacy migration
# ---------------------------------------------------------------------------


def test_legacy_snapshot_migrates() -> None:
    recipe, report = from_legacy_snapshot(copy.deepcopy(LEGACY_SNAPSHOT))
    assert isinstance(recipe, TrainingRecipe)
    assert isinstance(report, MigrationReport)

    assert recipe.recipe_schema_version == 1
    assert recipe.model.model_config_dict() == LEGACY_SNAPSHOT["model"]

    assert recipe.data.dataset_dir == "data/xml100/pyg"
    assert recipe.data.train_shards == [f"shard_{i:04d}" for i in range(1, 9)]
    assert recipe.data.val_shards == ["shard_0026"]
    assert recipe.data.val_limit == 2048
    assert recipe.data.batch_size == 16
    assert recipe.data.eval_batch_size == 16
    assert recipe.data.num_workers == 12
    assert recipe.data.pin_memory is True
    assert recipe.data.drop_last is False
    assert recipe.data.shuffle_mode == "global"
    assert recipe.data.support_rule == "x_gt_0"
    assert recipe.data.manifest == "dataset-manifest.json"

    assert recipe.optimizer.name == "Adam"
    assert recipe.optimizer.betas == [0.9, 0.999]
    assert recipe.scheduler.T_0 == 200

    assert recipe.trainer.seed == 42
    assert recipe.trainer.device == "cuda:0"
    assert recipe.trainer.grad_clip_norm == pytest.approx(0.8)
    assert recipe.trainer.max_steps is None
    assert recipe.trainer.resume_from is None

    assert recipe.logging.wandb is True
    assert recipe.logging.project == "cvrp-separator"
    assert recipe.logging.run_name == ""
    assert recipe.logging.tags == LEGACY_SNAPSHOT["logging"]["tags"]

    assert recipe_to_dict(parse_recipe(recipe_to_dict(recipe))) == recipe_to_dict(recipe)


def test_legacy_report_lists_only_allow_listed_keys() -> None:
    _recipe, report = from_legacy_snapshot(copy.deepcopy(LEGACY_SNAPSHOT))
    legacy = set(report.legacy_keys)
    assert legacy <= ALLOWED_LEGACY_KEYS, sorted(legacy - ALLOWED_LEGACY_KEYS)
    expected = ALLOWED_LEGACY_KEYS - {"trainer.run_id"}
    assert legacy == expected, sorted(expected.symmetric_difference(legacy))

    assert "data.dataset_path" in report.migrated
    assert "data.flat_files" in report.migrated
    assert "trainer.threshold" in report.dropped
    assert "trainer.checkpoint_dir" in report.dropped
    assert "scheduler.gamma" in report.dropped
    assert "logging.mode" in report.dropped
    assert "data.train_fraction" in report.dropped
    assert "data.support_rule" in report.defaulted
    assert "trainer.max_steps" in report.defaulted
    assert len(report.as_lines()) == len(report.notes)
    assert all(":" in line for line in report.as_lines())


def test_legacy_snapshot_from_a_yaml_file(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path / "config_snapshot.yaml", LEGACY_SNAPSHOT)
    recipe, report = from_legacy_snapshot(path)
    assert recipe.data.train_shards[0] == "shard_0001"
    assert set(report.legacy_keys) <= ALLOWED_LEGACY_KEYS


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p["trainer"].__setitem__("positive_weight", True), "positive_weight"),
        (lambda p: p["data"].__setitem__("pre_dataset_filter", {"mode": "tailoff"}), "pre_dataset_filter"),
        (lambda p: p["trainer"].__setitem__("amp", True), "amp"),
        (lambda p: p["trainer"].__setitem__("symmetric_loss", True), "symmetric_loss"),
        (lambda p: p["data"].__setitem__("train_fraction", 0.8), "train_fraction"),
        (lambda p: p["data"].__setitem__("shuffle", False), "shuffle"),
        (lambda p: p["data"].__setitem__("train_graph_keep_prob", 0.5), "train_graph_keep_prob"),
        (lambda p: p["data"].__setitem__("mfree_mode", True), "mfree_mode"),
        (lambda p: p["trainer"].__setitem__("label_mode", "freq"), "label_mode"),
        (lambda p: p["trainer"].__setitem__("precision", "fp64"), "precision"),
        (lambda p: p["trainer"].__setitem__("fork_from", "run/ckpt.pt"), "fork_from"),
        (lambda p: p["trainer"].__setitem__("resume_from", "run/ckpt.pt"), "resume_from"),
        (lambda p: p["data"].__setitem__("sharded", {"shard_dir": "x"}), "sharded"),
        (lambda p: p["data"]["flat_files"].__setitem__("train", ["shard_0001.pt"]), "flat_files"),
        (lambda p: p["data"]["flat_files"].__setitem__("train", [-1]), "flat_files"),
        (lambda p: p["data"]["flat_files"].__setitem__("test", [3]), "flat_files"),
        (lambda p: p.__setitem__("evaluation", {"preset": "x"}), "evaluation"),
        (lambda p: p["data"].__setitem__("persistent_workers", True), "persistent_workers"),
    ],
)
def test_legacy_snapshot_rejections(mutate, message: str) -> None:
    payload = copy.deepcopy(LEGACY_SNAPSHOT)
    mutate(payload)
    with pytest.raises(RecipeError) as excinfo:
        from_legacy_snapshot(payload)
    assert message in str(excinfo.value)


def test_legacy_flat_file_ranges() -> None:
    payload = copy.deepcopy(LEGACY_SNAPSHOT)
    payload["data"]["flat_files"] = {"train": [1, "3-5", 3], "val": ["26-27"]}
    recipe, _report = from_legacy_snapshot(payload)
    assert recipe.data.train_shards == ["shard_0001", "shard_0003", "shard_0004", "shard_0005"]
    assert recipe.data.val_shards == ["shard_0026", "shard_0027"]
