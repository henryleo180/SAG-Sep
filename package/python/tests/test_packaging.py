from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List

import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = PKG_ROOT / "pyproject.toml"
REQUIREMENTS = PKG_ROOT / "requirements.txt"
REQUIREMENTS_TRAIN = PKG_ROOT / "requirements-train.txt"


def _load_toml(path: Path) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def _requirement_lines(path: Path) -> List[str]:
    out: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        out.append(line)
    return out


def _normalise(requirement: str) -> str:
    return re.sub(r"\s+", "", requirement).lower()


def _named(requirements: List[str]) -> Dict[str, str]:
    return {
        re.split(r"[<>=!;\[]", item, maxsplit=1)[0].strip().lower(): _normalise(item)
        for item in requirements
    }


@pytest.fixture(scope="module")
def pyproject() -> dict:
    if not PYPROJECT.is_file():
        pytest.skip(f"not a source checkout: {PYPROJECT} is missing")
    return _load_toml(PYPROJECT)


def test_project_metadata(pyproject: dict) -> None:
    from sagsep_train import __version__

    project = pyproject["project"]
    assert project["name"] == "sagsep"
    assert project["version"] == __version__ == "0.1.0"
    assert project["requires-python"] == ">=3.10"
    assert pyproject["build-system"]["requires"] == ["setuptools>=68"]


def test_the_package_versions_agree() -> None:
    import sagsep
    import sagsep_train

    assert sagsep.__version__ == sagsep_train.__version__ == "0.1.0"


def test_runtime_pins_match_requirements(pyproject: dict) -> None:
    declared = _named(list(pyproject["project"]["dependencies"]))
    pinned = _named(_requirement_lines(REQUIREMENTS))
    assert declared == pinned, (
        "pyproject [project].dependencies and requirements.txt disagree:\n"
        f"  pyproject: {sorted(declared.values())}\n"
        f"  requirements.txt: {sorted(pinned.values())}"
    )


def test_train_extra_matches_requirements_train(pyproject: dict) -> None:
    extra = _named(list(pyproject["project"]["optional-dependencies"]["train"]))
    runtime = set(_named(_requirement_lines(REQUIREMENTS)))
    combined = _named(_requirement_lines(REQUIREMENTS_TRAIN))
    assert set(combined) - runtime == set(extra), (
        "requirements-train.txt adds different packages than the [train] extra:\n"
        f"  extra: {sorted(extra.values())}\n"
        f"  file:  {sorted(set(combined) - runtime)}"
    )
    for name, spec in extra.items():
        assert combined[name] == spec, f"{name}: {spec} in pyproject, {combined[name]} in the file"


def test_requirements_train_includes_the_runtime_file() -> None:
    text = REQUIREMENTS_TRAIN.read_text(encoding="utf-8")
    assert "-r requirements.txt" in text


def test_wandb_extra_is_declared_but_not_required(pyproject: dict) -> None:
    extras = pyproject["project"]["optional-dependencies"]
    assert extras["wandb"] == ["wandb>=0.22"]
    assert not any("wandb" in item for item in pyproject["project"]["dependencies"])


def test_package_discovery_and_data(pyproject: dict) -> None:
    find = pyproject["tool"]["setuptools"]["packages"]["find"]
    assert find["where"] == ["."]
    assert set(find["include"]) == {"sagsep*", "sagsep_train*"}
    patterns = pyproject["tool"]["setuptools"]["package-data"]["sagsep_train.sample"]
    assert "*.npz" in patterns and "*.toml" in patterns and "labels/*.json" in patterns


def test_the_sample_is_importable_as_package_data() -> None:
    from importlib import resources

    import sagsep_train.sample

    root = resources.files("sagsep_train.sample")
    for name in ("shard_0001.npz", "dataset-manifest.json", "smoke.toml", "manifest.csv"):
        assert root.joinpath(name).is_file(), name
    assert root.joinpath("labels").is_dir()


def test_every_declared_dependency_imports() -> None:
    for module in ("numpy", "scipy", "torch", "yaml"):
        __import__(module)
    if sys.version_info < (3, 11):
        __import__("tomli")


# ---------------------------------------------------------------------------
# the shipped recipes
# ---------------------------------------------------------------------------

RECIPES = ("cvrp-two-index-best", "smoke")


@pytest.fixture(scope="module")
def configs_dir(repo_root: Path) -> Path:
    path = repo_root / "configs" / "train"
    if not path.is_dir():
        pytest.skip(f"not a source checkout: {path} is missing")
    return path


@pytest.mark.parametrize("name", RECIPES)
def test_every_shipped_recipe_parses(configs_dir: Path, name: str) -> None:
    from sagsep_train.config import load_recipe

    recipe = load_recipe(configs_dir / f"{name}.toml")
    assert recipe.recipe_schema_version == 1
    assert recipe.data.dataset_dir.strip()
    assert recipe.data.train_shards


def test_reference_recipe_is_the_released_configuration(
    configs_dir: Path, repo_root: Path, models_dir: Path
) -> None:
    import yaml

    from sagsep_train.config import load_recipe

    recipe = load_recipe(configs_dir / "cvrp-two-index-best.toml")

    manifest = yaml.safe_load((models_dir / "cvrp-two-index-best.yaml").read_text("utf-8"))
    assert recipe.model.model_config_dict() == manifest["model"], (
        "the reference recipe no longer describes the architecture of the shipped model"
    )

    data = recipe.data
    assert data.train_shards == [f"shard_{i:04d}" for i in range(1, 9)]
    assert data.val_shards == ["shard_0026"]
    assert (data.val_limit, data.batch_size, data.eval_batch_size) == (2048, 16, 16)
    assert (data.shuffle_mode, data.num_workers, data.pin_memory) == ("global", 12, True)
    assert data.drop_last is False and data.support_rule == "x_gt_0"
    assert data.partial_connect_cross_depot is True

    opt, sched = recipe.optimizer, recipe.scheduler
    assert (opt.name, opt.lr, opt.weight_decay, opt.betas, opt.eps) == (
        "Adam", 1e-3, 0.0, [0.9, 0.999], 1e-8
    )
    assert (sched.name, sched.T_0, sched.T_mult, sched.eta_min) == (
        "CosineAnnealingWarmRestarts", 200, 1, 1e-5
    )

    trainer = recipe.trainer
    assert (trainer.seed, trainer.device, trainer.max_epochs) == (42, "cuda:0", 500)
    assert trainer.max_steps == 2563000
    assert (trainer.log_every, trainer.eval_every, trainer.checkpoint_every) == (100, 500, 1000)
    assert trainer.grad_clip_norm == 0.8 and trainer.deterministic is False
    assert trainer.loss == "BCEWithLogitsLoss"

    assert recipe.logging.jsonl is True and recipe.logging.wandb is False

    meta = recipe.metadata
    assert meta["recipe"] == "cvrp-two-index-best"
    assert meta["historical_run"]["steps"] == trainer.max_steps
    assert meta["data"]["train_instances"] == 3024
    assert isinstance(meta["deviations"], list) and len(meta["deviations"]) >= 9
    assert isinstance(meta["verification"], list) and meta["verification"]


@pytest.mark.parametrize("name", RECIPES)
def test_recipe_metadata_round_trips_through_a_run_snapshot(
    configs_dir: Path, tmp_path: Path, name: str
) -> None:
    from sagsep_train.config import load_recipe, recipe_to_dict, write_snapshot

    recipe = load_recipe(configs_dir / f"{name}.toml")
    snapshot = write_snapshot(recipe, tmp_path / f"{name}.snapshot.yaml")
    reloaded = load_recipe(snapshot)
    assert reloaded.metadata == recipe.metadata
    assert recipe_to_dict(reloaded) == recipe_to_dict(recipe)
