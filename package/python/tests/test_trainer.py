from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sagsep.model import build_policy_network, load_policy_model
from sagsep_train.checkpoint import (
    CHECKPOINT_STEMS,
    checkpoint_path,
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
    strip_orig_mod,
)
from sagsep_train.config import load_recipe, recipe_to_dict
from sagsep_train.data.collate import TrainBatch
from sagsep_train.data.format import (
    InstanceRecord,
    ShardBuilder,
    StateRecord,
    ceil_div,
    write_shard,
)
from sagsep_train.data.manifest import build_manifest, write_manifest
from sagsep_train.trainer import Trainer

MODEL_NAME = "cvrp-two-index-best"


@pytest.fixture(scope="module", autouse=True)
def _single_thread() -> None:
    torch.set_num_threads(1)


# ===========================================================================
# 1. fidelity against the historical trainer
# ===========================================================================


@pytest.fixture(scope="module")
def legacy() -> Dict[str, np.ndarray]:
    path = os.environ.get("SAGSEP_LEGACY_FIXTURE", "").strip()
    if not path:
        pytest.skip("SAGSEP_LEGACY_FIXTURE is not set (the fixture is not shipped)")
    if not Path(path).is_file():
        pytest.skip(f"SAGSEP_LEGACY_FIXTURE does not exist: {path}")
    with np.load(path, allow_pickle=True) as handle:
        return {key: handle[key] for key in handle.files}


@pytest.fixture(scope="module")
def legacy_batch(legacy: Dict[str, np.ndarray]) -> TrainBatch:
    is_depot = np.asarray(legacy["batch/is_depot"]).reshape(-1).astype(bool)
    counts = legacy.get("batch/vehicle_counts_per_instance")
    replicas_per_state = (
        np.asarray(counts, dtype=np.int64)
        if counts is not None
        else np.asarray([int(is_depot.sum())], dtype=np.int64)
    )
    return TrainBatch(
        edge_index=torch.from_numpy(np.ascontiguousarray(legacy["batch/edge_index"], dtype=np.int64)),
        node_attr=torch.from_numpy(np.ascontiguousarray(legacy["batch/nf"], dtype=np.float32)),
        edge_attr=torch.from_numpy(np.ascontiguousarray(legacy["batch/ef"], dtype=np.float32)),
        target=torch.from_numpy(np.ascontiguousarray(legacy["batch/s_hat"], dtype=np.float32).reshape(-1)),
        depot_mask=torch.from_numpy(np.ascontiguousarray(is_depot)),
        graph_id=torch.from_numpy(np.cumsum(is_depot.astype(np.int64)) - 1),
        n_states=int(replicas_per_state.shape[0]),
        n_replicas=int(is_depot.sum()),
        replicas_per_state=torch.from_numpy(replicas_per_state),
    )


@pytest.fixture(scope="module")
def released_net(models_dir: Path):
    ckpt = models_dir / f"{MODEL_NAME}.ckpt"
    manifest = models_dir / f"{MODEL_NAME}.yaml"
    if not ckpt.exists() or not manifest.exists():
        pytest.skip(f"released model not present under {models_dir}")
    with open(manifest, "r", encoding="utf-8") as handle:
        model_cfg = yaml.safe_load(handle)["model"]
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload

    def build():
        net = build_policy_network(model_cfg)
        net.load_state_dict(state)
        return net

    return build


def _masked_bce(logits: torch.Tensor, batch: TrainBatch) -> torch.Tensor:
    keep = ~batch.depot_mask
    return F.binary_cross_entropy_with_logits(logits.view(-1)[keep], batch.target[keep])


def test_reference_step_matches_legacy(legacy, legacy_batch, released_net) -> None:
    print(f"[versions] torch={torch.__version__} numpy={np.__version__}")
    print(f"[fixture] captured under torch={str(legacy['meta/torch'])}")

    net = released_net()
    net.eval()
    with torch.no_grad():
        logits = net.forward_batch(
            legacy_batch.edge_index, legacy_batch.node_attr, legacy_batch.edge_attr
        )
    want_logits = np.ascontiguousarray(legacy["eval/logits"], dtype=np.float32)
    got_logits = logits.detach().view(-1).numpy()
    if got_logits.tobytes() != want_logits.tobytes():
        diff = np.abs(got_logits - want_logits)
        first = int(np.argmax(diff > 0))
        raise AssertionError(
            f"eval logits differ: {int((diff > 0).sum())}/{diff.size} elements, "
            f"max |delta| {float(diff.max()):.3e}, first at node {first} "
            f"({got_logits[first]!r} vs {want_logits[first]!r})"
        )

    loss = _masked_bce(logits, legacy_batch)
    assert np.float32(loss.item()) == legacy["eval/loss"], (
        f"eval loss {np.float32(loss.item())!r} != {legacy['eval/loss']!r}"
    )


def test_reference_train_step_matches_legacy(legacy, legacy_batch, released_net) -> None:
    seed = int(legacy["meta/seed"])
    net = released_net()
    net.train()

    torch.manual_seed(seed)
    opt = torch.optim.Adam(
        net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    opt.zero_grad()
    logits = net.forward_batch(
        legacy_batch.edge_index, legacy_batch.node_attr, legacy_batch.edge_attr
    )
    want_logits = np.ascontiguousarray(legacy["train/logits"], dtype=np.float32)
    got_logits = logits.detach().view(-1).numpy()
    if got_logits.tobytes() != want_logits.tobytes():
        diff = np.abs(got_logits - want_logits)
        raise AssertionError(
            "first divergence: train-mode logits "
            f"({int((diff > 0).sum())}/{diff.size} elements differ, "
            f"max |delta| {float(diff.max()):.3e}) -- the dropout stream differs"
        )

    loss = _masked_bce(logits, legacy_batch)
    assert np.float32(loss.item()) == legacy["train/loss"], (
        f"first divergence: train loss {np.float32(loss.item())!r} != {legacy['train/loss']!r}"
    )

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 0.8)
    assert np.float32(grad_norm.item()) == legacy["train/grad_norm"], (
        f"first divergence: grad norm {np.float32(grad_norm.item())!r} "
        f"!= {legacy['train/grad_norm']!r}"
    )

    without_grad = {str(name) for name in legacy["train/params_without_grad"].tolist()}
    for name, param in net.named_parameters():
        if name in without_grad:
            assert param.grad is None, f"{name}: expected no gradient, got one"
            continue
        key = f"train/grad/{name}"
        assert key in legacy, f"fixture has no gradient for {name}"
        assert param.grad is not None, f"first divergence: {name} has no gradient"
        want = np.ascontiguousarray(legacy[key], dtype=np.float32)
        got = param.grad.detach().numpy().astype(np.float32, copy=False)
        if got.tobytes() != want.tobytes():
            diff = np.abs(got - want)
            raise AssertionError(
                f"first divergence: gradient of {name} "
                f"({int((diff > 0).sum())}/{diff.size} elements differ, "
                f"max |delta| {float(diff.max()):.3e})"
            )

    opt.step()
    for name, param in net.named_parameters():
        key = f"train/after_step/{name}"
        assert key in legacy, f"fixture has no post-step value for {name}"
        want = np.ascontiguousarray(legacy[key], dtype=np.float32)
        got = param.detach().numpy().astype(np.float32, copy=False)
        if got.tobytes() != want.tobytes():
            diff = np.abs(got - want)
            raise AssertionError(
                f"first divergence: {name} after the Adam step "
                f"({int((diff > 0).sum())}/{diff.size} elements differ, "
                f"max |delta| {float(diff.max()):.3e})"
            )


# ===========================================================================
# 2. the loop, on a synthetic dataset
# ===========================================================================

N_NODES = 11
CAPACITY = 12
SHARD_LAYOUT = ((2, 3), (2, 3), (2, 3))
SHARD_NAMES = ["shard_0001", "shard_0002", "shard_0003"]


def _instance(name: str, seed: int) -> InstanceRecord:
    rng = np.random.default_rng(seed)
    coords = rng.random((N_NODES, 2)) * 100.0
    demand = np.zeros(N_NODES, dtype=np.int64)
    demand[1:] = rng.integers(2, 4, size=N_NODES - 1)
    return InstanceRecord(
        id=name,
        source_sha256=hashlib.sha256(name.encode("utf-8")).hexdigest(),
        n_nodes=N_NODES,
        capacity=CAPACITY,
        k_fleet=3,
        k_label=ceil_div(int(demand.sum()), CAPACITY),
        termination="closed",
        coords=coords,
        demand=demand,
    )


def _state(rec: InstanceRecord, iteration: int, terminal: bool, seed: int) -> StateRecord:
    rng = np.random.default_rng(seed)
    pairs = [(u, v) for u in range(rec.n_nodes) for v in range(u + 1, rec.n_nodes)]
    u = np.asarray([p[0] for p in pairs], dtype=np.int32)
    v = np.asarray([p[1] for p in pairs], dtype=np.int32)
    x = rng.random(len(pairs)) * 0.9 + 0.1
    s_hat = rng.integers(0, 2, size=(rec.n_nodes, rec.k_label)).astype(np.uint8)
    s_hat[0, :] = 0
    return StateRecord(
        iteration=iteration,
        lp_value=100.0 + iteration,
        cuts_added=iteration,
        terminal=terminal,
        support_u=u,
        support_v=v,
        support_x=x,
        s_hat=s_hat,
        objective=rng.random(rec.k_label),
        rhs=np.ones(rec.k_label, dtype=np.int32),
        slot_violated=np.zeros(rec.k_label, dtype=np.uint8),
    )


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("train-shards")
    seed = 0
    for shard_row, (n_instances, n_states) in enumerate(SHARD_LAYOUT, start=1):
        builder = ShardBuilder()
        for i in range(n_instances):
            seed += 1
            rec = _instance(f"inst-{shard_row}-{i}", seed)
            row = builder.add_instance(rec)
            for iteration in range(1, n_states + 1):
                builder.add_state(
                    row, _state(rec, iteration, iteration == n_states, seed * 100 + iteration)
                )
        write_shard(root / f"{SHARD_NAMES[shard_row - 1]}.npz", builder.build())

    manifest = build_manifest(
        root,
        source={"kind": "generator", "name": "unit-test", "seed": 0},
        labeler={
            "version": "unit-test",
            "solver": "none",
            "solver_version": "0",
            "threads": 1,
            "lp_method": 1,
            "tolerances": {},
            "min_violation": 1e-6,
            "max_iter": None,
            "require_optimal": True,
        },
        splits={name: ("val" if name == SHARD_NAMES[-1] else "train") for name in SHARD_NAMES},
        created_by_version="0",
    )
    write_manifest(root / "dataset-manifest.json", manifest)
    return root


RECIPE_TEMPLATE = """
recipe_schema_version = 1

[model]
node_dim = 2
edge_dim = 2
latent_dim = 32
n_layer = 2
backbone_type = "transformer"

[model.transformer_params]
num_heads = 4
dropout = 0.1
use_edge_features = true

[model.policy_params]
hidden = [16]
out_act = "Identity"

[data]
dataset_dir = "{dataset_dir}"
manifest = "dataset-manifest.json"
train_shards = ["shard_0001", "shard_0002"]
val_shards = ["shard_0003"]
batch_size = 2
eval_batch_size = 2
shuffle_mode = "global"
num_workers = 0
support_rule = "x_gt_0"

[optimizer]
name = "Adam"
lr = 1e-3

[scheduler]
name = "CosineAnnealingWarmRestarts"
T_0 = 5
T_mult = 1
eta_min = 1e-5

[trainer]
seed = 7
device = "cpu"
max_epochs = {max_epochs}
max_steps = {max_steps}
log_every = 1
eval_every = {eval_every}
checkpoint_every = 4
grad_clip_norm = 0.8
resume_from = {resume_from}

[logging]
jsonl = true
wandb = false
"""


def _write_recipe(
    path: Path,
    dataset_dir: Path,
    *,
    max_epochs: int = 50,
    max_steps: str = "12",
    eval_every: int = 4,
    resume_from: Path | None = None,
) -> Path:
    text = RECIPE_TEMPLATE.format(
        dataset_dir=dataset_dir.as_posix(),
        max_epochs=max_epochs,
        max_steps=max_steps,
        eval_every=eval_every,
        resume_from=f'"{resume_from.as_posix()}"' if resume_from is not None else "false",
    )
    path.write_text(text.replace("resume_from = false\n", ""), encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _flat_state(net) -> bytes:
    return b"".join(
        param.detach().cpu().contiguous().numpy().tobytes()
        for _name, param in sorted(net.state_dict().items())
    )


def test_train_writes_a_complete_run_directory(tmp_path: Path, dataset_dir: Path) -> None:
    recipe_path = _write_recipe(tmp_path / "recipe.toml", dataset_dir)
    recipe = load_recipe(recipe_path)
    run_dir = tmp_path / "run"

    trainer = Trainer(recipe, run_dir=run_dir, recipe_path=recipe_path)
    before = _flat_state(trainer.net)
    trainer.train()

    assert trainer.global_step == 12

    for name in (
        "config_snapshot.yaml",
        "val_record_ids.json",
        "train_log.jsonl",
        "val_log.jsonl",
        CHECKPOINT_STEMS["latest"],
        CHECKPOINT_STEMS["best"],
    ):
        assert (run_dir / name).is_file(), f"missing {name}"

    train_rows = _read_jsonl(run_dir / "train_log.jsonl")
    assert len(train_rows) == 12, [row["step"] for row in train_rows]
    assert [row["step"] for row in train_rows] == list(range(1, 13))
    assert all(np.isfinite(row["loss"]) for row in train_rows)
    assert all(row["graphs"] > 0 and row["nodes"] > 0 for row in train_rows)

    val_rows = _read_jsonl(run_dir / "val_log.jsonl")
    assert [row["step"] for row in val_rows] == [4, 8, 12]
    assert all(np.isfinite(row["loss"]) for row in val_rows)

    ids = json.loads((run_dir / "val_record_ids.json").read_text(encoding="utf-8"))
    assert ids["count"] == len(ids["record_ids"]) == 6
    assert ids["record_ids"] == sorted(ids["record_ids"])

    assert _flat_state(trainer.net) != before, "no parameter moved in 12 steps"

    snapshot = yaml.safe_load((run_dir / "config_snapshot.yaml").read_text(encoding="utf-8"))
    assert len(snapshot["dataset_manifest_sha256"]) == 64
    assert snapshot["val_record_ids_file"] == "val_record_ids.json"
    reloaded = load_recipe(run_dir / "config_snapshot.yaml")
    assert reloaded.model.model_config_dict() == recipe.model.model_config_dict()


def test_run_directory_is_servable_by_the_released_loader(tmp_path: Path, dataset_dir: Path) -> None:
    recipe_path = _write_recipe(tmp_path / "recipe.toml", dataset_dir, max_steps="4")
    run_dir = tmp_path / "run"
    Trainer(load_recipe(recipe_path), run_dir=run_dir, recipe_path=recipe_path).train()

    model = load_policy_model(run_dir, device="cpu")
    assert model.device == "cpu"
    assert not model._net.training


def test_checkpoint_payload_round_trips(tmp_path: Path, dataset_dir: Path) -> None:
    recipe_path = _write_recipe(tmp_path / "recipe.toml", dataset_dir, max_steps="4")
    recipe = load_recipe(recipe_path)
    trainer = Trainer(recipe, run_dir=tmp_path / "run", recipe_path=recipe_path)
    trainer.train()

    payload = load_checkpoint(checkpoint_path(trainer.run_dir, "latest"))
    assert payload["recipe_schema_version"] == 1
    assert payload["global_step"] == 4
    assert payload["config"]["model"] == recipe.model.model_config_dict()
    assert payload["optimizer"] is not None and payload["scheduler"] is not None

    other = Trainer(load_recipe(recipe_path), run_dir=tmp_path / "run2", recipe_path=recipe_path)
    step, epoch = restore_training_state(
        payload, model=other.net, optimizer=other.optimizer, scheduler=other.scheduler
    )
    assert (step, epoch) == (payload["global_step"], payload["epoch"])
    assert _flat_state(other.net) == _flat_state(trainer.net)

    prefixed = {f"_orig_mod.{k}": v for k, v in payload["model"].items()}
    with pytest.raises(RuntimeError):
        other.net.load_state_dict(prefixed)
    other.net.load_state_dict(strip_orig_mod(prefixed))


def test_save_checkpoint_is_atomic(tmp_path: Path, dataset_dir: Path) -> None:
    recipe_path = _write_recipe(tmp_path / "recipe.toml", dataset_dir, max_steps="1")
    trainer = Trainer(load_recipe(recipe_path), run_dir=tmp_path / "run", recipe_path=recipe_path)
    target = tmp_path / "ckpt.pt"
    save_checkpoint(
        target,
        model=trainer.net,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        global_step=3,
        epoch=1,
        config=recipe_to_dict(trainer.recipe),
        extra={"note": "unit-test"},
    )
    assert target.is_file()
    assert not list(tmp_path.glob("ckpt.pt.tmp-*")), "the temporary file was left behind"
    assert load_checkpoint(target)["note"] == "unit-test"


def test_resume_reproduces_an_uninterrupted_run(tmp_path: Path, dataset_dir: Path) -> None:
    straight_recipe = _write_recipe(
        tmp_path / "straight.toml",
        dataset_dir,
        max_epochs=2,
        max_steps="1000",
        eval_every=1000,
    )
    straight_dir = tmp_path / "straight"
    straight = Trainer(load_recipe(straight_recipe), run_dir=straight_dir, recipe_path=straight_recipe)
    straight.train()
    assert straight.epochs_completed == 2

    first_recipe = _write_recipe(
        tmp_path / "first.toml", dataset_dir, max_epochs=1, max_steps="1000", eval_every=1000
    )
    first_dir = tmp_path / "split"
    first = Trainer(load_recipe(first_recipe), run_dir=first_dir, recipe_path=first_recipe)
    first.train()
    assert first.epochs_completed == 1

    resume_from = checkpoint_path(first_dir, "latest")
    assert load_checkpoint(resume_from)["epoch"] == 1, "a full epoch means one completed epoch"

    second_recipe = _write_recipe(
        tmp_path / "second.toml",
        dataset_dir,
        max_epochs=2,
        max_steps="1000",
        eval_every=1000,
        resume_from=resume_from,
    )
    resumed = Trainer(load_recipe(second_recipe), run_dir=tmp_path / "split2", recipe_path=second_recipe)
    assert resumed.start_epoch == 1 and resumed.global_step == first.global_step
    resumed.train()

    assert resumed.global_step == straight.global_step
    assert _flat_state(resumed.net) == _flat_state(straight.net), (
        "resuming on an epoch boundary must reproduce the uninterrupted run"
    )
    assert (
        resumed.optimizer.param_groups[0]["lr"] == straight.optimizer.param_groups[0]["lr"]
    ), "the cosine schedule did not continue where it stopped"


def test_evaluate_restores_the_training_mode(tmp_path: Path, dataset_dir: Path) -> None:
    recipe_path = _write_recipe(tmp_path / "recipe.toml", dataset_dir, max_steps="1")
    trainer = Trainer(load_recipe(recipe_path), run_dir=tmp_path / "run", recipe_path=recipe_path)
    trainer.net.train()
    metrics = trainer.evaluate()
    assert trainer.net.training is True
    assert metrics["batches"] == 3 and metrics["logits"] > 0
    assert np.isfinite(metrics["loss"])

    trainer.net.eval()
    trainer.evaluate()
    assert trainer.net.training is False


def test_shard_shuffle_mode_is_accepted(tmp_path: Path, dataset_dir: Path) -> None:
    recipe_path = _write_recipe(tmp_path / "recipe.toml", dataset_dir, max_steps="3")
    text = recipe_path.read_text(encoding="utf-8").replace(
        'shuffle_mode = "global"', 'shuffle_mode = "shard"'
    )
    recipe_path.write_text(text, encoding="utf-8")
    trainer = Trainer(load_recipe(recipe_path), run_dir=tmp_path / "run", recipe_path=recipe_path)
    trainer.train()
    assert trainer.global_step == 3
