from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sagsep_train.__main__ import build_parser, main as cli_main
from sagsep_train.cli import EXIT_ERROR, EXIT_OK, _parse_splits
from sagsep_train.data.format import (
    InstanceRecord,
    ShardBuilder,
    StateRecord,
    ceil_div,
    write_shard,
)


@pytest.fixture(scope="module", autouse=True)
def _single_thread() -> None:
    torch.set_num_threads(1)


# ===========================================================================
# 1. the parser
# ===========================================================================


def _parse(argv: list[str]):
    return build_parser().parse_args(argv)


def test_generate_parser() -> None:
    args = _parse(["generate", "--out", "d", "--n", "15", "--count", "4", "--seed", "11"])
    assert (args.command, args.out, args.n, args.count, args.seed) == ("generate", "d", 15, 4, 11)
    assert args.spec is None
    assert _parse(["generate", "--out", "d", "--n", "1", "--count", "1", "--seed", "0",
                   "--spec", "1163"]).spec == "1163"


def test_label_parser() -> None:
    args = _parse(["label", "--instances", "i", "--out", "o"])
    assert (args.shard, args.max_iter, args.threads) == ("1/1", None, 1)
    assert args.min_violation == pytest.approx(1e-6)
    assert args.vehicles_from is None and args.time_limit is None
    args = _parse(
        [
            "label", "--instances", "i", "--out", "o", "--shard", "2/8", "--max-iter", "50",
            "--threads", "4", "--min-violation", "1e-5", "--vehicles-from", "m.csv",
            "--time-limit", "30",
        ]
    )
    assert (args.shard, args.max_iter, args.threads) == ("2/8", 50, 4)
    assert args.min_violation == pytest.approx(1e-5)
    assert args.vehicles_from == "m.csv" and args.time_limit == pytest.approx(30.0)


def test_pack_parser() -> None:
    args = _parse(["pack", "--labels", "l", "--out", "o"])
    assert args.shard_size == 64 and args.split is None and args.instances is None
    args = _parse(
        ["pack", "--labels", "l", "--out", "o", "--shard-size", "8",
         "--split", "train=shard_0001,shard_0002", "--split", "val=shard_0003"]
    )
    assert args.shard_size == 8
    assert _parse_splits(args.split) == {
        "shard_0001": "train",
        "shard_0002": "train",
        "shard_0003": "val",
    }


def test_validate_train_export_smoke_parsers() -> None:
    assert _parse(["validate", "ds"]).dataset_dir == "ds"

    args = _parse(["train", "--config", "r.toml", "--out", "run"])
    assert (args.config, args.out, args.set, args.resume) == ("r.toml", "run", None, False)
    args = _parse(
        ["train", "--config", "r.toml", "--out", "run", "--set", "trainer.seed=7",
         "--set", "data.batch_size=4", "--resume"]
    )
    assert args.set == ["trainer.seed=7", "data.batch_size=4"] and args.resume is True

    args = _parse(["export", "--run", "run", "--name", "m", "--out", "o"])
    assert args.checkpoint == "best" and args.no_onnx is False
    args = _parse(
        ["export", "--run", "run", "--name", "m", "--out", "o", "--checkpoint", "latest",
         "--no-onnx"]
    )
    assert args.checkpoint == "latest" and args.no_onnx is True

    args = _parse(["smoke"])
    assert (args.licensed, args.onnx, args.workdir, args.keep) == (False, False, None, False)
    args = _parse(["smoke", "--licensed", "--onnx", "--workdir", "w", "--keep"])
    assert (args.licensed, args.onnx, args.workdir, args.keep) == (True, True, "w", True)


def test_verbose_is_accepted_on_both_sides() -> None:
    assert _parse(["validate", "ds"]).verbose is False
    assert _parse(["--verbose", "validate", "ds"]).verbose is True
    assert _parse(["validate", "ds", "--verbose"]).verbose is True


def test_a_command_is_required() -> None:
    with pytest.raises(SystemExit):
        _parse([])


def test_library_errors_are_one_line(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    recipe = tmp_path / "broken.toml"
    recipe.write_text('[data]\ndataset_dir = "x"\n[trianer]\nseed = 1\n', encoding="utf-8")
    assert cli_main(["train", "--config", str(recipe), "--out", str(tmp_path / "run")]) == EXIT_ERROR
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert captured.err.strip().startswith("error: RecipeError:")


# ===========================================================================
# 2. the pipeline, without a solver
# ===========================================================================

N_STATES = 3


def _fake_labels(instances_dir: Path, labels_dir: Path) -> None:
    from sagsep.io_vrp import load_instance

    labels_dir.mkdir(parents=True, exist_ok=True)
    for index, vrp in enumerate(sorted(instances_dir.glob("*.vrp"))):
        instance = load_instance(vrp, vehicles=0)
        n = int(instance.n_nodes)
        demand = np.asarray(instance.demand, dtype=np.int64)
        capacity = int(instance.capacity)
        k_label = ceil_div(int(demand.sum()), capacity)
        record = InstanceRecord(
            id=vrp.stem,
            source_sha256=hashlib.sha256(vrp.read_bytes()).hexdigest(),
            n_nodes=n,
            capacity=capacity,
            k_fleet=k_label,
            k_label=k_label,
            termination="closed",
            coords=np.asarray(instance.coords, dtype=np.float64),
            demand=demand,
        )
        builder = ShardBuilder()
        row = builder.add_instance(record)
        rng = np.random.default_rng(1000 + index)
        pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        u = np.asarray([p[0] for p in pairs], dtype=np.int32)
        v = np.asarray([p[1] for p in pairs], dtype=np.int32)
        for iteration in range(1, N_STATES + 1):
            s_hat = rng.integers(0, 2, size=(n, k_label)).astype(np.uint8)
            s_hat[0, :] = 0
            builder.add_state(
                row,
                StateRecord(
                    iteration=iteration,
                    lp_value=100.0 + iteration,
                    cuts_added=iteration,
                    terminal=iteration == N_STATES,
                    support_u=u,
                    support_v=v,
                    support_x=rng.random(len(pairs)) * 0.9 + 0.1,
                    s_hat=s_hat,
                    objective=rng.random(k_label),
                    rhs=np.ones(k_label, dtype=np.int32),
                    slot_violated=np.zeros(k_label, dtype=np.uint8),
                ),
            )
        write_shard(labels_dir / f"{vrp.stem}.labels.npz", builder.build())
        sidecar: Dict[str, Any] = {
            "name": vrp.stem,
            "source_sha256": record.source_sha256,
            "n_nodes": n,
            "capacity": capacity,
            "k_fleet": k_label,
            "k_label": k_label,
            "termination": "closed",
            "iterations": N_STATES,
            "total_cuts": 2,
            "base_lower_bound": 101.0,
            "lower_bound": 103.0,
            "runtime_s": 0.0,
            "n_states": N_STATES,
            "solver_info": {
                "gurobi_version": [13, 0, 1],
                "gurobi_method": 1,
                "gurobi_threads": 1,
                "min_violation": 1e-6,
                "oracle_tol": 1e-6,
                "max_iter": None,
                "time_limit": None,
                "support_rule": "x_gt_0_float64",
            },
        }
        (labels_dir / f"{vrp.stem}.labels.json").write_text(
            json.dumps(sidecar, indent=2), encoding="utf-8"
        )


def test_generate_pack_validate(tmp_path: Path) -> None:
    instances_dir = tmp_path / "instances"
    labels_dir = tmp_path / "labels"
    dataset_dir = tmp_path / "dataset"

    assert cli_main(
        ["generate", "--out", str(instances_dir), "--n", "12", "--count", "3", "--seed", "7"]
    ) == EXIT_OK
    generated = sorted(instances_dir.glob("*.vrp"))
    assert len(generated) == 3
    assert (instances_dir / "manifest.csv").is_file()

    _fake_labels(instances_dir, labels_dir)

    assert cli_main(
        ["pack", "--labels", str(labels_dir), "--out", str(dataset_dir), "--shard-size", "2",
         "--split", "train=shard_0001", "--split", "val=shard_0002",
         "--instances", str(instances_dir)]
    ) == EXIT_OK
    shards = sorted(dataset_dir.glob("shard_*.npz"))
    assert [p.name for p in shards] == ["shard_0001.npz", "shard_0002.npz"]

    manifest = json.loads((dataset_dir / "dataset-manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"] == {"instances": 3, "states": 3 * N_STATES}
    assert manifest["source"]["kind"] == "generator"
    assert sorted(manifest["source"]["instances"]) == sorted(p.stem for p in generated)
    assert manifest["labeler"]["solver_version"] == "13.0.1"
    assert manifest["labeler"]["lp_method"] == 1
    assert [entry["split"] for entry in manifest["shards"]] == ["train", "val"]

    assert cli_main(["validate", str(dataset_dir)]) == EXIT_OK


def test_validate_reports_a_broken_dataset(tmp_path: Path) -> None:
    assert cli_main(["validate", str(tmp_path / "nothing-here")]) == EXIT_ERROR


def test_pack_refuses_labels_without_a_sidecar(tmp_path: Path) -> None:
    instances_dir = tmp_path / "instances"
    labels_dir = tmp_path / "labels"
    assert cli_main(
        ["generate", "--out", str(instances_dir), "--n", "10", "--count", "1", "--seed", "3"]
    ) == EXIT_OK
    _fake_labels(instances_dir, labels_dir)
    for sidecar in labels_dir.glob("*.labels.json"):
        sidecar.unlink()
    assert cli_main(
        ["pack", "--labels", str(labels_dir), "--out", str(tmp_path / "ds")]
    ) == EXIT_ERROR


def test_train_and_export_on_the_packaged_sample(tmp_path: Path) -> None:
    from sagsep_train.smoke import SMOKE_RECIPE_NAME, sample_dir

    recipe = sample_dir() / SMOKE_RECIPE_NAME
    run_dir = tmp_path / "run"
    assert cli_main(
        ["train", "--config", str(recipe), "--out", str(run_dir),
         "--set", "trainer.max_steps=5", "--set", "trainer.eval_every=5",
         "--set", "trainer.checkpoint_every=5", "--set", "trainer.log_every=1"]
    ) == EXIT_OK
    for name in ("config_snapshot.yaml", "train_log.jsonl", "subset_prob_latest.pt"):
        assert (run_dir / name).is_file(), name
    rows = [
        json.loads(line)
        for line in (run_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 5
    assert all(np.isfinite(float(row["loss"])) for row in rows)

    out_dir = tmp_path / "export"
    assert cli_main(
        ["export", "--run", str(run_dir), "--name", "unit", "--out", str(out_dir),
         "--checkpoint", "latest", "--no-onnx"]
    ) == EXIT_OK
    assert (out_dir / "unit.ckpt").is_file()
    assert (out_dir / "unit.yaml").is_file()
    assert (out_dir / "unit.export.json").is_file()
    assert not (out_dir / "unit.onnx").exists()


def test_train_resume_without_a_checkpoint_fails(tmp_path: Path) -> None:
    from sagsep_train.smoke import SMOKE_RECIPE_NAME, sample_dir

    recipe = sample_dir() / SMOKE_RECIPE_NAME
    assert cli_main(
        ["train", "--config", str(recipe), "--out", str(tmp_path / "empty"), "--resume"]
    ) == EXIT_ERROR
