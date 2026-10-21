from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from sagsep_train import __version__
from sagsep_train.cli import EXIT_ERROR, EXIT_OK, EXIT_SKIPPED, CapabilityMissing

__all__ = [
    "REPORT_FILENAME",
    "SAMPLE_PACKAGE",
    "SMOKE_RECIPE_NAME",
    "LICENSED_SEED",
    "SmokeReport",
    "sample_dir",
    "copy_sample",
    "repo_root",
    "run_base_tier",
    "run_licensed_tier",
    "run_onnx_tier",
    "run",
]

REPORT_FILENAME = "smoke_report.json"

SAMPLE_PACKAGE = "sagsep_train.sample"

SMOKE_RECIPE_NAME = "smoke.toml"

LICENSED_SEED = 20260904

LICENSED_N = 15
LICENSED_COUNT = 4

EXPECTED_RUN_FILES = (
    "config_snapshot.yaml",
    "val_record_ids.json",
    "train_log.jsonl",
    "val_log.jsonl",
    "subset_prob_latest.pt",
    "subset_prob_best.pt",
)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


@dataclass
class Step:
    tier: str
    name: str
    ok: bool
    seconds: float
    detail: str = ""
    skipped: bool = False


@dataclass
class SmokeReport:
    workdir: Path
    tiers: List[str] = field(default_factory=list)
    steps: List[Step] = field(default_factory=list)
    ok: bool = True
    skipped: Optional[str] = None

    def record(self, step: Step) -> None:
        self.steps.append(step)
        if not step.ok:
            self.ok = False
        status = "SKIP" if step.skipped else ("PASS" if step.ok else "FAIL")
        suffix = f" -- {step.detail}" if step.detail else ""
        print(f"[smoke:{step.tier}] {status} {step.name} ({step.seconds:.1f}s){suffix}", flush=True)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "package_version": __version__,
            "python": sys.version.split()[0],
            "tiers": list(self.tiers),
            "ok": bool(self.ok),
            "skipped": self.skipped,
            "total_seconds": round(sum(step.seconds for step in self.steps), 3),
            "steps": [asdict(step) for step in self.steps],
        }

    def write(self) -> Path:
        path = Path(self.workdir) / REPORT_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_dict(), indent=2) + "\n", encoding="utf-8")
        return path


@contextmanager
def _step(report: SmokeReport, tier: str, name: str) -> Iterator[Dict[str, str]]:
    box: Dict[str, str] = {"detail": ""}
    started = time.perf_counter()
    try:
        yield box
    except CapabilityMissing as exc:
        report.record(Step(tier, name, True, time.perf_counter() - started, str(exc), skipped=True))
        raise
    except BaseException as exc:
        report.record(
            Step(tier, name, False, time.perf_counter() - started, f"{type(exc).__name__}: {exc}")
        )
        raise
    report.record(Step(tier, name, True, time.perf_counter() - started, box["detail"]))


# ---------------------------------------------------------------------------
# locating what the tiers need
# ---------------------------------------------------------------------------


def sample_dir() -> Path:
    from importlib import resources

    try:
        traversable = resources.files(SAMPLE_PACKAGE)
    except ModuleNotFoundError as exc:
        raise CapabilityMissing(
            f"the packaged sample dataset ({SAMPLE_PACKAGE}) is not installed: {exc}"
        ) from None
    path = Path(str(traversable))
    if not (path / "shard_0001.npz").is_file():
        raise CapabilityMissing(
            f"the packaged sample dataset is incomplete at {path} (no shard_0001.npz)"
        )
    return path


def copy_sample(dest: str | os.PathLike[str]) -> Path:
    destination = Path(dest)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(sample_dir(), destination, ignore=shutil.ignore_patterns("__pycache__"))
    return destination


def repo_root() -> Optional[Path]:
    override = os.environ.get("SAGSEP_REPO_ROOT", "").strip()
    if override:
        candidate = Path(override)
        return candidate if (candidate / "models").is_dir() else None
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "models").is_dir() and (parent / "package" / "python").is_dir():
            return parent
    return None


def _require_gurobi() -> str:
    try:
        from sagsep import gurobi_c
    except ImportError as exc:
        raise CapabilityMissing(f"the sagsep package is not importable: {exc}") from None
    try:
        version = gurobi_c.version()
        gurobi_c.GurobiEnv()
    except Exception as exc:
        raise CapabilityMissing(
            f"no usable Gurobi (set GUROBI_HOME and a license): {type(exc).__name__}: {exc}"
        ) from None
    return ".".join(str(part) for part in version)


def _require_onnx() -> str:
    try:
        import onnx
        import onnxruntime
    except ImportError as exc:
        raise CapabilityMissing(
            f"onnx/onnxruntime are not installed (pip install 'sagsep[train]'): {exc}"
        ) from None
    return f"onnx {onnx.__version__} / onnxruntime {onnxruntime.__version__}"


def _cli(argv: List[str]) -> None:
    from sagsep_train.__main__ import main as cli_main

    code = cli_main(argv)
    if code != EXIT_OK:
        raise RuntimeError(f"`sagsep_train {' '.join(argv)}` exited {code}")


# ---------------------------------------------------------------------------
# tier 1: the base tier
# ---------------------------------------------------------------------------


def run_base_tier(workdir: Path, report: SmokeReport) -> Dict[str, Any]:
    import torch

    from sagsep.model import PolicyModel, load_policy_model
    from sagsep_train.config import load_recipe
    from sagsep_train.data.collate import state_features
    from sagsep_train.data.format import read_shard
    from sagsep_train.export import export_run
    from sagsep_train.trainer import Trainer

    tier = "base"
    context: Dict[str, Any] = {}

    with _step(report, tier, "sample") as box:
        sample = copy_sample(workdir / "sample")
        shard = sample / "shard_0001.npz"
        data = read_shard(shard)
        box["detail"] = f"{data.n_instances} instances / {data.n_states} states in {sample}"
        context["sample"] = sample

    with _step(report, tier, "recipe") as box:
        recipe_path = sample / SMOKE_RECIPE_NAME
        recipe = load_recipe(recipe_path)
        if Path(recipe.data.dataset_dir).resolve() != sample.resolve():
            raise RuntimeError(
                f"{SMOKE_RECIPE_NAME} resolves dataset_dir to {recipe.data.dataset_dir}, "
                f"expected the copied sample at {sample}"
            )
        box["detail"] = (
            f"latent_dim={recipe.model.latent_dim} n_layer={recipe.model.n_layer} "
            f"max_steps={recipe.trainer.max_steps} device={recipe.trainer.device}"
        )

    with _step(report, tier, "train") as box:
        torch.set_num_threads(1)
        run_dir = workdir / "run"
        trainer = Trainer(recipe, run_dir=run_dir, recipe_path=recipe_path)
        before = {name: tensor.detach().clone() for name, tensor in trainer.net.state_dict().items()}
        trainer.train()
        after = trainer.net.state_dict()
        changed = [
            name for name, tensor in after.items() if not torch.equal(tensor, before[name])
        ]
        if not changed:
            raise RuntimeError("training changed no parameter at all")
        rows = [
            json.loads(line)
            for line in (run_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            raise RuntimeError("train_log.jsonl is empty: no step was logged")
        losses = [float(row["loss"]) for row in rows]
        if not all(math.isfinite(value) for value in losses):
            raise RuntimeError(f"non-finite training loss in {losses}")
        missing = [name for name in EXPECTED_RUN_FILES if not (run_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"the run directory is missing {', '.join(missing)}")
        box["detail"] = (
            f"{trainer.global_step} steps, loss {losses[0]:.4f} -> {losses[-1]:.4f}, "
            f"{len(changed)}/{len(after)} tensors changed"
        )
        context["run_dir"] = run_dir
        context["trainer"] = trainer

    with _step(report, tier, "export") as box:
        export_dir = workdir / "export"
        exported = export_run(
            run_dir,
            name="smoke",
            out_dir=export_dir,
            checkpoint="latest",
            onnx=False,
        )
        box["detail"] = f"{exported.ckpt_path.name} + {exported.manifest_path.name}"
        context["export_dir"] = export_dir
        context["export"] = exported

    with _step(report, tier, "serve") as box:
        state = read_shard(sample / "shard_0001.npz").state(0)
        feats, _labels = state_features(
            state,
            support_rule=recipe.data.support_rule,
            cross_depot=recipe.data.partial_connect_cross_depot,
        )
        net = trainer.net
        net.eval()
        want = PolicyModel.from_network(net, device="cpu").forward_probs(feats)
        got = load_policy_model(
            exported.ckpt_path,
            device="cpu",
            partial_connect_cross_depot=recipe.data.partial_connect_cross_depot,
        ).forward_probs(feats)
        if want.tobytes() != got.tobytes():
            delta = float(abs(want.astype("float64") - got.astype("float64")).max())
            raise RuntimeError(
                f"the exported checkpoint does not serve what was trained "
                f"(max |delta| = {delta:.3e})"
            )
        box["detail"] = f"forward_probs {want.shape} bitwise equal on state 0"
        context["net"] = net

    return context


# ---------------------------------------------------------------------------
# tier 2: the licensed tier
# ---------------------------------------------------------------------------


class _NoCuts:
    def separate(self, x_dense, *, iteration: int):
        from sagsep_train.rootloop import SeparationResult

        return SeparationResult(cuts=[])


def run_licensed_tier(workdir: Path, report: SmokeReport, context: Dict[str, Any]) -> None:
    from dataclasses import replace

    from sagsep.config import preset
    from sagsep.io_vrp import load_instance
    from sagsep.loop import run_instance
    from sagsep.model import load_policy_model
    from sagsep_train.rootloop import run_root_loop

    tier = "licensed"
    base = workdir / "licensed"
    instances_dir = base / "instances"
    labels_dir = base / "labels"
    dataset_dir = base / "dataset"

    with _step(report, tier, "solver") as box:
        box["detail"] = f"Gurobi {_require_gurobi()}"

    with _step(report, tier, "generate") as box:
        _cli(
            [
                "generate",
                "--out", str(instances_dir),
                "--n", str(LICENSED_N),
                "--count", str(LICENSED_COUNT),
                "--seed", str(LICENSED_SEED),
            ]
        )
        written = sorted(instances_dir.glob("*.vrp"))
        if len(written) != LICENSED_COUNT:
            raise RuntimeError(f"expected {LICENSED_COUNT} instances, got {len(written)}")
        box["detail"] = f"{len(written)} instances of n={LICENSED_N}, seed {LICENSED_SEED}"

    with _step(report, tier, "label") as box:
        _cli(
            [
                "label",
                "--instances", str(instances_dir),
                "--out", str(labels_dir),
                "--threads", "1",
                "--vehicles-from", str(instances_dir / "manifest.csv"),
            ]
        )
        sidecars = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(labels_dir.glob("*.labels.json"))
        ]
        if len(sidecars) != LICENSED_COUNT:
            raise RuntimeError(f"expected {LICENSED_COUNT} label files, got {len(sidecars)}")
        with_cuts = [side for side in sidecars if int(side["total_cuts"]) > 0]
        if not with_cuts:
            raise RuntimeError(
                f"seed {LICENSED_SEED} produced no separated cut in any of "
                f"{LICENSED_COUNT} instances -- pick another seed"
            )
        total_states = sum(int(side["n_states"]) for side in sidecars)
        box["detail"] = (
            f"{total_states} states, {len(with_cuts)}/{len(sidecars)} instances separated a cut"
        )
        context["licensed_sidecars"] = sidecars

    with _step(report, tier, "pack") as box:
        _cli(
            [
                "pack",
                "--labels", str(labels_dir),
                "--out", str(dataset_dir),
                "--shard-size", str(LICENSED_COUNT),
                "--split", "train=shard_0001",
                "--instances", str(instances_dir),
            ]
        )
        box["detail"] = f"{len(sorted(dataset_dir.glob('shard_*.npz')))} shard(s) in {dataset_dir}"

    with _step(report, tier, "validate") as box:
        _cli(["validate", str(dataset_dir)])
        box["detail"] = "dataset matches its manifest"

    root = repo_root()
    with _step(report, tier, "released-model") as box:
        if root is None:
            raise CapabilityMissing(
                "models/ is not next to this package (installed copy): the released "
                "checkpoint cannot be located"
            )
        ckpt = root / "models" / "cvrp-two-index-best.ckpt"
        if not ckpt.is_file():
            raise CapabilityMissing(f"released checkpoint not found: {ckpt}")

        chosen = with_cuts[0]
        vrp_path = instances_dir / f"{chosen['name']}.vrp"
        vehicles = int(chosen["k_fleet"])
        instance = load_instance(vrp_path, vehicles=vehicles)
        base_result = run_root_loop(
            instance, separator=_NoCuts(), vehicles=vehicles, max_iter=1
        )
        base_bound = float(base_result.lower_bound)

        cfg = preset("n50")
        cfg = replace(cfg, loop=replace(cfg.loop, max_iter=3))
        model = load_policy_model(ckpt, device="cpu")
        result = run_instance(instance, cfg, model)
        if not math.isfinite(result.objective):
            raise RuntimeError(f"the deployment loop returned {result.objective!r}")
        if result.objective < base_bound - 1e-6:
            raise RuntimeError(
                f"the deployment loop dropped below the base LP: "
                f"{result.objective!r} < {base_bound!r}"
            )
        box["detail"] = (
            f"{chosen['name']}: base LP {base_bound:.4f} -> {result.objective:.4f} "
            f"in {result.iteration_count} round(s), {result.total_cuts} cut(s)"
        )


# ---------------------------------------------------------------------------
# tier 3: the ONNX tier
# ---------------------------------------------------------------------------


def _c_engine(root: Optional[Path]) -> Optional[Path]:
    if root is None:
        return None
    for name in ("sagsep", "sagsep_gpu"):
        candidate = root / "package" / "c" / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def run_onnx_tier(workdir: Path, report: SmokeReport, context: Dict[str, Any]) -> None:
    from sagsep_train.export import export_run

    tier = "onnx"

    with _step(report, tier, "runtime") as box:
        box["detail"] = _require_onnx()

    with _step(report, tier, "export-onnx") as box:
        out_dir = workdir / "export-onnx"
        exported = export_run(
            context["run_dir"],
            name="smoke",
            out_dir=out_dir,
            checkpoint="latest",
            onnx=True,
        )
        parity = exported.parity or {}
        if not parity.get("ok"):
            raise RuntimeError(f"ONNX parity did not pass: {parity}")
        box["detail"] = (
            f"{exported.onnx_path.name}, {len(parity.get('cases', {}))} parity case(s), "
            f"worst |runtime - torch| = {parity['max_abs_diff']:.3e}"
        )
        context["onnx_path"] = exported.onnx_path

    with _step(report, tier, "c-engine") as box:
        root = repo_root()
        engine = _c_engine(root)
        if engine is None:
            box["detail"] = (
                "skipped: no built package/c/sagsep next to this package "
                "(build it with `make -C package/c`)"
            )
            return
        sample = context["sample"]
        instance = sorted(sample.glob("*.vrp"))[0].stem
        env = dict(os.environ)
        env["SAGSEP_INSTANCES_ROOT"] = str(sample)
        command = [
            str(engine),
            "--size-preset", "n50",
            "--instance", instance,
            "--onnx", str(context["onnx_path"]),
            "--device", "cpu",
            "--max-iter", "2",
            "--out", str(workdir / "c-engine"),
        ]
        completed = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=600
        )
        if completed.returncode != 0:
            tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-3:]
            raise RuntimeError(
                f"{engine.name} exited {completed.returncode}: {' | '.join(tail)}"
            )
        box["detail"] = f"{engine.name} ran {instance} to completion"


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run(
    *,
    workdir: str | os.PathLike[str] | None = None,
    licensed: bool = False,
    onnx: bool = False,
    keep: bool = False,
) -> int:
    temporary: Optional[str] = None
    if workdir is None:
        temporary = tempfile.mkdtemp(prefix="sagsep-smoke-")
        work = Path(temporary)
    else:
        work = Path(workdir)
        work.mkdir(parents=True, exist_ok=True)

    tiers = ["base"] + (["licensed"] if licensed else []) + (["onnx"] if onnx else [])
    report = SmokeReport(workdir=work, tiers=tiers)
    print(f"[smoke] workdir {work}")
    print(f"[smoke] tiers: {', '.join(tiers)}")

    code = EXIT_OK
    try:
        context = run_base_tier(work, report)
        if licensed:
            run_licensed_tier(work, report, context)
        if onnx:
            run_onnx_tier(work, report, context)
    except CapabilityMissing as exc:
        report.skipped = str(exc)
        code = EXIT_SKIPPED
        print(f"[smoke] SKIPPED: {exc}", file=sys.stderr)
    except Exception as exc:
        report.ok = False
        code = EXIT_ERROR
        print(f"[smoke] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)

    path = report.write()
    print(f"[smoke] report {path}")
    total = report.to_json_dict()["total_seconds"]
    verdict = {EXIT_OK: "PASS", EXIT_ERROR: "FAIL", EXIT_SKIPPED: "SKIPPED"}[code]
    print(f"[smoke] {verdict} -- {len(report.steps)} step(s) in {total:.1f}s")

    if temporary is not None and not keep:
        shutil.rmtree(temporary, ignore_errors=True)
    elif temporary is not None:
        print(f"[smoke] kept {temporary}")
    return code
