from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from sagsep_train import __version__

__all__ = [
    "EXIT_OK",
    "EXIT_ERROR",
    "EXIT_SKIPPED",
    "CapabilityMissing",
    "main",
    "cmd_generate",
    "cmd_label",
    "cmd_pack",
    "cmd_validate",
    "cmd_train",
    "cmd_export",
    "cmd_smoke",
]

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_SKIPPED = 3

LATEST_CHECKPOINT = "subset_prob_latest.pt"


class CapabilityMissing(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------


def _wrote(path: Path | str) -> None:
    print(f"[sagsep_train] wrote {path}")


def _parse_spec_code(code: str, n_customers: int):
    from sagsep_train.instances import XmlSpec

    text = str(code).strip()
    if len(text) != 4 or not text.isdigit():
        raise ValueError(
            f"--spec must be the four-digit class code "
            f"<depot><customers><demand><route>, got {code!r}"
        )
    depot, customer, demand, route = (int(ch) for ch in text)
    return XmlSpec(
        n_customers=int(n_customers),
        depot_pos=depot,
        customer_pos=customer,
        demand_type=demand,
        route_size=route,
    )


def _read_generator_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_generator_manifest(labels_dir: Path, instances_dir: Path | None) -> Path | None:
    if instances_dir is not None:
        candidate = Path(instances_dir)
        if candidate.is_dir():
            candidate = candidate / "manifest.csv"
        if not candidate.is_file():
            raise FileNotFoundError(f"no generator manifest at {candidate}")
        return candidate
    for candidate in (labels_dir / "manifest.csv", labels_dir.parent / "manifest.csv"):
        if candidate.is_file():
            return candidate
    return None


def _one(values: Iterable[Any], field: str) -> Any:
    seen: list[Any] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    if not seen:
        raise ValueError(f"no label sidecar carries {field}")
    if len(seen) > 1:
        raise ValueError(
            f"label sidecars disagree on {field}: {seen!r} -- pack one labelling run at a time"
        )
    return seen[0]


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


def cmd_generate(args: argparse.Namespace) -> int:
    from sagsep_train.instances import generate_set

    out_dir = Path(args.out)
    specs = None
    if args.spec:
        specs = [_parse_spec_code(args.spec, args.n)]
    rows = generate_set(
        out_dir,
        n_customers=int(args.n),
        count=int(args.count),
        seed=int(args.seed),
        specs=specs,
    )
    for row in rows:
        _wrote(out_dir / f"{row['instance']}.vrp")
    manifest = out_dir / "manifest.csv"
    _wrote(manifest)
    print(
        f"[sagsep_train] generated {len(rows)} instance(s) with n={int(args.n)} "
        f"from seed {int(args.seed)}"
    )
    print(f"[sagsep_train] manifest: {manifest}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# label
# ---------------------------------------------------------------------------


def cmd_label(args: argparse.Namespace) -> int:
    from sagsep_train.labeler import label_set

    instances_dir = Path(args.instances)
    out_dir = Path(args.out)
    vehicles_from = Path(args.vehicles_from) if args.vehicles_from else None
    written = label_set(
        instances_dir,
        out_dir,
        shard=str(args.shard),
        vehicles_from=vehicles_from,
        max_iter=None if args.max_iter is None else int(args.max_iter),
        threads=int(args.threads),
        min_violation=float(args.min_violation),
        time_limit=None if args.time_limit is None else float(args.time_limit),
    )
    for path in written:
        _wrote(path)
    print(
        f"[sagsep_train] labelled {len(written)} instance(s) of slice {args.shard} "
        f"into {out_dir}"
    )
    return EXIT_OK


# ---------------------------------------------------------------------------
# pack
# ---------------------------------------------------------------------------


def _labeler_block(sidecars: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    infos = [dict(side.get("solver_info") or {}) for side in sidecars]
    version = _one((tuple(info.get("gurobi_version") or ()) for info in infos), "gurobi_version")
    return {
        "version": __version__,
        "solver": "gurobi",
        "solver_version": ".".join(str(part) for part in version),
        "threads": int(_one((info.get("gurobi_threads") for info in infos), "gurobi_threads")),
        "lp_method": int(_one((info.get("gurobi_method") for info in infos), "gurobi_method")),
        "tolerances": {
            "oracle_tol": float(_one((info.get("oracle_tol") for info in infos), "oracle_tol")),
            "min_violation": float(
                _one((info.get("min_violation") for info in infos), "min_violation")
            ),
        },
        "min_violation": float(
            _one((info.get("min_violation") for info in infos), "min_violation")
        ),
        "max_iter": _one((info.get("max_iter") for info in infos), "max_iter"),
        "require_optimal": True,
        "time_limit": _one((info.get("time_limit") for info in infos), "time_limit"),
        "support_rule": str(_one((info.get("support_rule") for info in infos), "support_rule")),
    }


def _source_block(
    sidecars: Sequence[Mapping[str, Any]], generator_manifest: Path | None
) -> dict[str, Any]:
    names = sorted(str(side.get("name", "")) for side in sidecars)
    if generator_manifest is not None:
        rows = _read_generator_manifest(generator_manifest)
        wanted = set(names)
        kept = [row for row in rows if str(row.get("instance", "")) in wanted]
        return {
            "kind": "generator",
            "name": "sagsep_train.instances uchoa-style generator",
            "seed": [int(row["seed"]) for row in kept],
            "n_customers": sorted({int(row["n"]) for row in kept}),
            "specs": [str(row["spec"]) for row in kept],
            "instances": [str(row["instance"]) for row in kept],
            "generator_manifest": generator_manifest.name,
            "generator_manifest_sha256": _sha256_file(generator_manifest),
        }
    digest = hashlib.sha256()
    hashes = {
        str(side.get("name", "")): str(side.get("source_sha256", "")) for side in sidecars
    }
    for name in names:
        digest.update(f"{name}:{hashes[name]}\n".encode("utf-8"))
    return {
        "kind": "external",
        "archive": "instance files, as handed to the labeller",
        "archive_sha256": digest.hexdigest(),
        "instances": hashes,
    }


def _parse_splits(entries: Sequence[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in entries or ():
        name, sep, shards = str(entry).partition("=")
        if not sep or not name.strip() or not shards.strip():
            raise ValueError(f"--split must look like NAME=shard_0001,shard_0002, got {entry!r}")
        for shard in shards.split(","):
            shard = shard.strip()
            if not shard:
                continue
            if shard in out and out[shard] != name.strip():
                raise ValueError(f"--split assigns {shard!r} to two splits")
            out[shard] = name.strip()
    return out


def cmd_pack(args: argparse.Namespace) -> int:
    import json

    from sagsep_train.data.format import merge_shards
    from sagsep_train.data.manifest import (
        MANIFEST_FILENAME,
        build_manifest,
        write_manifest,
    )

    labels_dir = Path(args.labels)
    out_dir = Path(args.out)
    inputs = sorted(labels_dir.glob("*.labels.npz"))
    if not inputs:
        raise FileNotFoundError(f"no *.labels.npz under {labels_dir}")

    sidecars: list[dict[str, Any]] = []
    for npz in inputs:
        side = npz.with_name(npz.name[: -len(".npz")] + ".json")
        if not side.is_file():
            raise FileNotFoundError(
                f"{npz.name}: its {side.name} sidecar is missing -- the manifest's "
                "labeler block is read from the sidecars, not guessed"
            )
        sidecars.append(json.loads(side.read_text(encoding="utf-8")))

    generator_manifest = _find_generator_manifest(
        labels_dir, Path(args.instances) if args.instances else None
    )
    written = merge_shards(inputs, out_dir, shard_size=int(args.shard_size))
    for path in written:
        _wrote(path)

    manifest = build_manifest(
        out_dir,
        source=_source_block(sidecars, generator_manifest),
        labeler=_labeler_block(sidecars),
        splits=_parse_splits(args.split),
        created_by_version=__version__,
    )
    manifest_path = out_dir / MANIFEST_FILENAME
    write_manifest(manifest_path, manifest)
    _wrote(manifest_path)
    print(
        f"[sagsep_train] packed {manifest.n_instances} instance(s) / "
        f"{manifest.n_states} state(s) into {len(written)} shard(s)"
    )
    return EXIT_OK


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def cmd_validate(args: argparse.Namespace) -> int:
    from sagsep_train.data.validate import validate_dataset

    directory = Path(args.dataset_dir)
    problems = validate_dataset(directory)
    if problems:
        print(f"[sagsep_train] {directory}: {len(problems)} problem(s)", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return EXIT_ERROR
    print(f"[sagsep_train] {directory}: valid")
    return EXIT_OK


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    from sagsep_train.config import apply_overrides, load_recipe
    from sagsep_train.trainer import Trainer

    recipe_path = Path(args.config)
    run_dir = Path(args.out)
    recipe = load_recipe(recipe_path)
    if args.set:
        recipe = apply_overrides(recipe, args.set)
    if args.resume:
        resume_path = run_dir.resolve() / LATEST_CHECKPOINT
        if not resume_path.is_file():
            raise FileNotFoundError(f"--resume: no checkpoint at {resume_path}")
        recipe.trainer.resume_from = str(resume_path)
        print(f"[sagsep_train] resuming from {resume_path}")

    trainer = Trainer(recipe, run_dir=run_dir, recipe_path=recipe_path)
    trainer.train()
    print(f"[sagsep_train] run directory: {trainer.run_dir}")
    for name in sorted(p.name for p in trainer.run_dir.iterdir() if p.is_file()):
        _wrote(trainer.run_dir / name)
    return EXIT_OK


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def cmd_export(args: argparse.Namespace) -> int:
    from sagsep_train.export import export_run

    report = export_run(
        Path(args.run),
        name=str(args.name),
        out_dir=Path(args.out),
        checkpoint=str(args.checkpoint),
        onnx=not bool(args.no_onnx),
        verbose=bool(getattr(args, "verbose", False)),
    )
    _wrote(report.ckpt_path)
    _wrote(report.manifest_path)
    if report.onnx_path is not None:
        _wrote(report.onnx_path)
    if report.report_path is not None:
        _wrote(report.report_path)
    print(
        f"[sagsep_train] exported {report.name} from checkpoint "
        f"{report.checkpoint_kind} at step {report.global_step}"
    )
    if report.parity is not None:
        print(
            f"[sagsep_train] ONNX parity: PASS, worst |runtime - torch| = "
            f"{report.parity['max_abs_diff']:.3e}"
        )
    return EXIT_OK


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------


def cmd_smoke(args: argparse.Namespace) -> int:
    from sagsep_train import smoke

    return smoke.run(
        workdir=args.workdir,
        licensed=bool(args.licensed),
        onnx=bool(args.onnx),
        keep=bool(args.keep),
    )


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

COMMANDS = {
    "generate": cmd_generate,
    "label": cmd_label,
    "pack": cmd_pack,
    "validate": cmd_validate,
    "train": cmd_train,
    "export": cmd_export,
    "smoke": cmd_smoke,
}


def main(args: argparse.Namespace) -> int:
    command = getattr(args, "command", None)
    handler = COMMANDS.get(str(command))
    if handler is None:
        print(f"error: unknown command {command!r}", file=sys.stderr)
        return EXIT_ERROR
    verbose = bool(getattr(args, "verbose", False))
    try:
        return int(handler(args))
    except CapabilityMissing as exc:
        print(f"skipped: {exc}", file=sys.stderr)
        return EXIT_SKIPPED
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return EXIT_ERROR
    except Exception as exc:
        if verbose:
            traceback.print_exc()
        else:
            print(f"error: {type(exc).__name__}: {exc}".replace("\n", " "), file=sys.stderr)
        return EXIT_ERROR
