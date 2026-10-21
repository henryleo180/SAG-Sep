from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from sagsep_train import __version__, cli

__all__ = ["build_parser", "main"]


_VERBOSE_HELP = "Print the traceback of a failure instead of one line"


def _verbose_parent() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--verbose", action="store_true", default=argparse.SUPPRESS, help=_VERBOSE_HELP
    )
    return parent


def build_parser() -> argparse.ArgumentParser:
    common = _verbose_parent()
    parser = argparse.ArgumentParser(
        prog="sagsep_train",
        description="SAG-Sep training pipeline: generate, label, pack, validate, "
        "train, export, smoke.",
    )
    parser.add_argument("--verbose", action="store_true", default=False, help=_VERBOSE_HELP)
    parser.add_argument("--version", action="version", version=f"sagsep_train {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    gen = sub.add_parser(
        "generate",
        parents=[common],
        help="Generate CVRP instances and their manifest.csv",
        description="Generate a set of Uchoa-style CVRP instances into --out.",
    )
    gen.add_argument("--out", required=True, help="Output directory")
    gen.add_argument("--n", type=int, required=True, help="Customers per instance")
    gen.add_argument("--count", type=int, required=True, help="Number of instances")
    gen.add_argument("--seed", type=int, required=True, help="Master seed of the set")
    gen.add_argument(
        "--spec",
        default=None,
        help="Four-digit class code <depot><customers><demand><route> for every "
        "instance; omitted, the generator draws one per instance",
    )

    lab = sub.add_parser(
        "label",
        parents=[common],
        help="Run the exact oracle over the root loop of every instance",
        description="Label a directory of .vrp files into <name>.labels.{json,npz}.",
    )
    lab.add_argument("--instances", required=True, help="Directory of .vrp files")
    lab.add_argument("--out", required=True, help="Output directory for the label files")
    lab.add_argument(
        "--shard",
        default="1/1",
        help="This worker's slice of the directory, 'i/N' (default 1/1)",
    )
    lab.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Cap the root loop at this many rounds (default: run to closure)",
    )
    lab.add_argument("--threads", type=int, default=1, help="Solver threads (default 1)")
    lab.add_argument(
        "--min-violation",
        type=float,
        default=1e-6,
        help="LP admission threshold of a cut (default 1e-6)",
    )
    lab.add_argument(
        "--vehicles-from",
        default=None,
        help="manifest.csv whose 'vehicles' column pins the LP fleet size",
    )
    lab.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Wall-clock cap per instance, in seconds",
    )

    pack = sub.add_parser(
        "pack",
        parents=[common],
        help="Merge label files into shards and write the dataset manifest",
        description="Concatenate *.labels.npz into shard_*.npz plus "
        "dataset-manifest.json.",
    )
    pack.add_argument("--labels", required=True, help="Directory of *.labels.npz files")
    pack.add_argument("--out", required=True, help="Dataset directory to write")
    pack.add_argument(
        "--shard-size", type=int, default=64, help="Instances per shard (default 64)"
    )
    pack.add_argument(
        "--split",
        action="append",
        default=None,
        metavar="NAME=shard_0001,...",
        help="Assign shards to a split (train/val); repeatable",
    )
    pack.add_argument(
        "--instances",
        default=None,
        help="Directory (or file) holding the generator's manifest.csv; by default "
        "the labels directory and its parent are probed",
    )

    val = sub.add_parser(
        "validate",
        parents=[common],
        help="Check a dataset directory against its manifest",
        description="Re-hash every shard and check it against dataset-manifest.json.",
    )
    val.add_argument("dataset_dir", metavar="DIR", help="Dataset directory")

    train = sub.add_parser(
        "train",
        parents=[common],
        help="Train a model from a recipe",
        description="Run one training job; the run directory holds the snapshot, "
        "the logs and the checkpoints.",
    )
    train.add_argument("--config", required=True, help="Recipe file (.toml/.yaml)")
    train.add_argument("--out", required=True, help="Run directory")
    train.add_argument(
        "--set",
        action="append",
        default=None,
        metavar="KEY.PATH=VALUE",
        help="Override a recipe key; repeatable",
    )
    train.add_argument(
        "--resume",
        action="store_true",
        help=f"Continue from the run directory's {cli.LATEST_CHECKPOINT}",
    )

    exp = sub.add_parser(
        "export",
        parents=[common],
        help="Export a run's checkpoint as a released model",
        description="Write <name>.ckpt, <name>.yaml, <name>.onnx and the export "
        "report, proving each against the serving path.",
    )
    exp.add_argument("--run", required=True, help="Training run directory")
    exp.add_argument("--name", required=True, help="Released stem, e.g. cvrp-two-index-best")
    exp.add_argument("--out", required=True, help="Output directory")
    exp.add_argument(
        "--checkpoint",
        default="best",
        choices=["best", "latest"],
        help="Which of the run's checkpoints to publish: best (lowest validation "
        "loss, the default) or latest",
    )
    exp.add_argument(
        "--no-onnx", action="store_true", help="Skip the ONNX graph and its parity gates"
    )

    smoke = sub.add_parser(
        "smoke",
        parents=[common],
        help="Run the packaged end-to-end checks",
        description="Train on the packaged sample and export it; --licensed adds "
        "the solver pipeline, --onnx adds the graph export and the C engine.",
    )
    smoke.add_argument(
        "--licensed", action="store_true", help="Also run the tier that needs Gurobi"
    )
    smoke.add_argument(
        "--onnx", action="store_true", help="Also run the tier that needs onnx/onnxruntime"
    )
    smoke.add_argument(
        "--workdir", default=None, help="Where to work (default: a temporary directory)"
    )
    smoke.add_argument("--keep", action="store_true", help="Keep the working directory")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    return cli.main(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
