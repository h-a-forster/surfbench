"""Command-line interface: ``surfbench data | fit | study``.

Only ``--version`` works in the scaffold. Each subcommand names the milestone that
implements it (``docs/DESIGN.md`` section 9) and exits with status 2 until then.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from surfbench import __version__

PENDING = {
    ("data", "tardis"): "M4",
    ("data", "deribit"): "M4",
    ("data", "status"): "M4",
    ("fit", None): "M5",
    ("study", "run"): "M5",
    ("study", "report"): "M5",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="surfbench",
        description="Benchmark of implied volatility surface fits on Deribit BTC and ETH options.",
    )
    parser.add_argument("--version", action="version", version=f"surfbench {__version__}")
    sub = parser.add_subparsers(dest="command")

    data = sub.add_parser("data", help="extract and cache snapshots")
    data_sub = data.add_subparsers(dest="action", required=True)
    tardis = data_sub.add_parser(
        "tardis", help="stream first-of-month Tardis.dev files and cache t0/t1 snapshots"
    )
    tardis.add_argument("--config", type=Path, required=True)
    deribit = data_sub.add_parser("deribit", help="save a live snapshot from the Deribit API")
    deribit.add_argument("--out", type=Path, default=Path("cache/deribit"))
    deribit.add_argument("--horizon-min", type=float, default=0.0)
    status = data_sub.add_parser("status", help="list cached snapshots, sizes and quota state")
    status.add_argument("--config", type=Path, required=True)

    fit = sub.add_parser("fit", help="fit every model to one cached snapshot and print a table")
    fit.add_argument("snapshot", type=Path)
    fit.add_argument("--filter", type=Path, default=Path("configs/filter_v1.toml"))

    study = sub.add_parser("study", help="run | report")
    study_sub = study.add_subparsers(dest="action", required=True)
    run = study_sub.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--workers", type=int, default=None, help="worker processes (capped at 3)")
    run.add_argument("--force-new", action="store_true")
    report = study_sub.add_parser("report")
    report.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    key = (args.command, getattr(args, "action", None))
    milestone = PENDING.get(key)
    if milestone is not None:
        name = " ".join(k for k in key if k)
        print(f"surfbench {name}: not implemented yet (milestone {milestone})", file=sys.stderr)
        return 2
    return 0  # pragma: no cover - every command is pending in the scaffold


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
