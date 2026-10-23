"""Command-line interface: ``surfbench data | fit | study``."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from surfbench import __version__

PENDING: dict[tuple[str, str | None], str] = {
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
    fit.add_argument("--underlying", choices=["BTC", "ETH"], default="BTC")

    study = sub.add_parser("study", help="run | report")
    study_sub = study.add_subparsers(dest="action", required=True)
    run = study_sub.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--workers", type=int, default=None, help="worker processes (capped at 3)")
    run.add_argument("--force-new", action="store_true")
    run.add_argument("--limit", type=int, default=None, help="run at most this many units")
    report = study_sub.add_parser("report")
    report.add_argument("--config", type=Path, required=True)
    return parser


def load_toml(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as fh:
        return tomllib.load(fh)


def _filter_regex(study: dict[str, Any]) -> str | None:
    from surfbench.quotes import FilterConfig

    fpath = study.get("fit", {}).get("filter")
    return FilterConfig.load(Path(fpath)).symbol_regex if fpath else None


def cmd_data_tardis(args: argparse.Namespace) -> int:
    from surfbench.data import tardis

    study = load_toml(args.config)
    if study["data"].get("source", "tardis") != "tardis":
        print("config [data].source is not 'tardis'", file=sys.stderr)
        return 2
    cfg = tardis.TardisConfig.from_study(study, _filter_regex(study))
    tardis.run_session(cfg)
    return cmd_data_status(args)


def cmd_data_deribit(args: argparse.Namespace) -> int:
    from surfbench.data import deribit

    d = deribit.save_live_pair(args.out, horizon_min=args.horizon_min)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    print(f"saved {d.name}: {meta['n_t0']} instruments at t0, {meta.get('n_t1', 0)} at t1")
    return 0


def cmd_data_status(args: argparse.Namespace) -> int:
    from surfbench.data import tardis

    study = load_toml(args.config)
    if study["data"].get("source", "tardis") != "tardis":
        print(f"source: {study['data'].get('source')} (no download state)")
        return 0
    cfg = tardis.TardisConfig.from_study(study)
    counts: dict[str, int] = {}
    for month in sorted(cfg.months):
        st = tardis.month_status(cfg, month) or "pending"
        counts[st] = counts.get(st, 0) + 1
        print(f"{month}  {st}")
    nb = tardis.quota_not_before(cfg)
    total = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    print(f"months: {len(cfg.months)} ({total}); cache {tardis.cache_size_bytes(cfg) / 1e6:.1f} MB")
    if nb > datetime.now(UTC).timestamp():
        stamp = datetime.fromtimestamp(nb, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
        print(f"quota locked until {stamp}")
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    from surfbench.study.runner import fit_table

    print(fit_table(args.snapshot, args.filter, args.underlying))
    return 0


def cmd_study_run(args: argparse.Namespace) -> int:
    from surfbench.study.runner import run_study

    counts = run_study(
        args.config, workers=args.workers, force_new=args.force_new, limit=args.limit
    )
    print(f"units: ok {counts['ok']}, error {counts['error']}, skipped {counts['skipped']}")
    return 0 if counts["error"] == 0 else 1


def cmd_study_report(args: argparse.Namespace) -> int:
    from surfbench.study.report import write_report

    out = write_report(args.config)
    print(f"wrote {out}")
    return 0


HANDLERS = {
    ("data", "tardis"): cmd_data_tardis,
    ("data", "deribit"): cmd_data_deribit,
    ("data", "status"): cmd_data_status,
    ("fit", None): cmd_fit,
    ("study", "run"): cmd_study_run,
    ("study", "report"): cmd_study_report,
}


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
    return HANDLERS[key](args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
