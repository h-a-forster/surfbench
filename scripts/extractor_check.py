"""Check of the Tardis extractor on real files: download metadata and quote-convention checks.

For every month cached by ``surfbench data tardis`` prints what was streamed and, per
underlying, the data-quality diagnostics of DESIGN 5.3: median distance between our bid IVs
and the provider's, and the put-call-parity forward gap. No quote, level or fit leaves the
cache, so the output can be committed.

Run: ``uv run python scripts/extractor_check.py --out results/extractor-check``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from surfbench.quotes import FilterConfig, apply_filter, data_quality, read_snapshot

META_KEYS = ("month", "status", "bytes_streamed", "rows_read", "n_t0", "n_t1", "x_dataset_size")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("cache/tardis"))
    ap.add_argument("--filter", type=Path, default=Path("configs/filter_v1.toml"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    filt = FilterConfig.load(args.filter)
    rows = []
    for meta_p in sorted(args.cache.glob("*/meta.json")):
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        row = {key: meta.get(key) for key in META_KEYS}
        row["started_at"] = meta.get("started_at")
        row["sha256"] = meta.get("sha256")
        if meta.get("status") == "ok":
            snap = read_snapshot(meta_p.parent / "snap_t0.csv.gz")
            row["snapshot_bytes"] = sum(p.stat().st_size for p in meta_p.parent.glob("snap_*"))
            row["snapshot_bytes"] = sum(p.stat().st_size for p in meta_p.parent.glob("snap_*"))
            row["data_quality"] = {
                und: data_quality(snap, apply_filter(snap, und, filt)) for und in filt.underlyings
            }
        rows.append(row)
    text = json.dumps({"filter_sha256": filt.sha256, "months": rows}, indent=2)
    print(text)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
