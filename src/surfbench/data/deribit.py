"""Live snapshot adapter for the Deribit public API (development and filter freezing only).

Specification: ``docs/DESIGN.md`` section 5.1. One call per underlying to
``public/get_book_summary_by_currency`` (``kind=option``). Results computed from live
snapshots are never committed (DESIGN section 3).

The summary endpoint returns no per-instrument quote time; ``creation_timestamp`` is the
time the summary was produced, so the quote-age step of the filter is not informative on
live snapshots.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from surfbench import __version__
from surfbench.checkpoint import atomic_write_json
from surfbench.quotes import Snapshot, write_snapshot

__all__ = [
    "SUMMARY_URL",
    "fetch_summary",
    "parse_instrument",
    "save_live_pair",
    "snapshot_from_summaries",
]

SUMMARY_URL = (
    "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
    "?currency={currency}&kind=option"
)
_MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    )
}
_NAME = re.compile(r"^(BTC|ETH)-(\d{1,2})([A-Z]{3})(\d{2})-(\d+(?:d\d+)?)-([CP])$")
USER_AGENT = f"surfbench/{__version__} (+research; public data)"


def parse_instrument(name: str) -> tuple[str, int, float, bool] | None:
    """``(underlying, expiration_us, strike, is_call)``; ``None`` for other instruments.

    Deribit options expire at 08:00 UTC on the date in the name; decimal strikes use ``d``.
    """
    m = _NAME.match(name)
    if m is None:
        return None
    und, day, mon, yy, strike, cp = m.groups()
    if mon not in _MONTHS:
        return None
    exp = datetime(2000 + int(yy), _MONTHS[mon], int(day), 8, 0, tzinfo=UTC)
    return und, int(exp.timestamp()) * 1_000_000, float(strike.replace("d", ".")), cp == "C"


def snapshot_from_summaries(time_us: int, summaries: Iterable[Mapping[str, Any]]) -> Snapshot:
    """Build a snapshot from ``get_book_summary_by_currency`` result rows."""
    records = []
    for row in summaries:
        parsed = parse_instrument(str(row.get("instrument_name", "")))
        if parsed is None:
            continue
        _, exp_us, strike, is_call = parsed
        ts = row.get("creation_timestamp")
        records.append(
            {
                "symbol": row["instrument_name"],
                "timestamp_us": int(ts) * 1000 if ts is not None else None,
                "expiration_us": exp_us,
                "strike": strike,
                "option_type": "call" if is_call else "put",
                "bid_price": row.get("bid_price"),
                "ask_price": row.get("ask_price"),
                "mark_price": row.get("mark_price"),
                "mark_iv": row.get("mark_iv"),
                "underlying_price": row.get("underlying_price"),
                "bid_iv": None,
                "ask_iv": None,
            }
        )
    return Snapshot.from_records(time_us, records)


Opener = Callable[..., Any]


def fetch_summary(currency: str, *, opener: Opener = urllib.request.urlopen, timeout: float = 30.0):
    """Return the ``result`` list of the summary endpoint for ``currency``."""
    if currency not in ("BTC", "ETH"):
        raise ValueError("currency must be BTC or ETH")
    req = urllib.request.Request(
        SUMMARY_URL.format(currency=currency), headers={"User-Agent": USER_AGENT}
    )
    with opener(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    if "result" not in body:
        raise RuntimeError(f"unexpected Deribit response: {str(body)[:200]}")
    return body["result"]


def _take(currencies: tuple[str, ...], opener: Opener, timeout: float) -> Snapshot:
    t_us = int(time.time() * 1e6)
    rows: list[Mapping[str, Any]] = []
    for c in currencies:
        rows.extend(fetch_summary(c, opener=opener, timeout=timeout))
    return snapshot_from_summaries(t_us, rows)


def save_live_pair(
    out_dir: Path,
    *,
    horizon_min: float = 0.0,
    currencies: tuple[str, ...] = ("BTC", "ETH"),
    opener: Opener = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
    timeout: float = 30.0,
) -> Path:
    """Save ``snap_t0`` (and ``snap_t1`` after ``horizon_min`` min) under ``out_dir/<stamp>``."""
    snap0 = _take(currencies, opener, timeout)
    stamp = datetime.fromtimestamp(snap0.time_us / 1e6, tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    d = Path(out_dir) / stamp
    meta: dict[str, Any] = {
        "source": "deribit-live",
        "package_version": __version__,
        "t0_us": snap0.time_us,
        "n_t0": len(snap0),
        "sha256": {"snap_t0.csv.gz": write_snapshot(d / "snap_t0.csv.gz", snap0)},
        "status": "ok",
    }
    if horizon_min > 0:
        sleep(horizon_min * 60.0)
        snap1 = _take(currencies, opener, timeout)
        meta["t1_us"] = snap1.time_us
        meta["n_t1"] = len(snap1)
        meta["sha256"]["snap_t1.csv.gz"] = write_snapshot(d / "snap_t1.csv.gz", snap1)
    atomic_write_json(d / "meta.json", meta)
    return d
