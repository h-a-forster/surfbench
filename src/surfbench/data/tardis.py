"""Streaming extractor for Tardis.dev free first-of-month Deribit options_chain files.

Specification: ``docs/DESIGN.md`` section 5.1.

For one month the extractor streams ``.../options_chain/YYYY/MM/01/OPTIONS.csv.gz`` with a
plain ``GET`` (``HEAD`` and ranged requests are refused upstream), decompresses on the fly,
keeps the latest row of every inverse BTC/ETH option, freezes a copy of that state when the
first row with ``local_timestamp > t0`` arrives and again after ``t1``, then closes the
connection. Only the two snapshots and a metadata file are written.

Quota handling: ``429`` writes ``_quota.json`` with ``not_before`` and stops the session;
sessions refuse to start before ``not_before``. ``404`` marks the month ``unavailable``.
"""

from __future__ import annotations

import json
import random
import re
import time
import urllib.error
import urllib.request
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from surfbench import __version__
from surfbench.checkpoint import atomic_write_json, scrub_paths
from surfbench.quotes import Snapshot, write_snapshot

__all__ = [
    "TARDIS_COLUMNS",
    "InvalidData",
    "QuotaLocked",
    "TardisConfig",
    "extract_month",
    "month_status",
    "run_session",
    "stream_snapshots",
]

URL = "https://datasets.tardis.dev/v1/deribit/options_chain/{y}/{m}/01/OPTIONS.csv.gz"
TARDIS_COLUMNS = (
    "exchange",
    "symbol",
    "timestamp",
    "local_timestamp",
    "type",
    "strike_price",
    "expiration",
    "open_interest",
    "last_price",
    "bid_price",
    "bid_amount",
    "bid_iv",
    "ask_price",
    "ask_amount",
    "ask_iv",
    "mark_price",
    "mark_iv",
    "underlying_index",
    "underlying_price",
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
)
_IDX = {c: i for i, c in enumerate(TARDIS_COLUMNS)}
_NUMERIC_COLS = (
    "timestamp",
    "local_timestamp",
    "strike_price",
    "expiration",
    "bid_price",
    "bid_iv",
    "ask_price",
    "ask_iv",
    "mark_price",
    "mark_iv",
    "underlying_price",
)
USER_AGENT = f"surfbench/{__version__} (+research; first-of-month free sample)"
CHUNK = 1 << 20
NETWORK_RETRIES = 3


class QuotaLocked(RuntimeError):
    """The download quota is locked until ``not_before`` (UTC seconds)."""

    def __init__(self, not_before: float) -> None:
        super().__init__(
            "Tardis quota locked until "
            + datetime.fromtimestamp(not_before, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
        )
        self.not_before = not_before


class InvalidData(ValueError):
    """The stream violated the expected layout (columns, numbers, ordering)."""


@dataclass(frozen=True)
class TardisConfig:
    cache_dir: Path
    months: tuple[str, ...]
    t0: str = "00:01:00"
    horizon_min: float = 15.0
    fetch_order_seed: int = 7
    request_gap_s: float = 30.0
    timeout_s: float = 60.0
    max_requests_per_session: int = 12
    cache_cap_gb: float = 2.0
    symbol_regex: str = r"^(BTC|ETH)-\d{1,2}[A-Z]{3}\d{2}-\d+(d\d+)?-[CP]$"

    @classmethod
    def from_study(cls, study: dict[str, Any], symbol_regex: str | None = None) -> TardisConfig:
        d = study["data"]
        kw: dict[str, Any] = {
            "cache_dir": Path(d["cache_dir"]),
            "months": tuple(d["months"]),
            "t0": str(d.get("t0", "00:01:00")),
            "horizon_min": float(d["horizon_min"]),
            "fetch_order_seed": int(d.get("fetch_order_seed", 7)),
            "request_gap_s": float(d.get("request_gap_s", 30)),
            "timeout_s": float(d.get("timeout_s", 60)),
            "max_requests_per_session": int(d.get("max_requests_per_session", 12)),
            "cache_cap_gb": float(d.get("cache_cap_gb", 2.0)),
        }
        if symbol_regex:
            kw["symbol_regex"] = symbol_regex
        return cls(**kw)

    def times_us(self, month: str) -> tuple[int, int]:
        y, m = (int(x) for x in month.split("-"))
        hh, mm, ss = (int(x) for x in self.t0.split(":"))
        t0 = datetime(y, m, 1, hh, mm, ss, tzinfo=UTC)
        t0_us = int(t0.timestamp()) * 1_000_000
        return t0_us, t0_us + int(round(self.horizon_min * 60 * 1e6))

    def month_dir(self, month: str) -> Path:
        return self.cache_dir / f"{month}-01"


def _url(month: str) -> str:
    y, m = month.split("-")
    return URL.format(y=y, m=m)


def _record(fields: list[bytes]) -> dict[str, Any]:
    def g(name: str) -> str:
        return fields[_IDX[name]].decode("ascii")

    t = g("type")
    return {
        "symbol": g("symbol"),
        "timestamp_us": g("timestamp"),
        "expiration_us": g("expiration"),
        "strike": g("strike_price"),
        "option_type": "call" if t.startswith("c") else "put",
        "bid_price": g("bid_price"),
        "ask_price": g("ask_price"),
        "mark_price": g("mark_price"),
        "mark_iv": g("mark_iv"),
        "underlying_price": g("underlying_price"),
        "bid_iv": g("bid_iv"),
        "ask_iv": g("ask_iv"),
    }


@dataclass
class StreamResult:
    t0: Snapshot | None
    t1: Snapshot | None
    bytes_read: int
    rows_read: int
    rows_kept: int


def stream_snapshots(
    read: Callable[[int], bytes], t0_us: int, t1_us: int, symbol_regex: str
) -> StreamResult:
    """Parse a gzip CSV stream until ``t1`` is frozen; returns the snapshots and counters."""
    rx = re.compile(symbol_regex.encode("ascii"))
    dec = zlib.decompressobj(16 + zlib.MAX_WBITS)
    state: dict[bytes, list[bytes]] = {}
    snap0: Snapshot | None = None
    snap1: Snapshot | None = None
    buf = b""
    header_seen = False
    last_ts = -1
    bytes_read = rows = kept = 0
    n_cols = len(TARDIS_COLUMNS)
    num_idx = [_IDX[c] for c in _NUMERIC_COLS]

    def freeze(t_us: int) -> Snapshot:
        return Snapshot.from_records(t_us, (_record(f) for f in state.values()))

    while snap1 is None:
        chunk = read(CHUNK)
        if not chunk:
            break
        bytes_read += len(chunk)
        buf += dec.decompress(chunk)
        lines = buf.split(b"\n")
        buf = lines.pop()
        for line in lines:
            if not line:
                continue
            if not header_seen:
                cols = tuple(line.rstrip(b"\r").decode("ascii").split(","))
                if cols != TARDIS_COLUMNS:
                    raise InvalidData("unexpected Tardis columns")
                header_seen = True
                continue
            fields = line.rstrip(b"\r").split(b",")
            rows += 1
            try:
                ts = int(fields[3])
            except (ValueError, IndexError) as exc:
                raise InvalidData(f"row {rows}: bad local_timestamp") from exc
            if ts < last_ts:
                raise InvalidData(f"row {rows}: local_timestamp decreased")
            last_ts = ts
            if snap0 is None and ts > t0_us:
                snap0 = freeze(t0_us)
            if ts > t1_us:
                snap1 = freeze(t1_us)
                break
            if len(fields) < 2 or not rx.match(fields[1]):
                continue
            if len(fields) != n_cols:
                raise InvalidData(f"row {rows}: expected {n_cols} columns, got {len(fields)}")
            for i in num_idx:
                v = fields[i]
                if v:
                    try:
                        float(v)
                    except ValueError as exc:
                        raise InvalidData(f"row {rows}: non-numeric {TARDIS_COLUMNS[i]}") from exc
            state[fields[1]] = fields
            kept += 1
    return StreamResult(snap0, snap1, bytes_read, rows, kept)


# ------------------------------------------------------------------ cache and quota
def _quota_path(cfg: TardisConfig) -> Path:
    return cfg.cache_dir / "_quota.json"


def quota_not_before(cfg: TardisConfig) -> float:
    p = _quota_path(cfg)
    if not p.exists():
        return 0.0
    try:
        return float(json.loads(p.read_text(encoding="utf-8")).get("not_before", 0.0))
    except (OSError, ValueError):
        return 0.0


def cache_size_bytes(cfg: TardisConfig) -> int:
    if not cfg.cache_dir.exists():
        return 0
    return sum(p.stat().st_size for p in cfg.cache_dir.rglob("*") if p.is_file())


def month_status(cfg: TardisConfig, month: str) -> str | None:
    p = cfg.month_dir(month) / "meta.json"
    if not p.exists():
        return None
    try:
        return str(json.loads(p.read_text(encoding="utf-8")).get("status"))
    except (OSError, ValueError):
        return None


Opener = Callable[..., Any]


def extract_month(
    cfg: TardisConfig,
    month: str,
    *,
    opener: Opener = urllib.request.urlopen,
    now: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Fetch one month. Returns its metadata; raises :class:`QuotaLocked` on 429."""
    nb = quota_not_before(cfg)
    if nb > now():
        raise QuotaLocked(nb)
    cap = cfg.cache_cap_gb * 1e9
    if cache_size_bytes(cfg) > cap:
        raise RuntimeError(f"cache exceeds cache_cap_gb = {cfg.cache_cap_gb}; refusing to write")
    url = _url(month)
    t0_us, t1_us = cfg.times_us(month)
    out = cfg.month_dir(month)
    meta: dict[str, Any] = {
        "month": month,
        "url": url,
        "package_version": __version__,
        "t0_us": t0_us,
        "t1_us": t1_us,
    }
    last_err = ""
    for attempt in range(1, NETWORK_RETRIES + 1):
        started = now()
        meta["started_at"] = datetime.fromtimestamp(started, tz=UTC).isoformat()
        meta["attempt"] = attempt
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with opener(req, timeout=cfg.timeout_s) as resp:
                headers = resp.headers
                meta["http_status"] = int(getattr(resp, "status", 200))
                meta["x_dataset_size"] = headers.get("x-dataset-size")
                meta["x_md5"] = headers.get("x-md5")
                res = stream_snapshots(resp.read, t0_us, t1_us, cfg.symbol_regex)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                meta.update(status="unavailable", http_status=404)
                atomic_write_json(out / "meta.json", meta)
                return meta
            if exc.code == 429:
                retry = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    wait_s = float(retry) if retry is not None else 86400.0
                except ValueError:
                    wait_s = 86400.0
                not_before = now() + wait_s
                atomic_write_json(
                    _quota_path(cfg),
                    {
                        "not_before": not_before,
                        "observed_at": now(),
                        "retry_after_s": wait_s,
                        "month": month,
                    },
                )
                raise QuotaLocked(not_before) from exc
            last_err = f"HTTP {exc.code}"
        except InvalidData as exc:
            meta.update(status="invalid", error=str(exc))
            atomic_write_json(out / "meta.json", meta)
            return meta
        except (OSError, TimeoutError, zlib.error, EOFError) as exc:
            last_err = scrub_paths(f"{type(exc).__name__}: {exc}")
        else:
            meta["bytes_streamed"] = res.bytes_read
            meta["rows_read"] = res.rows_read
            meta["rows_kept"] = res.rows_kept
            meta["elapsed_s"] = round(now() - started, 3)
            if res.t0 is None or res.t1 is None:
                meta.update(status="invalid", error="stream ended before t1")
                atomic_write_json(out / "meta.json", meta)
                return meta
            meta["n_t0"] = len(res.t0)
            meta["n_t1"] = len(res.t1)
            meta["sha256"] = {
                "snap_t0.csv.gz": write_snapshot(out / "snap_t0.csv.gz", res.t0),
                "snap_t1.csv.gz": write_snapshot(out / "snap_t1.csv.gz", res.t1),
            }
            meta["status"] = "ok"
            atomic_write_json(out / "meta.json", meta)
            return meta
        if attempt < NETWORK_RETRIES:
            sleep(60.0)
    meta.update(status="error", error=last_err)
    atomic_write_json(out / "meta.json", meta)
    return meta


def fetch_order(cfg: TardisConfig) -> list[str]:
    months = sorted(cfg.months)
    random.Random(cfg.fetch_order_seed).shuffle(months)
    return months


def run_session(
    cfg: TardisConfig,
    *,
    opener: Opener = urllib.request.urlopen,
    now: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Fetch pending months in seeded random order, at most ``max_requests_per_session``."""
    summary: dict[str, Any] = {"requested": [], "stopped": None}
    nb = quota_not_before(cfg)
    if nb > now():
        summary["stopped"] = str(QuotaLocked(nb))
        log(summary["stopped"])
        return summary
    requests = 0
    for month in fetch_order(cfg):
        if month_status(cfg, month) in ("ok", "unavailable", "invalid"):
            continue
        if requests >= cfg.max_requests_per_session:
            summary["stopped"] = f"session request cap ({cfg.max_requests_per_session}) reached"
            break
        if requests > 0:
            sleep(cfg.request_gap_s)
        requests += 1
        try:
            meta = extract_month(cfg, month, opener=opener, now=now, sleep=sleep)
        except QuotaLocked as exc:
            summary["stopped"] = str(exc)
            log(f"{month}: 429, {exc}")
            break
        summary["requested"].append({"month": month, "status": meta["status"]})
        log(f"{month}: {meta['status']} ({meta.get('bytes_streamed', 0) / 1e6:.1f} MB streamed)")
    if summary["stopped"]:
        log(f"session stopped: {summary['stopped']}")
    return summary

