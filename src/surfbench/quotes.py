"""Normalised quote snapshots, the pre-registered quote filter, forwards and slices.

Specification: ``docs/DESIGN.md`` section 5.2 and 5.3.

A snapshot is the last known state of every option instrument at one time ``t``. It is
stored as a canonical gzip CSV (sorted by symbol, fixed columns, gzip ``mtime = 0``) so that
its SHA-256 identifies the content. Prices are in coin units (Deribit inverse options).
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import math
import re
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from surfbench.black import delta, implied_vol
from surfbench.checkpoint import atomic_write_bytes
from surfbench.models.base import FitInput

__all__ = [
    "SNAP_COLUMNS",
    "FilterConfig",
    "FilteredSnapshot",
    "Slice",
    "Snapshot",
    "apply_filter",
    "data_quality",
    "file_sha256",
    "merge_snapshots",
    "read_snapshot",
    "snapshot_bytes",
    "to_fit_input",
    "write_snapshot",
    "YEAR_SECONDS",
]

YEAR_SECONDS = 365.0 * 86400.0
SNAP_COLUMNS = (
    "symbol",
    "timestamp_us",
    "expiration_us",
    "strike",
    "option_type",
    "bid_price",
    "ask_price",
    "mark_price",
    "mark_iv",
    "underlying_price",
    "bid_iv",
    "ask_iv",
)
_NUMERIC = SNAP_COLUMNS[1:3] + SNAP_COLUMNS[3:4] + SNAP_COLUMNS[5:]
Float = NDArray[np.float64]


@dataclass(frozen=True)
class Snapshot:
    """Instrument states at ``time_us`` (microseconds since the Unix epoch, UTC)."""

    time_us: int
    symbol: NDArray[np.str_]
    timestamp_us: Float
    expiration_us: Float
    strike: Float
    is_call: NDArray[np.bool_]
    bid_price: Float
    ask_price: Float
    mark_price: Float
    mark_iv: Float  # percent, as published (diagnostics only)
    underlying_price: Float
    bid_iv: Float  # percent, provider value if any (diagnostics only)
    ask_iv: Float

    def __len__(self) -> int:
        return int(self.symbol.size)

    @classmethod
    def from_records(cls, time_us: int, records: Iterable[Mapping[str, Any]]) -> Snapshot:
        recs = sorted(records, key=lambda r: str(r["symbol"]))
        seen: set[str] = set()
        for r in recs:
            if r["symbol"] in seen:
                raise ValueError(f"duplicate symbol {r['symbol']!r}")
            seen.add(str(r["symbol"]))

        def num(key: str) -> Float:
            out = np.empty(len(recs))
            for i, r in enumerate(recs):
                v = r.get(key)
                out[i] = float(v) if v not in (None, "") else math.nan
            return out

        types = [str(r["option_type"]).lower() for r in recs]
        if any(t not in ("call", "put") for t in types):
            raise ValueError("option_type must be 'call' or 'put'")
        return cls(
            time_us=int(time_us),
            symbol=np.array([str(r["symbol"]) for r in recs], dtype=str),
            timestamp_us=num("timestamp_us"),
            expiration_us=num("expiration_us"),
            strike=num("strike"),
            is_call=np.array([t == "call" for t in types], dtype=bool),
            bid_price=num("bid_price"),
            ask_price=num("ask_price"),
            mark_price=num("mark_price"),
            mark_iv=num("mark_iv"),
            underlying_price=num("underlying_price"),
            bid_iv=num("bid_iv"),
            ask_iv=num("ask_iv"),
        )


def _fmt(v: float) -> str:
    if not math.isfinite(v):
        return ""
    if v == int(v) and abs(v) < 1e18:
        return str(int(v))
    return repr(float(v))


def snapshot_bytes(snap: Snapshot) -> bytes:
    """Canonical gzip CSV bytes (deterministic: sorted rows, ``mtime = 0``)."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["# time_us", str(snap.time_us)])
    w.writerow(SNAP_COLUMNS)
    order = np.argsort(snap.symbol, kind="stable")
    for i in order:
        w.writerow(
            [
                snap.symbol[i],
                _fmt(snap.timestamp_us[i]),
                _fmt(snap.expiration_us[i]),
                _fmt(snap.strike[i]),
                "call" if snap.is_call[i] else "put",
                _fmt(snap.bid_price[i]),
                _fmt(snap.ask_price[i]),
                _fmt(snap.mark_price[i]),
                _fmt(snap.mark_iv[i]),
                _fmt(snap.underlying_price[i]),
                _fmt(snap.bid_iv[i]),
                _fmt(snap.ask_iv[i]),
            ]
        )
    raw = buf.getvalue().encode("utf-8")
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", mtime=0, compresslevel=6, filename="") as gz:
        gz.write(raw)
    return out.getvalue()


def write_snapshot(path: Path, snap: Snapshot) -> str:
    """Write the canonical file atomically and return its SHA-256."""
    data = snapshot_bytes(snap)
    atomic_write_bytes(Path(path), data)
    return hashlib.sha256(data).hexdigest()


def read_snapshot(path: Path) -> Snapshot:
    with gzip.open(Path(path), "rt", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        first = next(reader)
        if not first or first[0] != "# time_us":
            raise ValueError("not a surfbench snapshot file")
        time_us = int(first[1])
        header = next(reader)
        if tuple(header) != SNAP_COLUMNS:
            raise ValueError("unexpected snapshot columns")
        records = [dict(zip(SNAP_COLUMNS, row, strict=True)) for row in reader]
    return Snapshot.from_records(time_us, records)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- filter
@dataclass(frozen=True)
class FilterConfig:
    version: int
    underlyings: tuple[str, ...]
    symbol_regex: str
    max_quote_age_s: float
    require_two_sided: bool
    min_bid_coin: float
    otm_only: bool
    min_abs_delta: float
    max_spread_iv: float
    min_iv: float
    max_iv: float
    min_days: float
    max_days: float
    max_forward_dispersion: float
    min_quotes: int
    min_quotes_each_side: int
    sha256: str = ""

    @classmethod
    def load(cls, path: Path) -> FilterConfig:
        raw = Path(path).read_bytes()
        cfg = tomllib.loads(raw.decode("utf-8"))
        q, e, ins = cfg["quote"], cfg["expiry"], cfg["instruments"]
        if e.get("forward_source", "underlying_price") != "underlying_price":
            raise ValueError("only forward_source = 'underlying_price' is supported")
        return cls(
            version=int(cfg["version"]),
            underlyings=tuple(ins["underlyings"]),
            symbol_regex=str(ins["symbol_regex"]),
            max_quote_age_s=float(q["max_quote_age_s"]),
            require_two_sided=bool(q["require_two_sided"]),
            min_bid_coin=float(q["min_bid_coin"]),
            otm_only=bool(q["otm_only"]),
            min_abs_delta=float(q["min_abs_delta"]),
            max_spread_iv=float(q["max_spread_iv"]),
            min_iv=float(q["min_iv"]),
            max_iv=float(q["max_iv"]),
            min_days=float(e["min_days"]),
            max_days=float(e["max_days"]),
            max_forward_dispersion=float(e["max_forward_dispersion"]),
            min_quotes=int(e["min_quotes"]),
            min_quotes_each_side=int(e["min_quotes_each_side"]),
            sha256=hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
        )


@dataclass(frozen=True)
class Slice:
    """Filtered quotes of one expiry, sorted by ``k``."""

    expiration_us: int
    T: float
    F: float
    symbol: NDArray[np.str_]
    strike: Float
    is_call: NDArray[np.bool_]
    k: Float
    bid_iv: Float
    ask_iv: Float
    mid_iv: Float
    provider_bid_iv: Float  # decimal; nan if not provided

    @property
    def half_spread(self) -> Float:
        return 0.5 * (self.ask_iv - self.bid_iv)

    @property
    def n(self) -> int:
        return int(self.k.size)


@dataclass
class FilteredSnapshot:
    underlying: str
    time_us: int
    slices: list[Slice]
    counts: dict[str, int] = field(default_factory=dict)
    dropped_expiries: dict[str, int] = field(default_factory=dict)

    @property
    def n_quotes(self) -> int:
        return sum(s.n for s in self.slices)


def apply_filter(snap: Snapshot, underlying: str, cfg: FilterConfig) -> FilteredSnapshot:
    """Apply filter steps 1-7 of DESIGN 5.2; record the instrument count after each step."""
    counts: dict[str, int] = {}
    dropped = {"forward": 0, "expiry_window": 0, "slice": 0}
    rx = re.compile(cfg.symbol_regex)
    prefix = f"{underlying}-"
    m = np.array([bool(rx.match(s)) and s.startswith(prefix) for s in snap.symbol], dtype=bool)
    counts["input"] = int(np.count_nonzero([s.startswith(prefix) for s in snap.symbol]))
    counts["symbol"] = int(m.sum())
    age_s = (snap.time_us - snap.timestamp_us) / 1e6
    m &= np.isfinite(age_s) & (age_s <= cfg.max_quote_age_s)
    counts["age"] = int(m.sum())
    bid, ask = snap.bid_price, snap.ask_price
    if cfg.require_two_sided:
        with np.errstate(invalid="ignore"):
            m &= np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask > bid)
    with np.errstate(invalid="ignore"):
        m &= bid >= cfg.min_bid_coin
    counts["two_sided"] = int(m.sum())

    slices: list[Slice] = []
    n_fwd = n_win = n_otm = n_iv = 0
    for exp in np.unique(snap.expiration_us[m]):
        idx = np.flatnonzero(m & (snap.expiration_us == exp))
        up = snap.underlying_price[idx]
        up = up[np.isfinite(up) & (up > 0)]
        if up.size == 0:
            dropped["forward"] += 1
            continue
        F = float(np.median(up))
        if (up.max() - up.min()) / F > cfg.max_forward_dispersion:
            dropped["forward"] += 1
            continue
        n_fwd += idx.size
        T = (float(exp) - snap.time_us) / 1e6 / YEAR_SECONDS
        days = T * 365.0
        if not (cfg.min_days <= days <= cfg.max_days):
            dropped["expiry_window"] += 1
            continue
        n_win += idx.size
        K = snap.strike[idx]
        k = np.log(K / F)
        is_call = snap.is_call[idx]
        if cfg.otm_only:
            keep = np.where(is_call, K >= F, K < F)
            idx, K, k, is_call = idx[keep], K[keep], k[keep], is_call[keep]
        n_otm += idx.size
        if idx.size == 0:
            dropped["slice"] += 1
            continue
        b, a = snap.bid_price[idx], snap.ask_price[idx]
        bid_iv = implied_vol(b, k, T, is_call)
        ask_iv = implied_vol(a, k, T, is_call)
        mid_iv = implied_vol(0.5 * (a + b), k, T, is_call)
        with np.errstate(invalid="ignore"):
            ok = (
                np.isfinite(bid_iv)
                & np.isfinite(ask_iv)
                & np.isfinite(mid_iv)
                & (bid_iv >= cfg.min_iv)
                & (ask_iv <= cfg.max_iv)
                & (mid_iv >= cfg.min_iv)
                & (mid_iv <= cfg.max_iv)
                & (ask_iv - bid_iv <= cfg.max_spread_iv)
            )
            d = delta(k, mid_iv * math.sqrt(T), is_call)
            ok &= np.abs(d) >= cfg.min_abs_delta
        n_iv += int(ok.sum())
        if not ok.any():
            dropped["slice"] += 1
            continue
        order = np.argsort(k[ok], kind="stable")
        sel = idx[ok][order]
        kk = k[ok][order]
        n_neg = int(np.count_nonzero(kk < 0))
        n_pos = kk.size - n_neg
        if (
            kk.size < cfg.min_quotes
            or n_neg < cfg.min_quotes_each_side
            or n_pos < cfg.min_quotes_each_side
            or np.any(np.diff(kk) <= 0)
        ):
            dropped["slice"] += 1
            continue
        slices.append(
            Slice(
                expiration_us=int(exp),
                T=T,
                F=F,
                symbol=snap.symbol[sel],
                strike=snap.strike[sel],
                is_call=snap.is_call[sel],
                k=kk,
                bid_iv=bid_iv[ok][order],
                ask_iv=ask_iv[ok][order],
                mid_iv=mid_iv[ok][order],
                provider_bid_iv=snap.bid_iv[sel] / 100.0,
            )
        )
    counts["forward"] = n_fwd
    counts["expiry_window"] = n_win
    counts["otm"] = n_otm
    counts["iv"] = n_iv
    counts["slice"] = sum(s.n for s in slices)
    slices.sort(key=lambda s: s.T)
    return FilteredSnapshot(underlying, snap.time_us, slices, counts, dropped)


def to_fit_input(
    slices: list[Slice],
    *,
    seed: int = 0,
    floor: float = 0.005,
    masks: list[NDArray[np.bool_]] | None = None,
) -> FitInput:
    if masks is None:
        masks = [np.ones(s.n, dtype=bool) for s in slices]
    return FitInput(
        expiries=tuple(s.T for s in slices),
        k=tuple(s.k[mk] for s, mk in zip(slices, masks, strict=True)),
        iv_mid=tuple(s.mid_iv[mk] for s, mk in zip(slices, masks, strict=True)),
        half_spread=tuple(s.half_spread[mk] for s, mk in zip(slices, masks, strict=True)),
        seed=seed,
        floor=floor,
    )


def data_quality(snap: Snapshot, filt: FilteredSnapshot) -> dict[str, float]:
    """Diagnostics of DESIGN 5.3 (reported, not filtered)."""
    out: dict[str, float] = {}
    diffs = []
    for s in filt.slices:
        ok = np.isfinite(s.provider_bid_iv)
        diffs.append(np.abs(s.bid_iv[ok] - s.provider_bid_iv[ok]))
    allx = np.concatenate(diffs) if diffs else np.array([])
    out["median_abs_bid_iv_diff_volpt"] = float(np.median(allx) * 100) if allx.size else math.nan
    # Put-call-parity forward from mid prices near the money: c - p = 1 - K/F (coin units).
    gaps = []
    prefix = f"{filt.underlying}-"
    for s in filt.slices:
        sel = np.flatnonzero(
            (snap.expiration_us == s.expiration_us) & np.char.startswith(snap.symbol, prefix)
        )
        by_strike: dict[float, dict[bool, float]] = {}
        for i in sel:
            b, a = snap.bid_price[i], snap.ask_price[i]
            if not (np.isfinite(b) and np.isfinite(a) and 0 < b < a):
                continue
            by_strike.setdefault(float(snap.strike[i]), {})[bool(snap.is_call[i])] = 0.5 * (a + b)
        for K, d in by_strike.items():
            if True in d and False in d and abs(math.log(K / s.F)) < 0.1:
                denom = 1.0 - (d[True] - d[False])
                if denom > 0:
                    gaps.append(abs(K / denom / s.F - 1.0))
    out["median_parity_forward_gap"] = float(np.median(gaps)) if gaps else math.nan
    return out


def merge_snapshots(snaps: list[Snapshot]) -> Snapshot:
    """Concatenate snapshots taken at the same time (symbols must not overlap)."""
    if not snaps:
        raise ValueError("nothing to merge")
    t = snaps[0].time_us
    if any(s.time_us != t for s in snaps):
        raise ValueError("snapshots must share time_us")
    fields = [f for f in Snapshot.__dataclass_fields__ if f != "time_us"]
    cat = {f: np.concatenate([getattr(s, f) for s in snaps]) for f in fields}
    if np.unique(cat["symbol"]).size != cat["symbol"].size:
        raise ValueError("duplicate symbols across snapshots")
    order = np.argsort(cat["symbol"], kind="stable")
    return Snapshot(time_us=t, **{f: v[order] for f, v in cat.items()})
