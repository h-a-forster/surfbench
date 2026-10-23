"""Snapshots, the quote filter and data-quality diagnostics (DESIGN 5.2, 5.3)."""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from surfbench import synthetic
from surfbench.quotes import (
    YEAR_SECONDS,
    FilterConfig,
    Snapshot,
    apply_filter,
    data_quality,
    read_snapshot,
    snapshot_bytes,
    to_fit_input,
    write_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]
T0 = 1_790_000_000 * 1_000_000  # an arbitrary UTC time in microseconds
DAY_US = 86_400 * 1_000_000


@pytest.fixture(scope="module")
def cfg() -> FilterConfig:
    return FilterConfig.load(ROOT / "configs" / "filter_v1.toml")


def _snapshot(seed: int = 0, days=(7, 30, 91)) -> tuple[Snapshot, object]:
    exps = tuple(T0 + d * DAY_US for d in days)
    T = tuple((e - T0) / 1e6 / YEAR_SECONDS for e in exps)
    surf = synthetic.random_surface("essvi", np.random.default_rng(seed), T)
    snap = synthetic.make_quote_snapshot(
        surf, underlying="BTC", forward=60000.0, time_us=T0, expirations_us=exps, seed=seed
    )
    return snap, surf


def test_filter_config_loads(cfg: FilterConfig) -> None:
    assert cfg.version == 1 and cfg.underlyings == ("BTC", "ETH")
    assert cfg.min_quotes == 8 and len(cfg.sha256) == 64


def test_canonical_bytes_are_deterministic(tmp_path: Path) -> None:
    snap, _ = _snapshot()
    b1 = snapshot_bytes(snap)
    rev = Snapshot.from_records(
        snap.time_us,
        reversed(
            [
                {
                    "symbol": snap.symbol[i],
                    "timestamp_us": snap.timestamp_us[i],
                    "expiration_us": snap.expiration_us[i],
                    "strike": snap.strike[i],
                    "option_type": "call" if snap.is_call[i] else "put",
                    "bid_price": snap.bid_price[i],
                    "ask_price": snap.ask_price[i],
                    "mark_price": snap.mark_price[i],
                    "mark_iv": snap.mark_iv[i],
                    "underlying_price": snap.underlying_price[i],
                    "bid_iv": snap.bid_iv[i],
                    "ask_iv": snap.ask_iv[i],
                }
                for i in range(len(snap))
            ]
        ),
    )
    assert snapshot_bytes(rev) == b1
    sha = write_snapshot(tmp_path / "s.csv.gz", snap)
    assert sha == hashlib.sha256(b1).hexdigest()
    back = read_snapshot(tmp_path / "s.csv.gz")
    assert back.time_us == snap.time_us and len(back) == len(snap)
    assert np.array_equal(back.bid_price, snap.bid_price)
    assert snapshot_bytes(back) == b1


def test_missing_values_round_trip(tmp_path: Path) -> None:
    rec = {
        "symbol": "BTC-1JAN30-50000-C",
        "timestamp_us": T0,
        "expiration_us": T0 + DAY_US,
        "strike": 50000,
        "option_type": "call",
        "bid_price": None,
        "ask_price": "0.01",
        "mark_price": 0.009,
        "mark_iv": 55.5,
        "underlying_price": 50000,
        "bid_iv": "",
        "ask_iv": None,
    }
    snap = Snapshot.from_records(T0, [rec])
    write_snapshot(tmp_path / "a.csv.gz", snap)
    back = read_snapshot(tmp_path / "a.csv.gz")
    assert math.isnan(back.bid_price[0]) and back.ask_price[0] == 0.01
    with pytest.raises(ValueError):
        Snapshot.from_records(T0, [rec, rec])
    with pytest.raises(ValueError):
        Snapshot.from_records(T0, [dict(rec, option_type="future")])


def test_filter_recovers_otm_quotes_and_ivs(cfg: FilterConfig) -> None:
    snap, surf = _snapshot(seed=1)
    filt = apply_filter(snap, "BTC", cfg)
    assert len(filt.slices) == 3
    c = filt.counts
    assert c["input"] == c["symbol"] == c["age"] == len(snap)
    assert c["otm"] <= c["expiry_window"] <= c["forward"] <= c["two_sided"] <= c["age"]
    assert c["slice"] == filt.n_quotes
    for j, s in enumerate(filt.slices):
        assert s.F == 60000.0
        assert np.all(np.where(s.is_call, s.strike >= s.F, s.strike < s.F))
        assert np.all(np.diff(s.k) > 0)
        # Bid/ask IVs recompute the generating IVs exactly (prices are exact Black-76).
        assert np.allclose(s.ask_iv - s.bid_iv, 0.02, atol=1e-9)
        assert np.allclose(s.bid_iv, s.provider_bid_iv, atol=1e-9)
        assert abs(s.T - surf.expiries[j]) < 1e-12
    assert filt.underlying == "BTC"
    assert apply_filter(snap, "ETH", cfg).n_quotes == 0
    dq = data_quality(snap, filt)
    assert dq["median_abs_bid_iv_diff_volpt"] < 1e-6
    assert dq["median_parity_forward_gap"] < 1e-9


def test_filter_steps_drop_what_they_should(cfg: FilterConfig) -> None:
    snap, _ = _snapshot(seed=2, days=(1, 7, 600))
    filt = apply_filter(snap, "BTC", cfg)
    assert [round(s.T * 365) for s in filt.slices] == [7]
    assert filt.dropped_expiries["expiry_window"] == 2
    # Stale quotes.
    stale = replace(snap, timestamp_us=snap.timestamp_us - 400 * 1_000_000)
    assert apply_filter(stale, "BTC", cfg).counts["age"] == 0
    # One-sided and sub-two-tick bids.
    one_sided = replace(snap, bid_price=np.where(snap.is_call, np.nan, snap.bid_price))
    f1 = apply_filter(one_sided, "BTC", cfg)
    assert f1.counts["two_sided"] == int(np.count_nonzero(~snap.is_call & (snap.bid_price >= 2e-4)))
    # Forward dispersion above 0.1% drops the expiry.
    up = snap.underlying_price.copy()
    up[0] *= 1.01
    f2 = apply_filter(replace(snap, underlying_price=up), "BTC", cfg)
    assert f2.dropped_expiries["forward"] == 1


def test_filter_spread_and_min_quotes(cfg: FilterConfig) -> None:
    snap, _ = _snapshot(seed=3)
    wide = replace(snap, ask_price=snap.ask_price * 3.0)
    f = apply_filter(wide, "BTC", cfg)
    assert f.n_quotes < apply_filter(snap, "BTC", cfg).n_quotes
    strict = replace(cfg, min_quotes=100)
    f2 = apply_filter(snap, "BTC", strict)
    assert f2.slices == [] and f2.dropped_expiries["slice"] == 3


def test_to_fit_input_masks(cfg: FilterConfig) -> None:
    snap, _ = _snapshot(seed=4)
    filt = apply_filter(snap, "BTC", cfg)
    data = to_fit_input(filt.slices, seed=5)
    assert data.n_slices == 3 and data.seed == 5
    masks = [np.arange(s.n) % 2 == 0 for s in filt.slices]
    sub = to_fit_input(filt.slices, masks=masks)
    assert all(sub.k[j].size == masks[j].sum() for j in range(3))
