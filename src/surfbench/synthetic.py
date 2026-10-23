"""Synthetic snapshots from known surfaces with bid-ask noise, for tests and model-recovery checks.

Specification: ``docs/DESIGN.md`` section 7.2.

:func:`random_surface` draws a plausible surface of a given family; :func:`make_snapshot`
samples quotes from any fitted surface on given ``k`` grids, adds Gaussian mid-IV noise and a
bid-ask spread, and returns a :class:`~surfbench.models.base.FitInput` plus bid/ask IVs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from surfbench.models.base import FitInput, FittedSurface, Float
from surfbench.models.baseline import BaselineModel
from surfbench.models.essvi import EssviModel, check_essvi_pair, check_essvi_slice
from surfbench.models.sabr import SabrModel
from surfbench.models.ssvi import SsviModel, phi_power
from surfbench.models.svi import SviModel, svi_constraints_ok

__all__ = [
    "DEFAULT_EXPIRIES",
    "SyntheticSnapshot",
    "k_grids",
    "make_snapshot",
    "random_surface",
]

DEFAULT_EXPIRIES = (7 / 365, 30 / 365, 91 / 365, 182 / 365, 365 / 365)


@dataclass(frozen=True)
class SyntheticSnapshot:
    data: FitInput
    bid_iv: tuple[Float, ...]
    ask_iv: tuple[Float, ...]
    true_iv: tuple[Float, ...]


def k_grids(
    surface: FittedSurface, n: int = 15, z: tuple[float, float] = (-2.5, 2.0)
) -> tuple[Float, ...]:
    """Per-slice ``k`` grids spanning ``z`` ATM standard deviations; ``k = 0`` is included."""
    out = []
    for j in range(len(surface.expiries)):
        s = float(np.sqrt(max(float(surface.w(j, np.array([0.0]))[0]), 1e-8)))
        n_lo = max(2, round(n * -z[0] / (z[1] - z[0])))
        n_hi = max(2, n - n_lo)
        left = np.linspace(z[0] * s, 0.0, n_lo, endpoint=False)
        right = np.linspace(0.0, z[1] * s, n_hi)
        out.append(np.concatenate([left, right]))
    return tuple(out)


def make_snapshot(
    surface: FittedSurface,
    grids: tuple[Float, ...] | None = None,
    *,
    noise: float = 0.0,
    spread: float = 0.02,
    seed: int = 0,
) -> SyntheticSnapshot:
    """Quotes from ``surface`` with mid-IV noise ``N(0, noise)`` and full spread ``spread``.

    ``noise`` and ``spread`` are decimal volatilities (0.005 = 0.5 vol point).
    """
    if noise < 0 or spread < 0:
        raise ValueError("noise and spread must be non-negative")
    rng = np.random.default_rng(seed)
    grids = grids if grids is not None else k_grids(surface)
    ks, mids, hs, bids, asks, trues = [], [], [], [], [], []
    for j, k in enumerate(grids):
        k = np.sort(np.asarray(k, dtype=float))
        true = surface.iv(j, k)
        mid = true + (rng.normal(0.0, noise, k.size) if noise > 0 else 0.0)
        mid = np.maximum(mid, 0.01)
        h = np.full(k.size, spread / 2.0)
        ks.append(k)
        mids.append(mid)
        hs.append(h)
        bids.append(mid - h)
        asks.append(mid + h)
        trues.append(true)
    data = FitInput(
        expiries=tuple(surface.expiries),
        k=tuple(ks),
        iv_mid=tuple(mids),
        half_spread=tuple(hs),
        seed=seed,
    )
    return SyntheticSnapshot(data, tuple(bids), tuple(asks), tuple(trues))


def _atm_vols(rng: np.random.Generator, n: int) -> Float:
    """ATM vols that keep total variance increasing in T."""
    base = rng.uniform(0.4, 0.9)
    slope = rng.uniform(-0.15, 0.1)
    return np.clip(base + slope * np.linspace(0, 1, n), 0.25, 1.2)


def random_surface(
    family: str,
    rng: np.random.Generator,
    expiries: tuple[float, ...] = DEFAULT_EXPIRIES,
) -> FittedSurface:
    """A random, plausible surface of the given family (crypto-like levels and skews)."""
    T = np.asarray(expiries, dtype=float)
    n = T.size
    atm = _atm_vols(rng, n)
    theta = np.maximum.accumulate(atm**2 * T)
    if family == "ssvi":
        rho = rng.uniform(-0.4, 0.3)
        u = rng.uniform(0.3, 0.9)
        eta = 2.0 * u / (1.0 + abs(rho))
        gamma = rng.uniform(0.2, 0.5)
        return SsviModel.surface(expiries, list(theta), rho, eta, gamma)
    if family == "essvi":
        for _ in range(1000):
            eta = rng.uniform(0.6, 1.4)
            gamma = rng.uniform(0.25, 0.5)
            rho0 = rng.uniform(-0.4, 0.2)
            drho = rng.uniform(-0.05, 0.05)
            rhos = np.clip(rho0 + drho * np.arange(n), -0.9, 0.9)
            psis = theta * phi_power(theta, eta, gamma)
            slices = [
                (float(t), float(r), float(p)) for t, r, p in zip(theta, rhos, psis, strict=True)
            ]
            ok = all(check_essvi_slice(*s) for s in slices) and all(
                check_essvi_pair(slices[i - 1], slices[i]) for i in range(1, n)
            )
            if ok:
                return EssviModel.surface(expiries, [list(s) for s in slices])
        raise RuntimeError("could not draw an admissible eSSVI surface")  # pragma: no cover
    if family == "svi":
        params = []
        for j in range(n):
            for _ in range(1000):
                v = float(theta[j])
                s = np.sqrt(v)
                rho = rng.uniform(-0.5, 0.3)
                m = rng.uniform(-0.3, 0.3) * s
                sig = rng.uniform(0.3, 1.5) * s
                b = rng.uniform(0.3, 0.8) * s
                a = v - b * (rho * (-m) + np.sqrt(m * m + sig * sig))
                if (
                    svi_constraints_ok(a, b, rho, m, sig)
                    and a + b * sig * np.sqrt(1 - rho * rho) > 0.3 * v
                ):
                    params.append([float(a), float(b), float(rho), float(m), float(sig)])
                    break
            else:  # pragma: no cover
                raise RuntimeError("could not draw an admissible SVI slice")
        return SviModel.surface(expiries, params)
    if family == "sabr":
        params = [
            [float(atm[j]), float(rng.uniform(-0.4, 0.3)), float(rng.uniform(0.5, 2.5))]
            for j in range(n)
        ]
        return SabrModel.surface(expiries, params)
    if family == "baseline":
        params = [
            [float(atm[j]), float(rng.uniform(-0.2, 0.1)), float(rng.uniform(0.05, 0.4))]
            for j in range(n)
        ]
        return BaselineModel.surface(expiries, params)
    raise ValueError(f"unknown family {family!r}")


def strike_step(F: float) -> float:
    """A 1-2-5 strike step of about ``F / 200`` (300 -> 500 for F = 60000)."""
    raw = F / 200.0
    mag = 10.0 ** np.floor(np.log10(raw))
    for m in (1.0, 2.0, 5.0, 10.0):
        if raw <= m * mag:
            return float(m * mag)
    return float(10.0 * mag)  # pragma: no cover


_MON = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def instrument_name(underlying: str, expiration_us: int, strike: float, is_call: bool) -> str:
    """Deribit-style option name, e.g. ``BTC-27NOV26-73000-C``."""
    from datetime import UTC, datetime

    d = datetime.fromtimestamp(expiration_us / 1e6, tz=UTC)
    ks = str(int(strike)) if float(strike).is_integer() else str(strike).replace(".", "d")
    return (
        f"{underlying}-{d.day}{_MON[d.month - 1]}{d.year % 100:02d}-{ks}-{'C' if is_call else 'P'}"
    )


@dataclass
class ScaledSurface(FittedSurface):
    """``w = factor^2 * w_base`` (every implied volatility multiplied by ``factor``)."""

    base: FittedSurface | None = None
    factor: float = 1.0

    def _w(self, j: int, k: Float) -> Float:
        assert self.base is not None
        return self.factor**2 * self.base.w(j, k)


def make_quote_snapshot(
    surface: FittedSurface,
    *,
    underlying: str,
    forward: float | tuple[float, ...],
    time_us: int,
    expirations_us: tuple[int, ...],
    n_strikes: int = 21,
    z: tuple[float, float] = (-2.5, 2.5),
    noise: float = 0.005,
    spread: float = 0.02,
    spread_wing: float = 0.0,
    strikes: tuple[Float, ...] | None = None,
    seed: int = 0,
):
    """Coin-priced quotes (calls and puts on every strike) from ``surface``.

    Slice ``j`` of ``surface`` must correspond to ``expirations_us[j]``; ``forward`` is one
    value or one per expiry. Mid IVs get ``N(0, noise)`` noise; the full spread is
    ``spread * (1 + spread_wing * |k| / sqrt(theta))`` and bid and ask prices are the prices
    at ``mid -+ spread/2``. Without ``strikes``, strikes are ``n_strikes`` points spanning
    ``z`` ATM standard deviations, rounded to a 1-2-5 step of about ``F / 200``.
    """
    from surfbench.black import call_price, put_price
    from surfbench.quotes import YEAR_SECONDS, Snapshot

    rng = np.random.default_rng(seed)
    fwd = (forward,) * len(expirations_us) if np.isscalar(forward) else tuple(forward)
    if len(fwd) != len(expirations_us):
        raise ValueError("forward must be a scalar or one value per expiry")
    records = []
    for j, exp in enumerate(expirations_us):
        F = float(fwd[j])
        T = (exp - time_us) / 1e6 / YEAR_SECONDS
        if abs(T - surface.expiries[j]) > 1e-9:
            raise ValueError("surface expiries do not match expirations_us")
        s_atm = float(np.sqrt(max(float(surface.w(j, np.array([0.0]))[0]), 1e-8)))
        if strikes is None:
            ks = np.linspace(z[0] * s_atm, z[1] * s_atm, n_strikes)
            step = strike_step(F)
            Ks = np.unique(np.maximum(np.round(F * np.exp(ks) / step), 1.0) * step)
        else:
            Ks = np.unique(np.asarray(strikes[j], dtype=float))
        k = np.log(Ks / F)
        mid = surface.iv(j, k) + rng.normal(0.0, noise, k.size)
        mid = np.maximum(mid, 0.05)
        half = 0.5 * spread * (1.0 + spread_wing * np.abs(k) / s_atm)
        for i, K in enumerate(Ks):
            lo = max(mid[i] - half[i], 1e-3) * np.sqrt(T)
            hi = (mid[i] + half[i]) * np.sqrt(T)
            for is_call in (True, False):
                f = call_price if is_call else put_price
                bid, ask = float(f(k[i], lo)), float(f(k[i], hi))
                records.append(
                    {
                        "symbol": instrument_name(underlying, exp, float(K), is_call),
                        "timestamp_us": time_us - 1_000_000,
                        "expiration_us": exp,
                        "strike": float(K),
                        "option_type": "call" if is_call else "put",
                        "bid_price": bid,
                        "ask_price": ask,
                        "mark_price": 0.5 * (bid + ask),
                        "mark_iv": 100 * mid[i],
                        "underlying_price": F,
                        "bid_iv": 100 * (mid[i] - half[i]),
                        "ask_iv": 100 * (mid[i] + half[i]),
                    }
                )
    return Snapshot.from_records(time_us, records)
