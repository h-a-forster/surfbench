"""Black-76 prices, vega and implied-volatility inversion for coin-settled (inverse) options.

Specification: ``docs/DESIGN.md`` section 4.1.

All prices here are *normalised*: the undiscounted Black-76 price divided by the forward
``F``. With zero rates this is exactly the coin price of a Deribit inverse option. Inputs are
log-forward moneyness ``k = ln(K/F)`` and total standard deviation ``s = sigma * sqrt(T)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

__all__ = [
    "S_MAX",
    "S_MIN",
    "black76",
    "call_price",
    "delta",
    "implied_vol",
    "implied_s",
    "otm_price",
    "put_price",
    "vega_s",
]

S_MIN = 1e-8
S_MAX = 20.0
_MAX_ITER = 100
_TOL_S = 1e-14
_INV_SQRT_2PI = 0.3989422804014327

Float = NDArray[np.float64]


def _npdf(x: Float) -> Float:
    return _INV_SQRT_2PI * np.exp(-0.5 * x * x)


def _d1(k: Float, s: Float) -> Float:
    with np.errstate(divide="ignore", invalid="ignore"):
        return -k / s + 0.5 * s


def call_price(k: ArrayLike, s: ArrayLike) -> Float:
    """Normalised call ``N(d1) - e^k N(d2)``."""
    k = np.asarray(k, dtype=float)
    s = np.asarray(s, dtype=float)
    d1 = _d1(k, s)
    return np.maximum(ndtr(d1) - np.exp(k) * ndtr(d1 - s), 0.0)


def put_price(k: ArrayLike, s: ArrayLike) -> Float:
    """Normalised put ``e^k N(-d2) - N(-d1)``."""
    k = np.asarray(k, dtype=float)
    s = np.asarray(s, dtype=float)
    d1 = _d1(k, s)
    return np.maximum(np.exp(k) * ndtr(s - d1) - ndtr(-d1), 0.0)


def otm_price(k: ArrayLike, s: ArrayLike) -> Float:
    """Out-of-the-money price: call for ``k >= 0``, put for ``k < 0``."""
    k = np.asarray(k, dtype=float)
    return np.where(k >= 0.0, call_price(k, s), put_price(k, s))


def vega_s(k: ArrayLike, s: ArrayLike) -> Float:
    """Derivative of the normalised price with respect to ``s``: ``n(d1)``."""
    k = np.asarray(k, dtype=float)
    s = np.asarray(s, dtype=float)
    return _npdf(_d1(k, s))


def delta(k: ArrayLike, s: ArrayLike, is_call: ArrayLike) -> Float:
    """Forward delta: ``N(d1)`` for calls, ``N(d1) - 1`` for puts."""
    k = np.asarray(k, dtype=float)
    s = np.asarray(s, dtype=float)
    nd1 = ndtr(_d1(k, s))
    return np.where(np.asarray(is_call, dtype=bool), nd1, nd1 - 1.0)


def black76(F: float, K: float, T: float, sigma: float, is_call: bool) -> float:
    """Undiscounted Black-76 price in currency units (zero rates)."""
    if F <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("F, K, T and sigma must be positive")
    k = np.log(K / F)
    s = sigma * np.sqrt(T)
    p = call_price(k, s) if is_call else put_price(k, s)
    return float(F * p)


def implied_s(price: ArrayLike, k: ArrayLike, is_call: ArrayLike) -> Float:
    """Invert a normalised price for ``s = sigma sqrt(T)``; ``nan`` where invalid.

    Valid prices satisfy ``max(intrinsic, 0) < price < 1`` for calls and
    ``max(intrinsic, 0) < price < e^k`` for puts. In-the-money prices are converted to the
    out-of-the-money side by put-call parity (``c - p = 1 - e^k``) before inversion.
    Method: vectorised safeguarded Newton-bisection on ``s`` in ``[1e-8, 20]``.
    """
    price, k, is_call = np.broadcast_arrays(
        np.asarray(price, dtype=float), np.asarray(k, dtype=float), np.asarray(is_call, dtype=bool)
    )
    price = price.astype(float, copy=True)
    k = k.astype(float, copy=True)
    is_call = is_call.astype(bool, copy=True)
    shape = price.shape
    price, k, is_call = price.ravel(), k.ravel(), is_call.ravel()
    ek = np.exp(k)
    upper = np.where(is_call, 1.0, ek)
    valid = np.isfinite(price) & np.isfinite(k) & (price > 0.0) & (price < upper)
    # Convert to the OTM side: calls for k >= 0, puts for k < 0.
    target = price.copy()
    to_put = is_call & (k < 0.0)
    to_call = ~is_call & (k >= 0.0)
    target[to_put] = price[to_put] - (1.0 - ek[to_put])
    target[to_call] = price[to_call] + (1.0 - ek[to_call])
    valid &= target > 0.0
    out = np.full(price.shape, np.nan)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return out.reshape(shape)
    kk = k[idx]
    tt = target[idx]
    lo = np.full(idx.size, S_MIN)
    hi = np.full(idx.size, S_MAX)
    p_lo = otm_price(kk, lo)
    p_hi = otm_price(kk, hi)
    ok = (p_lo < tt) & (tt < p_hi)
    # Targets outside (p(S_MIN), p(S_MAX)) have no solution in the bracket: nan.
    # Start from a Brenner-Subrahmanyam-like guess clipped into the bracket.
    guess = np.sqrt(2.0 * np.abs(kk)) + np.sqrt(2.0 * np.pi) * tt
    s = np.where(ok, np.clip(guess, 1e-4, 5.0), np.nan)
    active = ok.copy()
    for _ in range(_MAX_ITER):
        if not active.any():
            break
        a = np.flatnonzero(active)
        sa = s[a]
        f = otm_price(kk[a], sa) - tt[a]
        # Maintain the bracket: price is increasing in s.
        above = f > 0.0
        hi[a] = np.where(above, sa, hi[a])
        lo[a] = np.where(above, lo[a], sa)
        v = vega_s(kk[a], sa)
        with np.errstate(divide="ignore", invalid="ignore"):
            newton = sa - f / v
        inside = np.isfinite(newton) & (newton >= lo[a]) & (newton <= hi[a])
        new = np.where(inside, newton, 0.5 * (lo[a] + hi[a]))
        new = np.where(f == 0.0, sa, new)
        step = np.abs(new - sa)
        s[a] = new
        done = (step <= _TOL_S * np.maximum(1.0, new)) | (hi[a] - lo[a] <= _TOL_S) | (f == 0.0)
        active[a[done]] = False
    s[~ok] = np.nan
    out[idx] = s
    return out.reshape(shape)


def implied_vol(price: ArrayLike, k: ArrayLike, T: ArrayLike, is_call: ArrayLike) -> Float:
    """Implied volatility ``sigma = s / sqrt(T)`` from a normalised (coin) price."""
    T = np.asarray(T, dtype=float)
    if np.any(T <= 0):
        raise ValueError("T must be positive")
    return implied_s(price, k, is_call) / np.sqrt(T)
