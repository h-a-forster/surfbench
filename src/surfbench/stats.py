"""Month-level bootstrap confidence intervals, paired sign-flip tests and Holm adjustment.

Specification: ``docs/DESIGN.md`` section 6.4. Months are the independent unit; every
function takes one value per month (or one row per month) and a ``seed``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "Interval",
    "block_indices",
    "bootstrap_mean",
    "bootstrap_ratio",
    "holm",
    "rank_probabilities",
    "sign_flip_exact",
    "sign_flip_test",
]


@dataclass(frozen=True)
class Interval:
    """Point estimate with a percentile bootstrap interval."""

    estimate: float
    lo: float
    hi: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        return {"estimate": self.estimate, "lo": self.lo, "hi": self.hi, "n": self.n}


_NAN_INTERVAL = Interval(float("nan"), float("nan"), float("nan"), 0)


def block_indices(n: int, n_boot: int, block_length: int, rng: np.random.Generator) -> NDArray:
    """Resampling indices, shape ``(n_boot, n)``.

    ``block_length == 1`` is the ordinary i.i.d. bootstrap. Larger values give the moving
    block bootstrap: blocks of consecutive months start at uniform positions in
    ``0 .. n - block_length`` and are concatenated, then truncated to ``n``.
    """
    if n < 1:
        raise ValueError("need at least one observation")
    if block_length < 1:
        raise ValueError("block_length must be >= 1")
    if block_length == 1:
        return rng.integers(0, n, size=(n_boot, n))
    block_length = min(block_length, n)
    n_blocks = -(-n // block_length)
    starts = rng.integers(0, n - block_length + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block_length)[None, None, :]).reshape(n_boot, -1)
    return idx[:, :n]


def _clean(x: ArrayLike) -> NDArray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError("expected a 1-D array of monthly values")
    return a[np.isfinite(a)]


def bootstrap_mean(
    x: ArrayLike,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    block_length: int = 1,
    seed: int = 0,
) -> Interval:
    """Mean of monthly values with a percentile bootstrap interval (non-finite values dropped)."""
    a = _clean(x)
    if a.size == 0:
        return _NAN_INTERVAL
    rng = np.random.default_rng(seed)
    idx = block_indices(a.size, n_boot, block_length, rng)
    means = a[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return Interval(float(a.mean()), float(lo), float(hi), int(a.size))


def bootstrap_ratio(
    num: ArrayLike,
    den: ArrayLike,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    block_length: int = 1,
    seed: int = 0,
) -> Interval:
    """Pooled share ``sum(num) / sum(den)`` with a month-cluster bootstrap interval.

    Used for arbitrage shares: ``num`` is the number of violating slices in a month and
    ``den`` the number of slices checked. Months with ``den == 0`` are dropped.
    """
    n_ = np.asarray(num, dtype=float)
    d_ = np.asarray(den, dtype=float)
    if n_.shape != d_.shape or n_.ndim != 1:
        raise ValueError("num and den must be 1-D arrays of equal length")
    keep = np.isfinite(n_) & np.isfinite(d_) & (d_ > 0)
    n_, d_ = n_[keep], d_[keep]
    if n_.size == 0:
        return _NAN_INTERVAL
    rng = np.random.default_rng(seed)
    idx = block_indices(n_.size, n_boot, block_length, rng)
    ratios = n_[idx].sum(axis=1) / d_[idx].sum(axis=1)
    lo, hi = np.quantile(ratios, [alpha / 2.0, 1.0 - alpha / 2.0])
    return Interval(float(n_.sum() / d_.sum()), float(lo), float(hi), int(n_.size))


def sign_flip_test(d: ArrayLike, *, n_perm: int = 100_000, seed: int = 0) -> float:
    """Two-sided randomisation test of mean(d) = 0 by random sign flips.

    ``p = (1 + count) / (n_perm + 1)`` where ``count`` is the number of random sign vectors
    whose ``|mean|`` is at least the observed ``|mean|`` (a relative tolerance of 1e-12
    guards against ties lost to rounding). Non-finite values are dropped; an empty input or
    an all-zero input gives ``p = 1``.
    """
    a = _clean(d)
    if a.size == 0 or not np.any(a):
        return 1.0
    obs = abs(a.mean())
    thresh = obs * (1.0 - 1e-12)
    rng = np.random.default_rng(seed)
    count = 0
    chunk = max(1, min(n_perm, 2_000_000 // max(a.size, 1)))
    done = 0
    while done < n_perm:
        m = min(chunk, n_perm - done)
        signs = rng.integers(0, 2, size=(m, a.size), dtype=np.int8) * 2 - 1
        stats_ = np.abs(signs @ a) / a.size
        count += int(np.count_nonzero(stats_ >= thresh))
        done += m
    return (1.0 + count) / (n_perm + 1.0)


def sign_flip_exact(d: ArrayLike) -> float:
    """Exact two-sided sign-flip p-value over all ``2^n`` sign vectors (small ``n`` only)."""
    a = _clean(d)
    if a.size == 0 or not np.any(a):
        return 1.0
    if a.size > 20:
        raise ValueError("exact enumeration is limited to n <= 20")
    obs = abs(a.mean()) * (1.0 - 1e-12)
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=a.size)))
    stats_ = np.abs(signs @ a) / a.size
    return float(np.count_nonzero(stats_ >= obs) / signs.shape[0])


def holm(pvalues: ArrayLike) -> NDArray:
    """Holm (1979) step-down adjusted p-values, monotone and capped at 1."""
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1:
        raise ValueError("expected a 1-D array of p-values")
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be finite and in [0, 1]")
    m = p.size
    order = np.argsort(p, kind="stable")
    adj_sorted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adj_sorted, 1.0)
    return out


def rank_probabilities(
    values: ArrayLike,
    *,
    n_boot: int = 10_000,
    block_length: int = 1,
    seed: int = 0,
) -> NDArray:
    """Share of month resamples in which each column has the lowest mean.

    ``values`` has one row per month and one column per model and must be finite (use the
    common set of months). Ties are split equally.
    """
    v = np.asarray(values, dtype=float)
    if v.ndim != 2 or v.shape[0] < 1:
        raise ValueError("expected a (months, models) array")
    if not np.all(np.isfinite(v)):
        raise ValueError("values must be finite; restrict to the common set of months")
    rng = np.random.default_rng(seed)
    idx = block_indices(v.shape[0], n_boot, block_length, rng)
    means = v[idx].mean(axis=1)  # (n_boot, models)
    best = means.min(axis=1, keepdims=True)
    winners = np.isclose(means, best, rtol=0.0, atol=1e-15)
    share = (winners / winners.sum(axis=1, keepdims=True)).mean(axis=0)
    return share
