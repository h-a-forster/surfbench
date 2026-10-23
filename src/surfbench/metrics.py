"""Accuracy metrics (IV RMSE, MAE, inside-spread rate) and deterministic cross-validation folds.

Specification: ``docs/DESIGN.md`` sections 6.2 and 6.3. Errors are reported in vol points
(decimal volatility x 100).
"""

from __future__ import annotations

import zlib

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "fold_ids",
    "fold_offsets",
    "inside_spread",
    "mae_volpt",
    "rms_spread_units",
    "rmse_volpt",
    "wing_mask",
]

Float = NDArray[np.float64]


def fold_offsets(seed: int, unit_id: str, n_slices: int, n_folds: int = 5) -> NDArray[np.int64]:
    """Per-slice fold offsets drawn from ``seed`` and a CRC32 of the unit id."""
    rng = np.random.default_rng([int(seed), zlib.crc32(unit_id.encode("utf-8"))])
    return rng.integers(0, n_folds, size=n_slices)


def fold_ids(n: int, offset: int, n_folds: int = 5) -> NDArray[np.int64]:
    """Fold of the i-th quote of a slice sorted by ``k``: ``(i + offset) mod n_folds``."""
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    return (np.arange(n) + int(offset)) % n_folds


def wing_mask(n: int) -> NDArray[np.bool_]:
    """Training mask without the outermost quote on each side (quotes sorted by ``k``)."""
    if n < 3:
        raise ValueError("need at least 3 quotes")
    m = np.ones(n, dtype=bool)
    m[0] = m[-1] = False
    return m


def _err(pred: ArrayLike, obs: ArrayLike) -> Float:
    p = np.asarray(pred, dtype=float)
    o = np.asarray(obs, dtype=float)
    if p.shape != o.shape:
        raise ValueError("pred and obs must have equal shapes")
    return p - o


def rmse_volpt(pred: ArrayLike, obs: ArrayLike) -> float:
    e = _err(pred, obs)
    return float(100.0 * np.sqrt(np.mean(e * e))) if e.size else float("nan")


def mae_volpt(pred: ArrayLike, obs: ArrayLike) -> float:
    e = _err(pred, obs)
    return float(100.0 * np.mean(np.abs(e))) if e.size else float("nan")


def inside_spread(pred: ArrayLike, bid: ArrayLike, ask: ArrayLike) -> float:
    p = np.asarray(pred, dtype=float)
    if p.size == 0:
        return float("nan")
    b = np.asarray(bid, dtype=float)
    a = np.asarray(ask, dtype=float)
    return float(np.mean((p >= b) & (p <= a)))


def rms_spread_units(
    pred: ArrayLike, obs: ArrayLike, half_spread: ArrayLike, floor: float
) -> float:
    e = _err(pred, obs)
    if e.size == 0:
        return float("nan")
    s = np.maximum(np.asarray(half_spread, dtype=float), floor)
    return float(np.sqrt(np.mean((e / s) ** 2)))
