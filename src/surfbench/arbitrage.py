"""Static-arbitrage diagnostics: Durrleman g(k), call-price convexity, calendar spreads, slopes.

Specification: ``docs/DESIGN.md`` section 4.4. Diagnostics run on the fit to all ``t0``
quotes. For slice ``j`` with fitted ATM total variance ``theta_j = w_j(0)``:

- wide grid: ``n_grid`` points on ``[z0 sqrt(theta_j), z1 sqrt(theta_j)]`` (default +-4);
- in-range grid: wide-grid points inside the quoted ``k`` range plus the quoted ``k``;
- butterfly: ``w <= 0`` or ``g(k) < -g_tol`` anywhere on the grid;
- calendar (adjacent accepted slices): ``min_k (w_{j+1} - w_j) < -calendar_tol`` on the wide
  grid of slice ``j + 1``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from surfbench.black import otm_price
from surfbench.models.base import FittedSurface, accepted

__all__ = [
    "DEFAULT_CONFIG",
    "ArbitrageConfig",
    "SliceButterfly",
    "butterfly_slice",
    "calendar_pair",
    "diagnose",
    "durrleman_g",
    "inrange_grid",
    "price_second_differences",
    "wide_grid",
]

Float = NDArray[np.float64]


@dataclass(frozen=True)
class ArbitrageConfig:
    grid_z: tuple[float, float] = (-4.0, 4.0)
    n_grid: int = 401
    g_tol: float = 1e-6
    calendar_tol: float = 1e-8
    price_tol: float = 1e-10
    slope_bound: float = 4.0


DEFAULT_CONFIG = ArbitrageConfig()


def durrleman_g(k: ArrayLike, w: ArrayLike, w1: ArrayLike, w2: ArrayLike) -> Float:
    """Durrleman's function (Gatheral-Jacquier 2014, Lemma 2.2); ``nan`` where ``w <= 0``."""
    k, w, w1, w2 = (np.asarray(a, dtype=float) for a in (k, w, w1, w2))
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (1.0 - k * w1 / (2.0 * w)) ** 2 - 0.25 * w1 * w1 * (1.0 / w + 0.25) + 0.5 * w2
    return np.where(w > 0, g, np.nan)


def wide_grid(
    theta: float, cfg: ArbitrageConfig = DEFAULT_CONFIG, k_quotes: ArrayLike | None = None
) -> Float:
    """``n_grid`` points on ``[z0 sqrt(theta), z1 sqrt(theta)]``, widened to the quoted range."""
    s = float(np.sqrt(max(theta, 1e-12)))
    lo, hi = cfg.grid_z[0] * s, cfg.grid_z[1] * s
    if k_quotes is not None:
        kq = np.asarray(k_quotes, dtype=float)
        lo, hi = min(lo, float(kq.min())), max(hi, float(kq.max()))
    return np.linspace(lo, hi, cfg.n_grid)


def inrange_grid(wide: Float, k_quotes: ArrayLike) -> Float:
    kq = np.asarray(k_quotes, dtype=float)
    inside = wide[(wide >= kq.min()) & (wide <= kq.max())]
    return np.unique(np.concatenate([inside, kq]))


@dataclass(frozen=True)
class SliceButterfly:
    violated_wide: bool
    violated_inrange: bool
    min_g_wide: float
    min_g_inrange: float
    nonpositive_w: bool
    interval_lo: float | None
    interval_hi: float | None
    price_violated_wide: bool
    max_abs_slope: float


def _g_on(surface: FittedSurface, j: int, k: Float) -> tuple[Float, Float]:
    w = surface.w(j, k)
    return w, durrleman_g(k, w, surface.dw(j, k), surface.d2w(j, k))


def price_second_differences(surface: FittedSurface, j: int, k: Float) -> tuple[Float, Float]:
    """First and second divided differences of normalised call prices in ``K = e^k``.

    Returns ``(first, second)``; no arbitrage requires ``first <= 0`` and ``second >= 0``.
    The call price is split as ``c = o + (1 - K)^+`` with ``o`` the out-of-the-money price;
    the divided differences of the intrinsic part are taken exactly (slope -1, 0, or the
    chord through ``K = 1``), so deep in-the-money calls do not lose precision.
    """
    w = surface.w(j, k)
    o = otm_price(k, np.sqrt(np.maximum(w, 0.0)))
    K = np.exp(k)
    dK = np.diff(K)
    K1, K2 = K[:-1], K[1:]
    slope_intr = np.where(
        K2 <= 1.0, -1.0, np.where(K1 >= 1.0, 0.0, -np.maximum(1.0 - K1, 0.0) / dK)
    )
    first = np.diff(o) / dK + slope_intr
    second = np.diff(first) / (0.5 * (K[2:] - K[:-2]))
    return first, second


def butterfly_slice(
    surface: FittedSurface,
    j: int,
    k_quotes: ArrayLike,
    cfg: ArbitrageConfig = DEFAULT_CONFIG,
) -> SliceButterfly:
    theta = float(surface.w(j, np.array([0.0]))[0])
    wide = wide_grid(theta, cfg, k_quotes)
    w_wide, g_wide = _g_on(surface, j, wide)
    rng_k = inrange_grid(wide, k_quotes)
    w_in, g_in = _g_on(surface, j, rng_k)
    nonpos = bool(np.any(~(w_wide > 0)) or np.any(~(w_in > 0)) or theta <= 0)
    bad_wide = ~(w_wide > 0) | (g_wide < -cfg.g_tol)
    bad_in = ~(w_in > 0) | (g_in < -cfg.g_tol)
    lo = hi = None
    if bad_wide.any():
        lo, hi = float(wide[bad_wide].min()), float(wide[bad_wide].max())
    first, second = price_second_differences(surface, j, wide)
    price_bad = bool(np.any(first > cfg.price_tol) or np.any(second < -cfg.price_tol))
    slope = float(np.nanmax(np.abs(surface.dw(j, wide))))
    return SliceButterfly(
        violated_wide=bool(bad_wide.any()),
        violated_inrange=bool(bad_in.any()),
        min_g_wide=float(np.nanmin(g_wide)) if np.any(np.isfinite(g_wide)) else float("nan"),
        min_g_inrange=float(np.nanmin(g_in)) if np.any(np.isfinite(g_in)) else float("nan"),
        nonpositive_w=nonpos,
        interval_lo=lo,
        interval_hi=hi,
        price_violated_wide=price_bad,
        max_abs_slope=slope,
    )


def calendar_pair(
    surface: FittedSurface, j0: int, j1: int, cfg: ArbitrageConfig = DEFAULT_CONFIG
) -> tuple[bool, float]:
    """``(violated, worst gap)`` for slices ``j0 < j1`` on the wide grid of ``j1``."""
    theta1 = float(surface.w(j1, np.array([0.0]))[0])
    k = wide_grid(theta1, cfg)
    gap = surface.w(j1, k) - surface.w(j0, k)
    worst = float(np.nanmin(gap)) if np.any(np.isfinite(gap)) else float("nan")
    return bool(not np.isfinite(worst) or worst < -cfg.calendar_tol), worst


def diagnose(
    surface: FittedSurface,
    k_quotes: Sequence[ArrayLike],
    cfg: ArbitrageConfig = DEFAULT_CONFIG,
) -> dict[str, object]:
    """All diagnostics of one fitted surface; counts are over accepted slices only."""
    slices = []
    ok_idx = [j for j in range(len(surface.expiries)) if accepted(surface.status[j])]
    for j in ok_idx:
        rec = asdict(butterfly_slice(surface, j, k_quotes[j], cfg))
        rec["slice"] = j
        slices.append(rec)
    pairs = []
    for j0, j1 in zip(ok_idx[:-1], ok_idx[1:], strict=True):
        viol, worst = calendar_pair(surface, j0, j1, cfg)
        pairs.append({"j0": j0, "j1": j1, "violated": viol, "worst_gap": worst})

    def _min(vals: list[float]) -> float:
        finite = [v for v in vals if np.isfinite(v)]
        return float(min(finite)) if finite else float("nan")

    return {
        "n_slices": len(slices),
        "n_bfly_wide": sum(s["violated_wide"] for s in slices),
        "n_bfly_inrange": sum(s["violated_inrange"] for s in slices),
        "n_price_wide": sum(s["price_violated_wide"] for s in slices),
        "n_slope": sum(s["max_abs_slope"] > cfg.slope_bound for s in slices),
        "min_g_wide": _min([s["min_g_wide"] for s in slices]),
        "min_g_inrange": _min([s["min_g_inrange"] for s in slices]),
        "n_pairs": len(pairs),
        "n_cal": sum(p["violated"] for p in pairs),
        "worst_cal_gap": _min([p["worst_gap"] for p in pairs]),
        "slices": slices,
        "pairs": pairs,
    }
