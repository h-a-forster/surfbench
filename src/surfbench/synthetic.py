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
