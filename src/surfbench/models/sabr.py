"""SABR per expiry with Hagan et al. (2002) lognormal implied volatility, beta fixed.

Specification: ``docs/DESIGN.md`` section 4.3.4. The forward is normalised to ``f = 1`` and
``K = e^k``; with ``beta = 1`` (the default) the smile depends on ``K/F`` only, so this is
exact. Derivatives of ``w`` are central differences with step ``1e-4``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares

from surfbench.models.base import FitInput, FittedSurface, Float

__all__ = ["SabrModel", "SabrSurface", "hagan_lognormal_vol"]

ALPHA_BOUNDS = (1e-4, 10.0)
RHO_BOUNDS = (-0.999, 0.999)
NU_BOUNDS = (1e-4, 10.0)


def hagan_lognormal_vol(
    k: ArrayLike, T: float, alpha: float, rho: float, nu: float, beta: float = 1.0
) -> Float:
    """Hagan et al. (2002) eq. (2.17a) with ``f = 1`` and ``K = e^k`` (``ln(f/K) = -k``)."""
    k = np.asarray(k, dtype=float)
    omb = 1.0 - beta
    lfk = -k
    fk = np.exp(k)  # f * K with f = 1
    fk_b = fk ** (omb / 2.0)
    denom = fk_b * (1.0 + omb**2 / 24.0 * lfk**2 + omb**4 / 1920.0 * lfk**4)
    z = nu / alpha * fk_b * lfk
    # x(z) = ln((sqrt(1 - 2 rho z + z^2) + z - rho) / (1 - rho)), evaluated with log1p in
    # a form free of cancellation on both sides of z = rho; a third-order series of z / x(z)
    # is used for |z| < 1e-4 (truncation error below 1e-16).
    root = np.sqrt(1.0 - 2.0 * rho * z + z * z)
    am1 = (z * z - 2.0 * rho * z) / (root + 1.0)  # sqrt(1 - 2 rho z + z^2) - 1
    pos = z - rho >= 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        x_pos = np.log1p((am1 + z) / (1.0 - rho))
        x_neg = -np.log1p((am1 - z) / (1.0 + rho))
        xz = np.where(pos, x_pos, x_neg)
        series = 1.0 - 0.5 * rho * z + (1.0 / 6.0 - 0.25 * rho**2) * z**2
        series = series + (5.0 * rho / 24.0 - 0.25 * rho**3) * z**3
        zx = np.where(np.abs(z) < 1e-4, series, z / xz)
    corr = (
        1.0
        + (
            omb**2 / 24.0 * alpha**2 / fk**omb
            + 0.25 * rho * beta * nu * alpha / fk_b
            + (2.0 - 3.0 * rho**2) / 24.0 * nu**2
        )
        * T
    )
    return alpha / denom * zx * corr


@dataclass
class SabrSurface(FittedSurface):
    def _w(self, j: int, k: Float) -> Float:
        alpha, rho, nu = self.params["slices"][j]
        T = self.expiries[j]
        sig = hagan_lognormal_vol(k, T, alpha, rho, nu, self.params.get("beta", 1.0))
        # Hagan's expansion can turn negative for large T nu^2; such values are not squared.
        return np.maximum(sig, 0.0) ** 2 * T


class SabrModel:
    name = "sabr"

    def __init__(self, beta: float = 1.0) -> None:
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        self.beta = beta

    def _fit_slice(self, k: Float, iv: Float, scale: Float, T: float) -> tuple[list[float], str]:
        atm = float(iv[np.argmin(np.abs(k))])
        alpha0 = float(np.clip(atm, ALPHA_BOUNDS[0] * 1.01, ALPHA_BOUNDS[1] * 0.99))
        lb = [ALPHA_BOUNDS[0], RHO_BOUNDS[0], NU_BOUNDS[0]]
        ub = [ALPHA_BOUNDS[1], RHO_BOUNDS[1], NU_BOUNDS[1]]

        def resid(p: Float) -> Float:
            sig = hagan_lognormal_vol(k, T, p[0], p[1], p[2], self.beta)
            r = (sig - iv) / scale
            return np.where(np.isfinite(r), r, 1e3)

        best: tuple[float, list[float]] | None = None
        for rho0 in (-0.5, 0.0, 0.5):
            for nu0 in (0.5, 2.0):
                try:
                    sol = least_squares(resid, [alpha0, rho0, nu0], bounds=(lb, ub), method="trf")
                except (ValueError, FloatingPointError):  # pragma: no cover - defensive
                    continue
                cost = float(np.sum(resid(sol.x) ** 2))
                if np.isfinite(cost) and (best is None or cost < best[0] - 1e-15):
                    best = (cost, [float(v) for v in sol.x])
        if best is None:  # pragma: no cover - least_squares always returns a point
            return [float("nan")] * 3, "failed:optimizer"
        sig = hagan_lognormal_vol(k, T, *best[1], self.beta)
        if not np.all(np.isfinite(sig)) or np.any(sig <= 0):
            return best[1], "failed:invalid_vol"
        return best[1], "ok"

    def fit(self, data: FitInput) -> SabrSurface:
        slices: list[list[float]] = []
        status: list[str] = []
        for j in range(data.n_slices):
            p, st = self._fit_slice(data.k[j], data.iv_mid[j], data.scale(j), data.expiries[j])
            slices.append(p)
            status.append(st)
        return SabrSurface(data.expiries, tuple(status), {"slices": slices, "beta": self.beta})

    @staticmethod
    def surface(
        expiries: tuple[float, ...], slices: list[list[float]], beta: float = 1.0
    ) -> SabrSurface:
        return SabrSurface(
            tuple(expiries), ("ok",) * len(expiries), {"slices": slices, "beta": beta}
        )
