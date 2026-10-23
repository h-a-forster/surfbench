"""SSVI (Gatheral-Jacquier) with the arbitrage-free power-law curvature, fitted globally.

Specification: ``docs/DESIGN.md`` section 4.3.2.

``w(k, theta) = theta/2 (1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2))`` with
``phi(theta) = eta / (theta^gamma (1 + theta)^(1 - gamma))``. Shared ``(rho, eta, gamma)``,
``theta_j`` per expiry. Static-arbitrage freedom holds when ``theta_j`` is non-decreasing,
``gamma in (0, 1/2]`` and ``eta (1 + |rho|) <= 2``; these are asserted on every fit.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.optimize import least_squares

from surfbench.models.base import FitInput, FittedSurface, Float, iv_residuals

__all__ = ["SsviModel", "SsviSurface", "atm_total_variance", "phi_power", "ssvi_slice"]

RHO_SCALE = 0.999
U_BOUNDS = (0.01, 1.0)
GAMMA_BOUNDS = (0.01, 0.5)
X1_BOUND = 4.0
_TOL = 1e-12


def phi_power(theta: Float | float, eta: float, gamma: float) -> Float:
    theta = np.asarray(theta, dtype=float)
    return eta / (theta**gamma * (1.0 + theta) ** (1.0 - gamma))


def ssvi_slice(k: Float, theta: float, rho: float, phi: float) -> tuple[Float, Float, Float]:
    """SSVI slice total variance and its first two ``k``-derivatives."""
    u = phi * k + rho
    r = np.sqrt(u * u + 1.0 - rho * rho)
    w = 0.5 * theta * (1.0 + rho * phi * k + r)
    w1 = 0.5 * theta * phi * (rho + u / r)
    w2 = 0.5 * theta * phi * phi * (1.0 - rho * rho) / r**3
    return w, w1, w2


def atm_total_variance(k: Float, w: Float) -> float:
    """Total variance at ``k = 0`` by linear interpolation of the bracketing quotes.

    ``k`` must be sorted. Uses the nearest quote when ``k = 0`` is not bracketed.
    """
    if k[0] >= 0.0:
        return float(w[0])
    if k[-1] <= 0.0:
        return float(w[-1])
    i = int(np.searchsorted(k, 0.0))
    k0, k1, w0, w1 = k[i - 1], k[i], w[i - 1], w[i]
    if k1 == k0:  # pragma: no cover - duplicate strikes are removed by the filter
        return float(0.5 * (w0 + w1))
    return float(w0 + (w1 - w0) * (0.0 - k0) / (k1 - k0))


def check_ssvi(theta: Float, rho: float, eta: float, gamma: float) -> None:
    """Raise ``AssertionError`` if the SSVI no-arbitrage conditions fail."""
    theta = np.asarray(theta, dtype=float)
    if not (np.all(theta > 0) and np.all(np.diff(theta) >= -_TOL)):
        raise AssertionError("theta must be positive and non-decreasing")
    if not (0.0 < gamma <= 0.5 + _TOL):
        raise AssertionError("gamma must be in (0, 1/2]")
    if not (eta > 0 and eta * (1.0 + abs(rho)) <= 2.0 + _TOL and abs(rho) < 1.0):
        raise AssertionError("eta (1 + |rho|) <= 2 violated")


@dataclass
class SsviSurface(FittedSurface):
    analytic_derivatives: ClassVar[bool] = True

    def _args(self, j: int) -> tuple[float, float, float]:
        theta = float(self.params["theta"][j])
        rho, eta, gamma = self.params["rho"], self.params["eta"], self.params["gamma"]
        return theta, rho, float(phi_power(theta, eta, gamma))

    def _w(self, j: int, k: Float) -> Float:
        return ssvi_slice(k, *self._args(j))[0]

    def _dw(self, j: int, k: Float) -> Float:
        return ssvi_slice(k, *self._args(j))[1]

    def _d2w(self, j: int, k: Float) -> Float:
        return ssvi_slice(k, *self._args(j))[2]


def _unpack(x: Float) -> tuple[float, float, float]:
    rho = RHO_SCALE * float(np.tanh(x[0]))
    eta = 2.0 * float(x[1]) / (1.0 + abs(rho))
    return rho, eta, float(x[2])


class SsviModel:
    name = "ssvi"

    def fit(self, data: FitInput) -> SsviSurface:
        n = data.n_slices
        T = np.asarray(data.expiries)
        theta_star = np.array(
            [atm_total_variance(data.k[j], data.iv_mid[j] ** 2 * T[j]) for j in range(n)]
        )
        theta = np.maximum.accumulate(theta_star)
        if not np.all(theta > 0):  # pragma: no cover - iv_mid > 0 makes theta > 0
            return SsviSurface(data.expiries, ("failed:theta",) * n, {})
        k_all = data.k
        scales = [data.scale(j) for j in range(n)]

        def resid(x: Float) -> Float:
            rho, eta, gamma = _unpack(x)
            out = []
            for j in range(n):
                phi = float(phi_power(theta[j], eta, gamma))
                w = ssvi_slice(k_all[j], float(theta[j]), rho, phi)[0]
                out.append(iv_residuals(w, T[j], data.iv_mid[j], scales[j]))
            r = np.concatenate(out)
            return np.where(np.isfinite(r), r, 1e3)

        lb = [-X1_BOUND, U_BOUNDS[0], GAMMA_BOUNDS[0]]
        ub = [X1_BOUND, U_BOUNDS[1], GAMMA_BOUNDS[1]]
        best: tuple[float, Float] | None = None
        for rho0 in (-0.5, 0.0, 0.3):
            for u0 in (0.3, 0.7):
                for g0 in (0.25, 0.45):
                    x0 = np.array([np.arctanh(rho0 / RHO_SCALE), u0, g0])
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        sol = least_squares(resid, x0, bounds=(lb, ub), method="trf")
                    cost = float(np.sum(resid(sol.x) ** 2))
                    if best is None or cost < best[0] - 1e-15:
                        best = (cost, sol.x)
        assert best is not None
        rho, eta, gamma = _unpack(best[1])
        check_ssvi(theta, rho, eta, gamma)
        params = {
            "theta": [float(t) for t in theta],
            "theta_star": [float(t) for t in theta_star],
            "rho": rho,
            "eta": eta,
            "gamma": gamma,
        }
        return SsviSurface(data.expiries, ("ok",) * n, params)

    @staticmethod
    def surface(
        expiries: tuple[float, ...], theta: list[float], rho: float, eta: float, gamma: float
    ) -> SsviSurface:
        params = {"theta": list(theta), "rho": rho, "eta": eta, "gamma": gamma}
        return SsviSurface(tuple(expiries), ("ok",) * len(expiries), params)
