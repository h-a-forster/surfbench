"""Raw SVI per expiry, calibrated with the quasi-explicit method (Zeliade).

Specification: ``docs/DESIGN.md`` section 4.3.1.

``w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2))``. Imposed: ``b >= 0``,
``|rho| <= 0.999``, ``sigma >= 1e-4``, minimum variance ``a + b sigma sqrt(1 - rho^2) >= 0``
and Lee's bound ``b (1 + |rho|) <= 2``. Butterfly and calendar freedom are not imposed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.optimize import least_squares, minimize

from surfbench.models.base import FitInput, FittedSurface, Float, iv_residuals, w_weights

__all__ = ["SviModel", "SviSurface", "svi_constraints_ok", "svi_raw"]

RHO_MAX = 0.999
SIG_MIN = 1e-4
_TOL = 1e-12


def svi_raw(
    k: Float, a: float, b: float, rho: float, m: float, sig: float
) -> tuple[Float, Float, Float]:
    """Raw SVI total variance and its first two ``k``-derivatives."""
    x = k - m
    r = np.sqrt(x * x + sig * sig)
    w = a + b * (rho * x + r)
    w1 = b * (rho + x / r)
    w2 = b * sig * sig / r**3
    return w, w1, w2


def svi_constraints_ok(a: float, b: float, rho: float, m: float, sig: float) -> bool:
    """Constraints of DESIGN 4.3.1 (tolerance 1e-12)."""
    vals = (a, b, rho, m, sig)
    if not all(np.isfinite(v) for v in vals):
        return False
    return (
        b >= -_TOL
        and abs(rho) <= RHO_MAX + _TOL
        and sig >= SIG_MIN - _TOL
        and a + b * sig * np.sqrt(max(1.0 - rho * rho, 0.0)) >= -_TOL
        and b * (1.0 + abs(rho)) <= 2.0 + _TOL
    )


@dataclass
class SviSurface(FittedSurface):
    analytic_derivatives: ClassVar[bool] = True

    def _p(self, j: int) -> tuple[float, float, float, float, float]:
        a, b, rho, m, sig = self.params["slices"][j]
        return a, b, rho, m, sig

    def _w(self, j: int, k: Float) -> Float:
        return svi_raw(k, *self._p(j))[0]

    def _dw(self, j: int, k: Float) -> Float:
        return svi_raw(k, *self._p(j))[1]

    def _d2w(self, j: int, k: Float) -> Float:
        return svi_raw(k, *self._p(j))[2]


def _inner(y: Float, wt: Float, om: Float, sig: float) -> tuple[Float, float]:
    """Constrained WLS in ``(a, d, c)`` for fixed ``(m, sigma)``; returns params and loss."""
    sq = np.sqrt(y * y + 1.0)
    X = np.column_stack([np.ones_like(y), y, sq])
    Xw = X * om[:, None]
    yw = wt * om
    p0, *_ = np.linalg.lstsq(Xw, yw, rcond=None)

    def feasible(p: Float) -> bool:
        a, d, c = p
        return (
            c >= 0.0
            and abs(d) <= RHO_MAX * c + 1e-15
            and c + abs(d) <= 2.0 * sig + 1e-15
            and a + np.sqrt(max(c * c - d * d, 0.0)) >= 0.0
        )

    def loss(p: Float) -> float:
        r = Xw @ p - yw
        return float(r @ r)

    if feasible(p0):
        return p0, loss(p0)
    # Clip into the feasible region as a starting point.
    a0, d0, c0 = (float(v) for v in p0)
    c0 = float(np.clip(c0, 0.0, 2.0 * sig))
    d0 = float(np.clip(d0, -RHO_MAX * c0, RHO_MAX * c0))
    if c0 + abs(d0) > 2.0 * sig:
        d0 = float(np.sign(d0) * (2.0 * sig - c0))
    a0 = max(a0, -np.sqrt(max(c0 * c0 - d0 * d0, 0.0)))
    start = np.array([a0, d0, c0])
    H = Xw.T @ Xw
    g0 = Xw.T @ yw
    cons = [
        {"type": "ineq", "fun": lambda p: p[2], "jac": lambda p: np.array([0.0, 0.0, 1.0])},
        {
            "type": "ineq",
            "fun": lambda p: RHO_MAX * p[2] - p[1],
            "jac": lambda p: np.array([0.0, -1.0, RHO_MAX]),
        },
        {
            "type": "ineq",
            "fun": lambda p: RHO_MAX * p[2] + p[1],
            "jac": lambda p: np.array([0.0, 1.0, RHO_MAX]),
        },
        {
            "type": "ineq",
            "fun": lambda p: 2.0 * sig - p[2] - p[1],
            "jac": lambda p: np.array([0.0, -1.0, -1.0]),
        },
        {
            "type": "ineq",
            "fun": lambda p: 2.0 * sig - p[2] + p[1],
            "jac": lambda p: np.array([0.0, 1.0, -1.0]),
        },
        {
            "type": "ineq",
            "fun": lambda p: p[0] + np.sqrt(max(p[2] ** 2 - p[1] ** 2, 0.0)),
        },
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(
            lambda p: float(p @ H @ p - 2.0 * g0 @ p + yw @ yw),
            start,
            jac=lambda p: 2.0 * (H @ p - g0),
            constraints=cons,
            method="SLSQP",
            options={"maxiter": 200, "ftol": 1e-14},
        )
    p = np.asarray(res.x, dtype=float)
    # Repair tiny constraint violations from SLSQP's tolerance.
    a, d, c = (float(v) for v in p)
    c = float(np.clip(c, 0.0, 2.0 * sig))
    d = float(np.clip(d, -RHO_MAX * c, RHO_MAX * c))
    if c + abs(d) > 2.0 * sig:
        d = float(np.sign(d) * (2.0 * sig - c))
    a = max(a, -np.sqrt(max(c * c - d * d, 0.0)))
    p = np.array([a, d, c])
    if not np.all(np.isfinite(p)):
        return start, loss(start)
    if loss(start) < loss(p):
        return start, loss(start)
    return p, loss(p)


def _to_raw(p: Float, m: float, sig: float) -> tuple[float, float, float, float, float]:
    a, d, c = (float(v) for v in p)
    b = c / sig
    rho = d / c if c > 0 else 0.0
    return a, b, float(np.clip(rho, -RHO_MAX, RHO_MAX)), m, sig


class SviModel:
    name = "svi"

    def _fit_slice(self, k: Float, iv: Float, scale: Float, T: float) -> tuple[list[float], str]:
        wt = iv * iv * T
        om = w_weights(T, iv, scale)
        m_lo, m_hi = float(k.min()) - 1.0, float(k.max()) + 1.0
        s_lo, s_hi = 1e-3, 2.0

        def outer(x: Float) -> float:
            m = float(np.clip(x[0], m_lo, m_hi))
            sig = float(np.clip(x[1], s_lo, s_hi))
            _, loss = _inner((k - m) / sig, wt, om, sig)
            return loss

        grid_m = np.linspace(float(k.min()), float(k.max()), 5)
        grid_s = np.geomspace(0.01, 1.0, 5)
        best = min((outer(np.array([m, s])), m, s) for m in grid_m for s in grid_s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(
                outer,
                np.array([best[1], best[2]]),
                method="Nelder-Mead",
                options={"xatol": 1e-7, "fatol": 1e-12, "maxiter": 400},
            )
        x = res.x if res.fun <= best[0] else np.array([best[1], best[2]])
        m = float(np.clip(x[0], m_lo, m_hi))
        sig = float(np.clip(x[1], s_lo, s_hi))
        p, _ = _inner((k - m) / sig, wt, om, sig)
        raw = _to_raw(p, m, sig)
        if not svi_constraints_ok(*raw):
            return list(raw), "failed:constraints"

        def resid(q: Float, penalty: bool = True) -> Float:
            w = svi_raw(k, *q)[0]
            r = iv_residuals(w, T, iv, scale)
            if not penalty:
                return r
            a, b, rho, _, s = q
            minvar = a + b * s * np.sqrt(max(1.0 - rho * rho, 0.0))
            lee = b * (1.0 + abs(rho)) - 2.0
            pen = 1e4 * np.array([max(0.0, -minvar), max(0.0, lee)])
            return np.concatenate([r, pen])

        L0 = float(np.sum(resid(np.array(raw), False) ** 2))
        wmax = float(wt.max())
        lb = [-wmax - 1.0, 0.0, -RHO_MAX, m_lo, SIG_MIN]
        ub = [wmax + 1.0, 2.0, RHO_MAX, m_hi, s_hi]
        q0 = np.clip(np.array(raw), lb, ub)
        try:
            sol = least_squares(resid, q0, bounds=(lb, ub), method="trf", x_scale="jac")
            cand = tuple(float(v) for v in sol.x)
            L1 = float(np.sum(resid(np.array(cand), False) ** 2))
            if svi_constraints_ok(*cand) and np.isfinite(L1) and L1 < L0:
                raw = cand  # type: ignore[assignment]
        except (ValueError, FloatingPointError):  # pragma: no cover - keep the inner solution
            pass
        return list(raw), "ok"

    def fit(self, data: FitInput) -> SviSurface:
        slices: list[list[float]] = []
        status: list[str] = []
        for j in range(data.n_slices):
            p, st = self._fit_slice(data.k[j], data.iv_mid[j], data.scale(j), data.expiries[j])
            slices.append(p)
            status.append(st)
        return SviSurface(data.expiries, tuple(status), {"slices": slices})

    @staticmethod
    def surface(expiries: tuple[float, ...], slices: list[list[float]]) -> SviSurface:
        """Surface from raw parameters ``(a, b, rho, m, sigma)`` per slice."""
        return SviSurface(tuple(expiries), ("ok",) * len(expiries), {"slices": slices})
