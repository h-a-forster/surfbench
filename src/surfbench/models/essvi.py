"""eSSVI (Hendriks-Martini) fitted slice by slice with the Corbetta et al. robust algorithm.

Specification: ``docs/DESIGN.md`` section 4.3.3.

Slice ``j`` in ``(theta, rho, psi)`` form (``psi = theta phi``):
``w(k) = 1/2 (theta + rho psi k + sqrt((psi k + theta rho)^2 + theta^2 (1 - rho^2)))``.
Butterfly (sufficient): ``psi (1 + |rho|) < 4`` and ``psi <= 2 sqrt(theta / (1 + |rho|))``.
Calendar with the previous accepted slice (Mingone 2022, arXiv:2204.00312, section 2.1):
the necessary conditions ``theta >= theta_p``, ``psi >= psi_p`` and
``|rho psi - rho_p psi_p| <= psi - psi_p``, plus one of the two sufficient branches
``psi / psi_p <= theta / theta_p`` or
``(rho_p - (psi / psi_p) rho)^2 <= (theta / theta_p - 1)(psi^2 theta_p / (psi_p^2 theta) - 1)``.
The necessary conditions alone admit crossing slices. For each ``rho`` the fitter searches
the parts of the necessary ``psi`` interval where either branch holds. Every accepted slice
is checked to 1e-12.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from surfbench.models.base import FitInput, FittedSurface, Float, iv_residuals

__all__ = ["EssviModel", "EssviSurface", "check_essvi_pair", "check_essvi_slice", "essvi_slice"]

_TOL = 1e-12
_SHRINK = 1.0 - 1e-10
RHO_GRID = np.linspace(-0.95, 0.95, 21)
RHO_CLIP = 0.99
PSI_MIN = 1e-10
_N_SCAN = 33


def essvi_slice(k: Float, theta: float, rho: float, psi: float) -> tuple[Float, Float, Float]:
    """eSSVI slice total variance and its first two ``k``-derivatives."""
    u = psi * k + theta * rho
    r = np.sqrt(u * u + theta * theta * (1.0 - rho * rho))
    w = 0.5 * (theta + rho * psi * k + r)
    w1 = 0.5 * psi * (rho + u / r)
    w2 = 0.5 * psi * psi * theta * theta * (1.0 - rho * rho) / r**3
    return w, w1, w2


def check_essvi_slice(theta: float, rho: float, psi: float) -> bool:
    a = 1.0 + abs(rho)
    return (
        theta > 0.0
        and psi > 0.0
        and abs(rho) < 1.0
        and psi * a < 4.0 + _TOL
        and psi <= 2.0 * math.sqrt(theta / a) + _TOL
    )


def check_essvi_pair(prev: tuple[float, float, float], cur: tuple[float, float, float]) -> bool:
    """No calendar arbitrage between two eSSVI slices (necessary and sufficient conditions)."""
    tp, rp, pp = prev
    t, r, p = cur
    necessary = t >= tp - _TOL and p >= pp - _TOL and abs(r * p - rp * pp) <= p - pp + _TOL
    if not necessary:
        return False
    if p * tp <= pp * t * (1.0 + _TOL) + _TOL:
        return True
    q = p / pp
    return (rp - q * r) ** 2 <= (t / tp - 1.0) * (q * q * tp / t - 1.0) + _TOL


@dataclass
class EssviSurface(FittedSurface):
    analytic_derivatives: ClassVar[bool] = True

    def _p(self, j: int) -> tuple[float, float, float]:
        theta, rho, psi = self.params["slices"][j]
        return theta, rho, psi

    def _w(self, j: int, k: Float) -> Float:
        return essvi_slice(k, *self._p(j))[0]

    def _dw(self, j: int, k: Float) -> Float:
        return essvi_slice(k, *self._p(j))[1]

    def _d2w(self, j: int, k: Float) -> Float:
        return essvi_slice(k, *self._p(j))[2]


def _psi_interval(
    rho: float, k_star: float, theta_star: float, prev: tuple[float, float, float] | None
) -> tuple[float, float] | None:
    """Feasible ``psi`` interval for a given ``rho`` (DESIGN 4.3.3 step 2)."""
    a = 1.0 + abs(rho)
    rk = rho * k_star
    disc = 4.0 * rk * rk / (a * a) + 4.0 * theta_star / a
    if disc < 0.0:
        return None
    psi_plus = -2.0 * rk / a + math.sqrt(disc)
    hi = min(psi_plus, 4.0 / a) * _SHRINK
    lo = PSI_MIN
    if prev is not None:
        tp, rp, pp = prev
        psi_minus = max((pp - rp * pp) / (1.0 - rho), (pp + rp * pp) / (1.0 + rho), pp)
        lo = max(lo, psi_minus * (1.0 + 1e-12) + 1e-15)
        # theta = theta* - rho psi k* >= theta_p  <=>  rho k* psi <= theta* - theta_p
        gap = theta_star - tp
        if rk > 0.0:
            hi = min(hi, gap / rk * _SHRINK if gap > 0 else -1.0)
        elif rk < 0.0:
            lo = max(lo, gap / rk)
        elif gap < 0.0:
            return None
    if not (hi > lo and hi > 0.0):
        return None
    return lo, hi


def _sufficient_margin(
    psi: float, rho: float, k_star: float, theta_star: float, prev: tuple[float, float, float]
) -> float:
    """``>= 0`` iff one of the two sufficient calendar branches holds (``theta`` anchored)."""
    tp, rp, pp = prev
    theta = theta_star - rho * psi * k_star
    if theta <= 0.0:
        return -math.inf
    lin = (pp * theta - psi * tp) / (pp * tp)
    q = psi / pp
    quad = (theta / tp - 1.0) * (q * q * tp / theta - 1.0) - (rp - q * rho) ** 2
    return max(lin, quad)


def _psi_intervals(
    rho: float, k_star: float, theta_star: float, prev: tuple[float, float, float] | None
) -> list[tuple[float, float]]:
    """Sub-intervals of the necessary ``psi`` interval where a sufficient branch holds.

    The margin is sampled on ``_N_SCAN`` points; sign changes are refined with Brent's
    method and each end is pulled inside by a relative ``1e-9``. A feasible island narrower
    than the scan step can be missed, which only makes the fit more conservative.
    """
    iv_ = _psi_interval(rho, k_star, theta_star, prev)
    if iv_ is None:
        return []
    if prev is None:
        return [iv_]
    lo, hi = iv_

    def m(x: float) -> float:
        return _sufficient_margin(x, rho, k_star, theta_star, prev)

    grid = np.linspace(lo, hi, _N_SCAN)
    vals = np.array([m(float(x)) for x in grid])
    ok = vals >= 0.0
    out: list[tuple[float, float]] = []
    i = 0
    while i < grid.size:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < grid.size and ok[j + 1]:
            j += 1
        a = float(grid[i]) if i == 0 else brentq(m, float(grid[i - 1]), float(grid[i]), xtol=1e-15)
        b = (
            float(grid[j])
            if j == grid.size - 1
            else brentq(m, float(grid[j]), float(grid[j + 1]), xtol=1e-15)
        )
        width = b - a
        a, b = a + 1e-9 * width, b - 1e-9 * width
        if b > a and m(a) >= 0.0 and m(b) >= 0.0:
            out.append((a, b))
        elif width <= 0.0 and m(a) >= 0.0:
            out.append((a, a))
        i = j + 1
    return out


class EssviModel:
    name = "essvi"

    def _slice_loss(
        self, k: Float, iv: Float, scale: Float, T: float, k_star: float, theta_star: float
    ):
        def loss(rho: float, psi: float) -> float:
            theta = theta_star - rho * psi * k_star
            if theta <= 0.0:
                return math.inf
            w = essvi_slice(k, theta, rho, psi)[0]
            r = iv_residuals(w, T, iv, scale)
            return float(r @ r)

        return loss

    def _search(
        self,
        loss,
        rhos: Float,
        k_star: float,
        theta_star: float,
        prev: tuple[float, float, float] | None,
    ) -> tuple[float, float, float] | None:
        best: tuple[float, float, float] | None = None

        def admissible(rho: float, psi: float) -> bool:
            cur = (theta_star - rho * psi * k_star, rho, psi)
            return check_essvi_slice(*cur) and (prev is None or check_essvi_pair(prev, cur))

        for rho in rhos:
            rho = float(rho)
            for lo, hi in _psi_intervals(rho, k_star, theta_star, prev):
                cands: list[float] = [lo, hi]  # bounded Brent never evaluates the end points
                if hi - lo >= 1e-12:
                    res = minimize_scalar(
                        lambda p, rho=rho: loss(rho, p),
                        bounds=(lo, hi),
                        method="bounded",
                        options={"xatol": 1e-10 * max(1.0, hi)},
                    )
                    cands.insert(0, float(res.x))
                for psi in cands:
                    val = loss(rho, psi)
                    if (
                        math.isfinite(val)
                        and (best is None or val < best[0])
                        and admissible(rho, psi)
                    ):
                        best = (val, rho, psi)
        return best

    def _polish_rho(
        self,
        loss,
        best: tuple[float, float, float],
        k_star: float,
        theta_star: float,
        prev: tuple[float, float, float] | None,
    ) -> tuple[float, float, float]:
        """Profile Brent over ``rho`` on one refine-grid step around the best grid point."""
        _, rho_b, _ = best
        lo = max(rho_b - 0.02, -RHO_CLIP)
        hi = min(rho_b + 0.02, RHO_CLIP)
        if hi - lo < 1e-9:
            return best

        def profile(rho: float) -> float:
            r = self._search(loss, np.array([rho]), k_star, theta_star, prev)
            return 1e300 if r is None else r[0]  # finite, so Brent arithmetic stays defined

        res = minimize_scalar(profile, bounds=(lo, hi), method="bounded", options={"xatol": 1e-9})
        cand = self._search(loss, np.array([float(res.x)]), k_star, theta_star, prev)
        if cand is not None and cand[0] < best[0]:
            return cand
        return best

    def _fit_slice(
        self,
        k: Float,
        iv: Float,
        scale: Float,
        T: float,
        prev: tuple[float, float, float] | None,
    ) -> tuple[tuple[float, float, float] | None, str]:
        i = int(np.argmin(np.abs(k)))
        k_star = float(k[i])
        theta_star = float(iv[i] ** 2 * T)
        status = "ok"
        for attempt in range(2):
            loss = self._slice_loss(k, iv, scale, T, k_star, theta_star)
            best = self._search(loss, RHO_GRID, k_star, theta_star, prev)
            if best is not None:
                rho_b = best[1]
                fine = np.clip(np.linspace(rho_b - 0.1, rho_b + 0.1, 11), -RHO_CLIP, RHO_CLIP)
                best2 = self._search(loss, fine, k_star, theta_star, prev)
                if best2 is not None and best2[0] < best[0]:
                    best = best2
                best = self._polish_rho(loss, best, k_star, theta_star, prev)
                _, rho, psi = best
                theta = theta_star - rho * psi * k_star
                cur = (theta, rho, psi)
                if not check_essvi_slice(*cur) or (
                    prev is not None and not check_essvi_pair(prev, cur)
                ):
                    raise AssertionError(f"eSSVI constraints violated by accepted slice {cur}")
                return cur, status
            if attempt == 0 and prev is not None:
                theta_star = prev[0] * (1.0 + 1e-6)
                status = "anchor_lifted"
                continue
            break
        return None, "failed:infeasible"

    def fit(self, data: FitInput) -> EssviSurface:
        slices: list[list[float]] = []
        status: list[str] = []
        prev: tuple[float, float, float] | None = None
        for j in range(data.n_slices):
            cur, st = self._fit_slice(
                data.k[j], data.iv_mid[j], data.scale(j), data.expiries[j], prev
            )
            if cur is None:
                slices.append([float("nan")] * 3)
            else:
                slices.append([float(v) for v in cur])
                prev = cur
            status.append(st)
        return EssviSurface(data.expiries, tuple(status), {"slices": slices})

    @staticmethod
    def surface(expiries: tuple[float, ...], slices: list[list[float]]) -> EssviSurface:
        """Surface from ``(theta, rho, psi)`` per slice."""
        return EssviSurface(tuple(expiries), ("ok",) * len(expiries), {"slices": slices})
