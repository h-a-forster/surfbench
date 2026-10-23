"""Baseline: per-expiry weighted quadratic of implied volatility in log-moneyness.

Specification: ``docs/DESIGN.md`` section 4.3.5. ``sigma(k) = c0 + c1 k + c2 k^2`` by
weighted linear least squares with weights ``1 / max(h, floor)^2`` (closed form), which is
exactly the common objective. Predictions below ``1e-4`` are clipped (and counted).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from surfbench.models.base import FitInput, FittedSurface, Float

__all__ = ["BaselineModel", "BaselineSurface"]

SIGMA_MIN = 1e-4


@dataclass
class BaselineSurface(FittedSurface):
    analytic_derivatives: ClassVar[bool] = True

    def _coef(self, j: int) -> Float:
        return np.asarray(self.params["coef"][j], dtype=float)

    def _sigma(self, j: int, k: Float) -> tuple[Float, Float, Float]:
        c0, c1, c2 = self._coef(j)
        raw = c0 + c1 * k + c2 * k * k
        clipped = raw < SIGMA_MIN
        sig = np.where(clipped, SIGMA_MIN, raw)
        ds = np.where(clipped, 0.0, c1 + 2.0 * c2 * k)
        d2s = np.where(clipped, 0.0, 2.0 * c2)
        return sig, ds, d2s

    def _w(self, j: int, k: Float) -> Float:
        sig, _, _ = self._sigma(j, k)
        return sig * sig * self.expiries[j]

    def _dw(self, j: int, k: Float) -> Float:
        sig, ds, _ = self._sigma(j, k)
        return 2.0 * sig * ds * self.expiries[j]

    def _d2w(self, j: int, k: Float) -> Float:
        sig, ds, d2s = self._sigma(j, k)
        return 2.0 * (ds * ds + sig * d2s) * self.expiries[j]

    def n_clipped(self, j: int, k: Float) -> int:
        c0, c1, c2 = self._coef(j)
        return int(np.count_nonzero(c0 + c1 * k + c2 * k * k < SIGMA_MIN))


class BaselineModel:
    name = "baseline"

    def fit(self, data: FitInput) -> BaselineSurface:
        coefs: list[list[float]] = []
        status: list[str] = []
        for j in range(data.n_slices):
            k = data.k[j]
            sw = 1.0 / data.scale(j)
            X = np.column_stack([np.ones_like(k), k, k * k]) * sw[:, None]
            y = data.iv_mid[j] * sw
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            if np.all(np.isfinite(coef)):
                coefs.append([float(c) for c in coef])
                status.append("ok")
            else:  # pragma: no cover - lstsq on finite input returns finite output
                coefs.append([float("nan")] * 3)
                status.append("failed:lstsq")
        return BaselineSurface(data.expiries, tuple(status), {"coef": coefs})

    @staticmethod
    def surface(expiries: tuple[float, ...], coef: list[list[float]]) -> BaselineSurface:
        return BaselineSurface(tuple(expiries), ("ok",) * len(expiries), {"coef": coef})
