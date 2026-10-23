"""Model protocol and fitted-surface container shared by all models.

Specification: ``docs/DESIGN.md`` section 4.2.

Every model fits a :class:`FitInput` (one underlying, one snapshot, training quotes only)
and returns a :class:`FittedSurface` that evaluates total implied variance ``w`` and its
first two ``k``-derivatives on each listed expiry. All models minimise the same objective:

    L = sum_i ((sigma_model(k_i) - sigma_mid_i) / max(h_i, floor))^2

with ``h_i`` the half bid-ask spread in implied volatility and ``floor = 0.005``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "FD_STEP",
    "STATUS_OK",
    "FitInput",
    "FittedSurface",
    "SurfaceModel",
    "accepted",
    "iv_objective",
    "iv_residuals",
    "w_weights",
]

Float = NDArray[np.float64]
FD_STEP = 1e-4
STATUS_OK = ("ok", "anchor_lifted")
DEFAULT_FLOOR = 0.005


def accepted(status: str) -> bool:
    """True for slice statuses whose fit is used (``ok`` or ``anchor_lifted``)."""
    return status in STATUS_OK


@dataclass(frozen=True)
class FitInput:
    """Training quotes of one underlying at one snapshot, slices in increasing ``T``."""

    expiries: tuple[float, ...]
    k: tuple[Float, ...]
    iv_mid: tuple[Float, ...]
    half_spread: tuple[Float, ...]
    seed: int = 0
    floor: float = DEFAULT_FLOOR

    def __post_init__(self) -> None:
        n = len(self.expiries)
        if not (len(self.k) == len(self.iv_mid) == len(self.half_spread) == n):
            raise ValueError("expiries, k, iv_mid and half_spread must have equal length")
        if n == 0:
            raise ValueError("FitInput needs at least one slice")
        t = np.asarray(self.expiries, dtype=float)
        if np.any(t <= 0) or np.any(np.diff(t) <= 0):
            raise ValueError("expiries must be positive and strictly increasing")
        for j in range(n):
            kj, vj, hj = self.k[j], self.iv_mid[j], self.half_spread[j]
            if not (kj.shape == vj.shape == hj.shape) or kj.ndim != 1:
                raise ValueError(f"slice {j}: k, iv_mid and half_spread must be equal 1-D arrays")
            if kj.size < 3:
                raise ValueError(f"slice {j}: need at least 3 quotes")
            if not (np.all(np.isfinite(kj)) and np.all(np.isfinite(vj)) and np.all(vj > 0)):
                raise ValueError(f"slice {j}: k and iv_mid must be finite, iv_mid positive")
            if np.any(np.diff(kj) < 0):
                raise ValueError(f"slice {j}: k must be sorted")

    @property
    def n_slices(self) -> int:
        return len(self.expiries)

    def scale(self, j: int) -> Float:
        """Residual scale ``max(h, floor)`` of slice ``j``."""
        return np.maximum(np.nan_to_num(self.half_spread[j], nan=0.0), self.floor)

    def subset(self, masks: tuple[NDArray[np.bool_], ...]) -> FitInput:
        """Keep the quotes selected by one boolean mask per slice."""
        return FitInput(
            expiries=self.expiries,
            k=tuple(a[m] for a, m in zip(self.k, masks, strict=True)),
            iv_mid=tuple(a[m] for a, m in zip(self.iv_mid, masks, strict=True)),
            half_spread=tuple(a[m] for a, m in zip(self.half_spread, masks, strict=True)),
            seed=self.seed,
            floor=self.floor,
        )


def w_weights(T: float, iv_mid: Float, scale: Float) -> Float:
    """First-order total-variance weights ``1 / (2 sigma T max(h, floor))``."""
    return 1.0 / (2.0 * iv_mid * T * scale)


def iv_residuals(w_model: Float, T: float, iv_mid: Float, scale: Float) -> Float:
    """Exact IV-space residuals of the common objective."""
    sig = np.sqrt(np.maximum(w_model, 0.0) / T)
    return (sig - iv_mid) / scale


def iv_objective(surface: FittedSurface, data: FitInput) -> float:
    """Common objective ``L`` summed over the accepted slices of ``surface``."""
    total = 0.0
    for j in range(data.n_slices):
        if not accepted(surface.status[j]):
            continue
        r = iv_residuals(surface.w(j, data.k[j]), data.expiries[j], data.iv_mid[j], data.scale(j))
        total += float(np.sum(r * r))
    return total


@dataclass
class FittedSurface:
    """Base class of fitted surfaces.

    Subclasses implement :meth:`_w` (and optionally analytic :meth:`_dw`, :meth:`_d2w`) for
    accepted slices; failed slices evaluate to ``nan``.
    """

    expiries: tuple[float, ...]
    status: tuple[str, ...]
    params: dict[str, Any] = field(default_factory=dict)
    analytic_derivatives: ClassVar[bool] = False

    def _check(self, j: int) -> bool:
        if not 0 <= j < len(self.expiries):
            raise IndexError(f"slice {j} out of range")
        return accepted(self.status[j])

    def _w(self, j: int, k: Float) -> Float:  # pragma: no cover - abstract
        raise NotImplementedError

    def _dw(self, j: int, k: Float) -> Float:
        h = FD_STEP
        return (self._w(j, k + h) - self._w(j, k - h)) / (2.0 * h)

    def _d2w(self, j: int, k: Float) -> Float:
        h = FD_STEP
        return (self._w(j, k + h) - 2.0 * self._w(j, k) + self._w(j, k - h)) / (h * h)

    def _eval(self, fn: Any, j: int, k: ArrayLike) -> Float:
        kk = np.asarray(k, dtype=float)
        if not self._check(j):
            return np.full(kk.shape, np.nan)
        return np.asarray(fn(j, kk), dtype=float)

    def w(self, j: int, k: ArrayLike) -> Float:
        return self._eval(self._w, j, k)

    def dw(self, j: int, k: ArrayLike) -> Float:
        return self._eval(self._dw, j, k)

    def d2w(self, j: int, k: ArrayLike) -> Float:
        return self._eval(self._d2w, j, k)

    def iv(self, j: int, k: ArrayLike) -> Float:
        return np.sqrt(np.maximum(self.w(j, k), 0.0) / self.expiries[j])

    @property
    def n_accepted(self) -> int:
        return sum(accepted(s) for s in self.status)


class SurfaceModel(Protocol):
    name: str

    def fit(self, data: FitInput) -> FittedSurface: ...
