"""Surface models. Each model fits a filtered snapshot and returns a fitted surface that
evaluates total variance and its first two k-derivatives on every listed expiry.

Specification: ``docs/DESIGN.md`` section 4.2 and 4.3.
"""

from __future__ import annotations

from surfbench.models.base import FitInput, FittedSurface, SurfaceModel, accepted
from surfbench.models.baseline import BaselineModel
from surfbench.models.essvi import EssviModel
from surfbench.models.sabr import SabrModel
from surfbench.models.ssvi import SsviModel
from surfbench.models.svi import SviModel

__all__ = [
    "MODEL_NAMES",
    "FitInput",
    "FittedSurface",
    "SurfaceModel",
    "accepted",
    "get_model",
]

MODEL_NAMES = ("baseline", "svi", "ssvi", "essvi", "sabr")


def get_model(name: str, *, sabr_beta: float = 1.0) -> SurfaceModel:
    """Model instance by name (``baseline``, ``svi``, ``ssvi``, ``essvi``, ``sabr``)."""
    if name == "baseline":
        return BaselineModel()
    if name == "svi":
        return SviModel()
    if name == "ssvi":
        return SsviModel()
    if name == "essvi":
        return EssviModel()
    if name == "sabr":
        return SabrModel(beta=sabr_beta)
    raise ValueError(f"unknown model {name!r}; expected one of {', '.join(MODEL_NAMES)}")
