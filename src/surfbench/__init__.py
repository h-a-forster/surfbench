"""surfbench: a benchmark of arbitrage-free implied volatility surface fits on crypto options.

Conventions (used everywhere, see ``docs/DESIGN.md`` section 4): log-forward moneyness
``k = ln(K / F)``, time to expiry ``T`` in years (ACT/365, 24/7 calendar), total implied
variance ``w = sigma**2 * T``, volatilities as decimals (0.55, not 55), zero rates. Deribit
inverse option prices are in coin units: ``price = Black76(F, K, T, sigma) / F``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("surfbench")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0"

__all__ = ["__version__"]
