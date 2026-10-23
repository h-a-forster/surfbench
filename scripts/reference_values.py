"""Independent recomputation of the reference values quoted in docs/DESIGN.md section 7.1.

Self-contained (numpy and scipy only, no surfbench imports) so that it can serve as an
oracle for the package implementation. Run: ``uv run python scripts/reference_values.py``.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def black76(F: float, K: float, T: float, sigma: float, call: bool) -> float:
    """Undiscounted Black-76 price in currency units (zero rates)."""
    s = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * s * s) / s
    d2 = d1 - s
    c = F * norm.cdf(d1) - K * norm.cdf(d2)
    return c if call else c - (F - K)


def svi_raw(k: np.ndarray, a: float, b: float, m: float, rho: float, sig: float):
    """Raw SVI total variance and its first two k-derivatives."""
    x = k - m
    r = np.sqrt(x * x + sig * sig)
    w = a + b * (rho * x + r)
    w1 = b * (rho + x / r)
    w2 = b * sig * sig / r**3
    return w, w1, w2


def ssvi_w(k: np.ndarray, theta: float, rho: float, phi: float):
    """SSVI slice total variance and its first two k-derivatives."""
    u = phi * k + rho
    r = np.sqrt(u * u + 1.0 - rho * rho)
    w = 0.5 * theta * (1.0 + rho * phi * k + r)
    w1 = 0.5 * theta * phi * (rho + u / r)
    w2 = 0.5 * theta * phi * phi * (1.0 - rho * rho) / r**3
    return w, w1, w2


def essvi_w(k: np.ndarray, theta: float, rho: float, psi: float):
    """eSSVI slice in (theta, rho, psi) form, psi = theta * phi."""
    return ssvi_w(k, theta, rho, psi / theta)


def g_durrleman(k: np.ndarray, w: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    return (1.0 - k * w1 / (2.0 * w)) ** 2 - 0.25 * w1 * w1 * (1.0 / w + 0.25) + 0.5 * w2


def phi_power(theta: float, eta: float, gamma: float) -> float:
    return eta / (theta**gamma * (1.0 + theta) ** (1.0 - gamma))


def hagan_lognormal(F: float, K: float, T: float, alpha: float, beta: float, rho: float, nu: float):
    """Hagan et al. (2002) eq. (2.17a): lognormal implied volatility."""
    omb = 1.0 - beta
    fk = F * K
    lfk = math.log(F / K)
    fk_b = fk ** (omb / 2.0)
    denom = fk_b * (1.0 + omb**2 / 24.0 * lfk**2 + omb**4 / 1920.0 * lfk**4)
    z = nu / alpha * fk_b * lfk
    if abs(z) < 1e-12:
        zx = 1.0
    else:
        xz = math.log((math.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho))
        zx = z / xz
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


def main() -> None:
    print("Black-76 (undiscounted, zero rates)")
    print("  F=100 K=100 T=1 sigma=0.2 call:", f"{black76(100, 100, 1, 0.2, True):.15f}")
    c = black76(5e4, 6e4, 0.25, 0.6, True) / 5e4
    print("  F=50000 K=60000 T=0.25 sigma=0.6 call/F:", f"{c:.15f}")
    p = black76(5e4, 4e4, 0.25, 0.8, False) / 5e4
    print("  F=50000 K=40000 T=0.25 sigma=0.8 put/F:", f"{p:.15f}")

    k = np.linspace(-1.5, 1.5, 30001)
    w, w1, w2 = svi_raw(k, -0.0410, 0.1331, 0.3586, 0.3060, 0.4153)
    g = g_durrleman(k, w, w1, w2)
    neg = k[g < 0]
    print("Vogt raw SVI (a,b,m,rho,sigma)=(-0.0410,0.1331,0.3586,0.3060,0.4153), T=1")
    print(f"  min g = {g.min():.6f} at k = {k[g.argmin()]:.4f}")
    print(f"  g < 0 on k in [{neg.min():.4f}, {neg.max():.4f}]")

    kk = np.linspace(-3, 3, 6001)
    for eta, rho, gamma in [(1.2, -0.3, 0.4), (1.5, -0.33, 0.5), (1.0, 0.9, 0.2)]:
        gmin = np.inf
        for theta in [0.001, 0.01, 0.05, 0.1, 0.3, 1.0, 3.0]:
            ws = ssvi_w(kk, theta, rho, phi_power(theta, eta, gamma))
            gmin = min(gmin, float(g_durrleman(kk, *ws).min()))
        bound = eta * (1 + abs(rho))
        print(f"SSVI power law eta={eta} rho={rho} gamma={gamma}: eta(1+|rho|)={bound:.3f}")
        print(f"  min g over the theta grid = {gmin:.6f}")

    # Butterfly violation: psi(1+|rho|) > 4 breaks the necessary condition (Lemma 4.2).
    ws = essvi_w(kk, 0.04, -0.7, 2.6)
    print("eSSVI slice theta=0.04 rho=-0.7 psi=2.6 (psi(1+|rho|) = 4.42 > 4)")
    print(f"  min g = {g_durrleman(kk, *ws).min():.6f}")

    # Calendar: the first pair meets the Hendriks-Martini sufficient conditions, the second
    # breaks the |d(rho psi)/d psi| <= 1 condition.
    s1 = (0.02, -0.4, 0.20)
    for s2 in [(0.05, -0.3, 0.30), (0.05, 0.6, 0.30)]:
        ratio = abs((s2[1] * s2[2] - s1[1] * s1[2]) / (s2[2] - s1[2]))
        diff = essvi_w(kk, *s2)[0] - essvi_w(kk, *s1)[0]
        print(f"eSSVI slices {s1} -> {s2}: |d(rho psi)/d psi| = {ratio:.3f}")
        print(f"  min w2 - w1 on [-3, 3] = {diff.min():.6e}")

    print("SABR Hagan (2002) lognormal vol, F=1, T=0.25, alpha=0.6, rho=-0.2, nu=1.5")
    for beta in (1.0, 0.5):
        vals = [hagan_lognormal(1.0, K, 0.25, 0.6, beta, -0.2, 1.5) for K in (0.7, 1.0, 1.4)]
        print(f"  beta={beta}: K=0.7,1.0,1.4 -> " + ", ".join(f"{v:.12f}" for v in vals))


if __name__ == "__main__":
    main()
