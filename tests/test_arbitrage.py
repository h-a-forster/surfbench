"""Static-arbitrage diagnostics (DESIGN 4.4, 7.1, M3 done-check)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from surfbench import arbitrage, synthetic
from surfbench.arbitrage import ArbitrageConfig
from surfbench.models import get_model
from surfbench.models.essvi import EssviModel
from surfbench.models.ssvi import SsviModel
from surfbench.models.svi import SviModel

ROOT = Path(__file__).resolve().parents[1]


def _oracle():
    spec = importlib.util.spec_from_file_location(
        "reference_values", ROOT / "scripts" / "reference_values.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["reference_values"] = mod
    spec.loader.exec_module(mod)
    return mod


REF = _oracle()
VOGT = (-0.0410, 0.1331, 0.3060, 0.3586, 0.4153)  # (a, b, rho, m, sigma)


def _g(surface, j, k):
    return arbitrage.durrleman_g(k, surface.w(j, k), surface.dw(j, k), surface.d2w(j, k))


def test_vogt_minimum_and_interval() -> None:
    surf = SviModel.surface((1.0,), [list(VOGT)])
    k = np.linspace(-1.5, 1.5, 30001)
    g = _g(surf, 0, k)
    assert g.min() == pytest.approx(-0.032864, abs=5e-7)
    assert k[g.argmin()] == pytest.approx(0.8793, abs=1e-4)
    neg = k[g < 0]
    assert neg.min() == pytest.approx(0.6425, abs=1e-4)
    assert neg.max() == pytest.approx(1.2569, abs=1e-4)
    # Independent oracle on the same grid.
    w, w1, w2 = REF.svi_raw(k, -0.0410, 0.1331, 0.3586, 0.3060, 0.4153)
    assert np.allclose(g, REF.g_durrleman(k, w, w1, w2), atol=1e-12, rtol=0)


def test_vogt_flagged_by_butterfly_slice() -> None:
    surf = SviModel.surface((1.0,), [list(VOGT)])
    res = arbitrage.butterfly_slice(surf, 0, np.linspace(-0.5, 1.5, 20))
    assert res.violated_wide and res.violated_inrange and res.price_violated_wide
    assert res.interval_lo is not None and 0.6 < res.interval_lo < 0.7
    assert res.interval_hi is not None and 1.2 < res.interval_hi < 1.3


@pytest.mark.parametrize(
    ("eta", "rho", "gamma", "ref"),
    [(1.2, -0.3, 0.4, 0.257777), (1.5, -0.33, 0.5, 0.250849), (1.0, 0.9, 0.2, 0.290163)],
)
def test_admissible_ssvi_is_clean(eta: float, rho: float, gamma: float, ref: float) -> None:
    thetas = [0.001, 0.01, 0.05, 0.1, 0.3, 1.0, 3.0]
    T = tuple(np.linspace(0.01, 2.0, len(thetas)))
    surf = SsviModel.surface(T, thetas, rho, eta, gamma)
    kk = np.linspace(-3, 3, 6001)
    gmin = min(float(_g(surf, j, kk).min()) for j in range(len(thetas)))
    assert gmin == pytest.approx(ref, abs=5e-7)
    diag = arbitrage.diagnose(surf, [kk[::100]] * len(thetas))
    assert diag["n_bfly_wide"] == 0 and diag["n_cal"] == 0 and diag["n_price_wide"] == 0


def test_essvi_butterfly_violation() -> None:
    surf = EssviModel.surface((0.1,), [[0.04, -0.7, 2.6]])
    kk = np.linspace(-3, 3, 6001)
    assert float(_g(surf, 0, kk).min()) == pytest.approx(-11.930666, abs=5e-7)
    assert arbitrage.butterfly_slice(surf, 0, kk[::500]).violated_wide


@pytest.mark.parametrize(
    ("s2", "ref", "violated"),
    [((0.05, -0.3, 0.3), 2.809898e-2, False), ((0.05, 0.6, 0.3), -2.436786e-1, True)],
)
def test_essvi_calendar(s2: tuple[float, float, float], ref: float, violated: bool) -> None:
    surf = EssviModel.surface((0.1, 0.2), [[0.02, -0.4, 0.2], list(s2)])
    kk = np.linspace(-3, 3, 6001)
    gap = surf.w(1, kk) - surf.w(0, kk)
    assert gap.min() == pytest.approx(ref, abs=5e-9)
    flag, _ = arbitrage.calendar_pair(
        surf, 0, 1, ArbitrageConfig(grid_z=(-3 / np.sqrt(0.05), 3 / np.sqrt(0.05)), n_grid=6001)
    )
    assert flag is violated


def test_g_and_price_check_agree_on_random_svi() -> None:
    rng = np.random.default_rng(42)
    checked = disagree = 0
    for _ in range(200):
        a = rng.uniform(-0.05, 0.05)
        b = rng.uniform(0.02, 0.6)
        rho = rng.uniform(-0.9, 0.9)
        m = rng.uniform(-0.3, 0.5)
        sig = rng.uniform(0.02, 0.6)
        if a + b * sig * np.sqrt(1 - rho * rho) <= 0.005 or b * (1 + abs(rho)) > 2:
            continue
        surf = SviModel.surface((1.0,), [[a, b, rho, m, sig]])
        k = np.linspace(-1.5, 1.5, 3001)
        g = _g(surf, 0, k)
        _, second = arbitrage.price_second_differences(surf, 0, k)
        # Density = second derivative in K; sign must match g where the density is not tiny.
        w = surf.w(0, k)[1:-1]
        s = np.sqrt(w)
        d2 = -k[1:-1] / s - s / 2
        scale = np.exp(-0.5 * d2 * d2) / np.sqrt(2 * np.pi * w) / np.exp(k[1:-1])
        band = (np.abs(g[1:-1]) > 1e-3) & (scale > 1e-4)
        checked += 1
        disagree += int(np.any(np.sign(g[1:-1][band]) != np.sign(second[band])))
    assert checked >= 100
    assert disagree == 0


def test_density_matches_g_formula() -> None:
    # Breeden-Litzenberger: d2c/dK2 = g(k) n(d2) / (K sqrt(w)) at K = e^k.
    surf = SviModel.surface((1.0,), [list(VOGT)])
    k = np.linspace(-1.0, 1.5, 2501)
    _, second = arbitrage.price_second_differences(surf, 0, k)
    w = surf.w(0, k)[1:-1]
    s = np.sqrt(w)
    d2 = -k[1:-1] / s - s / 2
    dens = (
        _g(surf, 0, k)[1:-1] * np.exp(-0.5 * d2 * d2) / np.sqrt(2 * np.pi) / (np.exp(k[1:-1]) * s)
    )
    assert np.allclose(second, dens, atol=2e-5)


@pytest.mark.parametrize("family", ["svi", "ssvi", "essvi", "baseline"])
def test_fd_derivatives_match_analytic(family: str) -> None:
    from surfbench.models.base import FittedSurface

    surf = synthetic.random_surface(family, np.random.default_rng(9))
    k = np.linspace(-0.6, 0.6, 25)
    for j in range(len(surf.expiries)):
        fd1 = FittedSurface._dw(surf, j, k)
        fd2 = FittedSurface._d2w(surf, j, k)
        assert np.allclose(fd1, surf.dw(j, k), atol=1e-6, rtol=0)
        assert np.allclose(fd2, surf.d2w(j, k), atol=1e-6, rtol=0)


def test_noisy_ssvi_essvi_fits_have_no_violations() -> None:
    rng = np.random.default_rng(77)
    for i in range(6):
        surf = synthetic.random_surface(["ssvi", "essvi", "svi"][i % 3], rng)
        snap = synthetic.make_snapshot(surf, noise=0.005, spread=0.02, seed=i)
        for name in ("ssvi", "essvi"):
            fit = get_model(name).fit(snap.data)
            diag = arbitrage.diagnose(fit, snap.data.k)
            assert diag["n_bfly_wide"] == 0, (name, i, diag["min_g_wide"])
            assert diag["n_cal"] == 0, (name, i, diag["worst_cal_gap"])
            assert diag["n_price_wide"] == 0


def test_nonpositive_variance_is_a_violation() -> None:
    surf = SviModel.surface((1.0,), [[-0.05, 0.1, 0.0, 0.0, 0.1]])  # w(0) = -0.04
    res = arbitrage.butterfly_slice(surf, 0, np.array([-0.1, 0.0, 0.1]))
    assert res.violated_wide and res.nonpositive_w


def test_diagnose_skips_failed_slices() -> None:
    from surfbench.models.essvi import EssviSurface

    surf = EssviSurface(
        (0.1, 0.2, 0.3),
        ("ok", "failed:infeasible", "ok"),
        {"slices": [[0.02, -0.4, 0.2], [np.nan] * 3, [0.05, -0.3, 0.3]]},
    )
    diag = arbitrage.diagnose(surf, [np.array([-0.1, 0.0, 0.1])] * 3)
    assert diag["n_slices"] == 2 and diag["n_pairs"] == 1 and diag["n_cal"] == 0


@pytest.mark.slow
def test_fifty_noisy_fits_ssvi_essvi_clean() -> None:
    rng = np.random.default_rng(2024)
    for i in range(50):
        surf = synthetic.random_surface(["ssvi", "essvi", "svi", "sabr", "baseline"][i % 5], rng)
        snap = synthetic.make_snapshot(surf, noise=0.005, spread=0.02, seed=1000 + i)
        for name in ("ssvi", "essvi"):
            diag = arbitrage.diagnose(get_model(name).fit(snap.data), snap.data.k)
            assert diag["n_bfly_wide"] == diag["n_cal"] == diag["n_price_wide"] == 0, (name, i)


def test_calendar_pair_flags_a_tiny_crossing() -> None:
    base = [0.01, 0.1, -0.3, 0.0, 0.2]
    below = [base[0] - 1e-6, *base[1:]]
    above = [base[0] + 1e-6, *base[1:]]
    flag, worst = arbitrage.calendar_pair(SviModel.surface((0.1, 0.2), [base, below]), 0, 1)
    assert flag and worst == pytest.approx(-1e-6, rel=1e-6)
    flag, worst = arbitrage.calendar_pair(SviModel.surface((0.1, 0.2), [base, above]), 0, 1)
    assert not flag and worst == pytest.approx(1e-6, rel=1e-6)


def test_price_check_flags_monotonicity_without_convexity_failure() -> None:
    # A quadratic-in-k volatility grows so fast that call prices rise again far right of the
    # money: the first difference turns positive while second differences stay >= 0 and
    # Durrleman's g stays non-negative on the wide grid.
    surf = get_model("baseline").surface((0.1,), [[0.5, 0.0, 2.0]])
    kq = np.array([-0.1, 0.1])
    cfg = arbitrage.DEFAULT_CONFIG
    theta = float(surf.w(0, np.array([0.0]))[0])
    first, second = arbitrage.price_second_differences(surf, 0, arbitrage.wide_grid(theta, cfg, kq))
    assert first.max() > cfg.price_tol and second.min() >= -cfg.price_tol
    rec = arbitrage.butterfly_slice(surf, 0, kq)
    assert rec.price_violated_wide and not rec.violated_wide
