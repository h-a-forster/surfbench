"""Models: reference values, derivatives, recovery and constraints (DESIGN 4.3, 7.1, 7.2)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from surfbench import synthetic
from surfbench.models import MODEL_NAMES, FitInput, accepted, get_model
from surfbench.models.base import iv_objective
from surfbench.models.essvi import (
    EssviModel,
    _psi_interval,
    check_essvi_pair,
    check_essvi_slice,
    essvi_slice,
)
from surfbench.models.sabr import SabrModel, hagan_lognormal_vol
from surfbench.models.ssvi import SsviModel, atm_total_variance, check_ssvi, phi_power
from surfbench.models.svi import SviModel, svi_constraints_ok, svi_raw

SABR_REF = {
    1.0: (0.729816776146, 0.619687500000, 0.637525623611),
    0.5: (0.788289252793, 0.623625000000, 0.595862833063),
}


@pytest.mark.parametrize("beta", [1.0, 0.5])
def test_sabr_reference_values(beta: float) -> None:
    k = np.log([0.7, 1.0, 1.4])
    vols = hagan_lognormal_vol(k, 0.25, 0.6, -0.2, 1.5, beta)
    assert np.allclose(vols, SABR_REF[beta], atol=1e-10, rtol=0)


def test_sabr_small_z_branch_continuous() -> None:
    k = np.array([-1e-9, 0.0, 1e-9, 1e-6])
    v = hagan_lognormal_vol(k, 0.5, 0.6, -0.3, 1.2)
    assert np.all(np.isfinite(v)) and np.ptp(v[:3]) < 1e-9


def test_sabr_extreme_z_is_finite() -> None:
    k = np.linspace(-6, 6, 101)
    for rho in (-0.999, 0.0, 0.999):
        v = hagan_lognormal_vol(k, 0.1, 0.5, rho, 3.0)
        assert np.all(np.isfinite(v)) and np.all(v > 0)


def test_sabr_series_matches_closed_form() -> None:
    # Either side of the |z| = 1e-4 switch the two expressions agree to rounding.
    for rho in (-0.7, 0.0, 0.6):
        for edge in (-1e-4, 1e-4):
            z = edge + np.array([-2e-8, -1e-8, 0.0, 1e-8, 2e-8])
            v = hagan_lognormal_vol(-z * 0.6 / 1.5, 0.25, 0.6, rho, 1.5)
            assert np.abs(np.diff(v, 2)).max() < 1e-13


def test_sabr_beta_validation() -> None:
    with pytest.raises(ValueError):
        SabrModel(beta=1.5)


def test_svi_formula_hand_value() -> None:
    # w(k) at k = m is a + b sigma; slope at k = m is b rho.
    w, w1, w2 = svi_raw(np.array([0.1]), 0.02, 0.1, -0.3, 0.1, 0.2)
    assert w[0] == pytest.approx(0.02 + 0.1 * 0.2, abs=1e-15)
    assert w1[0] == pytest.approx(-0.03, abs=1e-15)
    assert w2[0] == pytest.approx(0.1 / 0.2, abs=1e-12)


def test_ssvi_and_essvi_forms_agree() -> None:
    k = np.linspace(-2, 2, 41)
    theta, rho, eta, gamma = 0.09, -0.35, 1.1, 0.4
    phi = float(phi_power(theta, eta, gamma))
    surf = SsviModel.surface((0.25,), [theta], rho, eta, gamma)
    w_e = essvi_slice(k, theta, rho, theta * phi)[0]
    assert np.allclose(surf.w(0, k), w_e, atol=1e-15)
    # ATM total variance equals theta.
    assert float(surf.w(0, np.array([0.0]))[0]) == pytest.approx(theta, abs=1e-15)


def test_atm_total_variance_interp() -> None:
    k = np.array([-0.2, -0.1, 0.1, 0.3])
    w = np.array([0.05, 0.04, 0.06, 0.08])
    assert atm_total_variance(k, w) == pytest.approx(0.05)
    assert atm_total_variance(np.array([0.1, 0.2]), np.array([1.0, 2.0])) == 1.0
    assert atm_total_variance(np.array([-0.3, -0.1]), np.array([1.0, 2.0])) == 2.0


@pytest.mark.parametrize("family", MODEL_NAMES)
def test_analytic_or_fd_derivatives(family: str) -> None:
    rng = np.random.default_rng(5)
    surf = synthetic.random_surface(family, rng)
    k = np.linspace(-0.8, 0.8, 33)
    h = 1e-5
    for j in range(len(surf.expiries)):
        fd1 = (surf.w(j, k + h) - surf.w(j, k - h)) / (2 * h)
        fd2 = (surf.w(j, k + h) - 2 * surf.w(j, k) + surf.w(j, k - h)) / h**2
        assert np.allclose(surf.dw(j, k), fd1, atol=1e-6)
        assert np.allclose(surf.d2w(j, k), fd2, atol=1e-4)


@pytest.mark.parametrize("family", MODEL_NAMES)
def test_own_family_noise_free_recovery(family: str) -> None:
    rng = np.random.default_rng(11)
    for i in range(3):
        surf = synthetic.random_surface(family, rng)
        snap = synthetic.make_snapshot(surf, seed=i)
        fit = get_model(family).fit(snap.data)
        assert all(accepted(s) for s in fit.status)
        for j in range(snap.data.n_slices):
            err = fit.iv(j, snap.data.k[j]) - snap.true_iv[j]
            rmse_volpt = 100 * float(np.sqrt(np.mean(err**2)))
            assert rmse_volpt < 0.01, (family, i, j, rmse_volpt)


def _assert_constraints(name: str, fit) -> None:
    if name == "ssvi":
        p = fit.params
        check_ssvi(np.array(p["theta"]), p["rho"], p["eta"], p["gamma"])
    elif name == "essvi":
        prev = None
        for st, sl in zip(fit.status, fit.params["slices"], strict=True):
            if not accepted(st):
                continue
            assert check_essvi_slice(*sl)
            if prev is not None:
                assert check_essvi_pair(prev, tuple(sl))
            prev = tuple(sl)
    elif name == "svi":
        for st, sl in zip(fit.status, fit.params["slices"], strict=True):
            if accepted(st):
                assert svi_constraints_ok(*sl)


def _noisy_fits(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    families = list(MODEL_NAMES)
    for i in range(n):
        surf = synthetic.random_surface(families[i % len(families)], rng)
        snap = synthetic.make_snapshot(surf, noise=0.005, spread=0.02, seed=seed + i)
        for name in MODEL_NAMES:
            fit = get_model(name).fit(snap.data)
            assert len(fit.status) == snap.data.n_slices
            _assert_constraints(name, fit)
            assert math.isfinite(iv_objective(fit, snap.data))


def test_noisy_surfaces_fit_without_exceptions() -> None:
    _noisy_fits(5, seed=100)


@pytest.mark.slow
def test_fifty_noisy_surfaces_fit_without_exceptions() -> None:
    _noisy_fits(50, seed=200)


def test_shared_objective_ranks_own_family_first() -> None:
    # On noise-free SVI data the flexible SVI fit cannot be worse than the baseline.
    rng = np.random.default_rng(3)
    snap = synthetic.make_snapshot(synthetic.random_surface("svi", rng), seed=0)
    l_svi = iv_objective(SviModel().fit(snap.data), snap.data)
    l_base = iv_objective(get_model("baseline").fit(snap.data), snap.data)
    assert l_svi <= l_base + 1e-12


def test_essvi_psi_interval_hand_case() -> None:
    # rho = 0, k* = 0: psi_plus = sqrt(4 theta*), capped by 4.
    lo, hi = _psi_interval(0.0, 0.0, 0.04, None)
    assert hi == pytest.approx(0.4, rel=1e-9) and lo > 0
    # A previous slice with larger ATM variance and rho k* = 0 is infeasible.
    assert _psi_interval(0.0, 0.0, 0.04, (0.05, 0.0, 0.1)) is None


def test_essvi_anchor_lifted_when_market_variance_decreases() -> None:
    k = np.linspace(-0.4, 0.4, 9)
    T = (0.1, 0.2)
    iv1 = 0.8 + 0.2 * k**2
    iv2 = np.sqrt((0.8**2 * 0.1 * 0.95) / 0.2) + 0.2 * k**2  # ATM total variance falls
    data = FitInput(T, (k, k), (iv1, iv2), (np.full(9, 0.01), np.full(9, 0.01)))
    fit = EssviModel().fit(data)
    assert fit.status[0] == "ok"
    assert fit.status[1] in ("anchor_lifted", "failed:infeasible")
    if fit.status[1] == "anchor_lifted":
        assert check_essvi_pair(tuple(fit.params["slices"][0]), tuple(fit.params["slices"][1]))


def test_failed_slice_evaluates_to_nan() -> None:
    from surfbench.models.essvi import EssviSurface

    surf = EssviSurface((0.1,), ("failed:infeasible",), {"slices": [[np.nan] * 3]})
    assert np.all(np.isnan(surf.w(0, np.array([0.0, 0.1]))))
    assert surf.n_accepted == 0
    with pytest.raises(IndexError):
        surf.w(1, np.array([0.0]))


def test_fit_input_validation() -> None:
    k = np.array([-0.1, 0.0, 0.1])
    v = np.array([0.5, 0.5, 0.5])
    h = np.full(3, 0.01)
    with pytest.raises(ValueError):
        FitInput((0.2, 0.1), (k, k), (v, v), (h, h))
    with pytest.raises(ValueError):
        FitInput((0.1,), (k[:2],), (v[:2],), (h[:2],))
    with pytest.raises(ValueError):
        FitInput((0.1,), (k[::-1],), (v,), (h,))
    with pytest.raises(ValueError):
        FitInput((0.1,), (k,), (-v,), (h,))
    with pytest.raises(ValueError):
        FitInput((), (), (), ())
    with pytest.raises(ValueError):
        get_model("gp")


def test_baseline_clipping_counted() -> None:
    from surfbench.models.baseline import BaselineModel

    surf = BaselineModel.surface((0.5,), [[0.1, 0.0, -1.0]])
    k = np.array([-1.0, 0.0, 1.0])
    assert surf.n_clipped(0, k) == 2
    assert np.all(surf.iv(0, k) > 0)
