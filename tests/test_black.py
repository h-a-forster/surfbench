"""Black-76 prices and implied-volatility inversion (DESIGN 4.1, 7.1)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from surfbench import black


def test_reference_call_currency_units() -> None:
    assert black.black76(100.0, 100.0, 1.0, 0.2, True) == pytest.approx(
        7.965567455405804, abs=1e-12
    )


def test_reference_coin_call_and_put() -> None:
    c = black.black76(5e4, 6e4, 0.25, 0.6, True) / 5e4
    p = black.black76(5e4, 4e4, 0.25, 0.8, False) / 5e4
    assert c == pytest.approx(0.054405634678143, abs=1e-12)
    assert p == pytest.approx(0.063911835245144, abs=1e-12)


def test_put_call_parity() -> None:
    k = np.linspace(-2, 2, 41)
    s = 0.7
    assert np.allclose(black.call_price(k, s) - black.put_price(k, s), 1 - np.exp(k), atol=1e-14)


def test_vega_identity_and_fd() -> None:
    k = np.linspace(-1.5, 1.5, 31)
    s = 0.4
    v = black.vega_s(k, s)
    d2 = -k / s - s / 2
    assert np.allclose(v, np.exp(k) * np.exp(-0.5 * d2 * d2) / math.sqrt(2 * math.pi), atol=1e-14)
    h = 1e-6
    fd = (black.call_price(k, s + h) - black.call_price(k, s - h)) / (2 * h)
    assert np.allclose(v, fd, atol=1e-8)


def test_delta_signs() -> None:
    d_c = float(black.delta(0.0, 0.5, True))
    d_p = float(black.delta(0.0, 0.5, False))
    assert 0.5 < d_c < 1 and -0.5 < d_p < 0
    assert d_c - d_p == pytest.approx(1.0)


def test_iv_round_trip_grid() -> None:
    # OTM prices (the side the inversion works on): exact to 1e-10 where vega_s > 1e-8.
    k, s = np.meshgrid(np.linspace(-2, 2, 81), np.geomspace(1e-3, 3, 60))
    k, s = k.ravel(), s.ravel()
    price = black.otm_price(k, s)
    s_hat = black.implied_s(price, k, k >= 0)
    mask = black.vega_s(k, s) > 1e-8
    assert mask.sum() > 1000
    assert np.all(np.isfinite(s_hat[mask]))
    err = np.abs(s_hat[mask] - s[mask])
    assert err.max() < 1e-10, err.max()


def test_iv_round_trip_itm_inputs() -> None:
    # ITM inputs lose the time value to rounding; require 1e-10 where it is >= 1e-6.
    k, s = np.meshgrid(np.linspace(-2, 2, 81), np.geomspace(1e-3, 3, 60))
    k, s = k.ravel(), s.ravel()
    itm_price = np.where(k < 0, black.call_price(k, s), black.put_price(k, s))
    s_hat = black.implied_s(itm_price, k, k < 0)
    mask = (black.otm_price(k, s) >= 1e-6) & (black.vega_s(k, s) > 1e-8)
    assert mask.sum() > 1000
    assert np.abs(s_hat[mask] - s[mask]).max() < 1e-10


def test_iv_scales_with_t() -> None:
    p = black.black76(5e4, 6e4, 0.25, 0.6, True) / 5e4
    iv = black.implied_vol(p, math.log(6e4 / 5e4), 0.25, True)
    assert float(iv) == pytest.approx(0.6, abs=1e-12)


@pytest.mark.parametrize(
    ("price", "k", "is_call"),
    [
        (0.0, 0.1, True),
        (-0.01, 0.1, True),
        (1.0, 0.1, True),
        (1.2, 0.1, True),
        (math.exp(-0.2), -0.2, False),
        (0.0, -0.2, False),
        (0.1, -0.5, True),  # below the intrinsic value 1 - e^-0.5 = 0.393
        (float("nan"), 0.0, True),
    ],
)
def test_invalid_prices_give_nan(price: float, k: float, is_call: bool) -> None:
    assert np.isnan(black.implied_s(price, k, is_call))


def test_t_must_be_positive() -> None:
    with pytest.raises(ValueError):
        black.implied_vol(0.05, 0.0, 0.0, True)
    with pytest.raises(ValueError):
        black.black76(1.0, 1.0, 1.0, -0.1, True)


def test_agrees_with_brentq_random() -> None:
    rng = np.random.default_rng(1)
    k = rng.uniform(-1.5, 1.5, 200)
    s = rng.uniform(0.02, 2.0, 200)
    is_call = k >= 0
    price = black.otm_price(k, s)
    ours = black.implied_s(price, k, is_call)
    checked = 0
    for i in range(200):
        f = black.call_price if is_call[i] else black.put_price
        ref = brentq(lambda x, i=i, f=f: float(f(k[i], x)) - price[i], 1e-8, 20, xtol=1e-15)
        if black.vega_s(k[i], s[i]) > 1e-6:
            assert abs(ours[i] - ref) < 1e-9
            checked += 1
    assert checked > 150


def test_itm_prices_use_parity() -> None:
    s = 0.5
    itm_call = black.call_price(-0.3, s)
    assert float(black.implied_s(itm_call, -0.3, True)) == pytest.approx(s, abs=1e-11)
    itm_put = black.put_price(0.3, s)
    assert float(black.implied_s(itm_put, 0.3, False)) == pytest.approx(s, abs=1e-11)


def test_shape_preserved() -> None:
    p = np.full((3, 4), 0.05)
    assert black.implied_s(p, 0.0, True).shape == (3, 4)
