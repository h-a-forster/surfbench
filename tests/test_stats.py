"""Bootstrap, sign-flip and Holm against hand-computed cases (DESIGN 6.4)."""

from __future__ import annotations

import numpy as np
import pytest

from surfbench import stats


def test_holm_hand_case() -> None:
    p = [0.01, 0.04, 0.03, 0.005]
    # Sorted: 0.005*4 = 0.02, 0.01*3 = 0.03, 0.03*2 = 0.06, 0.04*1 = 0.04 -> monotone 0.06.
    assert np.allclose(stats.holm(p), [0.03, 0.06, 0.06, 0.02])


def test_holm_caps_and_validates() -> None:
    assert np.allclose(stats.holm([0.5, 0.9]), [1.0, 1.0])
    with pytest.raises(ValueError):
        stats.holm([0.1, float("nan")])
    with pytest.raises(ValueError):
        stats.holm([1.2])


def test_sign_flip_exact_hand_case() -> None:
    # d = (1, 2, 3): |mean| = 2; only (+++) and (---) reach |sum| = 6: p = 2/8.
    assert stats.sign_flip_exact([1.0, 2.0, 3.0]) == pytest.approx(0.25)
    # d = (1, -1): observed mean 0, every sign vector qualifies.
    assert stats.sign_flip_exact([1.0, -1.0]) == pytest.approx(1.0)


def test_sign_flip_random_matches_exact() -> None:
    d = np.array([0.4, 1.1, -0.2, 0.9, 0.7, 1.5, 0.3, -0.1, 0.8, 0.6])
    exact = stats.sign_flip_exact(d)
    p = stats.sign_flip_test(d, n_perm=200_000, seed=3)
    assert abs(p - exact) < 0.003
    assert stats.sign_flip_test(d, n_perm=1000, seed=3) >= 1 / 1001


def test_sign_flip_degenerate() -> None:
    assert stats.sign_flip_test([]) == 1.0
    assert stats.sign_flip_test([0.0, 0.0]) == 1.0
    assert stats.sign_flip_test([float("nan"), 0.0]) == 1.0


def test_sign_flip_is_seeded() -> None:
    d = [0.1, 0.3, -0.2, 0.5, 0.4]
    a = stats.sign_flip_test(d, n_perm=5000, seed=1)
    assert a == stats.sign_flip_test(d, n_perm=5000, seed=1)


def test_bootstrap_mean_constant_and_coverage() -> None:
    iv = stats.bootstrap_mean([2.0] * 10, n_boot=500, seed=1)
    assert (iv.estimate, iv.lo, iv.hi, iv.n) == (2.0, 2.0, 2.0, 10)
    # Seeded Monte Carlo coverage: normal data, n = 30, nominal 95%.
    rng = np.random.default_rng(0)
    cover = 0
    for i in range(200):
        x = rng.normal(1.0, 1.0, 30)
        r = stats.bootstrap_mean(x, n_boot=1000, seed=i)
        cover += r.lo <= 1.0 <= r.hi
    assert 0.88 <= cover / 200 <= 0.99


def test_bootstrap_drops_nan_and_empty() -> None:
    r = stats.bootstrap_mean([1.0, float("nan"), 3.0], n_boot=200, seed=0)
    assert r.n == 2 and r.estimate == 2.0
    assert np.isnan(stats.bootstrap_mean([float("nan")]).estimate)


def test_block_indices_shapes() -> None:
    rng = np.random.default_rng(0)
    idx = stats.block_indices(10, 7, 3, rng)
    assert idx.shape == (7, 10) and idx.min() >= 0 and idx.max() <= 9
    assert np.all(np.diff(idx[:, :3], axis=1) == 1)  # consecutive within a block
    assert np.all(stats.block_indices(2, 4, 3, rng) == [0, 1])
    with pytest.raises(ValueError):
        stats.block_indices(0, 1, 1, rng)


def test_bootstrap_ratio_pooled() -> None:
    r = stats.bootstrap_ratio([1, 0, 2, 0], [4, 4, 0, 2], n_boot=500, seed=0)
    assert r.estimate == pytest.approx(1 / 10)
    assert r.n == 3
    assert 0.0 <= r.lo <= r.estimate <= r.hi <= 0.25


def test_rank_probabilities() -> None:
    v = np.array([[1.0, 2.0, 3.0]] * 5)
    assert np.allclose(stats.rank_probabilities(v, n_boot=100, seed=0), [1, 0, 0])
    tie = np.array([[1.0, 1.0]] * 3)
    assert np.allclose(stats.rank_probabilities(tie, n_boot=50, seed=0), [0.5, 0.5])
    with pytest.raises(ValueError):
        stats.rank_probabilities(np.array([[np.nan, 1.0]]))
