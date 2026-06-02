"""Tests for the population observation model."""

from __future__ import annotations

import numpy as np
import pytest

from popsim.observation import project_population, random_loadings


def test_loadings_unit_norm_columns():
    W = random_loadings(50, 4, rng=0)
    assert W.shape == (50, 4)
    np.testing.assert_allclose(np.linalg.norm(W, axis=0), 1.0)


def test_gaussian_shape_and_determinism():
    z = np.random.default_rng(0).standard_normal((300, 3))
    W = random_loadings(20, 3, rng=1)
    x1 = project_population(z, W, model="gaussian", rng=5)
    x2 = project_population(z, W, model="gaussian", rng=5)
    assert x1.shape == (300, 20)
    np.testing.assert_array_equal(x1, x2)


def test_gaussian_snr_is_respected():
    z = np.random.default_rng(0).standard_normal((20000, 2))
    W = random_loadings(1, 2, rng=2)
    snr = 4.0
    x = project_population(z, W, model="gaussian", snr=snr, rng=3)
    signal = (z @ W.T)[:, 0]
    noise = x[:, 0] - signal
    assert abs(signal.var() / noise.var() - snr) / snr < 0.1


def test_poisson_returns_nonneg_integer_counts():
    z = np.random.default_rng(0).standard_normal((500, 3))
    W = random_loadings(10, 3, rng=1)
    x = project_population(z, W, model="poisson", baseline=2.0, rng=4)
    assert x.dtype == np.int64
    assert np.all(x >= 0)
    assert x.sum() > 0


def test_unknown_model_raises():
    W = random_loadings(5, 2, rng=0)
    with pytest.raises(ValueError):
        project_population(np.zeros((10, 2)), W, model="banana")
