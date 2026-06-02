"""Tests for intrinsic latent dynamics."""

from __future__ import annotations

import numpy as np
import pytest

from popsim.latents import (
    ar1_latents,
    generate_latents,
    lds_latents,
    oscillatory_latents,
)


def test_ar1_shape_and_determinism():
    z1 = ar1_latents(500, 4, tau=20.0, rng=0)
    z2 = ar1_latents(500, 4, tau=20.0, rng=0)
    assert z1.shape == (500, 4)
    np.testing.assert_array_equal(z1, z2)


def test_ar1_unit_marginal_variance():
    z = ar1_latents(20000, 3, tau=15.0, rng=1)
    assert np.allclose(z.var(axis=0), 1.0, atol=0.15)


def test_ar1_autocorrelation_matches_tau():
    tau = 25.0
    z = ar1_latents(50000, 1, tau=tau, rng=2)[:, 0]
    z0 = z - z.mean()
    ac1 = np.dot(z0[:-1], z0[1:]) / np.dot(z0, z0)
    assert abs(ac1 - np.exp(-1.0 / tau)) < 0.02


def test_larger_tau_is_smoother():
    z_smooth = ar1_latents(20000, 1, tau=50.0, rng=3)[:, 0]
    z_rough = ar1_latents(20000, 1, tau=5.0, rng=3)[:, 0]
    assert np.abs(np.diff(z_smooth)).mean() < np.abs(np.diff(z_rough)).mean()


def test_lds_shape_and_unit_variance():
    z = lds_latents(20000, 5, rng=0)
    assert z.shape == (20000, 5)
    assert np.allclose(z.var(axis=0), 1.0, atol=0.05)


def test_lds_is_oscillatory():
    # A rotational LDS block should produce a signal whose autocorrelation goes
    # negative at some lag (a smooth AR(1) process never does).
    z = lds_latents(40000, 2, freqs=[0.02], decay=0.005, rng=1)[:, 0]
    z0 = z - z.mean()
    ac = np.correlate(z0, z0, mode="full")[z0.size - 1 :]
    ac = ac / ac[0]
    assert ac[:200].min() < -0.1


def test_oscillatory_shape_and_unit_variance():
    z = oscillatory_latents(2000, 4, rng=0)
    assert z.shape == (2000, 4)
    assert np.allclose(z.var(axis=0), 1.0, atol=1e-6)


def test_generate_latents_dispatch():
    for kind in ("ar1", "lds", "oscillatory"):
        z = generate_latents(kind, 300, 3, rng=0)
        assert z.shape == (300, 3)


def test_generate_latents_unknown_kind_raises():
    with pytest.raises(ValueError):
        generate_latents("banana", 100, 3, rng=0)


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        ar1_latents(0, 3)
    with pytest.raises(ValueError):
        ar1_latents(10, 0)
    with pytest.raises(ValueError):
        ar1_latents(10, 3, tau=-1.0)
