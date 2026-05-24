"""Tests for canonical subspaces and principal angles."""

from __future__ import annotations

import numpy as np

from striatum_cca import subspace


def test_canonical_weights_shape():
    components = np.random.default_rng(0).standard_normal((8, 5))
    coef = np.random.default_rng(1).standard_normal((5, 4))
    w = subspace.canonical_weights(components, coef, d=3)
    assert w.shape == (8, 3)


def test_principal_angles_identical_subspaces_are_zero():
    basis = np.random.default_rng(2).standard_normal((10, 3))
    angles = subspace.principal_angles(basis, basis.copy())
    assert np.allclose(angles, 0.0, atol=1e-9)


def test_principal_angles_orthogonal_subspaces_are_pi_over_2():
    eye = np.eye(6)
    angles = subspace.principal_angles(eye[:, :2], eye[:, 3:5])
    assert np.allclose(angles, np.pi / 2, atol=1e-9)


def test_principal_angles_bounded():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((12, 3))
    b = rng.standard_normal((12, 3))
    angles = subspace.principal_angles(a, b)
    assert angles.shape == (3,)
    assert np.all((angles >= 0) & (angles <= np.pi / 2 + 1e-9))
    assert np.all(np.diff(angles) >= -1e-9)            # ascending


def test_split_half_angle_small_for_stable_subspace():
    # A strong, trial-stable shared structure -> either half recovers the same
    # canonical subspace -> small split-half angle.
    rng = np.random.default_rng(4)
    n_tr, n_bins, k = 16, 50, 6
    latent = rng.standard_normal((n_tr, n_bins, 2))
    load_x = rng.standard_normal((2, k))
    load_y = rng.standard_normal((2, k))
    sx = latent @ load_x + 0.15 * rng.standard_normal((n_tr, n_bins, k))
    sy = latent @ load_y + 0.15 * rng.standard_normal((n_tr, n_bins, k))
    comp = np.eye(k)
    ang_x, _ = subspace.split_half_angles(sx, sy, comp, comp, d=1, n_splits=8)
    assert ang_x[0] < np.pi / 4          # well below orthogonal
