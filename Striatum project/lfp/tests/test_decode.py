"""Ground-truth tests for the decoding / CCA primitives."""

import numpy as np

from striatum_lfp.decode import (
    bin_by_position,
    heldout_cca,
    residualise_by_group,
    ridge_cv_decode,
)


def test_bin_by_position_averages_within_bins():
    vals = np.array([[0.0], [2.0], [10.0], [12.0]])
    pos = np.array([10.0, 30.0, 60.0, 90.0])  # -> bins 0,0,1,1 over 0..100, 2 bins
    out = bin_by_position(vals, pos, n_bins=2, pos_max=100)
    np.testing.assert_allclose(out[:, 0], [1.0, 11.0])


def test_bin_by_position_empty_bin_is_nan():
    out = bin_by_position(np.array([[5.0]]), np.array([5.0]), n_bins=3, pos_max=100)
    assert np.isnan(out[1, 0]) and np.isnan(out[2, 0])


def test_ridge_recovers_a_linear_position_code():
    rng = np.random.default_rng(0)
    n_trials, n_bins = 40, 20
    pos = np.tile(np.linspace(0, 200, n_bins), n_trials)
    groups = np.repeat(np.arange(n_trials), n_bins)
    # two features linearly related to position + noise
    X = np.column_stack([pos, 200 - pos]) + rng.normal(0, 8, (len(pos), 2))
    r2, mae, _ = ridge_cv_decode(X, pos, groups)
    assert r2 > 0.9 and mae < 15


def test_ridge_at_chance_for_unrelated_features():
    rng = np.random.default_rng(1)
    n = 600
    y = rng.uniform(0, 200, n)
    X = rng.normal(size=(n, 3))
    groups = np.repeat(np.arange(60), 10)
    r2, _, _ = ridge_cv_decode(X, y, groups)
    assert r2 < 0.1  # no information -> ~0 or negative


def test_residualise_removes_group_mean():
    vals = np.array([[1.0], [3.0], [10.0], [14.0]])
    grp = np.array([0, 0, 1, 1])
    out = residualise_by_group(vals, grp)
    np.testing.assert_allclose(out[:, 0], [-1.0, 1.0, -2.0, 2.0])


def test_heldout_cca_detects_shared_latent_and_rejects_independent():
    rng = np.random.default_rng(2)
    n = 400
    latent = rng.normal(size=(n, 1))
    A = latent @ rng.normal(size=(1, 6)) + 0.3 * rng.normal(size=(n, 6))
    B = latent @ rng.normal(size=(1, 5)) + 0.3 * rng.normal(size=(n, 5))
    shared = heldout_cca(A, B, k=3)
    indep = heldout_cca(rng.normal(size=(n, 6)), rng.normal(size=(n, 5)), k=3)
    assert shared > 0.7
    assert indep < 0.3
