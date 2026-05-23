"""Unit tests for core numerical primitives, against synthetic ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from striatum_cca import config, core

CFG = config.DEFAULT


# ---------------------------------------------------------------------------
# residualise
# ---------------------------------------------------------------------------
def test_residualise_zeroes_trial_mean():
    rng = np.random.default_rng(0)
    tensor = rng.standard_normal((12, 50, 7))
    res = core.residualise(tensor)
    assert res.shape == tensor.shape
    # Per (bin, unit) the mean across trials must be ~0.
    assert np.allclose(res.mean(axis=0), 0.0, atol=1e-12)


def test_residualise_preserves_fluctuations():
    # A pure trial-mean tensor (no trial-to-trial variation) residualises to 0.
    profile = np.random.default_rng(1).standard_normal((1, 50, 5))
    tensor = np.repeat(profile, 8, axis=0)
    assert np.allclose(core.residualise(tensor), 0.0, atol=1e-12)


def test_residualise_rejects_non_3d():
    with pytest.raises(ValueError):
        core.residualise(np.zeros((3, 4)))


# ---------------------------------------------------------------------------
# missing-bin imputation
# ---------------------------------------------------------------------------
def test_impute_fills_nan_with_bin_unit_mean():
    tensor = np.ones((4, 3, 2))
    tensor[1, 0, 0] = 5.0
    tensor[2, 0, 0] = np.nan          # visited trials at (bin0,unit0): 1, 5, 1
    out = core.impute_missing_bins(tensor)
    assert not np.isnan(out).any()
    assert np.isclose(out[2, 0, 0], (1.0 + 5.0 + 1.0) / 3.0)
    assert out[3, 2, 1] == 1.0        # untouched entry unchanged


def test_impute_all_nan_bin_becomes_zero():
    tensor = np.ones((3, 2, 2))
    tensor[:, 1, 0] = np.nan          # bin 1, unit 0 never visited
    out = core.impute_missing_bins(tensor)
    assert np.all(out[:, 1, 0] == 0.0)


def test_impute_then_residualise_zeroes_missing_samples():
    rng = np.random.default_rng(11)
    tensor = rng.standard_normal((8, 5, 3))
    tensor[2, 1, 0] = np.nan
    res = core.residualise(core.impute_missing_bins(tensor))
    assert np.isclose(res[2, 1, 0], 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
def test_pca_full_rank_reconstructs():
    rng = np.random.default_rng(2)
    tensor = rng.standard_normal((10, 50, 8))   # 500 samples > 8 units
    state = core.pca_fit(tensor, k=8)
    scores = core.pca_transform(tensor, state)
    flat = tensor.reshape(500, 8)
    recon = scores.reshape(500, 8) @ state.components.T + state.mean
    assert np.allclose(recon, flat, atol=1e-10)
    assert np.isclose(state.explained_variance_ratio.sum(), 1.0, atol=1e-10)


def test_pca_partial_rank_is_monotone_and_incomplete():
    rng = np.random.default_rng(3)
    tensor = rng.standard_normal((10, 50, 8))
    state = core.pca_fit(tensor, k=3)
    evr = state.explained_variance_ratio
    assert evr.shape == (3,)
    assert np.all(np.diff(evr) <= 1e-12)        # descending
    assert evr.sum() < 1.0                      # 3 of 8 components


# ---------------------------------------------------------------------------
# CCA — correctness against known ground truth
# ---------------------------------------------------------------------------
def test_cca_recovers_known_canonical_correlation():
    rng = np.random.default_rng(4)
    n, rho = 8000, 0.7
    x = rng.standard_normal((n, 4))
    y = rng.standard_normal((n, 4))
    y[:, 0] = rho * x[:, 0] + np.sqrt(1 - rho**2) * rng.standard_normal(n)
    res = core.cca_fit(x, y)
    assert abs(res.r[0] - rho) < 0.03           # CC1 ~ rho
    assert res.r[1] < 0.1                       # remaining dims ~ 0


def test_cca_invariant_to_invertible_linear_mixing():
    # Canonical correlations are unchanged by invertible linear transforms.
    rng = np.random.default_rng(5)
    n = 6000
    x = rng.standard_normal((n, 4))
    y = rng.standard_normal((n, 4))
    y[:, 0] = 0.6 * x[:, 0] + 0.8 * rng.standard_normal(n)
    r_plain = core.cca_fit(x, y).r
    mx = rng.standard_normal((4, 4))
    my = rng.standard_normal((4, 4))
    r_mixed = core.cca_fit(x @ mx, y @ my).r
    assert np.allclose(np.sort(r_plain), np.sort(r_mixed), atol=1e-8)


def test_cca_rejects_rank_deficient_input():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((200, 3))
    x = np.column_stack([x, x[:, 0]])           # duplicated column -> rank 3
    y = rng.standard_normal((200, 4))
    with pytest.raises(ValueError):
        core.cca_fit(x, y)


# ---------------------------------------------------------------------------
# choose_k
# ---------------------------------------------------------------------------
def test_choose_k_samples_rule():
    # 500 samples / 25 = 20, within unit counts and the cap.
    assert core.choose_k(60, 60, 500, CFG) == 20


def test_choose_k_capped_by_units():
    # The smaller area has only 8 units.
    assert core.choose_k(60, 8, 500, CFG) == 8


def test_choose_k_capped_by_k_cap():
    assert core.choose_k(200, 200, 100_000, CFG) == CFG.k_cap


# ---------------------------------------------------------------------------
# trial folds
# ---------------------------------------------------------------------------
def test_trial_folds_partition_all_trials():
    folds = core.trial_folds(10, 5, seed=0)
    assert len(folds) == 5
    combined = np.sort(np.concatenate(folds))
    assert np.array_equal(combined, np.arange(10))
    assert all(len(f) == 2 for f in folds)


# ---------------------------------------------------------------------------
# cross-validated CCA
# ---------------------------------------------------------------------------
def test_cv_in_sample_is_biased_above_held_out_on_noise():
    # Independent data: in-sample CC1 is inflated; held-out CC1 ~ 0.
    rng = np.random.default_rng(7)
    px = rng.standard_normal((10, 50, 20))
    py = rng.standard_normal((10, 50, 20))
    cv = core.cca_cv(px, py, CFG)
    assert cv.in_sample_r[0] > 0.3              # upward-biased
    assert abs(cv.held_out_r[0]) < 0.2          # generalises to ~0
    assert cv.in_sample_r[0] - cv.held_out_r[0] > 0.2


def test_cv_recovers_true_correlation():
    # A genuine shared dimension at correlation rho generalises to held-out.
    rng = np.random.default_rng(8)
    n_tr, n_bin, k, rho = 12, 50, 6, 0.6
    shared = rng.standard_normal((n_tr, n_bin))
    px = rng.standard_normal((n_tr, n_bin, k))
    py = rng.standard_normal((n_tr, n_bin, k))
    px[:, :, 0] = shared
    py[:, :, 0] = rho * shared + np.sqrt(1 - rho**2) * rng.standard_normal((n_tr, n_bin))
    cv = core.cca_cv(px, py, CFG)
    assert abs(cv.held_out_r[0] - rho) < 0.15
    assert cv.n_samples == n_tr * n_bin
