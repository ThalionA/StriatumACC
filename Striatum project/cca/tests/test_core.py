"""Unit tests for core numerical primitives, against synthetic ground truth."""

from __future__ import annotations

import dataclasses

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
# missing-data handling (NaN samples are dropped, not imputed)
# ---------------------------------------------------------------------------
def test_residualise_preserves_nan():
    rng = np.random.default_rng(11)
    tensor = rng.standard_normal((8, 5, 3))
    tensor[2, 1, :] = np.nan                       # unvisited (trial, bin)
    res = core.residualise(tensor)
    assert np.all(np.isnan(res[2, 1, :]))          # missing sample stays NaN
    assert np.all(np.isfinite(res[0, 0, :]))       # observed samples finite


def test_n_valid_samples_counts_finite_trial_bins():
    tensor = np.zeros((4, 5, 3))                   # 20 (trial, bin) samples
    tensor[1, 2, :] = np.nan
    tensor[3, 0, :] = np.nan
    assert core.n_valid_samples(tensor) == 18


def test_cca_fit_drops_missing_rows():
    # cca_fit on data with NaN rows must equal cca_fit on the clean subset.
    rng = np.random.default_rng(14)
    n, rho = 4000, 0.6
    x = rng.standard_normal((n, 4))
    y = rng.standard_normal((n, 4))
    y[:, 0] = rho * x[:, 0] + np.sqrt(1 - rho**2) * rng.standard_normal(n)
    r_clean = core.cca_fit(x, y).r
    x_nan, y_nan = x.copy(), y.copy()
    x_nan[::10] = np.nan                           # 10% missing rows
    r_dropped = core.cca_fit(x_nan, y_nan).r
    keep = np.arange(n) % 10 != 0
    r_subset = core.cca_fit(x[keep], y[keep]).r
    assert np.allclose(r_dropped, r_subset, atol=1e-9)
    assert abs(r_dropped[0] - r_clean[0]) < 0.03


def test_pca_fit_handles_missing_rows():
    rng = np.random.default_rng(15)
    tensor = rng.standard_normal((10, 50, 6))
    tensor[3, 7, :] = np.nan
    state = core.pca_fit(tensor, k=6)
    assert np.all(np.isfinite(state.components))
    assert np.all(np.isfinite(state.mean))


# ---------------------------------------------------------------------------
# zscore_units
# ---------------------------------------------------------------------------
def test_zscore_units_gives_unit_variance():
    rng = np.random.default_rng(12)
    tensor = rng.standard_normal((8, 30, 4)) * np.array([1.0, 5.0, 0.2, 3.0])
    z = core.zscore_units(tensor)
    assert np.allclose(z.std(axis=(0, 1)), 1.0, atol=1e-9)


def test_zscore_units_leaves_constant_unit_finite():
    rng = np.random.default_rng(13)
    tensor = rng.standard_normal((6, 10, 3))
    tensor[:, :, 1] = 2.0                          # constant (zero-variance)
    z = core.zscore_units(tensor)
    assert np.all(np.isfinite(z))
    assert np.allclose(z[:, :, 1], 2.0)            # zero-std unit divided by 1


def test_zscore_units_preserves_nan():
    rng = np.random.default_rng(16)
    tensor = rng.standard_normal((6, 10, 3))
    tensor[2, 4, :] = np.nan
    z = core.zscore_units(tensor)
    assert np.all(np.isnan(z[2, 4, :]))


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


def test_cca_handles_rank_deficient_input():
    # A duplicated column makes x rank 3 (of 4). cca_fit must not error; it
    # returns min(rank_x, rank_y) = 3 finite canonical correlations.
    rng = np.random.default_rng(6)
    x = rng.standard_normal((200, 3))
    x = np.column_stack([x, x[:, 0]])           # duplicated column -> rank 3
    y = rng.standard_normal((200, 4))
    res = core.cca_fit(x, y)
    assert res.r.shape == (3,)
    assert np.all(np.isfinite(res.r))
    assert np.all((res.r >= 0) & (res.r <= 1))


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


def test_k_for_variance_counts_dominant_components():
    # Rank-2 data + tiny noise: ~all variance is in 2 components.
    rng = np.random.default_rng(0)
    latents = rng.standard_normal((500, 2))
    loadings = rng.standard_normal((2, 8))
    data = latents @ loadings + 0.001 * rng.standard_normal((500, 8))
    assert core.k_for_variance(data, 0.90) == 2
    assert core.k_for_variance(data, 0.999) == 2


def test_choose_k_fixed_mode():
    cfg = dataclasses.replace(config.DEFAULT, k_mode="fixed", k_fixed=7)
    assert core.choose_k(50, 50, 100_000, cfg, max_rank=40) == 7
    assert core.choose_k(5, 50, 100_000, cfg, max_rank=40) == 5   # unit-capped


def test_choose_k_variance_mode_uses_variance_k():
    cfg = dataclasses.replace(config.DEFAULT, k_mode="variance")
    assert core.choose_k(50, 50, 100_000, cfg, max_rank=40, variance_k=6) == 6
    assert core.choose_k(50, 50, 100_000, cfg, max_rank=4, variance_k=6) == 4


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


def test_cca_cv_with_missing_samples():
    # cca_cv must run with missing (NaN) (trial, bin) samples and still
    # recover a genuine shared correlation from the observed ones.
    rng = np.random.default_rng(17)
    n_tr, n_bin, k, rho = 12, 50, 6, 0.6
    shared = rng.standard_normal((n_tr, n_bin))
    px = rng.standard_normal((n_tr, n_bin, k))
    py = rng.standard_normal((n_tr, n_bin, k))
    px[:, :, 0] = shared
    py[:, :, 0] = rho * shared + np.sqrt(1 - rho**2) * rng.standard_normal((n_tr, n_bin))
    miss = rng.random((n_tr, n_bin)) < 0.25        # 25% missing (trial,bin)
    px[miss] = np.nan
    py[miss] = np.nan
    cv = core.cca_cv(px, py, CFG)
    assert np.isfinite(cv.held_out_r[0])
    assert abs(cv.held_out_r[0] - rho) < 0.2
    assert cv.n_samples == int((~miss).sum())


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
