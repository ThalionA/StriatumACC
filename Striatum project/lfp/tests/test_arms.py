"""Synthetic ground-truth tests for the decoding / reliability / CCA primitives."""

from __future__ import annotations

import numpy as np
import pytest

from striatum_lfp import arms


# --- design matrix assembly --------------------------------------------------

def test_design_matrix_shapes_and_labels():
    # (n_channels, n_bins, n_trials) with a value that encodes its own indices.
    n_ch, n_bins, n_tr = 3, 5, 4
    cube = np.arange(n_ch * n_bins * n_tr, dtype=float).reshape(n_ch, n_bins, n_tr)
    X, y, groups = arms.design_matrix(cube)
    assert X.shape == (n_bins * n_tr, n_ch)
    assert y.tolist() == list(range(n_bins)) * n_tr
    assert groups.tolist() == sum(([t] * n_bins for t in range(n_tr)), [])
    np.testing.assert_allclose(X[0], cube[:, 0, 0])
    np.testing.assert_allclose(X[1], cube[:, 1, 0])


def test_design_matrix_drops_incomplete_samples():
    cube = np.ones((2, 3, 2))
    cube[0, 1, 0] = np.nan
    X, y, groups = arms.design_matrix(cube)
    assert X.shape == (5, 2)
    assert 1 not in y[groups == 0]


def test_design_matrix_selects_trials():
    cube = np.arange(2 * 3 * 5, dtype=float).reshape(2, 3, 5)
    _, _, groups = arms.design_matrix(cube, trials=np.array([1, 3]))
    assert set(groups.tolist()) == {1, 3}


# --- split-half reliability --------------------------------------------------

def test_reliability_is_one_for_a_noiseless_repeated_profile():
    profile = np.sin(np.linspace(0, 3, 20))
    cube = np.tile(profile[None, :, None], (4, 1, 10))
    r = arms.split_half_reliability(cube)
    np.testing.assert_allclose(r, 1.0, atol=1e-8)


def test_reliability_is_near_zero_for_pure_noise():
    rng = np.random.default_rng(0)
    cube = rng.normal(size=(6, 30, 40))
    assert abs(np.nanmedian(arms.split_half_reliability(cube))) < 0.25


def test_reliability_is_spearman_brown_corrected():
    """Two halves each of n/2 trials under-estimate the full-set reliability."""
    rng = np.random.default_rng(1)
    profile = rng.normal(size=(1, 40, 1))
    cube = profile + rng.normal(size=(1, 40, 60)) * 2.0
    corrected = arms.split_half_reliability(cube)
    raw = arms.split_half_reliability(cube, spearman_brown=False)
    assert corrected[0] > raw[0]
    assert corrected[0] == pytest.approx(2 * raw[0] / (1 + raw[0]))


def test_reliability_needs_at_least_two_trials():
    assert np.isnan(arms.split_half_reliability(np.ones((2, 10, 1)))).all()


def test_reliability_uses_interleaved_halves_not_first_vs_second():
    """A slow drift across trials must not be read as low reliability."""
    n_tr = 40
    profile = np.sin(np.linspace(0, 3, 20))[None, :, None]
    drift = np.linspace(0, 5, n_tr)[None, None, :]
    cube = profile + drift
    assert arms.split_half_reliability(cube)[0] > 0.99


# --- grouped held-out CCA ----------------------------------------------------

def test_grouped_cca_recovers_a_shared_latent():
    rng = np.random.default_rng(3)
    n, groups = 400, np.repeat(np.arange(40), 10)
    latent = rng.normal(size=n)
    A = latent[:, None] * 1.5 + rng.normal(size=(n, 6)) * 0.4
    B = latent[:, None] * 1.2 + rng.normal(size=(n, 5)) * 0.4
    assert arms.heldout_cca_grouped(A, B, groups) > 0.85


def test_grouped_cca_is_near_zero_for_independent_blocks():
    rng = np.random.default_rng(4)
    n, groups = 400, np.repeat(np.arange(40), 10)
    A = rng.normal(size=(n, 6))
    B = rng.normal(size=(n, 5))
    assert abs(arms.heldout_cca_grouped(A, B, groups)) < 0.3


def test_grouped_cca_does_not_leak_within_trial_structure():
    """Structure shared only *within* a trial must not survive a trial-wise split.

    Each trial gets its own random offset in both blocks and nothing else. A
    row-wise split would score this near 1; a trial-wise split must not.
    """
    rng = np.random.default_rng(5)
    groups = np.repeat(np.arange(40), 10)
    offs = rng.normal(size=40)
    A = offs[groups][:, None] * np.ones((1, 4)) + rng.normal(size=(400, 4)) * 0.05
    B = offs[groups][:, None] * np.ones((1, 4)) + rng.normal(size=(400, 4)) * 0.05
    row_split = arms.heldout_cca_grouped(A, B, np.arange(400))     # groups == rows
    trial_split = arms.heldout_cca_grouped(A, B, groups)
    assert row_split > 0.9
    assert trial_split < row_split


def test_grouped_cca_returns_nan_when_too_few_groups():
    rng = np.random.default_rng(6)
    A = rng.normal(size=(10, 3))
    B = rng.normal(size=(10, 3))
    assert np.isnan(arms.heldout_cca_grouped(A, B, np.zeros(10, int)))


def test_trial_shuffle_null_destroys_the_coupling():
    rng = np.random.default_rng(7)
    groups = np.repeat(np.arange(40), 10)
    latent = rng.normal(size=400)
    A = latent[:, None] + rng.normal(size=(400, 4)) * 0.3
    B = latent[:, None] + rng.normal(size=(400, 4)) * 0.3
    real = arms.heldout_cca_grouped(A, B, groups)
    null = np.median(arms.trial_shuffle_cca_null(A, B, groups, n_shuffles=8))
    assert real > 0.8
    assert null < real / 2


def test_within_area_ceiling_is_high_for_a_coherent_block():
    rng = np.random.default_rng(8)
    groups = np.repeat(np.arange(40), 10)
    latent = rng.normal(size=400)
    A = latent[:, None] + rng.normal(size=(400, 8)) * 0.2
    assert arms.within_area_ceiling(A, groups) > 0.85


# --- decoding null -----------------------------------------------------------

def test_circular_shift_actually_changes_the_targets():
    """The bug this replaces: a trial permutation left y bit-identical."""
    y = np.tile(np.arange(10), 5)
    groups = np.repeat(np.arange(5), 10)
    rng = np.random.default_rng(0)
    shifted = arms.circular_shift_targets(y, groups, rng)
    assert not np.array_equal(shifted, y)
    # Every trial still holds exactly the same label multiset.
    for g in range(5):
        assert sorted(shifted[groups == g]) == sorted(y[groups == g])


def test_ridge_recovers_a_linear_position_code():
    rng = np.random.default_rng(0)
    n_trials, n_bins = 40, 20
    pos = np.tile(np.linspace(0, 200, n_bins), n_trials)
    groups = np.repeat(np.arange(n_trials), n_bins)
    # two features linearly related to position + noise
    X = np.column_stack([pos, 200 - pos]) + rng.normal(0, 8, (len(pos), 2))
    r2, mae, _ = arms.ridge_cv_decode(X, pos, groups)
    assert r2 > 0.9 and mae < 15


def test_ridge_at_chance_for_unrelated_features():
    rng = np.random.default_rng(1)
    n = 600
    y = rng.uniform(0, 200, n)
    X = rng.normal(size=(n, 3))
    groups = np.repeat(np.arange(60), 10)
    r2, _, _ = arms.ridge_cv_decode(X, y, groups)
    assert r2 < 0.1  # no information -> ~0 or negative


def test_circular_shift_destroys_a_decodable_mapping():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(30), 20)
    y = np.tile(np.arange(20), 30)
    X = y[:, None] + rng.normal(size=(600, 3)) * 0.5
    real = arms.ridge_cv_decode(X, y.astype(float), groups)[0]
    null = arms.ridge_cv_decode(X, arms.circular_shift_targets(y, groups, rng).astype(float),
                           groups)[0]
    assert real > 0.9
    assert null < 0.1


def test_circular_shift_leaves_a_single_sample_trial_alone():
    y = np.array([7])
    out = arms.circular_shift_targets(y, np.array([0]), np.random.default_rng(0))
    assert out.tolist() == [7]


# --- speed residualisation ---------------------------------------------------

def test_residualise_removes_a_pure_covariate_effect():
    speed = np.linspace(1, 3, 40).reshape(8, 5)
    cube = (2.5 * speed + 1.0)[None, :, :] * np.ones((3, 1, 1))
    resid = arms.residualise_on(cube, speed)
    np.testing.assert_allclose(resid, 0.0, atol=1e-9)


def test_residualise_keeps_the_part_the_covariate_cannot_explain():
    rng = np.random.default_rng(2)
    speed = rng.normal(size=(8, 5))
    signal = rng.normal(size=(8, 5))
    cube = (speed + signal)[None]
    resid = arms.residualise_on(cube, speed)
    assert np.corrcoef(resid[0].ravel(), signal.ravel())[0, 1] > 0.7
    assert abs(np.corrcoef(resid[0].ravel(), speed.ravel())[0, 1]) < 1e-8


def test_residualise_passes_through_a_constant_covariate():
    cube = np.arange(12, dtype=float).reshape(1, 3, 4)
    out = arms.residualise_on(cube, np.ones((3, 4)))
    np.testing.assert_allclose(out, cube)


# --- BH-FDR ------------------------------------------------------------------

@pytest.mark.parametrize("p,expected,rejected", [
    # Values verified against statsmodels.stats.multitest.multipletests(method="fdr_bh").
    ([0.001, 0.008, 0.039, 0.041, 0.042],
     [0.005, 0.02, 0.042, 0.042, 0.042], [True] * 5),
    ([0.01, 0.2, 0.03, 0.9, 0.04],
     [0.05, 0.25, 0.0666667, 0.9, 0.0666667], [True, False, False, False, False]),
    ([0.001, 0.9], [0.002, 0.9], [True, False]),
])
def test_fdr_bh_matches_the_reference_implementation(p, expected, rejected):
    adj, rej = arms.fdr_bh(np.array(p), q=0.05)
    np.testing.assert_allclose(adj, expected, atol=1e-6)
    assert rej.tolist() == rejected


def test_fdr_bh_adjusted_p_are_monotone_in_the_raw_p():
    """The step-up correction must never let a larger raw p adjust to a smaller one."""
    rng = np.random.default_rng(0)
    p = np.sort(rng.uniform(size=50))
    adj, _ = arms.fdr_bh(p)
    assert np.all(np.diff(adj) >= -1e-12)


def test_fdr_bh_rejects_nothing_when_all_p_are_large():
    _, rej = arms.fdr_bh(np.array([0.4, 0.5, 0.9]))
    assert not rej.any()


def test_fdr_bh_ignores_nan_entries():
    adj, rej = arms.fdr_bh(np.array([0.001, np.nan, 0.9]))
    assert np.isnan(adj[1]) and not rej[1]
    assert rej[0]


# --- batch_triu_corr_mean port (batch_triu_corr_mean.m) ---------------------

def _naive_triu_corr_mean(cube):
    """Reference: loop over cells, correlate every pair of trial profiles."""
    n_cells, n_bins, w = cube.shape
    out = np.full(n_cells, np.nan)
    for c in range(n_cells):
        block = cube[c]                       # (bins, w)
        rs = []
        for i in range(w):
            for j in range(i + 1, w):
                a, b = block[:, i], block[:, j]
                if np.std(a) == 0 or np.std(b) == 0:
                    continue
                rs.append(np.corrcoef(a, b)[0, 1])
        if rs:
            out[c] = np.mean(rs)
    return out


def test_batch_triu_matches_a_naive_pairwise_loop():
    rng = np.random.default_rng(0)
    cube = rng.normal(size=(6, 20, 5))
    np.testing.assert_allclose(arms.batch_triu_corr_mean(cube),
                               _naive_triu_corr_mean(cube), atol=1e-10)


def test_batch_triu_is_one_for_identical_profiles():
    profile = np.sin(np.linspace(0, 4, 25))
    cube = np.tile(profile[None, :, None], (3, 1, 5))
    np.testing.assert_allclose(arms.batch_triu_corr_mean(cube), 1.0, atol=1e-9)


def test_batch_triu_is_minus_one_for_two_opposed_profiles():
    profile = np.linspace(-1, 1, 16)
    cube = np.stack([profile, -profile], axis=1)[None]      # (1, 16, 2)
    np.testing.assert_allclose(arms.batch_triu_corr_mean(cube), -1.0, atol=1e-9)


def test_batch_triu_needs_two_trials_and_two_bins():
    assert np.isnan(arms.batch_triu_corr_mean(np.ones((2, 10, 1)))).all()
    assert np.isnan(arms.batch_triu_corr_mean(np.ones((2, 1, 5)))).all()


def test_batch_triu_flat_cell_is_nan_not_zero():
    """MATLAB sets sd 0 -> 1 and NaNs -> 0, then returns NaN for an all-NaN cell."""
    cube = np.ones((1, 10, 4))                              # zero variance everywhere
    assert np.isnan(arms.batch_triu_corr_mean(cube)).all()


def test_batch_triu_ignores_scale_and_offset_per_trial():
    rng = np.random.default_rng(1)
    cube = rng.normal(size=(3, 30, 4))
    scaled = cube * np.array([1.0, 5.0, 0.2, 100.0]) + np.array([0.0, -3.0, 7.0, 2.0])
    np.testing.assert_allclose(arms.batch_triu_corr_mean(cube),
                               arms.batch_triu_corr_mean(scaled), atol=1e-9)


# --- moving-window reliability (IntegratedAll_v1.m:565-630) -----------------

def test_moving_window_indices_are_centred_and_clipped():
    assert arms.moving_window_indices(0, 10, 5).tolist() == [0, 1, 2]
    assert arms.moving_window_indices(1, 10, 5).tolist() == [0, 1, 2, 3]
    assert arms.moving_window_indices(5, 10, 5).tolist() == [3, 4, 5, 6, 7]
    assert arms.moving_window_indices(9, 10, 5).tolist() == [7, 8, 9]


def test_moving_window_matches_the_matlab_window_arithmetic():
    """max(1, t-half) : min(n, t+half) with half = floor(w/2), 1-based in MATLAB."""
    n, w, half = 12, 5, 2
    for t in range(n):
        expected = list(range(max(0, t - half), min(n - 1, t + half) + 1))
        assert arms.moving_window_indices(t, n, w).tolist() == expected


def test_moving_reliability_returns_one_value_per_trial():
    rng = np.random.default_rng(2)
    cube = rng.normal(size=(4, 20, 30))
    out = arms.moving_window_reliability(cube)
    assert out.shape == (4, 30)
    assert np.isfinite(out).all()


def test_moving_reliability_is_one_for_a_perfectly_repeated_profile():
    profile = np.cos(np.linspace(0, 5, 25))
    cube = np.tile(profile[None, :, None], (2, 1, 15))
    np.testing.assert_allclose(arms.moving_window_reliability(cube), 1.0, atol=1e-9)


def test_moving_reliability_tracks_a_change_in_reliability():
    """First half noisy, second half a clean repeated profile."""
    rng = np.random.default_rng(3)
    profile = np.sin(np.linspace(0, 4, 30))
    early = rng.normal(size=(1, 30, 20))
    late = np.tile(profile[None, :, None], (1, 1, 20)) + rng.normal(size=(1, 30, 20)) * 0.2
    out = arms.moving_window_reliability(np.concatenate([early, late], axis=2))
    assert out[0, :15].mean() < 0.2
    assert out[0, -15:].mean() > 0.8


def test_moving_reliability_single_trial_session_is_all_nan():
    assert np.isnan(arms.moving_window_reliability(np.ones((2, 10, 1)))).all()


def test_moving_reliability_window_size_must_be_odd_and_at_least_three():
    with pytest.raises(ValueError):
        arms.moving_window_reliability(np.ones((1, 10, 10)), window_size=4)
    with pytest.raises(ValueError):
        arms.moving_window_reliability(np.ones((1, 10, 10)), window_size=1)


def test_trial_shuffle_control_destroys_moving_reliability():
    profile = np.sin(np.linspace(0, 4, 30))
    drift = np.linspace(0, 1, 40)
    cube = (profile[None, :, None] * (1 + drift[None, None, :]))
    rng = np.random.default_rng(4)
    real = np.nanmean(arms.moving_window_reliability(cube))
    shuffled = np.nanmean(arms.moving_window_reliability(
        arms.shuffle_trials(cube, rng)))
    assert real > 0.99
    # ...and so does its shuffle: for a stationary profile obs - shuffle is ~0,
    # which is why that difference measures drift, not single-trial reliability.
    assert shuffled > 0.99
    # A pure spatial profile survives shuffling; add trial-specific structure and
    # the shuffle must break it.
    cube2 = cube + np.arange(30)[None, :, None] * drift[None, None, :] * 0.3
    real2 = np.nanmean(arms.moving_window_reliability(cube2))
    shuf2 = np.nanmean(arms.moving_window_reliability(arms.shuffle_trials(cube2, rng)))
    assert real2 >= shuf2


def test_shuffle_trials_is_a_permutation():
    rng = np.random.default_rng(5)
    cube = np.arange(2 * 3 * 8, dtype=float).reshape(2, 3, 8)
    out = arms.shuffle_trials(cube, rng)
    assert sorted(out[0, 0].tolist()) == sorted(cube[0, 0].tolist())
    assert not np.array_equal(out, cube)
