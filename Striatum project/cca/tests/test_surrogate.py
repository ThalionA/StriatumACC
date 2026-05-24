"""Tests for the held-out-CC surrogate significance test."""

from __future__ import annotations

import dataclasses

import numpy as np

from striatum_cca import config, core, surrogate

CFG = config.DEFAULT


# ---------------------------------------------------------------------------
# permute_trials
# ---------------------------------------------------------------------------
def test_permute_trials_preserves_trial_multiset():
    rng = np.random.default_rng(0)
    tensor = rng.standard_normal((10, 8, 3))
    shuffled = surrogate.permute_trials(tensor, rng)
    assert shuffled.shape == tensor.shape
    orig = {tensor[i].tobytes() for i in range(10)}
    new = {shuffled[i].tobytes() for i in range(10)}
    assert orig == new


# ---------------------------------------------------------------------------
# circshift_bins
# ---------------------------------------------------------------------------
def test_circshift_bins_rolls_each_trial():
    rng = np.random.default_rng(0)
    tensor = rng.standard_normal((6, 40, 3))
    out = surrogate.circshift_bins(tensor, rng, min_shift=10)
    assert out.shape == tensor.shape
    for t in range(6):
        # a circular roll permutes the bin rows -> sorted columns unchanged
        assert np.allclose(np.sort(out[t], axis=0), np.sort(tensor[t], axis=0))
        # min_shift=10 -> no trial is left unshifted
        assert not np.allclose(out[t], tensor[t])


# ---------------------------------------------------------------------------
# p_value
# ---------------------------------------------------------------------------
def test_p_value_real_above_all_surrogates_is_minimal():
    assert surrogate.p_value(1.0, np.zeros(99)) == 1 / 100


def test_p_value_real_below_all_surrogates_is_one():
    assert surrogate.p_value(0.0, np.ones(99)) == 1.0


# ---------------------------------------------------------------------------
# build_null  (held-out per-dimension significance)
# ---------------------------------------------------------------------------
def _shared_latent_scores(n_shared, rho_noise, seed):
    """Score tensors sharing `n_shared` genuine latent dimensions."""
    rng = np.random.default_rng(seed)
    n_tr, n_bins, k = 12, 50, 6
    latents = rng.standard_normal((n_tr, n_bins, n_shared))
    load_x = rng.standard_normal((n_shared, k))
    load_y = rng.standard_normal((n_shared, k))
    x = latents @ load_x + 0.3 * rng.standard_normal((n_tr, n_bins, k))
    y = latents @ load_y + rho_noise * rng.standard_normal((n_tr, n_bins, k))
    return x, y


def test_build_null_flags_genuine_shared_dimensions():
    cfg = dataclasses.replace(CFG, n_shuffles=60, null_type="trials")
    x, y = _shared_latent_scores(n_shared=2, rho_noise=0.3, seed=2)
    real = core.cca_cv(x, y, cfg).held_out_r
    null = surrogate.build_null(x, y, real, cfg)
    assert null.null_held_out.shape == (60, real.shape[0])
    assert null.p_per_dim.shape == real.shape
    assert null.n_significant >= 1            # genuine shared structure


def test_build_null_independent_data_not_significant():
    cfg = dataclasses.replace(CFG, n_shuffles=60, null_type="trials")
    rng = np.random.default_rng(3)
    x = rng.standard_normal((12, 50, 6))
    y = rng.standard_normal((12, 50, 6))
    real = core.cca_cv(x, y, cfg).held_out_r
    null = surrogate.build_null(x, y, real, cfg)
    # No shared structure -> at most a chance hit among 6 dimensions.
    assert null.n_significant <= 1


def test_build_null_does_not_over_call_with_strong_dim1():
    # One very strong shared dimension + 5 pure-noise dimensions. The held-out
    # test must NOT flag the noise dimensions (the in-sample test did -- the
    # spectrum-shift bug this rework fixes).
    cfg = dataclasses.replace(CFG, n_shuffles=60, null_type="trials")
    x, y = _shared_latent_scores(n_shared=1, rho_noise=0.3, seed=5)
    real = core.cca_cv(x, y, cfg).held_out_r
    null = surrogate.build_null(x, y, real, cfg)
    assert 1 <= null.n_significant <= 2        # the one real dim, not all six


def test_build_null_circshift_runs_and_flags_shared_dimensions():
    cfg = dataclasses.replace(CFG, n_shuffles=60, null_type="circshift",
                              circshift_min_bins=10)
    x, y = _shared_latent_scores(n_shared=2, rho_noise=0.3, seed=2)
    real = core.cca_cv(x, y, cfg).held_out_r
    null = surrogate.build_null(x, y, real, cfg)
    assert null.null_held_out.shape == (60, real.shape[0])
    assert null.n_significant >= 1
