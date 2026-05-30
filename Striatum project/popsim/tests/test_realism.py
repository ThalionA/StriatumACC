"""Tests for biological-realism primitives and the trialised simulation path."""

from __future__ import annotations

import numpy as np
import pytest

from popsim import scenarios, simulate_trials
from popsim.metrics import lag_of_peak_xcorr
from popsim.observation import RealismParams, project_population
from popsim.realism import (
    global_fluctuation,
    lognormal_gains,
    slow_drift,
    subpoisson_counts,
)


# --- realism primitives ----------------------------------------------------
def test_lognormal_gains_mean_and_cv():
    g = lognormal_gains(50000, cv=0.5, rng=0)
    assert abs(g.mean() - 1.0) < 0.02
    assert abs(g.std() / g.mean() - 0.5) < 0.03


def test_lognormal_gains_zero_cv_is_ones():
    np.testing.assert_array_equal(lognormal_gains(20, cv=0.0), np.ones(20))


def test_global_fluctuation_is_shared_across_neurons():
    g = global_fluctuation(2000, 10, strength=1.0, rng=0)
    assert g.shape == (2000, 10)
    # Every neuron sees a scaled copy of one shared signal -> rank-1, so all
    # pairwise correlations are +/-1 (weights are non-negative here -> +1).
    cols = g[:, g.std(0) > 0]
    corr = np.corrcoef(cols.T)
    assert np.allclose(np.abs(corr), 1.0, atol=1e-6)


def test_global_fluctuation_zero_strength():
    assert not global_fluctuation(100, 5, strength=0.0).any()


def test_slow_drift_endpoint_std():
    d = slow_drift(4000, 4000, drift_std=0.3, rng=0)
    assert d.shape == (4000, 4000)
    # Std across neurons of the final-timestep gain ~ drift_std.
    assert abs(d[-1].std() - 0.3) < 0.05
    assert np.all(d > 0)


def test_subpoisson_counts_fano_below_one():
    rate = np.full((40000, 1), 3.0)
    counts = subpoisson_counts(rate, regularity=0.8, rng=0)
    fano = counts.var() / counts.mean()
    # Target Fano ~ 1 - regularity = 0.2; assert clearly sub-Poisson.
    assert fano < 0.5
    # Mean is preserved (moment-matched binomial).
    assert abs(counts.mean() - 3.0) < 0.05
    assert np.all(counts >= 0)


def test_subpoisson_counts_zero_is_poisson():
    rate = np.full((40000, 1), 4.0)
    counts = subpoisson_counts(rate, regularity=0.0, rng=0)
    fano = counts.var() / counts.mean()
    assert abs(fano - 1.0) < 0.05  # Poisson


def test_subpoisson_counts_rejects_bad_regularity():
    with pytest.raises(ValueError):
        subpoisson_counts(np.ones(5), regularity=1.0)
    with pytest.raises(ValueError):
        subpoisson_counts(np.ones(5), regularity=-0.1)


def test_realism_off_matches_clean_model():
    z = np.random.default_rng(0).standard_normal((300, 3))
    from popsim.observation import random_loadings

    W = random_loadings(15, 3, rng=1)
    a = project_population(z, W, rng=7)
    b = project_population(z, W, realism_params=RealismParams(), rng=7)
    np.testing.assert_array_equal(a, b)


# --- trialised simulation --------------------------------------------------
def test_simulate_trials_shapes():
    cfg = scenarios.lagged(n_timesteps=1)  # n_timesteps unused by trial path
    r = simulate_trials(cfg, n_trials=20, n_bins=40)
    for spec in cfg.areas:
        assert r.latents[spec.name].shape == (20, 40, spec.n_latents)
        assert r.neural[spec.name].shape == (20, 40, spec.n_neurons)
        assert r.loadings[spec.name].shape == (spec.n_neurons, spec.n_latents)


def test_simulate_trials_deterministic():
    cfg = scenarios.zero_lag(n_timesteps=1)
    r1 = simulate_trials(cfg, n_trials=10, n_bins=30)
    r2 = simulate_trials(cfg, n_trials=10, n_bins=30)
    for area in r1.area_names:
        np.testing.assert_array_equal(r1.neural[area], r2.neural[area])


def test_simulate_trials_independent_trials():
    # Concatenating one area's latents across trials should show no coupling
    # across the trial boundary: the lag-0 cross-trial autocorrelation of the
    # trial-mean is near zero (trials are independent draws).
    cfg = scenarios.no_coupling(n_timesteps=1)
    r = simulate_trials(cfg, n_trials=200, n_bins=30)
    trial_means = r.latents["A"][:, :, 0].mean(axis=1)  # (n_trials,)
    ac1 = np.corrcoef(trial_means[:-1], trial_means[1:])[0, 1]
    assert abs(ac1) < 0.2


def test_simulate_trials_preserves_lag_structure():
    cfg = scenarios.lagged(n_timesteps=1, lag_ab=8, lag_cb=20)
    # Long trials so the lag is resolvable within a trial.
    r = simulate_trials(cfg, n_trials=40, n_bins=200)
    zA = r.latents["A"][:, :, 0].reshape(-1)
    zB = r.latents["B"][:, :, 0].reshape(-1)
    assert abs(lag_of_peak_xcorr(zA, zB, 40) - 8) <= 3


def test_simulate_trials_poisson_and_realism_runs():
    cfg = scenarios.zero_lag(
        n_timesteps=1,
        observation="poisson",
    )
    # Turn on every realism knob via the area specs.
    for spec in cfg.areas:
        spec.realism = RealismParams(
            neuron_gain_cv=0.4, global_noise=0.2, drift_std=0.2, refractory=0.5
        )
        spec.baseline = 1.5
    r = simulate_trials(cfg, n_trials=15, n_bins=50)
    for spec in cfg.areas:
        x = r.neural[spec.name]
        assert x.dtype == np.int64
        assert np.all(x >= 0)


def test_simulate_trials_rejects_bad_sizes():
    cfg = scenarios.no_coupling(n_timesteps=1)
    with pytest.raises(ValueError):
        simulate_trials(cfg, n_trials=0, n_bins=10)
    with pytest.raises(ValueError):
        simulate_trials(cfg, n_trials=10, n_bins=0)
