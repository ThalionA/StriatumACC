"""Ground-truth tests: each scenario must exhibit its configured coupling."""

from __future__ import annotations

import json

import numpy as np
import pytest

from popsim import scenarios, simulate
from popsim.metrics import cca, lag_of_peak_xcorr, partial_cca, partial_correlation


def _canonical_variates(X, Y):
    corrs, A, B = cca(X, Y)
    return corrs, (X - X.mean(0)) @ A[:, 0], (Y - Y.mean(0)) @ B[:, 0]


def test_simulation_is_deterministic_in_seed():
    cfg = scenarios.zero_lag(n_timesteps=500)
    r1, r2 = simulate(cfg), simulate(cfg)
    for area in r1.area_names:
        np.testing.assert_array_equal(r1.neural[area], r2.neural[area])
        np.testing.assert_array_equal(r1.latents[area], r2.latents[area])


def test_output_shapes_match_config():
    cfg = scenarios.lagged(n_timesteps=400)
    r = simulate(cfg)
    for spec in cfg.areas:
        assert r.latents[spec.name].shape == (400, spec.n_latents)
        assert r.neural[spec.name].shape == (400, spec.n_neurons)
        assert r.loadings[spec.name].shape == (spec.n_neurons, spec.n_latents)


def test_no_coupling_has_low_cross_area_correlation():
    r = simulate(scenarios.no_coupling(n_timesteps=6000))
    assert cca(r.latents["A"], r.latents["B"])[0][0] < 0.2
    assert cca(r.latents["A"], r.latents["C"])[0][0] < 0.2


def test_zero_lag_peaks_at_zero():
    r = simulate(scenarios.zero_lag(n_timesteps=6000))
    corrs, ua, ub = _canonical_variates(r.latents["A"], r.latents["B"])
    assert corrs[0] > 0.5
    assert abs(lag_of_peak_xcorr(ua, ub, max_lag=40)) <= 2


def test_lagged_positive_and_negative_lags():
    lag_ab, lag_cb = 10, 25
    r = simulate(scenarios.lagged(n_timesteps=8000, lag_ab=lag_ab, lag_cb=lag_cb))
    zA, zB, zC = r.latents["A"], r.latents["B"], r.latents["C"]
    assert abs(lag_of_peak_xcorr(zA[:, 0], zB[:, 0], 50) - lag_ab) <= 3
    assert abs(lag_of_peak_xcorr(zC[:, 0], zB[:, 1], 50) - lag_cb) <= 3
    # Reversing pair order flips the sign of the A->B lag.
    assert abs(lag_of_peak_xcorr(zB[:, 0], zA[:, 0], 50) + lag_ab) <= 3


def test_mediated_collapses_under_partial_cca():
    r = simulate(scenarios.mediated(n_timesteps=8000))
    zA, zB, zC = r.latents["A"], r.latents["B"], r.latents["C"]
    marginal, _, _ = cca(zA, zB)
    partial, _, _ = partial_cca(zA, zB, zC)
    # Marginal A-B coupling is real, but collapses once C is controlled for.
    assert marginal[0] > 0.45
    assert partial[0] < 0.3
    assert partial[0] < 0.5 * marginal[0]
    # On the single linked dimensions the partial correlation is ~zero.
    assert abs(partial_correlation(zA[:, 0], zB[:, 0], zC)) < 0.1


def test_epoch_varying_changes_direction_and_orientation():
    lag = 5
    r = simulate(scenarios.epoch_varying(n_timesteps=9000, lag=lag))
    b1, b2 = r.config.epoch_boundaries
    zA, zB = r.latents["A"], r.latents["B"]
    m = lag + 20  # skip a margin after each epoch start to avoid edge effects

    # Epoch 1: A -> B on dim0 -> dim0 (A leads B, positive lag).
    assert lag_of_peak_xcorr(zA[m:b1, 0], zB[m:b1, 0], 30) > 0
    # Epoch 2: B -> A on dim0 -> dim0 (B leads A, so the A,B pair peaks negative).
    assert lag_of_peak_xcorr(zA[b1 + m:b2, 0], zB[b1 + m:b2, 0], 30) < 0
    # Epoch 3: orientation changes to dim1 -> dim2; the dim0->dim0 link is off.
    seg = slice(b2 + m, r.config.n_timesteps)
    corr_new = abs(np.corrcoef(zA[seg, 1], zB[seg, 2])[0, 1])
    corr_old = abs(np.corrcoef(zA[seg, 0], zB[seg, 0])[0, 1])
    assert corr_new > corr_old


def test_metadata_is_json_serialisable():
    r = simulate(scenarios.epoch_varying(n_timesteps=300))
    meta = r.metadata()
    assert "edges" in json.loads(json.dumps(meta))


def test_poisson_observation_runs():
    cfg = scenarios.zero_lag(n_timesteps=400, observation="poisson")
    r = simulate(cfg)
    for spec in cfg.areas:
        assert r.neural[spec.name].dtype == np.int64
        assert np.all(r.neural[spec.name] >= 0)


@pytest.mark.parametrize("dynamics", ["ar1", "lds", "oscillatory"])
def test_all_dynamics_run(dynamics):
    cfg = scenarios.zero_lag(n_timesteps=400, dynamics=dynamics)
    r = simulate(cfg)
    assert r.latents["A"].shape == (400, 4)


@pytest.mark.parametrize("name", list(scenarios.SCENARIOS))
def test_all_scenarios_run(name):
    r = simulate(scenarios.SCENARIOS[name](n_timesteps=300))
    assert set(r.area_names) == {"A", "B", "C"}
