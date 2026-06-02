"""Ground-truth tests for the extended coupling scenarios (Phase 2)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from popsim import scenarios, simulate
from popsim.metrics import cca, lag_of_peak_xcorr, partial_cca, partial_correlation


def test_bidirectional_opposite_lags_on_separate_dims():
    lag_ab, lag_ba = 5, 12
    r = simulate(
        scenarios.bidirectional(n_timesteps=9000, lag_ab=lag_ab, lag_ba=lag_ba)
    )
    zA, zB = r.latents["A"], r.latents["B"]
    # Forward A0 -> B0: A leads B (positive lag).
    assert abs(lag_of_peak_xcorr(zA[:, 0], zB[:, 0], 40) - lag_ab) <= 3
    # Backward B1 -> A1: B leads A, so the (A1, B1) pair peaks at negative lag.
    assert abs(lag_of_peak_xcorr(zA[:, 1], zB[:, 1], 40) + lag_ba) <= 3


def test_common_input_collapses_under_partial_cca():
    r = simulate(scenarios.common_input(n_timesteps=8000))
    zA, zB, zC = r.latents["A"], r.latents["B"], r.latents["C"]
    marginal, _, _ = cca(zA, zB)
    partial, _, _ = partial_cca(zA, zB, zC)
    assert marginal[0] > 0.45
    assert partial[0] < 0.3
    # On the shared dimension the partial correlation is ~zero (pure confound).
    assert abs(partial_correlation(zA[:, 0], zB[:, 0], zC)) < 0.1


def test_rotated_subspace_recovers_rank():
    rank = 2
    r = simulate(scenarios.rotated_subspace(n_timesteps=8000, rank=rank))
    corrs, _, _ = cca(r.latents["A"], r.latents["B"])
    # Exactly `rank` strong canonical correlations; a clear drop at the next.
    assert corrs[rank - 1] > 0.4
    assert corrs[rank] < 0.25


def test_partial_mediation_is_graded():
    r = simulate(scenarios.partial_mediation(n_timesteps=8000))
    zA, zB, zC = r.latents["A"], r.latents["B"], r.latents["C"]
    marginal, _, _ = cca(zA, zB)
    partial, _, _ = partial_cca(zA, zB, zC)
    # Conditioning on C reduces but does not remove the A-B coupling: the direct
    # channel survives (contrast with `mediated`, where partial collapses).
    assert partial[0] < marginal[0]
    assert partial[0] > 0.25
    # The mediated dim (A0,B0) collapses; the direct dim (A1,B1) survives.
    assert abs(partial_correlation(zA[:, 0], zB[:, 0], zC)) < 0.15
    assert abs(partial_correlation(zA[:, 1], zB[:, 1], zC)) > 0.15


def test_noise_correlation_is_observation_level_only():
    r = simulate(scenarios.noise_correlation(n_timesteps=8000, strength=0.8))
    # Latents are NOT coupled.
    lat_cca, _, _ = cca(r.latents["A"], r.latents["B"])
    assert lat_cca[0] < 0.2
    # But the populations are correlated through shared additive noise.
    mean_a = r.neural["A"].mean(axis=1)
    mean_b = r.neural["B"].mean(axis=1)
    assert abs(np.corrcoef(mean_a, mean_b)[0, 1]) > 0.3
    # A third uninvolved area (C) is not dragged along.
    mean_c = r.neural["C"].mean(axis=1)
    assert abs(np.corrcoef(mean_a, mean_c)[0, 1]) < 0.2


def test_new_scenarios_in_registry_run():
    for name in ["bidirectional", "common_input", "rotated_subspace",
                 "partial_mediation", "noise_correlation"]:
        r = simulate(scenarios.SCENARIOS[name](n_timesteps=300))
        assert set(r.area_names) == {"A", "B", "C"}


def test_noise_correlation_metadata_records_shared_noise():
    r = simulate(scenarios.noise_correlation(n_timesteps=200))
    meta = json.loads(json.dumps(r.metadata()))
    assert meta["shared_noise"]
    assert meta["shared_noise"][0]["areas"] == ["A", "B"]


def test_shared_noise_rejects_poisson_area():
    cfg = scenarios.noise_correlation(n_timesteps=200, observation="poisson")
    with pytest.raises(ValueError):
        simulate(cfg)
