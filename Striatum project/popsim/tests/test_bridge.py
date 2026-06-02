"""Tests for the striatum_cca bridge.

These are skipped automatically if striatum_cca (or h5py) is not importable, so
the popsim suite stays self-contained while still validating the bridge wherever
the sibling package is present.
"""

from __future__ import annotations

import pytest

from popsim import scenarios, simulate_trials
from popsim.bridge import (
    analyse_pair,
    pca_scores,
    striatum_cca_available,
    to_area_tensor,
)

pytestmark = pytest.mark.skipif(
    not striatum_cca_available(), reason="striatum_cca not importable"
)


def test_to_area_tensor_shape_and_unknown_area():
    r = simulate_trials(scenarios.zero_lag(n_timesteps=1), n_trials=10, n_bins=30)
    t = to_area_tensor(r, "A")
    assert t.shape == (10, 30, 60)
    with pytest.raises(KeyError):
        to_area_tensor(r, "Z")


def test_pca_scores_shape():
    r = simulate_trials(scenarios.zero_lag(n_timesteps=1), n_trials=12, n_bins=30)
    scores, state = pca_scores(to_area_tensor(r, "B"), k=5)
    assert scores.shape == (12, 30, 5)
    assert state.components.shape[1] == 5


def test_bridge_zero_lag_high_cc_peak_at_zero():
    r = simulate_trials(scenarios.zero_lag(n_timesteps=1), n_trials=60, n_bins=50)
    out = analyse_pair(r, "A", "B", k=4, max_lag=20)
    # Cross-validated canonical correlation is clearly present.
    assert out.held_out_cc[0] > 0.4
    # Instantaneous coupling -> lag curve peaks at 0.
    assert abs(out.peak_lag) <= 3


def test_bridge_lagged_recovers_peak_lag():
    r = simulate_trials(
        scenarios.lagged(n_timesteps=1, lag_ab=8), n_trials=80, n_bins=60
    )
    out = analyse_pair(r, "A", "B", k=4, max_lag=25)
    assert abs(out.peak_lag - 8) <= 4


def test_bridge_mediated_collapses_under_partial():
    r = simulate_trials(scenarios.mediated(n_timesteps=1), n_trials=80, n_bins=50)
    out = analyse_pair(r, "A", "B", k=4, partial_area="C", max_lag=15)
    assert out.partial_cc is not None
    # The held-out A-B coupling is real but drops sharply once C is removed.
    assert out.held_out_cc[0] > 0.3
    assert out.partial_cc[0] < 0.6 * out.held_out_cc[0]


def test_bridge_no_coupling_low_cc():
    r = simulate_trials(scenarios.no_coupling(n_timesteps=1), n_trials=60, n_bins=50)
    out = analyse_pair(r, "A", "B", k=4, max_lag=15)
    assert out.held_out_cc[0] < 0.3
