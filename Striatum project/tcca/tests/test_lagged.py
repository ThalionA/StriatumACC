"""Tests for lagged CCA and the Information Flow Index."""

from __future__ import annotations

import numpy as np
import pytest

from striatum_tcca import config, lagged

CFG = config.DEFAULT


# ---------------------------------------------------------------------------
# lag_slice
# ---------------------------------------------------------------------------
def test_lag_slice_zero_returns_full():
    x = np.random.default_rng(0).standard_normal((6, 20, 3))
    y = np.random.default_rng(1).standard_normal((6, 20, 3))
    xl, yl = lagged.lag_slice(x, y, 0)
    assert np.array_equal(xl, x) and np.array_equal(yl, y)


def test_lag_slice_positive_pairs_x_with_later_y():
    x = np.random.default_rng(2).standard_normal((6, 20, 3))
    y = np.random.default_rng(3).standard_normal((6, 20, 3))
    xl, yl = lagged.lag_slice(x, y, 4)
    assert xl.shape == (6, 16, 3) and yl.shape == (6, 16, 3)
    assert np.array_equal(xl, x[:, :16, :])     # x bins 0..15
    assert np.array_equal(yl, y[:, 4:, :])      # paired with y bins 4..19


def test_lag_slice_negative_pairs_x_with_earlier_y():
    x = np.random.default_rng(4).standard_normal((6, 20, 3))
    y = np.random.default_rng(5).standard_normal((6, 20, 3))
    xl, yl = lagged.lag_slice(x, y, -4)
    assert np.array_equal(xl, x[:, 4:, :])
    assert np.array_equal(yl, y[:, :16, :])


def test_lag_slice_rejects_oversized_lag():
    x = np.zeros((2, 10, 2))
    with pytest.raises(ValueError):
        lagged.lag_slice(x, x, 10)


# ---------------------------------------------------------------------------
# information_flow_index
# ---------------------------------------------------------------------------
def test_ifi_symmetric_curve_is_zero():
    lags = np.arange(-3, 4)
    cc1 = np.array([0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2])
    assert abs(lagged.information_flow_index(lags, cc1)) < 1e-12


def test_ifi_positive_when_x_leads():
    lags = np.arange(-3, 4)
    cc1 = np.array([0.0, 0.0, 0.0, 0.3, 0.6, 0.6, 0.6])   # mass at L>0
    assert lagged.information_flow_index(lags, cc1) > 0.5


def test_ifi_clips_negative_correlations():
    lags = np.arange(-3, 4)
    cc1 = np.array([-0.5, -0.5, -0.5, 0.0, 0.4, 0.4, 0.4])
    ifi = lagged.information_flow_index(lags, cc1)
    assert ifi == 1.0      # negatives clipped to 0 -> all flow is X->Y


# ---------------------------------------------------------------------------
# lag_curve — planted lag
# ---------------------------------------------------------------------------
def test_lag_curve_recovers_planted_lead():
    # Y is X shifted forward by 3 bins: Y leads/lags such that pairing X[b]
    # with Y[b+3] aligns them -> peak at lag +3, IFI strongly positive.
    rng = np.random.default_rng(7)
    x = rng.standard_normal((14, 30, 4))
    y = np.empty_like(x)
    y[:, 3:, :] = x[:, :-3, :] + 0.1 * rng.standard_normal((14, 27, 4))
    y[:, :3, :] = rng.standard_normal((14, 3, 4))
    result = lagged.lag_curve(x, y, CFG, max_lag=5)
    assert result.peak_lag == 3
    assert result.ifi > 0.3
    assert result.cc1[result.lags == 3][0] > 0.8


# ---------------------------------------------------------------------------
# ifi_by_window  (point 4)
# ---------------------------------------------------------------------------
def test_ifi_by_window_symmetric_curve_zero_at_every_window():
    lags = np.arange(-5, 6)
    cc = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    windows = lagged.ifi_by_window(lags, cc)
    assert windows.shape == (5,)
    assert np.allclose(windows, 0.0, atol=1e-12)


def test_ifi_by_window_detects_one_sided_mass():
    lags = np.arange(-5, 6)
    cc = np.array([0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=float)
    windows = lagged.ifi_by_window(lags, cc)
    assert np.all(windows > 0.9)            # all flow is X->Y at every window


# ---------------------------------------------------------------------------
# per-dimension lag curve  (point 1)
# ---------------------------------------------------------------------------
def test_lag_curve_returns_all_dimensions():
    rng = np.random.default_rng(9)
    x = rng.standard_normal((14, 30, 4))
    y = rng.standard_normal((14, 30, 4))
    result = lagged.lag_curve(x, y, CFG, held_out=False)
    n_lags, d = result.cc_per_dim.shape
    assert n_lags == result.lags.size
    assert result.ifi_per_dim.shape == (d,)
    assert result.peak_lag_per_dim.shape == (d,)
    assert result.ifi_windows.shape[0] == d
    # back-compatible dominant-dimension accessors
    assert result.cc1.shape == (n_lags,)
    assert result.cc1[0] == result.cc_per_dim[0, 0]


# ---------------------------------------------------------------------------
# heldout_lag_curve_flat (continuous-regime, segment-aware, held-out) + window IFI
# ---------------------------------------------------------------------------
def _flat_lead_data(rng, n_trials=12, n_bins=120, k=3, lead=3):
    """Flat continuous scores where X leads Y by `lead` bins WITHIN each trial."""
    groups = np.repeat(np.arange(n_trials), n_bins)
    Sx = np.zeros((n_trials * n_bins, k))
    Sy = np.zeros((n_trials * n_bins, k))
    for t in range(n_trials):
        sl = slice(t * n_bins, (t + 1) * n_bins)
        s = rng.standard_normal(n_bins + lead)
        x = s[lead:]                      # X carries the signal at time b
        y = s[:n_bins]                    # Y carries it `lead` bins LATER (b+lead)
        Sx[sl, 0] = x + 0.2 * rng.standard_normal(n_bins)
        Sy[sl, 0] = y + 0.2 * rng.standard_normal(n_bins)
        Sx[sl, 1:] = 0.3 * rng.standard_normal((n_bins, k - 1))
        Sy[sl, 1:] = 0.3 * rng.standard_normal((n_bins, k - 1))
    return Sx, Sy, groups


def test_heldout_lag_curve_recovers_known_lead():
    rng = np.random.default_rng(0)
    Sx, Sy, groups = _flat_lead_data(rng, lead=3)
    lags, cc = lagged.heldout_lag_curve_flat(Sx, Sy, groups, max_lag=8, n_folds=4)
    peak = int(lags[np.nanargmax(cc)])
    assert peak == 3                                    # held-out curve peaks at the true lead
    assert lagged.information_flow_index(lags, cc) > 0  # IFI positive => X leads Y


def test_ifi_window_sweep_recovers_direction():
    rng = np.random.default_rng(1)
    Sx, Sy, groups = _flat_lead_data(rng, lead=2)
    lags, cc = lagged.heldout_lag_curve_flat(Sx, Sy, groups, max_lag=10, n_folds=4)
    win = lagged.ifi_by_window(lags, cc)                # IFI for |lag|<=w, w=1..10
    assert win.shape[0] == 10
    # every window WIDE ENOUGH to contain the +2 peak reports X-leads (IFI>0); the
    # w=1 window (only +/-1, peak excluded) is legitimately the noisiest — that IS
    # the "too-tight window is unclean" effect the sweep exists to reveal.
    assert np.all(win[1:] > 0)
    assert np.nanmax(win) > 0.5                          # a clean directionality is recovered


def test_segment_aware_no_cross_trial_pairing():
    # within-trial signal but trials are independent; a lag beyond a trial cannot
    # borrow signal across the boundary -> curve must not peak at a spurious far lag
    rng = np.random.default_rng(2)
    Sx, Sy, groups = _flat_lead_data(rng, n_bins=40, lead=2)
    lags, cc = lagged.heldout_lag_curve_flat(Sx, Sy, groups, max_lag=6, n_folds=4)
    assert int(lags[np.nanargmax(cc)]) == 2
