"""Tests for lagged CCA and the Information Flow Index."""

from __future__ import annotations

import numpy as np
import pytest

from striatum_cca import config, lagged

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
