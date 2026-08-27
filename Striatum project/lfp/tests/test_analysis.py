"""Tests for the epoch / learning-point layer that the LFP analyses share with MATLAB.

The learning point and the epoch windows are the axis every "across learning"
claim is plotted against, so they are checked against MATLAB's own logged output
(processed_data/corridordark3_2026-08-27.log) rather than against themselves.
"""

from __future__ import annotations

import numpy as np
import pytest

from striatum_lfp import analysis

# MATLAB's learning points for the 16 task animals, in cohort order, as printed
# by CorridorVsDarkActivity.m on 2026-08-27.
MATLAB_LPS = {
    523: 22, 614: 44, 624: 32, 727: 53, 730: 54, 731: 36, 822: 33, 823: 14,
    1105: 44, 1106: 67, 1201: 84, 1206: None, 1212: 23, 409: 39, 418: 26, 703: None,
}


# --- learning point (find_learning_points.m) --------------------------------

def test_learning_point_needs_the_trial_itself_below_threshold():
    """The 2026-05-24 fix: the LP is a sub-threshold trial, not a window start."""
    z = np.zeros(30)
    z[10] = 0.0                      # above threshold
    z[11:21] = -3.0
    assert analysis.learning_point(z) == 12      # 1-based: first sub-threshold trial


def test_learning_point_requires_sustained_performance():
    z = np.zeros(40)
    z[5] = -3.0                                   # one good trial, not sustained
    z[20:30] = -3.0
    assert analysis.learning_point(z) == 21


def test_learning_point_is_none_for_a_non_learner():
    assert analysis.learning_point(np.zeros(50)) is None


def test_learning_point_is_none_for_a_short_session():
    assert analysis.learning_point(np.full(5, -3.0)) is None


def test_learning_point_ignores_nans():
    z = np.full(30, np.nan)
    z[10:25] = -3.0
    assert analysis.learning_point(z) == 11


def test_learning_point_matches_matlab_on_the_real_cohort():
    got = analysis.cohort_learning_points()
    for mouse, expected in MATLAB_LPS.items():
        assert got[mouse] == expected, f"{mouse}: got {got[mouse]}, MATLAB says {expected}"


# --- epoch windows (epoch_indices.m) ----------------------------------------

def test_epochs_with_naive_split_give_four_unequal_windows():
    idx = analysis.epoch_indices(lp=30, n_trials=100, naive_split=3)
    assert [len(x) for x in idx] == [3, 7, 10, 10]
    assert idx[0].tolist() == [1, 2, 3]
    assert idx[1].tolist() == list(range(4, 11))
    assert idx[2].tolist() == list(range(20, 30))      # the 10 trials before LP
    assert idx[3].tolist() == list(range(30, 40))      # expert starts AT lp


def test_epochs_without_split_give_three_windows():
    idx = analysis.epoch_indices(lp=30, n_trials=100)
    assert [len(x) for x in idx] == [10, 10, 10]
    assert idx[0].tolist() == list(range(1, 11))


def test_non_learner_keeps_naive_and_loses_both_lp_windows():
    idx = analysis.epoch_indices(lp=None, n_trials=100, naive_split=3)
    assert len(idx[0]) == 3 and len(idx[1]) == 7
    assert idx[2].size == 0 and idx[3].size == 0


def test_learning_point_beyond_the_session_yields_no_lp_windows():
    """epoch_indices.m:74-76 -- lp > n_trials means the LP is not in the data."""
    idx = analysis.epoch_indices(lp=120, n_trials=100, naive_split=3)
    assert idx[2].size == 0 and idx[3].size == 0


def test_expert_window_running_past_the_session_is_dropped():
    idx = analysis.epoch_indices(lp=95, n_trials=100, naive_split=3)
    assert idx[3].size == 0                     # 95..104 does not fit
    assert idx[2].tolist() == list(range(85, 95))


def test_intermediate_window_before_trial_one_is_dropped():
    idx = analysis.epoch_indices(lp=5, n_trials=100, naive_split=3)
    assert idx[2].size == 0                     # -5..4 does not fit
    assert idx[3].tolist() == list(range(5, 15))


def test_expert_starts_at_lp1_shifts_by_one():
    idx = analysis.epoch_indices(lp=30, n_trials=100, expert_starts_at="lp1")
    assert idx[2].tolist() == list(range(31, 41))


def test_naive_split_must_be_inside_the_epoch():
    with pytest.raises(ValueError):
        analysis.epoch_indices(lp=30, n_trials=100, naive_split=10)


# --- speed from bin timing ---------------------------------------------------

def test_bin_speed_recovers_a_known_constant_speed():
    """A 5 cm bin crossed in 250 ms is 20 cm/s."""
    start = np.array([[0], [250]], dtype=np.int32)
    stop = np.array([[250], [500]], dtype=np.int32)
    speed = analysis.bin_speed_cm_s(start, stop)
    np.testing.assert_allclose(speed, 20.0)


def test_bin_speed_is_nan_for_an_absent_bin():
    speed = analysis.bin_speed_cm_s(np.array([[-1]], np.int32), np.array([[-1]], np.int32))
    assert np.isnan(speed).all()


def test_bin_speed_is_nan_for_a_zero_duration_bin():
    speed = analysis.bin_speed_cm_s(np.array([[10]], np.int32), np.array([[10]], np.int32))
    assert np.isnan(speed).all()


# --- joint z-scoring ---------------------------------------------------------

def test_joint_zscore_keeps_a_corridor_dark_difference():
    """Scoring the states separately would force both means to zero."""
    rng = np.random.default_rng(0)
    corridor = rng.normal(10.0, 1.0, size=(3, 50, 20))
    dark = rng.normal(5.0, 1.0, size=(3, 50, 20))
    zc, zd = analysis.joint_zscore(corridor, dark)
    assert np.nanmean(zc) > 0.5 and np.nanmean(zd) < -0.5
    pooled = np.concatenate([zc.reshape(3, -1), zd.reshape(3, -1)], axis=1)
    np.testing.assert_allclose(np.nanmean(pooled, axis=1), 0.0, atol=1e-9)
    np.testing.assert_allclose(np.nanstd(pooled, axis=1), 1.0, atol=1e-9)


def test_joint_zscore_tolerates_nan_cells():
    corridor = np.ones((2, 4, 3))
    corridor[0, 0, 0] = np.nan
    dark = np.zeros((2, 4, 3))
    zc, zd = analysis.joint_zscore(corridor, dark)
    assert np.isnan(zc[0, 0, 0])
    assert np.isfinite(zc[0, 1, 0])


def test_joint_zscore_flat_channel_gives_nan_not_infinity():
    corridor = np.ones((1, 2, 2))
    dark = np.ones((1, 2, 2))
    zc, _ = analysis.joint_zscore(corridor, dark)
    assert np.isnan(zc).all()
