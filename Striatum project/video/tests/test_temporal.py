"""Per-frame signals -> the cca temporal arm's corridor time bins."""

import numpy as np

from striatum_video.temporal import corridor_frame_ms, lag_within_trial, to_time_bins


def test_corridor_frame_times_are_zeroed_on_the_second_corridor_row():
    # MATLAB aligns spike column 0 with the corridor's 2nd VR row
    # (separate_dark_and_corridor_periods.m uses trial_times_zeroed(idx + 1)).
    t_ms = np.array([0, 30, 60, 95, 125, 160.0])
    rows = np.array([2, 3, 4, 5])  # corridor rows of one trial
    np.testing.assert_allclose(corridor_frame_ms(t_ms, rows), [-35, 0, 30, 65])


def test_to_time_bins_interpolates_at_bin_centres():
    frame_ms = np.array([0.0, 30.0, 60.0, 90.0])
    values = np.array([0.0, 3.0, 6.0, 9.0])  # 0.1 per ms
    out = to_time_bins(frame_ms, values, n_bins=4, bin_ms=20)
    np.testing.assert_allclose(out, [1.0, 3.0, 5.0, 7.0])  # centres 10, 30, 50, 70 ms


def test_to_time_bins_ignores_nan_frames_and_holds_the_ends():
    frame_ms = np.array([0.0, 30.0, 60.0])
    values = np.array([1.0, np.nan, 3.0])
    out = to_time_bins(frame_ms, values, n_bins=5, bin_ms=20)  # centres up to 90 ms
    np.testing.assert_allclose(out, [1 + 2 * 10 / 60, 1 + 2 * 30 / 60, 1 + 2 * 50 / 60, 3.0, 3.0])


def test_lag_within_trial_shifts_without_crossing_trials():
    x = np.arange(10.0)
    trial = np.array([0] * 5 + [1] * 5)
    # lag +1: row i takes the value from row i - 1 (the past); first row of each trial holds its own value
    np.testing.assert_array_equal(lag_within_trial(x, trial, 1), [0, 0, 1, 2, 3, 5, 5, 6, 7, 8])
    # lag -2: the future, held at each trial's end
    np.testing.assert_array_equal(lag_within_trial(x, trial, -2), [2, 3, 4, 4, 4, 7, 8, 9, 9, 9])
