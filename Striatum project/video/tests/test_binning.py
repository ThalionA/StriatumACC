"""Trial cutting and spatial binning must reproduce the MATLAB pipeline's rules
(cut_data_per_trial.m, separate_dark_and_corridor_periods.m, spatial_binning.m,
ProcessStriatumTask.m bin edges)."""

import numpy as np
import pytest

from striatum_video.binning import (
    bin_session,
    corridor_start,
    position_bin_index,
    spatial_bin_edges,
    trial_bounds,
)


def test_bin_edges_match_process_striatum_task():
    e = spatial_bin_edges()
    assert e.size == 51
    assert e[0] == 0 and e[1] == 4 and e[-2] == 196
    assert e[-1] == 204  # MATLAB widens the last edge: bin_edges(end) = 200 + 4


def test_position_bin_index_follows_histcounts():
    e = spatial_bin_edges()
    x = np.array([-0.1, 0.0, 3.99, 4.0, 199.0, 203.9, 204.0, 204.1, np.nan])
    # histcounts: [e_k, e_k+1) except the LAST bin, which includes its right edge;
    # outside or NaN -> no bin (-1 here, 0 in MATLAB).
    np.testing.assert_array_equal(position_bin_index(x, e), [-1, 0, 0, 1, 49, 49, 49, -1, -1])


def test_trial_bounds_cut_where_the_trial_column_changes():
    starts, ends = trial_bounds(np.array([1, 1, 1, 2, 2, 3, 3, 3, 3]))
    np.testing.assert_array_equal(starts, [0, 3, 5])
    np.testing.assert_array_equal(ends, [2, 4, 8])  # inclusive


def test_corridor_starts_at_the_first_world_above_six():
    assert corridor_start(np.array([6, 6, 12, 12, 6])) == 2
    assert corridor_start(np.array([6, 6])) is None


def _one_trial_session():
    # 2 dark rows, then corridor rows walking through bins 0 and 1.
    world = np.array([6, 6, 12, 12, 12, 12, 12])
    x = np.array([0.0, 0.0, 0.5, 2.0, 4.5, 6.0, 7.0])
    t_ms = np.array([0, 30, 60, 90, 120, 150, 210.0])
    trial = np.ones(7)
    feature = np.array([100, 100, 1.0, 3.0, 10.0, 20.0, 30.0])
    return {"trial": trial, "world": world, "x": x}, t_ms, {"f": feature}


def test_bin_session_durations_and_feature_means_per_bin():
    vr, t_ms, feats = _one_trial_session()
    out = bin_session(vr, t_ms, feats)
    # bin 0: corridor rows 2,3 (t 60, 90) -> 0.030 s, mean feature 2
    # bin 1: rows 4,5,6 (t 120..210) -> 0.090 s, mean 20; dark rows ignored
    assert out["durations"][0, 0] == pytest.approx(0.030)
    assert out["durations"][0, 1] == pytest.approx(0.090)
    assert out["f"][0, 0] == pytest.approx(2.0)
    assert out["f"][0, 1] == pytest.approx(20.0)
    assert np.all(np.isnan(out["durations"][0, 2:]))


def test_a_bin_needs_at_least_two_rows():
    vr, t_ms, feats = _one_trial_session()
    vr["x"] = np.array([0.0, 0.0, 0.5, 4.5, 4.6, 8.1, 8.2])  # bin 0 gets ONE corridor row
    out = bin_session(vr, t_ms, feats)
    assert np.isnan(out["durations"][0, 0]) and np.isnan(out["f"][0, 0])
    assert out["durations"][0, 1] == pytest.approx(0.030)


def test_feature_nans_are_ignored_within_a_bin():
    vr, t_ms, feats = _one_trial_session()
    feats["f"][4] = np.nan
    out = bin_session(vr, t_ms, feats)
    assert out["f"][0, 1] == pytest.approx(25.0)


def test_a_trial_without_a_corridor_is_all_nan():
    vr, t_ms, feats = _one_trial_session()
    vr["world"][:] = 6
    out = bin_session(vr, t_ms, feats)
    assert np.all(np.isnan(out["durations"])) and np.all(np.isnan(out["f"]))


def test_still_frames_need_no_motion_and_no_lick_in_the_whole_window():
    from striatum_video.binning import still_frames
    v = np.zeros(20)
    lick = np.zeros(20)
    v[3] = 5.0
    lick[15] = 1
    still = still_frames(v, lick, half_window=2)
    # frames within 2 of the movement (1..5) or of the lick (13..17) are not still;
    # nor are the first/last 2, whose window runs off the session
    np.testing.assert_array_equal(np.flatnonzero(still), [6, 7, 8, 9, 10, 11, 12])


def test_per_trial_floor_is_the_median_over_still_frames_with_a_minimum_count():
    from striatum_video.binning import per_trial_still_median
    trial = np.array([1] * 6 + [2] * 6)
    feature = np.array([1, 2, 3, 100, 100, 100, 7, 8, 9, 10, 11, 12], float)
    still = np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0], bool)
    out = per_trial_still_median(feature, still, trial, min_frames=2)
    assert out[0] == 2.0 and np.isnan(out[1])  # trial 2 has only one still frame
