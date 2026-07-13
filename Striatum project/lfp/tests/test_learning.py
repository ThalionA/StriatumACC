"""Ground-truth tests for the learning-evolution helpers."""

import numpy as np

from striatum_lfp.learning import (
    Traversal,
    corridor_traversals,
    phase_indices,
    robust_log_outliers,
    segment_band_power,
    valid_traversals,
)


def test_corridor_traversals_groups_by_trial_and_world():
    # two corridor traversals (world 6) separated by dark (world 12)
    world = np.array([6, 6, 6, 12, 12, 6, 6, 6, 6])
    trial = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2])
    frame_ms = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800])
    trav = corridor_traversals(world, trial, frame_ms, min_frames=2)
    assert trav == [Traversal(1, 0, 200), Traversal(2, 500, 800)]


def test_corridor_traversals_skips_short_runs():
    world = np.array([6, 6, 12, 12])
    trial = np.array([1, 1, 1, 1])
    frame_ms = np.array([0, 100, 200, 300])
    assert corridor_traversals(world, trial, frame_ms, min_frames=5) == []


def test_valid_traversals_clips_bounds_and_preserves_filtered_indexing():
    traversals = [
        Traversal(1, -100, 700),
        Traversal(2, 800, 1_100),  # only 200 samples remain after clipping
        Traversal(3, 200, 900),
    ]
    got = valid_traversals(traversals, n_samples=1_000, min_samples=500)
    assert got == [Traversal(1, 0, 700), Traversal(3, 200, 900)]


def test_segment_band_power_recovers_band_of_a_tone():
    fs, n = 1000, 4000
    t = np.arange(n) / fs
    seg = np.column_stack([np.sin(2 * np.pi * 6 * t), 0.5 * np.sin(2 * np.pi * 6 * t)])
    bands = {"theta": (4, 8), "beta": (15, 30), "low_gamma": (30, 80)}
    power = segment_band_power(seg, fs, bands)
    assert power["theta"][0] > 20 * power["beta"][0]
    assert power["theta"][0] > 20 * power["low_gamma"][0]
    # channel 2 has half the amplitude -> ~1/4 the power
    assert 0.15 < power["theta"][1] / power["theta"][0] < 0.35


def test_segment_band_power_is_scale_covariant():
    fs, n = 1000, 2000
    t = np.arange(n) / fs
    seg = np.sin(2 * np.pi * 20 * t)[:, None]
    bands = {"beta": (15, 30)}
    p1 = segment_band_power(seg, fs, bands)["beta"][0]
    p2 = segment_band_power(seg * 10, fs, bands)["beta"][0]
    assert np.isclose(p2 / p1, 100.0, rtol=0.05)  # power scales with amplitude^2


def test_robust_log_outliers_flags_the_artefact_scale():
    # ordinary traversal peaks ~1e-4, artefact peaks ~50 -> flagged
    peaks = np.array([1e-4, 1.2e-4, 0.9e-4, 1.1e-4, 50.0, 1.0e-4])
    flagged = robust_log_outliers(peaks, z_threshold=4.0)
    assert flagged[4] and flagged.sum() == 1


def test_robust_log_outliers_scale_invariant():
    peaks = np.array([1e-4, 1.2e-4, 0.9e-4, 50.0])
    a = robust_log_outliers(peaks)
    b = robust_log_outliers(peaks * 1e6)  # multiplicative gain
    np.testing.assert_array_equal(a, b)


def test_phase_indices_lp_aligned_when_valid():
    idx, scheme = phase_indices(n_trials=100, learning_point=40, trials_per_epoch=10)
    assert scheme == "lp"
    np.testing.assert_array_equal(idx["naive"], np.arange(0, 10))
    # MATLAB trials 30:39 and 40:49 -> Python indices 29:38 and 39:48.
    np.testing.assert_array_equal(idx["intermediate"], np.arange(29, 39))
    np.testing.assert_array_equal(idx["expert"], np.arange(39, 49))


def test_rolling_mean_sem_matches_plain_stats_in_a_flat_window():
    from striatum_lfp.learning import rolling_mean_sem
    x = np.array([1.0, 2, 3, 4, 5, 6, 7])
    mean, sem = rolling_mean_sem(x, window=3)
    # interior point averages its 3-neighbourhood
    assert np.isclose(mean[3], 4.0)
    assert np.isclose(sem[3], np.std([3.0, 4, 5]) / np.sqrt(3))
    # NaNs are skipped, not propagated
    y = np.array([np.nan, 2.0, np.nan, 4.0, np.nan])
    m2, _ = rolling_mean_sem(y, window=3)
    assert np.isclose(m2[1], 2.0)


def test_phase_indices_falls_back_to_thirds():
    idx, scheme = phase_indices(n_trials=30, learning_point=None)
    assert scheme == "thirds"
    assert idx["naive"][0] == 0 and idx["expert"][-1] == 29
    # too-early LP that would not fit its window also falls back
    _, scheme2 = phase_indices(n_trials=30, learning_point=3, trials_per_epoch=10)
    assert scheme2 == "thirds"
