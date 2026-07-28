"""Tests for the temporal data layer: derived velocity + global time-binned streams.

The striatum has no velocity channel, so running speed is derived from the
corridor position stream (``dataio.trial_velocity``) and the per-trial time bins
are concatenated into one global timeline whose ``trial_idx`` channel keeps the
running mask / sliding window aligned to traversals
(``dataio.build_temporal_streams``). Both are pure functions tested here on
synthetic ground truth -- no h5py.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from striatum_cca import config, dataio

CFG25 = dataclasses.replace(config.DEFAULT, temporal_bin_ms=25)


# ---------------------------------------------------------------------------
# trial_velocity
# ---------------------------------------------------------------------------
def test_velocity_constant_speed_ramp():
    # Position ramps 0 -> 100 a.u. over 1000 ms => constant speed.
    # speed = 100 a.u. * 1.25 cm/a.u. / 1.0 s = 125 cm/s.
    times = np.arange(0, 1001, 10, dtype=float)        # VR rate ~100 Hz
    position = times / 1000.0 * 100.0                  # linear 0..100 a.u.
    vel = dataio.trial_velocity(position, times, n_bins=40, cfg=CFG25)
    assert vel.shape == (40,)
    assert np.allclose(vel, 125.0, atol=1e-6), vel


def test_velocity_zero_when_stationary():
    times = np.arange(0, 1001, 10, dtype=float)
    position = np.full_like(times, 42.0)               # not moving
    vel = dataio.trial_velocity(position, times, n_bins=40, cfg=CFG25)
    assert np.allclose(vel, 0.0)


def test_velocity_rezeroes_times_to_corridor_onset():
    # Same ramp, but the time stream starts at a 5 s offset (corridor onset is
    # not zero in corridorData.trial_times). Re-zeroing must give the same speed.
    times = np.arange(5000, 6001, 10, dtype=float)
    position = (times - times[0]) / 1000.0 * 100.0
    vel = dataio.trial_velocity(position, times, n_bins=40, cfg=CFG25)
    assert np.allclose(vel, 125.0, atol=1e-6)


def test_velocity_empty_or_degenerate_returns_zeros():
    assert np.array_equal(
        dataio.trial_velocity(np.array([]), np.array([]), 5, CFG25), np.zeros(5)
    )
    assert np.array_equal(
        dataio.trial_velocity([1.0], [0.0], 1, CFG25), np.zeros(1)
    )


# ---------------------------------------------------------------------------
# build_temporal_streams
# ---------------------------------------------------------------------------
def _trial(n_1ms, n_units, total_ms, end_pos_au):
    """One synthetic trial: all-ones 1 ms spikes + a linear position ramp."""
    spikes = np.ones((n_1ms, n_units), dtype=float)
    times = np.arange(0, total_ms, 10, dtype=float)
    position = np.linspace(0.0, end_pos_au, times.size)
    return spikes, position, times


def test_build_streams_concatenates_and_labels_trials():
    n_units = 3
    trials = [
        _trial(100, n_units, total_ms=110, end_pos_au=80),   # 4 x 25 ms bins
        _trial(150, n_units, total_ms=160, end_pos_au=120),  # 6 x 25 ms bins
    ]
    s = dataio.build_temporal_streams(iter(trials), n_units, CFG25)
    assert s.spikes.shape == (10, n_units)
    # all-ones 1 ms spikes -> each 25 ms bin sums to 25.
    assert np.allclose(s.spikes, 25.0)
    assert np.array_equal(s.trial_idx,
                          np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=float))
    assert s.vel.shape == (10,)
    assert np.all(s.vel > 0)


def test_build_streams_drops_overlong_trial_but_keeps_index():
    n_units = 2
    cfg = dataclasses.replace(CFG25, temporal_max_trial_ms=100)  # max 4 bins
    trials = [
        _trial(100, n_units, 110, 80),    # 4 bins  -> kept, trial id 0
        _trial(150, n_units, 160, 120),   # 6 bins  -> over-long, dropped
        _trial(100, n_units, 110, 80),    # 4 bins  -> kept, trial id 2
    ]
    s = dataio.build_temporal_streams(iter(trials), n_units, cfg)
    assert s.spikes.shape == (8, n_units)
    # Trial 1 contributes no bins but consumes its index: labels are 0*4, 2*4.
    assert np.array_equal(s.trial_idx,
                          np.array([0, 0, 0, 0, 2, 2, 2, 2], dtype=float))


def test_build_streams_skips_malformed_unit_count():
    n_units = 3
    bad = (np.ones((100, 5)), np.linspace(0, 80, 11), np.arange(0, 110, 10.0))
    good = _trial(100, n_units, 110, 80)
    s = dataio.build_temporal_streams(iter([bad, good]), n_units, CFG25)
    assert s.spikes.shape == (4, n_units)            # only the good trial
    assert np.array_equal(s.trial_idx, np.array([1, 1, 1, 1], dtype=float))


def test_running_mask_and_trial_ids_over_two_trials():
    """The running mask + per-bin trial id index the sliding window correctly."""
    n_units = 2
    trials = [
        _trial(800, n_units, 810, 200),   # 32 bins, fast -> running throughout
        _trial(800, n_units, 810, 200),   # 32 bins
    ]
    s = dataio.build_temporal_streams(iter(trials), n_units, CFG25)
    run = (~np.isnan(s.trial_idx)) & (s.vel >= CFG25.velocity_thresh_cm_s)
    assert np.all(run)                                # both trials are running
    assert set(np.unique(s.trial_idx[run]).tolist()) == {0.0, 1.0}
