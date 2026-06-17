"""Ground-truth tests for the striatum_tcca data layer (pure -- no h5py)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from striatum_tcca import config, dataio

CFG = config.DEFAULT
CFG_RAW25 = dataclasses.replace(CFG, temporal_bin_ms=25, gaussian_sd_ms=0.0)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def make_animal(animal_id=1, n_units=5, n_trials=40, change_point=np.nan,
                zerr=None, area_masks=None, ntypes=None):
    if zerr is None:
        zerr = np.zeros(n_trials)
    if area_masks is None:
        area_masks = {a: np.ones(n_units, dtype=bool) for a in config.AREAS}
    if ntypes is None:
        ntypes = np.ones((n_units, 5))            # all MSN (type 1)
    return dataio.Animal(
        animal_id=animal_id, neurontypes=ntypes, area_masks=area_masks,
        change_point=change_point, zscored_lick_errors=zerr, n_trials=n_trials,
        corridor_path=Path("/dev/null"),
    )


def _trial(n_1ms, n_units, total_ms, end_pos_au):
    """One synthetic trial: all-ones 1 ms spikes + a linear position ramp."""
    spikes = np.ones((n_1ms, n_units), dtype=float)
    times = np.arange(0, total_ms, 10, dtype=float)
    position = np.linspace(0.0, end_pos_au, times.size)
    return spikes, position, times


# ---------------------------------------------------------------------------
# rebin_trial: smooth + bin
# ---------------------------------------------------------------------------
def test_rebin_sums_within_bins_raw():
    spikes = np.zeros((100, 2))
    spikes[:, 0] = 1.0
    spikes[:, 1] = 2.0
    out = dataio.rebin_trial(spikes, bin_ms=50, gaussian_sd_ms=0.0)
    assert out.shape == (2, 2)
    np.testing.assert_array_equal(out[:, 0], [50, 50])
    np.testing.assert_array_equal(out[:, 1], [100, 100])


def test_rebin_drops_incomplete_final_bin():
    out = dataio.rebin_trial(np.ones((110, 1)), bin_ms=50, gaussian_sd_ms=0.0)
    assert out.shape == (2, 1)            # last 10 ms dropped


def test_rebin_gaussian_conserves_mass_in_interior():
    # Spikes confined to the interior so the kernel never leaks past the edges:
    # smoothing must preserve the total count.
    spikes = np.zeros((300, 1))
    spikes[100:200, 0] = 1.0              # 100 spikes, well inside
    raw = dataio.rebin_trial(spikes, bin_ms=50, gaussian_sd_ms=0.0)
    smooth = dataio.rebin_trial(spikes, bin_ms=50, gaussian_sd_ms=2.5)
    assert smooth.shape == raw.shape == (6, 1)
    assert smooth.sum() == raw.sum() == 100
    # Smoothing redistributes but the totals match.
    np.testing.assert_allclose(smooth.sum(), 100.0, atol=1e-4)


def test_rebin_sigma_zero_equals_raw():
    spikes = (np.arange(200) % 3).astype(float)[:, None]
    a = dataio.rebin_trial(spikes, 50, gaussian_sd_ms=0.0)
    b = dataio.rebin_trial(spikes, 50)    # default sigma 0
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# trial_velocity (ported from the round-17 test_velocity.py)
# ---------------------------------------------------------------------------
def test_velocity_constant_speed_ramp():
    times = np.arange(0, 1001, 10, dtype=float)
    position = times / 1000.0 * 100.0     # 0..100 a.u. over 1 s
    vel = dataio.trial_velocity(position, times, n_bins=40, cfg=CFG_RAW25)
    # 100 a.u. * 1.25 cm/a.u. / 1 s = 125 cm/s.
    assert vel.shape == (40,)
    np.testing.assert_allclose(vel, 125.0, atol=1e-6)


def test_velocity_zero_when_stationary():
    times = np.arange(0, 1001, 10, dtype=float)
    position = np.full_like(times, 42.0)
    np.testing.assert_allclose(
        dataio.trial_velocity(position, times, 40, CFG_RAW25), 0.0)


def test_velocity_rezeroes_times_to_corridor_onset():
    times = np.arange(5000, 6001, 10, dtype=float)        # 5 s offset
    position = (times - times[0]) / 1000.0 * 100.0
    vel = dataio.trial_velocity(position, times, 40, CFG_RAW25)
    np.testing.assert_allclose(vel, 125.0, atol=1e-6)


def test_velocity_degenerate_returns_zeros():
    assert np.array_equal(dataio.trial_velocity([], [], 5, CFG_RAW25), np.zeros(5))
    assert np.array_equal(dataio.trial_velocity([1.0], [0.0], 1, CFG_RAW25),
                          np.zeros(1))


# ---------------------------------------------------------------------------
# build_temporal_streams
# ---------------------------------------------------------------------------
def test_build_streams_concatenates_and_labels_trials():
    n_units = 3
    trials = [_trial(100, n_units, 110, 80), _trial(150, n_units, 160, 120)]
    s = dataio.build_temporal_streams(iter(trials), n_units, CFG_RAW25)
    assert s.spikes.shape == (10, n_units)            # 4 + 6 bins at 25 ms
    np.testing.assert_allclose(s.spikes, 25.0)        # all-ones -> 25/bin
    np.testing.assert_array_equal(
        s.trial_idx, np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=float))
    assert s.vel.shape == (10,) and np.all(s.vel > 0)


def test_build_streams_drops_overlong_but_keeps_index():
    n_units = 2
    cfg = dataclasses.replace(CFG_RAW25, temporal_max_trial_ms=100)   # max 4 bins
    trials = [_trial(100, n_units, 110, 80),     # kept, id 0
              _trial(150, n_units, 160, 120),    # over-long, dropped
              _trial(100, n_units, 110, 80)]     # kept, id 2
    s = dataio.build_temporal_streams(iter(trials), n_units, cfg)
    assert s.spikes.shape == (8, n_units)
    np.testing.assert_array_equal(
        s.trial_idx, np.array([0, 0, 0, 0, 2, 2, 2, 2], dtype=float))


def test_build_streams_skips_malformed_unit_count():
    n_units = 3
    bad = (np.ones((100, 5)), np.linspace(0, 80, 11), np.arange(0, 110, 10.0))
    good = _trial(100, n_units, 110, 80)
    s = dataio.build_temporal_streams(iter([bad, good]), n_units, CFG_RAW25)
    assert s.spikes.shape == (4, n_units)            # only the good trial
    np.testing.assert_array_equal(s.trial_idx, np.array([1, 1, 1, 1], dtype=float))


def test_build_streams_start_index_offsets_trial_ids():
    n_units = 2
    trials = [_trial(100, n_units, 110, 80)]
    s = dataio.build_temporal_streams(iter(trials), n_units, CFG_RAW25,
                                      start_index=7)
    np.testing.assert_array_equal(s.trial_idx, np.full(4, 7.0))


# ---------------------------------------------------------------------------
# learning point + cohort
# ---------------------------------------------------------------------------
def test_learning_point_returns_first_sub_threshold_sustained_trial():
    # Trials 0,1 above threshold; sustained learning from trial idx 2 (1-indexed 3).
    z = np.array([1, 1, -3, -3, -3, -3, -3, -3, -3, 1, 1, 1], dtype=float)
    assert dataio.find_learning_point(z, CFG) == 3


def test_learning_point_none_when_never_sustained():
    z = np.zeros(30)
    assert dataio.find_learning_point(z, CFG) is None


def test_classify_cohort_learner_nonlearner_yoked():
    cfg = dataclasses.replace(CFG, manual_nonlearners=frozenset({3}))
    z_learn = np.array([1, 1, -3, -3, -3, -3, -3, -3, -3, 1, 1, 1], dtype=float)
    a1 = make_animal(animal_id=1, zerr=z_learn)            # learner, lp=3
    a2 = make_animal(animal_id=2, zerr=np.zeros(12))       # no lp -> nonlearner
    a3 = make_animal(animal_id=3, zerr=z_learn)            # lp=3 but manual non
    entries, yoked = dataio.classify_cohort([a1, a2, a3], cfg)
    assert entries[1].role == "learner" and entries[1].lp == 3
    assert entries[2].role == "nonlearner" and entries[2].lp == yoked
    assert entries[3].role == "nonlearner"                 # forced
    assert yoked == 3                                       # mean of learners = {3}


# ---------------------------------------------------------------------------
# epochs + engagement periods
# ---------------------------------------------------------------------------
def test_epoch_windows_three_disjoint_blocks():
    w = dataio.epoch_windows(lp=21, n_usable=35, cfg=CFG)
    np.testing.assert_array_equal(w["naive"], np.arange(0, 10))
    np.testing.assert_array_equal(w["intermediate"], np.arange(11, 21))
    np.testing.assert_array_equal(w["expert"], np.arange(21, 31))


def test_epoch_windows_none_when_overlap_or_overrun():
    assert dataio.epoch_windows(lp=15, n_usable=40, cfg=CFG) is None     # overlap
    assert dataio.epoch_windows(lp=30, n_usable=35, cfg=CFG) is None     # overrun


def test_n_usable_trials_truncates_at_change_point():
    assert dataio.n_usable_trials(make_animal(n_trials=40, change_point=np.nan)) == 40
    assert dataio.n_usable_trials(make_animal(n_trials=40, change_point=25.4)) == 25
    assert dataio.n_usable_trials(make_animal(n_trials=40, change_point=50)) == 40


def test_period_trial_range():
    a = make_animal(n_trials=40, change_point=25)
    assert dataio.period_trial_range(a, "engaged") == (0, 25)
    assert dataio.period_trial_range(a, "disengaged") == (25, 40)
    assert dataio.period_trial_range(a, "all") == (0, 40)
    # No disengagement -> empty disengaged range.
    b = make_animal(n_trials=40, change_point=np.nan)
    assert dataio.period_trial_range(b, "disengaged") == (40, 40)


# ---------------------------------------------------------------------------
# unit selection
# ---------------------------------------------------------------------------
def test_select_units_excludes_fast_spiking_in_every_area():
    masks = {a: np.zeros(5, dtype=bool) for a in config.AREAS}
    masks["DMS"] = np.array([1, 1, 1, 0, 0], dtype=bool)
    ntypes = np.ones((5, 5))
    ntypes[:, config.NTYPE_COL] = [1, 2, 1, 2, 1]         # units 1,3 are FS
    a = make_animal(area_masks=masks, ntypes=ntypes)
    excl = dataio.select_units(a, "DMS", dataclasses.replace(CFG, exclude_fast_spiking=True))
    incl = dataio.select_units(a, "DMS", dataclasses.replace(CFG, exclude_fast_spiking=False))
    np.testing.assert_array_equal(excl, [0, 2])           # FS unit 1 dropped
    np.testing.assert_array_equal(incl, [0, 1, 2])
