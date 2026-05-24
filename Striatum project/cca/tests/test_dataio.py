"""Unit tests for data IO and cohort logic, against synthetic animals."""

from __future__ import annotations

import numpy as np

from striatum_cca import config, dataio

CFG = config.DEFAULT


# ---------------------------------------------------------------------------
# synthetic animal builder
# ---------------------------------------------------------------------------
def make_animal(animal_id, n_trials, units_per_area, change_point, z, fs_units=()):
    """Construct an Animal with contiguous per-area unit blocks.

    ``units_per_area`` maps area name -> unit count. ``fs_units`` is a set of
    global unit indices to mark as fast-spiking (neurontype 2).
    """
    n_units = sum(units_per_area.values())
    rng = np.random.default_rng(animal_id)
    spatial_fr = rng.standard_normal((n_trials, config.N_BINS, n_units))

    masks = {a: np.zeros(n_units, dtype=bool) for a in config.AREAS}
    start = 0
    for area in config.AREAS:
        count = units_per_area.get(area, 0)
        masks[area][start : start + count] = True
        start += count

    neurontypes = np.full((n_units, 5), 5.0)    # default type 5 (RS)
    for u in fs_units:
        neurontypes[u, config.NTYPE_COL] = config.FS_TYPE_CODE

    return dataio.Animal(
        animal_id=animal_id,
        spatial_fr=spatial_fr,
        neurontypes=neurontypes,
        area_masks=masks,
        change_point=change_point,
        zscored_lick_errors=np.asarray(z, dtype=float),
        n_trials=n_trials,
    )


# ---------------------------------------------------------------------------
# learning point detection
# ---------------------------------------------------------------------------
def test_find_learning_point_is_first_sustained_below_trial():
    # Error drops to -3 from 0-indexed trial 30; that trial is itself below
    # threshold and starts a sustained run -> LP = 31 (1-indexed).
    z = np.concatenate([np.zeros(30), np.full(40, -3.0)])
    lp = dataio.find_learning_point(z, CFG)
    assert lp == 31


def test_find_learning_point_skips_above_threshold_window_start():
    # The window [t, t+10) qualifies, but trial t is ABOVE threshold; the LP
    # must be the first below-threshold trial, not the window start.
    z = np.concatenate([np.zeros(30), [1.0], np.full(40, -3.0)])
    lp = dataio.find_learning_point(z, CFG)
    assert lp == 32                              # 0-indexed 31, z = -3
    assert z[lp - 1] <= CFG.lp_z_threshold


def test_find_learning_point_returns_none_when_never_learned():
    z = np.full(80, 1.0)
    assert dataio.find_learning_point(z, CFG) is None


def test_find_learning_point_needs_enough_within_window():
    # Only 6 of every 10 trials are below threshold -> never reaches 7.
    z = np.tile([-3, -3, -3, -3, -3, -3, 1, 1, 1, 1], 8).astype(float)
    assert dataio.find_learning_point(z, CFG) is None


# ---------------------------------------------------------------------------
# cohort classification
# ---------------------------------------------------------------------------
def test_classify_cohort_splits_learners_and_yokes_nonlearners():
    # First below-threshold trial at 0-indexed 39 / 59 -> LP 40 / 60.
    learner_a = make_animal(1, 130, {"DMS": 10}, np.nan,
                            np.concatenate([np.zeros(39), np.full(80, -3.0)]))
    learner_b = make_animal(2, 130, {"DMS": 10}, np.nan,
                            np.concatenate([np.zeros(59), np.full(60, -3.0)]))
    non_learner = make_animal(3, 130, {"DMS": 10}, np.nan, np.full(120, 1.0))
    entries, yoked = classify_with([learner_a, learner_b, non_learner])

    assert entries[1].role == "learner" and entries[1].raw_lp == 40
    assert entries[2].role == "learner" and entries[2].raw_lp == 60
    assert entries[3].role == "nonlearner"
    assert yoked == 50                           # mean of LPs 40 and 60
    assert entries[3].lp == 50
    assert entries[3].raw_lp is None


def test_classify_cohort_forces_manual_nonlearner():
    # Animal 8 is in cfg.manual_nonlearners even though it has a detected LP.
    a8 = make_animal(8, 130, {"DMS": 10}, np.nan,
                     np.concatenate([np.zeros(39), np.full(80, -3.0)]))   # LP 40
    other = make_animal(1, 130, {"DMS": 10}, np.nan,
                        np.concatenate([np.zeros(49), np.full(70, -3.0)]))  # LP 50
    entries, yoked = classify_with([a8, other])
    assert entries[8].role == "nonlearner"
    assert entries[8].raw_lp == 40               # LP was detected ...
    assert yoked == 50                           # ... excluded from the yoke ...
    assert entries[8].lp == 50                   # ... and yoked instead


def classify_with(animals):
    return dataio.classify_cohort(animals, CFG)


# ---------------------------------------------------------------------------
# usable trials / epoch windows
# ---------------------------------------------------------------------------
def test_n_usable_trials_truncates_at_change_point():
    a = make_animal(1, 200, {"DMS": 5}, 120.4, np.zeros(200))
    assert dataio.n_usable_trials(a) == 120


def test_n_usable_trials_uses_full_length_when_no_change_point():
    a = make_animal(1, 90, {"DMS": 5}, np.nan, np.zeros(90))
    assert dataio.n_usable_trials(a) == 90


def test_epoch_windows_are_disjoint_and_correct():
    win = dataio.epoch_windows(lp=42, n_usable=100, cfg=CFG)
    assert win is not None
    assert set(win) == {"naive", "expert"}
    assert np.array_equal(win["naive"], np.arange(0, 10))
    assert np.array_equal(win["expert"], np.arange(42, 52))


def test_epoch_windows_valid_for_fast_learner():
    # Intermediate is gone (round 7): naive vs expert only needs the two
    # windows not to overlap, so a fast learner (10 <= lp < 20) is now valid.
    win = dataio.epoch_windows(lp=12, n_usable=100, cfg=CFG)
    assert win is not None
    assert set(win) == {"naive", "expert"}
    assert np.array_equal(win["naive"], np.arange(0, 10))
    assert np.array_equal(win["expert"], np.arange(12, 22))


def test_epoch_windows_boundary_lp_equals_trials_per_epoch():
    # lp == trials_per_epoch: naive [0, 10) and expert [10, 20) just abut.
    win = dataio.epoch_windows(lp=10, n_usable=100, cfg=CFG)
    assert win is not None
    assert np.array_equal(win["expert"], np.arange(10, 20))


def test_epoch_windows_reject_overlap_when_lp_too_early():
    # lp < trials_per_epoch would make naive and expert overlap.
    assert dataio.epoch_windows(lp=9, n_usable=100, cfg=CFG) is None


def test_epoch_windows_reject_when_expert_exceeds_trials():
    assert dataio.epoch_windows(lp=42, n_usable=48, cfg=CFG) is None


# ---------------------------------------------------------------------------
# temporal re-binning (round 8)
# ---------------------------------------------------------------------------
def test_rebin_trial_sums_1ms_counts():
    # 6 ms of one unit, binned to 2 ms -> 3 bins, each the sum of its 2 ms.
    spikes = np.array([[1], [0], [2], [1], [0], [0]], dtype=np.uint16)
    out = dataio.rebin_trial(spikes, bin_ms=2)
    assert out.shape == (3, 1)
    assert np.array_equal(out.ravel(), [1.0, 3.0, 0.0])


def test_rebin_trial_drops_incomplete_final_bin():
    spikes = np.ones((7, 2), dtype=np.uint16)        # 7 ms, bin 2 -> 3 full bins
    out = dataio.rebin_trial(spikes, bin_ms=2)
    assert out.shape == (3, 2)
    assert np.all(out == 2.0)


def test_pad_trials_nan_pads_to_longest_no_truncation():
    # variable-length per-trial arrays -> NaN-padded to the longest; the long
    # trial is kept whole (no truncation, no window).
    a = np.ones((3, 2), dtype=np.float32)
    b = np.full((5, 2), 2.0, dtype=np.float32)
    out = dataio.pad_trials([a, b])
    assert out.shape == (2, 5, 2)
    assert np.all(out[0, :3] == 1.0)
    assert np.all(np.isnan(out[0, 3:]))
    assert np.all(out[1] == 2.0)


# ---------------------------------------------------------------------------
# unit selection / area tensors
# ---------------------------------------------------------------------------
def test_select_units_excludes_fast_spiking():
    # DMS = global units 0..9; mark unit 3 and 7 as fast-spiking.
    a = make_animal(1, 60, {"DMS": 10, "ACC": 10}, np.nan, np.zeros(60),
                    fs_units=(3, 7))
    kept = dataio.select_units(a, "DMS", CFG)
    assert np.array_equal(kept, np.array([0, 1, 2, 4, 5, 6, 8, 9]))


def test_select_units_keeps_fast_spiking_when_disabled():
    a = make_animal(1, 60, {"DMS": 10}, np.nan, np.zeros(60), fs_units=(3,))
    cfg = config.Config(exclude_fast_spiking=False)
    assert dataio.select_units(a, "DMS", cfg).size == 10


def test_area_tensor_shape_and_truncation():
    a = make_animal(1, 200, {"DMS": 12, "ACC": 8}, 80.0, np.zeros(200),
                    fs_units=(0,))
    tensor, idx = dataio.area_tensor(a, "DMS", CFG)
    assert tensor.shape == (80, config.N_BINS, 11)   # 80 usable trials, 1 FS dropped
    assert idx.size == 11
