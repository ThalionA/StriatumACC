"""The one trial layer: MATLAB's good-trial numbering, the learning point, the
disengagement point and the three epoch windows, mapped onto raw trial indices.

Synthetic sessions with known answers, then one parity check against the real
preprocessed struct where it is on this machine.
"""

import numpy as np
import pytest

from striatum_lfp import analysis, config, trials


def _session(n_raw=120, bad=(), lp=40, dp=np.nan, has_data=None, lp_source="measured"):
    good = np.ones(n_raw, bool)
    good[list(bad)] = False
    return trials.SessionTrials(mouse=1, matlab_good=good, lp=lp, lp_source=lp_source,
                                dp=dp, has_data=has_data)


def test_clean_session_reproduces_epoch_indices():
    s = _session(lp=40)
    naive, inter, expert = analysis.epoch_indices(40, 120)
    ep = s.epochs()
    assert list(ep) == list(trials.EPOCHS)
    np.testing.assert_array_equal(ep["Naive"], naive - 1)
    np.testing.assert_array_equal(ep["Intermediate"], inter - 1)
    np.testing.assert_array_equal(ep["Expert"], expert - 1)


def test_windows_count_good_trials_and_skip_a_bad_one():
    """1212's case: raw trial 5 is not good, so every later good trial sits one
    raw index further on. The learning point is on the good numbering."""
    s = _session(bad=(5,), lp=12)
    ep = s.epochs()
    np.testing.assert_array_equal(ep["Naive"], [0, 1, 2, 3, 4, 6, 7, 8, 9, 10])
    np.testing.assert_array_equal(ep["Intermediate"], [1, 2, 3, 4, 6, 7, 8, 9, 10, 11])
    np.testing.assert_array_equal(ep["Expert"], np.arange(12, 22))
    assert 5 not in s.usable()


def test_disengagement_clips_usable_trials_and_drops_a_crossing_window():
    """DP is a raw trial number: trials after it are not analysed, and a window
    that would reach past it does not exist (418: LP 26, DP 27)."""
    s = _session(lp=26, dp=27)
    np.testing.assert_array_equal(s.usable(), np.arange(27))
    ep = s.epochs()
    assert ep["Expert"].size == 0
    np.testing.assert_array_equal(ep["Intermediate"], np.arange(15, 25))


def test_missing_disengagement_point_means_no_clip():
    """MATLAB's convention: ``min([change_point_mean, n_trials])`` ignores NaN."""
    s = _session(dp=np.nan)
    np.testing.assert_array_equal(s.usable(), np.arange(120))


def test_window_without_recording_coverage_does_not_exist():
    """407's export stops before its session does: a window that needs a trial
    the recording does not cover is dropped, not silently shortened."""
    has_data = np.ones(120, bool)
    has_data[45:] = False
    s = _session(lp=40, has_data=has_data)
    ep = s.epochs()
    assert ep["Expert"].size == 0
    assert ep["Intermediate"].size == 10
    np.testing.assert_array_equal(s.usable(), np.arange(45))


def test_short_coverage_array_is_padded_as_no_data():
    """The cubes store at most 200 trials; anything beyond has no data."""
    s = _session(n_raw=260, lp=40, has_data=np.ones(200, bool))
    assert s.usable().max() == 199


def test_non_learner_without_a_learning_point_has_only_naive():
    ep = _session(lp=None).epochs()
    assert ep["Naive"].size == 10
    assert ep["Intermediate"].size == 0 and ep["Expert"].size == 0


def test_with_data_returns_a_new_session():
    s = _session()
    covered = s.with_data(np.zeros(120, bool))
    assert s.usable().size == 120
    assert covered.usable().size == 0


def test_epoch_of_labels_raw_trials():
    s = _session(lp=40)
    labels = s.epoch_of()
    assert labels.shape == (120,)
    assert labels[0] == "Naive" and labels[39] == "Expert" and labels[29] == "Intermediate"
    assert labels[60] == ""


# --- real preprocessed struct -------------------------------------------------

@pytest.mark.parametrize("cohort_name", ["task", "control"])
def test_matlab_good_mask_matches_matlab_trial_count(cohort_name):
    ch = config.get_cohort(cohort_name)
    if not ch.preproc_mat.exists():
        pytest.skip("preprocessed struct absent")
    sessions = trials.cohort_sessions(ch)
    counts = analysis.cohort_trial_counts(ch)
    assert set(sessions) == set(ch.mouse_ids)
    for mouse, s in sessions.items():
        assert int(s.matlab_good.sum()) == counts[mouse], mouse


def test_1212_raw_trial_102_is_not_good():
    if not config.TASK.preproc_mat.exists():
        pytest.skip("preprocessed struct absent")
    s = trials.cohort_sessions(config.TASK)[1212]
    assert not s.matlab_good[102]
    assert np.isnan(s.dp)


def test_only_the_trial_layer_builds_windows():
    """Seven drivers each re-derived epochs, DP and trial counts before 2026-09-25,
    and they disagreed. Any driver that builds its own windows again fails here."""
    from pathlib import Path

    project = Path(__file__).resolve().parents[2]
    drivers = [*project.glob("lfp/scripts/*.py"), *project.glob("infotheory/scripts/*.py")]
    forbidden = ("epoch_indices(", "disengagement_points(", "cohort_learning_points(",
                 "cohort_trial_counts(", "NAIVE_SPLIT", "EPOCH_NAMES")
    offenders = [f"{p.relative_to(project)}: {tok}" for p in drivers
                 for tok in forbidden if tok in p.read_text()]
    assert not offenders, offenders
