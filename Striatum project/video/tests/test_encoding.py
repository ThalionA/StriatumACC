"""Cross-validated position vs position+movement encoding, on synthetic units
whose drive is known."""

import numpy as np

from striatum_video.encoding import cv_delta_r2

N_TRIALS, N_BINS = 60, 50


# The exchangeable within-bin trial shuffle, kept HERE only as the counterexample:
# it ignores slow drift shared by firing and movement (see the drift tests).
def permute_within_bin(movement, bins, trials, rng):
    """Null: each trial's movement rows are replaced, bin by bin, by those of a
    randomly permuted partner trial. Every bin keeps its own movement values
    (so position-linked movement survives) but the trial-specific link to
    firing is broken. Rows must be ordered consistently within trial."""
    out = movement.copy()
    uniq = np.unique(trials)
    partner = dict(zip(uniq, rng.permutation(uniq)))
    index = {(t, b): i for i, (t, b) in enumerate(zip(trials, bins))}
    for i, (t, b) in enumerate(zip(trials, bins)):
        j = index.get((partner[t], b))
        out[i] = movement[j] if j is not None else movement[i]
    return out


def _session(seed=0):
    rng = np.random.default_rng(seed)
    bins = np.tile(np.arange(N_BINS), N_TRIALS)
    trials = np.repeat(np.arange(N_TRIALS), N_BINS)
    # movement varies trial to trial on top of a position profile
    speed = 10 + 5 * np.sin(bins / 8) + rng.normal(0, 3, bins.size) + np.repeat(rng.normal(0, 3, N_TRIALS), N_BINS)
    other = rng.normal(0, 1, bins.size)
    movement = np.column_stack([speed, other])
    tuning = np.exp(-((bins - 20) / 5.0) ** 2) * 10
    return rng, bins, trials, movement, tuning


def test_a_speed_driven_unit_gains_r2_from_movement():
    rng, bins, trials, movement, tuning = _session()
    fr = tuning + 0.8 * (movement[:, 0] - movement[:, 0].mean()) + rng.normal(0, 1, bins.size)
    r2_pos, r2_full, _ = cv_delta_r2(fr, bins, movement, trials, n_folds=5)
    assert r2_full - r2_pos > 0.2


def test_a_position_only_unit_gains_nothing():
    rng, bins, trials, movement, tuning = _session(1)
    fr = tuning + rng.normal(0, 1, bins.size)
    r2_pos, r2_full, _ = cv_delta_r2(fr, bins, movement, trials, n_folds=5)
    assert r2_pos > 0.5
    assert abs(r2_full - r2_pos) < 0.01


def test_heldout_predictions_cover_every_row_once():
    rng, bins, trials, movement, tuning = _session(2)
    fr = tuning + rng.normal(0, 1, bins.size)
    _, _, preds = cv_delta_r2(fr, bins, movement, trials, n_folds=5)
    assert preds["pos"].shape == fr.shape and np.all(np.isfinite(preds["full"]))


def test_permute_within_bin_keeps_each_bins_values_but_moves_trials():
    rng, bins, trials, movement, _ = _session(3)
    perm = permute_within_bin(movement, bins, trials, rng)
    for b in (0, 17, 49):
        m = bins == b
        np.testing.assert_allclose(np.sort(perm[m, 0]), np.sort(movement[m, 0]))
    assert not np.allclose(perm[:, 0], movement[:, 0])


def test_the_null_kills_a_real_movement_effect():
    rng, bins, trials, movement, tuning = _session(4)
    fr = tuning + 0.8 * (movement[:, 0] - movement[:, 0].mean()) + rng.normal(0, 1, bins.size)
    r2_pos, _, _ = cv_delta_r2(fr, bins, movement, trials, n_folds=5)
    _, r2_null, _ = cv_delta_r2(fr, bins, permute_within_bin(movement, bins, trials, rng), trials, n_folds=5)
    assert r2_null - r2_pos < 0.02


def _drifting_session(seed=5):
    """Firing and movement both drift slowly across trials; no within-session
    coupling beyond the shared drift."""
    rng, bins, trials, movement, tuning = _session(seed)
    drift = np.repeat(np.linspace(-1, 1, N_TRIALS), N_BINS)
    movement = movement.copy()
    movement[:, 0] += 6 * drift
    fr = tuning + 3 * drift + rng.normal(0, 1, bins.size)
    return rng, bins, trials, movement, fr


def test_shared_slow_drift_fools_the_exchangeable_null():
    """Documents the bug the drift terms fix: without them, drift alone passes."""
    rng, bins, trials, movement, fr = _drifting_session()
    r2_pos, r2_full, _ = cv_delta_r2(fr, bins, movement, trials)
    null = [cv_delta_r2(fr, bins, permute_within_bin(movement, bins, trials, rng), trials)[1] - r2_pos
            for _ in range(20)]
    assert r2_full - r2_pos > np.quantile(null, 0.95)


def test_drift_terms_and_circular_null_reject_shared_drift():
    from striatum_video.encoding import circular_shift_trials, trial_drift_basis
    _rng, bins, trials, movement, fr = _drifting_session()
    drift = trial_drift_basis(trials, n_basis=5)
    r2_pos, r2_full, _ = cv_delta_r2(fr, bins, movement, trials, nuisance=drift)
    uniq = np.unique(trials).size
    null = [cv_delta_r2(fr, bins, circular_shift_trials(movement, bins, trials, k), trials, nuisance=drift)[1] - r2_pos
            for k in range(10, uniq - 10, 2)]
    assert r2_full - r2_pos <= np.quantile(null, 0.95)


def test_drift_terms_keep_a_real_trial_by_trial_movement_effect():
    from striatum_video.encoding import circular_shift_trials, trial_drift_basis
    _rng, bins, trials, movement, fr = _drifting_session(6)
    fr = fr + 0.8 * (movement[:, 0] - movement[:, 0].mean())
    drift = trial_drift_basis(trials, n_basis=5)
    r2_pos, r2_full, _ = cv_delta_r2(fr, bins, movement, trials, nuisance=drift)
    null = [cv_delta_r2(fr, bins, circular_shift_trials(movement, bins, trials, k), trials, nuisance=drift)[1] - r2_pos
            for k in range(10, np.unique(trials).size - 10, 2)]
    assert r2_full - r2_pos > np.quantile(null, 0.95)


def test_circular_shift_moves_whole_trials_and_keeps_bins():
    from striatum_video.encoding import circular_shift_trials
    _, bins, trials, movement, _ = _session(7)
    out = circular_shift_trials(movement, bins, trials, 3)
    first_trial = trials == 0
    np.testing.assert_allclose(out[first_trial], movement[trials == 3])
