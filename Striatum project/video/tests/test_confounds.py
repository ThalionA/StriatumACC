"""Movement confounds in the cca pipeline's layouts (spatial tensor; temporal
ragged per-trial list with the spike bins' lengths)."""

import numpy as np

from striatum_video.confounds import spatial_confound, temporal_confound


def test_spatial_confound_stacks_named_covariates_per_trial_and_bin():
    binned = {"a": np.arange(6.0).reshape(2, 3), "b": -np.arange(6.0).reshape(2, 3)}
    out = spatial_confound(binned, ("a", "b"))
    assert out.shape == (2, 3, 2)
    np.testing.assert_array_equal(out[1, 2], [5.0, -5.0])


def test_temporal_confound_matches_spike_bin_lengths_and_lags_within_trial():
    # one session, two trials; trial 1 has no spike bins (empty traversal)
    trial = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2])
    world = np.array([6, 12, 12, 12, 12, 6, 12, 12, 12])
    t_ms = np.array([0, 10, 30, 70, 110, 150, 160, 200, 240.0])
    vr = {"trial": trial, "world": world}
    sig = {"s": np.array([0, 0, 0, 40, 80, 0, 0, 40, 80.0])}  # 1 per ms after the 2nd corridor row
    out = temporal_confound(vr, t_ms, sig, ("s",), n_bins_per_trial=[4, 0], bin_ms=20, lags_ms=(0, 20))
    assert len(out) == 2 and out[1].shape == (0, 2)
    # corridor frames at -20, 0, 40, 80 ms carry 0, 0, 40, 80 -> centres 10..70 ms read 10..70
    np.testing.assert_allclose(out[0][:, 0], [10, 30, 50, 70])
    np.testing.assert_allclose(out[0][:, 1], [10, 10, 30, 50])  # lag +20 ms = one bin into the past


def test_shift_confound_keeps_nan_layout_and_trial_lengths_but_moves_values():
    from striatum_video.confounds import shift_confound
    tensor = np.arange(24.0).reshape(4, 3, 2)
    tensor[1, 2] = np.nan
    out = shift_confound(tensor, 0.5)
    np.testing.assert_array_equal(np.isnan(out), np.isnan(tensor))
    np.testing.assert_array_equal(np.sort(out[np.isfinite(out)]), np.sort(tensor[np.isfinite(tensor)]))
    assert not np.array_equal(out[np.isfinite(out)], tensor[np.isfinite(tensor)])
    ragged = [np.arange(6.0).reshape(3, 2), np.zeros((0, 2)), np.arange(6.0, 14.0).reshape(4, 2)]
    out_r = shift_confound(ragged, 0.5)
    assert [a.shape for a in out_r] == [a.shape for a in ragged]
    np.testing.assert_array_equal(np.sort(np.concatenate(out_r).ravel()), np.arange(14.0))
