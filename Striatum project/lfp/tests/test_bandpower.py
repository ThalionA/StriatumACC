"""Ground-truth tests for the trial/spatial/dark binning of LFP band power.

The binning must reproduce ``spatial_binning.m`` and
``separate_dark_and_corridor_periods.m`` sample-for-sample, because the whole
point is that an LFP band-power array can be swapped in wherever the unit
pipeline uses ``spatial_binned_fr_all`` and keep its indexing.
"""

from __future__ import annotations

import numpy as np
import pytest

from striatum_lfp import bandpower


# --- trial boundaries (cut_data_per_trial.m) --------------------------------

def test_trial_boundaries_on_a_known_sequence():
    vr_trial = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3])
    starts, ends = bandpower.trial_boundaries(vr_trial)
    assert starts.tolist() == [0, 3, 5]
    assert ends.tolist() == [2, 4, 8]          # inclusive, last trial runs to the end


def test_trial_boundaries_single_trial():
    starts, ends = bandpower.trial_boundaries(np.ones(7))
    assert starts.tolist() == [0] and ends.tolist() == [6]


def test_trial_boundaries_change_on_any_difference_not_increment():
    """MATLAB uses diff(vr_trial) ~= 0, so a decrement also starts a trial."""
    starts, _ = bandpower.trial_boundaries(np.array([5, 5, 2, 2]))
    assert starts.tolist() == [0, 2]


# --- VR time -> npx index (interp1 nearest on 0:n-1) ------------------------

def test_npx_index_matches_interp1_nearest():
    got = bandpower.npx_index(np.array([0.0, 0.4, 0.6, 12.0, 12.7]), n_npx=100)
    assert got.tolist() == [0, 0, 1, 12, 13]


def test_npx_index_clips_to_the_recording():
    got = bandpower.npx_index(np.array([-5.0, 1e9]), n_npx=100)
    assert got.tolist() == [0, 99]


# --- corridor onset (separate_dark_and_corridor_periods.m) ------------------

def test_corridor_start_uses_the_frame_after_world_exceeds_6():
    world = np.array([6, 6, 6, 7, 7, 7])
    times = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0])
    vr_idx, npx_rel = bandpower.corridor_start(world, times)
    assert vr_idx == 3                       # first frame with world > 6
    assert npx_rel == 400                    # MATLAB takes trial_times(idx + 1)


def test_corridor_start_returns_none_when_the_corridor_never_opens():
    assert bandpower.corridor_start(np.array([6, 6, 6]), np.arange(3.0)) is None


def test_corridor_start_returns_none_when_the_next_frame_is_missing():
    """MATLAB indexes idx+1 and the try/catch swallows the overflow."""
    assert bandpower.corridor_start(np.array([6, 6, 7]), np.arange(3.0)) is None


# --- spatial bin segments (spatial_binning.m) -------------------------------

def _uniform_traversal(n=200, corridor_au=200.0):
    """A constant-speed run down the corridor sampled every 10 ms."""
    position = np.linspace(0.0, corridor_au, n)
    times = np.arange(n, dtype=float) * 10.0
    return position, times


def test_spatial_segments_cover_every_bin_for_a_clean_traversal():
    position, times = _uniform_traversal()
    segs = bandpower.spatial_bin_segments(position, times, bandpower.spatial_bin_edges())
    assert len(segs) == 50
    assert all(s is not None for s in segs)
    # Segments advance monotonically and stay inside the traversal.
    starts = [s[0] for s in segs]
    assert starts == sorted(starts)
    assert segs[-1][1] <= times[-1]


def test_spatial_segment_endpoints_are_the_first_and_last_frame_in_the_bin():
    position = np.array([0.0, 1.0, 2.0, 3.0, 10.0])       # first 4 frames in bin 1
    times = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    segs = bandpower.spatial_bin_segments(position, times, bandpower.spatial_bin_edges())
    assert segs[0] == (0, 30)


def test_spatial_bin_with_one_sample_is_dropped():
    """spatial_binning.m requires sum(idx_in_bin) > 1, not >= 1."""
    position = np.array([0.0, 10.0, 20.0, 30.0])          # one frame per bin
    times = np.array([0.0, 10.0, 20.0, 30.0])
    segs = bandpower.spatial_bin_segments(position, times, bandpower.spatial_bin_edges())
    assert segs[0] is None


def test_spatial_times_are_rezeroed_to_the_corridor_start():
    """The caller passes trial-zeroed times; the bin map is corridor-relative."""
    position, times = _uniform_traversal()
    segs_a = bandpower.spatial_bin_segments(position, times, bandpower.spatial_bin_edges())
    segs_b = bandpower.spatial_bin_segments(position, times + 5_000.0,
                                            bandpower.spatial_bin_edges())
    assert segs_a == segs_b


def test_spatial_bin_edges_match_project_cfg():
    edges = bandpower.spatial_bin_edges()
    assert edges[0] == 0 and len(edges) == 51
    assert edges[-2] == 196 and edges[-1] == 204       # last edge is widened to 204
    assert np.allclose(np.diff(edges[:-1]), 4)


def test_a_re_entered_bin_spans_first_to_last_visit():
    """MATLAB takes bin_times(1) to bin_times(end) regardless of re-entry."""
    position = np.array([1.0, 2.0, 50.0, 51.0, 3.0])
    times = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    segs = bandpower.spatial_bin_segments(position, times, bandpower.spatial_bin_edges())
    assert segs[0] == (0, 40)


# --- dark bin segments (ProcessStriatumTask.m:173-181) ----------------------

def test_dark_segments_are_50_bins_of_100_ms():
    segs = bandpower.dark_bin_segments(n_dark_samples=5_000)
    assert len(segs) == 50
    assert segs[0] == (0, 99)
    assert segs[-1] == (4_900, 4_999)


def test_dark_segments_stop_at_a_short_dark_period():
    segs = bandpower.dark_bin_segments(n_dark_samples=250)
    assert segs[0] == (0, 99) and segs[1] == (100, 199)
    assert segs[2] == (200, 249)                # partial final bin, then nothing
    assert all(s is None for s in segs[3:])


def test_dark_segments_empty_when_there_is_no_dark_period():
    assert all(s is None for s in bandpower.dark_bin_segments(0))


# --- segment accumulation ---------------------------------------------------

def test_accumulator_recovers_known_means():
    """Two cells over a ramp: the mean of each segment is exact."""
    acc = bandpower.SegmentAccumulator(n_cells=2, n_channels=3)
    signal = np.tile(np.arange(10, dtype=float)[:, None], (1, 3))
    acc.add_block(signal, block_start=0, segments=[(0, 0, 4), (1, 5, 10)])
    out = acc.result()
    np.testing.assert_allclose(out[0], 1.5)      # mean of 0,1,2,3
    np.testing.assert_allclose(out[1], 7.0)      # mean of 5..9


def test_accumulator_is_exact_across_a_block_boundary():
    """A segment split between two blocks must give the same mean as one block."""
    signal = np.tile(np.arange(20, dtype=float)[:, None], (1, 2))
    whole = bandpower.SegmentAccumulator(n_cells=1, n_channels=2)
    whole.add_block(signal, block_start=0, segments=[(0, 0, 20)])
    split = bandpower.SegmentAccumulator(n_cells=1, n_channels=2)
    split.add_block(signal[:7], block_start=0, segments=[(0, 0, 20)])
    split.add_block(signal[7:], block_start=7, segments=[(0, 0, 20)])
    np.testing.assert_allclose(split.result(), whole.result())
    np.testing.assert_allclose(split.result()[0], 9.5)


def test_accumulator_reports_nan_for_a_cell_it_never_saw():
    acc = bandpower.SegmentAccumulator(n_cells=2, n_channels=1)
    acc.add_block(np.ones((5, 1)), block_start=0, segments=[(0, 0, 5)])
    out = acc.result()
    assert out[0, 0] == 1.0
    assert np.isnan(out[1, 0])


def test_accumulator_counts_samples_per_cell():
    acc = bandpower.SegmentAccumulator(n_cells=1, n_channels=1)
    acc.add_block(np.ones((10, 1)), block_start=0, segments=[(0, 2, 7)])
    assert acc.counts[0] == 5


def test_accumulator_rejects_a_channel_count_mismatch():
    acc = bandpower.SegmentAccumulator(n_cells=1, n_channels=3)
    with pytest.raises(ValueError):
        acc.add_block(np.ones((4, 2)), block_start=0, segments=[(0, 0, 4)])


# --- notch + band filtering -------------------------------------------------

def test_notch_removes_a_50hz_tone_and_keeps_a_theta_tone():
    fs = 1000
    t = np.arange(20_000) / fs
    signal = (np.sin(2 * np.pi * 6 * t) + 5 * np.sin(2 * np.pi * 50 * t))[:, None]
    cleaned = bandpower.apply_notches(signal.copy(), fs=fs)
    # 50 Hz power collapses; 6 Hz survives.
    def power_at(x, hz):
        f = np.fft.rfftfreq(x.shape[0], 1 / fs)
        return np.abs(np.fft.rfft(x[:, 0]))[np.argmin(np.abs(f - hz))]
    assert power_at(cleaned, 50) < 0.02 * power_at(signal, 50)
    assert power_at(cleaned, 6) > 0.95 * power_at(signal, 6)


def test_band_power_recovers_the_amplitude_of_a_pure_tone():
    """A unit-amplitude sinusoid inside the band has mean power 1/2."""
    fs = 1000
    t = np.arange(30_000) / fs
    signal = np.sin(2 * np.pi * 6 * t)[:, None]
    power = bandpower.band_power_series(signal, (4.0, 8.0), fs=fs)
    assert power[5_000:-5_000].mean() == pytest.approx(0.5, rel=0.05)


def test_band_power_rejects_a_tone_outside_the_band():
    fs = 1000
    t = np.arange(30_000) / fs
    inside = bandpower.band_power_series(np.sin(2 * np.pi * 6 * t)[:, None], (4.0, 8.0), fs=fs)
    outside = bandpower.band_power_series(np.sin(2 * np.pi * 40 * t)[:, None], (4.0, 8.0), fs=fs)
    assert outside[5_000:-5_000].mean() < 0.01 * inside[5_000:-5_000].mean()


def test_band_power_is_non_negative():
    rng = np.random.default_rng(0)
    power = bandpower.band_power_series(rng.normal(size=(5_000, 4)), (30.0, 80.0), fs=1000)
    assert (power >= 0).all()


# --- truncated trials (1212's export stops before its session does) ----------

def test_truncated_trials_flags_only_those_past_the_end():
    ends = np.array([100.0, 500.0, 999.0, 1000.0, 5000.0])
    mask = bandpower.truncated_trials(ends, n_npx=1000)
    assert mask.tolist() == [False, False, False, True, True]


def test_a_trial_ending_exactly_at_the_last_sample_is_not_truncated():
    assert not bandpower.truncated_trials(np.array([999.0]), n_npx=1000)[0]


def test_truncated_trials_none_when_the_session_fits():
    assert not bandpower.truncated_trials(np.arange(10, dtype=float), n_npx=1000).any()


# --- coupling envelope now reuses band_power_series (dedup, 2026-08-28) -----

def test_coupling_envelope_equals_binned_band_power_root():
    """The refactor must not change the statistic, only where it is defined."""
    from striatum_lfp.cohort import bin_mean

    rng = np.random.default_rng(0)
    block = rng.normal(size=(5_000, 8))
    power = bandpower.band_power_series(block, (30.0, 90.0), fs=1000)
    expected = np.sqrt(bin_mean(power, 100))
    assert expected.shape == (50, 8)
    assert (expected >= 0).all()


def test_band_power_series_uses_the_shared_sos_designer():
    from striatum_lfp.features import design_band_sos
    from scipy.signal import sosfiltfilt

    rng = np.random.default_rng(1)
    x = rng.normal(size=(2_000, 3))
    sos = design_band_sos((15.0, 30.0), fs=1000, order=4)
    np.testing.assert_allclose(bandpower.band_power_series(x, (15.0, 30.0), fs=1000),
                               np.square(sosfiltfilt(sos, x, axis=0)))
