"""Synthetic ground-truth tests for the LFP inventory diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from striatum_lfp import inventory


# --- depth vs geometry -------------------------------------------------------

def test_depth_matches_the_export_convention():
    depth = (np.arange(384) // 2) * 20.0
    ok, err = inventory.check_depth_against_geometry(depth, 384)
    assert ok and err == 0.0


def test_depth_mismatch_is_quantified_not_just_flagged():
    depth = (np.arange(384) // 2) * 20.0
    depth[10] += 40.0
    ok, err = inventory.check_depth_against_geometry(depth, 384)
    assert not ok
    assert err == pytest.approx(40.0)


def test_depth_wrong_length_is_not_a_match():
    ok, err = inventory.check_depth_against_geometry(np.zeros(10), 384)
    assert not ok and np.isnan(err)


# --- terminal padding --------------------------------------------------------

def test_padding_onset_finds_terminal_run():
    zeros = np.zeros(100)
    zeros[80:] = 1.0
    assert inventory.padding_onset_s(zeros) == 80.0


def test_padding_onset_none_when_file_ends_with_signal():
    zeros = np.zeros(100)
    zeros[40:50] = 1.0            # mid-file dropout, not padding
    assert inventory.padding_onset_s(zeros) is None


def test_padding_onset_whole_file_zero():
    assert inventory.padding_onset_s(np.ones(50)) == 0.0


# --- window placement --------------------------------------------------------

def test_window_starts_never_run_past_the_end():
    starts = inventory.window_starts(10_000, 5, 1_000)
    assert starts[0] == 0 and starts[-1] == 9_000
    assert np.all(starts + 1_000 <= 10_000)


def test_window_starts_respects_first_and_last():
    starts = inventory.window_starts(10_000, 3, 100, first=2_000, last=5_000)
    assert starts[0] == 2_000 and starts[-1] == 4_900


def test_window_starts_single_window():
    assert inventory.window_starts(10_000, 1, 100).tolist() == [0]


# --- spectra -----------------------------------------------------------------

def _psd_power_law(exponent: float, fmax: int = 500):
    freqs = np.arange(fmax + 1, dtype=float)
    with np.errstate(divide="ignore"):
        curve = np.where(freqs > 0, freqs ** exponent, 1.0)
    return freqs, np.column_stack([curve, curve])


def test_loglog_slope_recovers_known_exponent():
    freqs, psd = _psd_power_law(-2.0)
    assert inventory.loglog_slope(freqs, psd, (2.0, 40.0)) == pytest.approx(-2.0, abs=1e-6)


def test_loglog_slope_flat_for_white_noise_spectrum():
    freqs = np.arange(501, dtype=float)
    psd = np.ones((501, 4))
    assert inventory.loglog_slope(freqs, psd, (2.0, 40.0)) == pytest.approx(0.0, abs=1e-9)


def test_band_power_averages_only_inside_the_band():
    freqs = np.arange(11, dtype=float)
    psd = np.tile(freqs[:, None], (1, 3))
    np.testing.assert_allclose(inventory.band_power(freqs, psd, (2.0, 4.0)), 3.0)


def test_line_ratio_is_one_without_a_peak():
    freqs, psd = _psd_power_law(-1.0)
    assert inventory.line_ratio(freqs, psd, 50.0) == pytest.approx(1.0, abs=0.05)


def test_line_ratio_recovers_an_injected_peak():
    freqs, psd = _psd_power_law(-1.0)
    psd[freqs == 50.0] *= 8.0
    assert inventory.line_ratio(freqs, psd, 50.0) == pytest.approx(8.0, rel=0.1)


# --- spatial structure -------------------------------------------------------

def test_spatial_correlations_recover_a_known_depth_gradient():
    """Neighbours share a latent, channels 100 apart share nothing."""
    rng = np.random.default_rng(0)
    n, n_ch = 4_000, 200
    groups = rng.normal(size=(n, n_ch // 10))
    block = np.repeat(groups, 10, axis=1) + rng.normal(size=(n, n_ch)) * 0.3
    adjacent, distant = inventory.spatial_correlations(block, distant_offset=100)
    assert adjacent > 0.8
    assert abs(distant) < 0.1


def test_spatial_correlations_flag_a_scrambled_layout():
    """A shared global signal makes distant channels as correlated as neighbours."""
    rng = np.random.default_rng(1)
    n, n_ch = 4_000, 200
    common = rng.normal(size=(n, 1))
    block = common + rng.normal(size=(n, n_ch)) * 0.5
    adjacent, distant = inventory.spatial_correlations(block, distant_offset=100)
    assert distant > 0.5
    assert abs(adjacent - distant) < 0.1


def test_spatial_correlations_tolerate_dead_channels():
    rng = np.random.default_rng(2)
    block = rng.normal(size=(1_000, 150))
    block[:, 7] = 0.0
    adjacent, distant = inventory.spatial_correlations(block, distant_offset=100)
    assert np.isfinite(adjacent) and np.isfinite(distant)


# --- referencing -------------------------------------------------------------

def test_common_mode_residual_zero_when_reference_already_removed():
    rng = np.random.default_rng(3)
    block = rng.normal(size=(2_000, 100))
    block -= block.mean(axis=1, keepdims=True)
    mean_res, _ = inventory.common_mode_residuals(block)
    assert mean_res == pytest.approx(0.0, abs=1e-9)


def test_common_mode_residual_detects_an_injected_common_signal():
    rng = np.random.default_rng(4)
    common = rng.normal(size=(2_000, 1))
    block = rng.normal(size=(2_000, 100)) + common
    mean_res, _ = inventory.common_mode_residuals(block)
    assert mean_res > 0.5
