"""Tests for per-trial alignment and behavioural features.

The alignment is the part of this analysis most likely to be silently wrong, so
it is pinned hardest. VR position keeps increasing during the 5 s dark
inter-trial period as well as in the corridor, so a bare "first sample at or past
the reward-zone boundary" fires in the DARK on some trials. Measured on task
animal 1 trial 1 (2026-09-17): the naive rule returned 3185 ms, the corridor-
restricted rule 10478 ms, and the corridor does not begin until 5001 ms.

Created 2026-09-17.
"""
from __future__ import annotations

import numpy as np
import pytest

from striatum_info import trials

CORRIDOR_WORLD = 12.0
DARK_WORLD = 6.0


def _trial(dark_ms=5000, corridor_ms=10000, dark_peak=150.0, corridor_peak=200.0):
    """A trial whose dark period runs the wheel past the reward-zone boundary."""
    t = np.arange(dark_ms + corridor_ms, dtype=float)
    world = np.where(t < dark_ms, DARK_WORLD, CORRIDOR_WORLD)
    pos = np.concatenate([
        np.linspace(0, dark_peak, dark_ms),
        np.linspace(0, corridor_peak, corridor_ms),
    ])
    return world, pos, t


def test_reward_zone_entry_ignores_the_dark_period():
    world, pos, t = _trial()
    naive = t[np.flatnonzero(pos >= trials.REWARD_ZONE_AU)[0]]
    entry = trials.reward_zone_entry(world, pos, t)
    assert naive < 5000, "the fixture must reproduce the trap: naive rule fires in the dark"
    assert entry > 5000, f"entry must be in the corridor, got {entry}"
    # Position at the reported entry really is at the boundary.
    k = int(np.argmin(np.abs(t - entry)))
    assert pos[k] == pytest.approx(trials.REWARD_ZONE_AU, abs=1.0)


def test_reward_zone_entry_is_none_when_the_corridor_never_reaches_it():
    world, pos, t = _trial(corridor_peak=60.0)
    assert trials.reward_zone_entry(world, pos, t) is None


def test_reward_zone_entry_is_none_without_a_corridor():
    t = np.arange(5000, dtype=float)
    world = np.full(t.size, DARK_WORLD)
    pos = np.linspace(0, 150.0, t.size)
    assert trials.reward_zone_entry(world, pos, t) is None


def test_alignment_window_is_centred_on_the_event():
    n_units, n_ms = 4, 20_000
    spikes = np.zeros((n_ms, n_units))
    entry = 10_000.0
    # One spike in every unit exactly at the event.
    spikes[int(entry), :] = 1
    npx = np.arange(n_ms, dtype=float)
    out, centre = trials.align_spikes(spikes, npx, entry, window_ms=(-1000, 500),
                                      bin_ms=10)
    assert out.shape == (n_units, 150)
    assert centre == 100, "the event belongs at the boundary between bin 99 and 100"
    assert out[:, centre].all(), "the spike at t=0 must land in the centre bin"
    assert out[:, :centre].sum() == 0 and out[:, centre + 1:].sum() == 0


def test_alignment_binarises_rather_than_counting():
    n_ms = 20_000
    spikes = np.zeros((n_ms, 1))
    spikes[10_000:10_005, 0] = 1            # five spikes inside one 10 ms bin
    out, centre = trials.align_spikes(spikes, np.arange(n_ms, dtype=float), 10_000.0,
                                      window_ms=(-1000, 500), bin_ms=10)
    assert out[0, centre] == 1, "the paper binarises: any spike in the bin is 1"
    assert out.sum() == 1


def test_alignment_refuses_a_window_running_off_the_trial():
    n_ms = 1_200
    spikes = np.zeros((n_ms, 2))
    npx = np.arange(n_ms, dtype=float)
    assert trials.align_spikes(spikes, npx, 200.0, window_ms=(-1000, 500),
                               bin_ms=10) is None


def test_features_are_finite_and_named():
    world, pos, t = _trial()
    licks = np.zeros(t.size)
    licks[[6000, 6500, 12000]] = 1
    feats = trials.behavioural_features(world, pos, t, licks,
                                        lick_error_z=-1.2, success=1.0)
    assert set(feats) == set(trials.FEATURE_NAMES)
    for name in ("max_velocity_cm_s", "mean_velocity_cm_s", "path_length_au",
                 "time_to_reward_zone_ms", "corridor_duration_ms"):
        assert np.isfinite(feats[name]), f"{name} should be finite on a clean trial"
    assert feats["n_licks"] == 3
    assert feats["lick_error_z"] == pytest.approx(-1.2)


def test_velocity_uses_the_corridor_only():
    """A fast dark period must not inflate the corridor velocity features."""
    slow = trials.behavioural_features(*_trial(dark_peak=10.0), np.zeros(15_000),
                                       lick_error_z=0.0, success=1.0)
    fast_dark = trials.behavioural_features(*_trial(dark_peak=900.0), np.zeros(15_000),
                                            lick_error_z=0.0, success=1.0)
    assert slow["mean_velocity_cm_s"] == pytest.approx(
        fast_dark["mean_velocity_cm_s"], rel=1e-6), (
        "dark-period running must not enter the corridor velocity")


def test_pooled_sample_order_is_tile_not_repeat():
    """The convention the MI driver depends on, pinned here because it is silent.

    A (units, window, trials) block ravelled in C order puts sample s at
    trial ``s % n_trials``, so the per-trial labels must be TILED. Using
    ``repeat`` instead pairs each sample with the wrong trial and still returns
    perfectly plausible information values.
    """
    n_units, n_win, n_tr = 2, 5, 7
    block = np.zeros((n_units, n_win, n_tr))
    trial_id = np.arange(n_tr)
    # Stamp each element with its trial identity.
    block[:] = trial_id[None, None, :]
    flat = block.reshape(n_units, -1)
    tiled = np.tile(trial_id, n_win)
    assert np.array_equal(flat[0], tiled), "C-order ravel means tile, not repeat"
    assert not np.array_equal(flat[0], np.repeat(trial_id, n_win))
