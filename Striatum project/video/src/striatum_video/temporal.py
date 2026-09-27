"""Put per-frame signals on the cca temporal arm's time bins.

The temporal arm re-bins corridorData.binned_spikes (1 ms counts) into
cfg.temporal_bin_ms bins per trial (striatum_cca.dataio.rebin_trial). Spike
column 0 is the corridor's SECOND VR row: MATLAB takes the corridor start time
as trial_times_zeroed(idx + 1) (separate_dark_and_corridor_periods.m), and in
every trial checked n_1ms = (last - 2nd corridor row time) + 1 ms. So a frame's
time on the spike timeline is its VR time minus that of the 2nd corridor row.
(striatum_cca.dataio.trial_velocity zeroes on the FIRST row instead: one VR
frame, ~30 ms, early. Flagged in NOTES, not changed here.)
"""

import numpy as np


def corridor_frame_ms(t_ms, corridor_rows):
    """Time (ms) of each corridor frame on the trial's 1 ms spike timeline."""
    t = np.asarray(t_ms, float)[corridor_rows]
    return t - t[1]


def to_time_bins(frame_ms, values, n_bins, bin_ms):
    """Per-frame values linearly interpolated at the bin centres
    (bin_ms * k + bin_ms / 2); NaN frames are skipped; ends are held."""
    ok = np.isfinite(values)
    centres = bin_ms * np.arange(n_bins) + bin_ms / 2
    return np.interp(centres, np.asarray(frame_ms)[ok], np.asarray(values)[ok])


def lag_within_trial(x, trial, lag):
    """x shifted by `lag` rows inside each trial (lag > 0 = the past); rows
    whose source falls outside the trial hold the trial's edge value."""
    x = np.asarray(x, float)
    out = np.empty_like(x)
    for t in np.unique(trial):
        idx = np.flatnonzero(trial == t)
        src = np.clip(np.arange(idx.size) - lag, 0, idx.size - 1)
        out[idx] = x[idx][src]
    return out
