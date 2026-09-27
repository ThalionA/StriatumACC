"""Per-frame video features -> raw trial x 5 cm position bin, by the MATLAB
pipeline's own rules, so they sit on the same grid as spatial_binned_fr_all.

Rules mirrored (file:line in 'Striatum project/'):
- trials: a new trial wherever the VR trial column changes (cut_data_per_trial.m:7);
- corridor: from the first row with world > 6 to the trial's end
  (separate_dark_and_corridor_periods.m:21); a trial with none is left empty;
- bins: edges 0:4:200 with the last widened to 204 (ProcessStriatumTask.m:19-20),
  histcounts semantics, a bin needs >= 2 rows (spatial_binning.m:39);
- duration of a bin: last minus first row time in it (spatial_binning.m:45).

Works on raw VR rows, which are video frames 1:1 (see sessions.py).
"""

import numpy as np

BIN_SIZE_AU = 4
CORRIDOR_END_AU = 200
CORRIDOR_WORLD_ABOVE = 6


def spatial_bin_edges():
    edges = np.arange(0, CORRIDOR_END_AU + BIN_SIZE_AU / 2, BIN_SIZE_AU, dtype=float)
    edges[-1] = CORRIDOR_END_AU + BIN_SIZE_AU
    return edges


def position_bin_index(x, edges):
    """0-based bin per sample with MATLAB histcounts semantics; -1 = no bin."""
    x = np.asarray(x, float)
    idx = np.searchsorted(edges, x, side="right") - 1
    idx[x == edges[-1]] = edges.size - 2
    idx[~np.isfinite(x) | (x < edges[0]) | (x > edges[-1])] = -1
    return idx


def trial_bounds(vr_trial):
    """(starts, ends) row indices of each raw trial, ends inclusive."""
    vr_trial = np.asarray(vr_trial)
    change = np.flatnonzero(np.diff(vr_trial) != 0)
    return np.concatenate([[0], change + 1]), np.concatenate([change, [vr_trial.size - 1]])


def corridor_start(world):
    above = np.flatnonzero(np.asarray(world) > CORRIDOR_WORLD_ABOVE)
    return int(above[0]) if above.size else None


def bin_session(vr, t_ms, features, edges=None):
    """{'durations' (s), <feature>: nanmean per bin}, each (n_raw_trials, n_bins).

    vr: {'trial', 'world', 'x'} per row; t_ms: row times in ms; features:
    {name: per-row array}. NaN wherever a bin has < 2 rows or the trial has no
    corridor."""
    edges = spatial_bin_edges() if edges is None else edges
    n_bins = edges.size - 1
    starts, ends = trial_bounds(vr["trial"])
    out = {k: np.full((starts.size, n_bins), np.nan) for k in ["durations", *features]}
    for i, (s, e) in enumerate(zip(starts, ends)):
        c = corridor_start(vr["world"][s:e + 1])
        if c is None:
            continue
        rows = np.arange(s + c, e + 1)
        b = position_bin_index(vr["x"][rows], edges)
        for k in range(n_bins):
            in_bin = rows[b == k]
            if in_bin.size < 2:
                continue
            out["durations"][i, k] = (t_ms[in_bin[-1]] - t_ms[in_bin[0]]) / 1000
            for name, values in features.items():
                v = values[in_bin]
                if np.isfinite(v).any():
                    out[name][i, k] = np.nanmean(v)
    return out


def still_frames(velocity, lick, half_window):
    """True where VR velocity is exactly 0 and there is no lick for
    half_window frames on either side: frames where the mouse is not
    running or licking, whose ME is camera noise plus fidgeting."""
    moving = (np.asarray(velocity) != 0) | (np.asarray(lick) > 0)
    k = 2 * half_window + 1
    busy = np.convolve(moving.astype(float), np.ones(k), mode="same") > 0
    busy[:half_window] = True
    busy[-half_window:] = True
    return ~busy


def per_trial_still_median(feature, still, vr_trial, min_frames=30):
    """Median of feature over still frames in each raw trial; NaN if fewer
    than min_frames still frames."""
    starts, ends = trial_bounds(vr_trial)
    out = np.full(starts.size, np.nan)
    for i, (s, e) in enumerate(zip(starts, ends)):
        v = feature[s:e + 1][still[s:e + 1]]
        v = v[np.isfinite(v)]
        if v.size >= min_frames:
            out[i] = np.median(v)
    return out
