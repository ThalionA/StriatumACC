"""Movement covariates in the layouts striatum_cca.pipeline.prepare_pair_confounded
expects: a spatial (n_raw_trials, n_bins, n) tensor, or a temporal ragged
per-trial list whose lengths equal the spike bins of dataio.area_tensor."""

import numpy as np

from .binning import corridor_start, trial_bounds
from .temporal import corridor_frame_ms, lag_within_trial, to_time_bins


def spatial_confound(binned, names):
    """Stack per-(trial, 5 cm bin) covariates from a *_binned.npz."""
    return np.stack([np.asarray(binned[n], float) for n in names], axis=-1)


def temporal_confound(vr, t_ms, sig, names, n_bins_per_trial, bin_ms, lags_ms):
    """Per raw trial t < len(n_bins_per_trial): the named per-frame signals on
    that trial's corridor spike bins (spike column 0 = 2nd corridor row), each
    at every lag in lags_ms (> 0 = the past). Trials with 0 bins -> (0, n)."""
    starts, ends = trial_bounds(vr["trial"])
    n_cols = len(names) * len(lags_ms)
    out = []
    for t, nb in enumerate(n_bins_per_trial):
        if nb == 0:
            out.append(np.zeros((0, n_cols)))
            continue
        c = corridor_start(vr["world"][starts[t]:ends[t] + 1])
        if c is None or ends[t] - starts[t] - c < 1:  # no usable corridor rows: missing, dropped at fit
            out.append(np.full((nb, n_cols), np.nan))
            continue
        rows = np.arange(starts[t] + c, ends[t] + 1)
        fms = corridor_frame_ms(t_ms, rows)
        if abs(fms[-1] + 1 - nb * bin_ms) > bin_ms + 1:
            raise ValueError(f"trial {t}: {nb} spike bins of {bin_ms} ms vs corridor span {fms[-1]:.0f} ms")
        trial_id = np.zeros(nb, int)
        cols = []
        for n in names:
            x = to_time_bins(fms, np.asarray(sig[n])[rows], nb, bin_ms)
            cols += [lag_within_trial(x, trial_id, round(lag / bin_ms)) for lag in lags_ms]
        out.append(np.column_stack(cols))
    return out


def shift_confound(confound, fraction):
    """Control confound: the same values circularly shifted along the session
    by `fraction` of its finite samples, keeping the layout (spatial NaN
    pattern, temporal per-trial lengths). Same dimensionality and marginal
    statistics as the real confound, but misaligned with the activity -- so it
    measures how much CC falls from regressing out that many unrelated signals."""
    if isinstance(confound, np.ndarray):
        out = confound.copy()
        rows = np.isfinite(confound).all(-1)
        vals = confound[rows]
        out[rows] = np.roll(vals, int(fraction * vals.shape[0]), axis=0)
        return out
    lengths = [a.shape[0] for a in confound]
    flat = np.concatenate(confound, axis=0)
    flat = np.roll(flat, int(fraction * flat.shape[0]), axis=0)
    return np.split(flat, np.cumsum(lengths)[:-1])
