"""Ingest real task-mouse behaviour from `preprocessed_data.mat` (v7.3 / HDF5).

Each mouse's `spatial_binned_data` holds, per spatial bin × trial:
  * `licks`     — lick **counts** per bin (integers, with occasional NaN)
  * `durations` — time spent in each bin, seconds (→ velocity)

We convert this to the model's input format — per-(trial × bin) lick counts,
log-velocities, and a validity mask — applying:
  * a validity mask where licks are NaN or a bin was never occupied;
  * velocity = bin_size_cm / duration, clipped to a physiological range so
    that pauses / grooming do not blow up the log-normal velocity term.

This module only *reads* `data/raw`-equivalent processed data; it never writes.
"""
from __future__ import annotations

import numpy as np
import h5py

from .config import TaskConfig

VEL_CLIP_CM_S = (2.0, 150.0)   # physiological clip for head-fixed wheel running


def load_real_cohort(mat_path: str, cfg: TaskConfig | None = None,
                     truncate_at_disengagement: bool = True):
    """Return a list of per-mouse dicts: {mouse, licks, logv, mask, n_trials, ...}.

    `licks`, `logv`, `mask` are float arrays of shape (n_trials, n_bins).
    `mask` is 1.0 on bins with valid behavioural data, 0.0 otherwise.

    With `truncate_at_disengagement` (default), each session is cut at the
    behaviourally defined disengagement point (`change_point_mean`), so only
    task-engaged trials are fit.  Mice whose disengagement point is NaN (the
    detector found none) keep all trials.
    """
    cfg = cfg or TaskConfig()
    out = []
    with h5py.File(mat_path, "r") as f:
        pd = f["preprocessed_data"]
        n_animals = pd["n_trials"].shape[0]
        for i in range(n_animals):
            sbd = f[pd["spatial_binned_data"][i, 0]]
            licks = np.asarray(sbd["licks"], dtype=np.float64).T      # (trials, bins)
            dur = np.asarray(sbd["durations"], dtype=np.float64).T    # (trials, bins)

            if licks.shape[1] != cfg.n_bins:           # guard against geometry drift
                raise ValueError(f"mouse {i}: {licks.shape[1]} bins, expected {cfg.n_bins}")

            n_total = int(licks.shape[0])
            cp = np.asarray(f[pd["change_point_mean"][i, 0]]).squeeze()
            disengage = int(cp) if np.isfinite(cp) else n_total
            n_keep = min(disengage, n_total) if truncate_at_disengagement else n_total
            n_keep = max(n_keep, 1)
            licks, dur = licks[:n_keep], dur[:n_keep]

            valid = np.isfinite(licks) & np.isfinite(dur) & (dur > 0)
            licks_clean = np.where(valid, licks, 0.0)

            vel = np.where(dur > 0, cfg.bin_size_cm / np.maximum(dur, 1e-9), np.nan)
            vel = np.clip(vel, *VEL_CLIP_CM_S)
            logv = np.where(valid, np.log(vel), 0.0)

            out.append(dict(
                mouse=f"M{i + 1:02d}",
                licks=licks_clean,
                logv=logv,
                mask=valid.astype(np.float64),
                n_trials=int(n_keep),
                n_total=n_total,
                disengage=disengage,
            ))
    return out


def cohort_summary(cohort):
    """One-line-per-mouse description, for sanity logging."""
    lines = []
    for m in cohort:
        v = np.exp(m["logv"][m["mask"] > 0])
        lr = m["licks"][m["mask"] > 0]
        lines.append(
            f"  {m['mouse']}: {m['n_trials']:3d} trials | "
            f"valid bins {m['mask'].mean()*100:5.1f}% | "
            f"lick/bin {lr.mean():.3f} | vel {np.median(v):5.1f} cm/s")
    return "\n".join(lines)
