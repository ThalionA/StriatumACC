"""Shared analysis layer: learning points, epoch windows, area aggregation.

Every "across learning" statement in this project is plotted against the same
axis -- a per-animal learning point and the four epoch windows around it. These
are ports of ``find_learning_points.m`` and ``epoch_indices.m`` rather than
re-derivations, and ``tests/test_analysis.py`` checks the port against MATLAB's
own logged learning points for all 16 animals.
"""

from __future__ import annotations

import numpy as np

from . import config

# project_cfg.m:65-68
LP_Z_THRESHOLD = -2.0
LP_WINDOW = 10
LP_MIN_CONSECUTIVE = 7
TRIALS_PER_EPOCH = 10
# CorridorVsDarkActivity.m:50 and SpatioTemporalActivityEvolution's {1:3, 4:10, ...}
NAIVE_SPLIT = 3
EPOCH_NAMES = ("Trials 1-3", "Trials 4-10", "Intermediate", "Expert")

BIN_SIZE_CM = 5.0       # project_cfg cfg.bin_size_cm


def learning_point(zscored_lick_errors: np.ndarray, *,
                   z_threshold: float = LP_Z_THRESHOLD,
                   window: int = LP_WINDOW,
                   min_consecutive: int = LP_MIN_CONSECUTIVE) -> int | None:
    """First trial (1-based) that is itself below threshold and sustains it.

    Port of ``find_learning_points.m``. The trial must satisfy two conditions
    together: its own z-scored lick error is at or below ``z_threshold``, and its
    forward window of ``window`` trials contains at least ``min_consecutive``
    sub-threshold trials. Returning the window *start* instead -- which could be
    an above-threshold trial -- was the bug fixed on 2026-05-24. ``None`` for a
    non-learner, matching MATLAB's NaN.
    """
    z = np.asarray(zscored_lick_errors, dtype=float).ravel()
    if z.size < window:
        return None
    passes = np.nan_to_num(z, nan=np.inf) <= z_threshold
    # Forward-looking moving sum over [t, t+window-1], as movsum([0, w-1]) does.
    padded = np.concatenate([passes.astype(int), np.zeros(window - 1, dtype=int)])
    cumulative = np.concatenate([[0], np.cumsum(padded)])
    win_counts = cumulative[window:window + z.size] - cumulative[:z.size]
    hits = np.flatnonzero(passes & (win_counts >= min_consecutive))
    return int(hits[0]) + 1 if hits.size else None


def epoch_indices(lp: int | None, n_trials: int, *,
                  trials_per_epoch: int = TRIALS_PER_EPOCH,
                  naive_start: int = 1,
                  expert_starts_at: str = "lp",
                  naive_split: int | None = None) -> list[np.ndarray]:
    """Naive / (split) / Intermediate / Expert trial indices, 1-based.

    Port of ``epoch_indices.m``. A window that does not fit inside
    ``[1, n_trials]`` comes back empty rather than clipped, and a learning point
    beyond the session yields no LP-relative window at all -- both behaviours
    matter, because a clipped window would silently compare unequal amounts of
    data across animals.
    """
    w = trials_per_epoch
    if naive_split is not None and not (1 <= naive_split < w):
        raise ValueError(
            f"naive_split must be in [1, {w - 1}]; got {naive_split}")

    if naive_split is None:
        idx = [np.array([], int)] * 3
        if n_trials >= w:
            idx[0] = np.arange(naive_start, naive_start + w)
        k = 1
    else:
        idx = [np.array([], int)] * 4
        if n_trials >= naive_split:
            idx[0] = np.arange(naive_start, naive_start + naive_split)
        if n_trials >= w:
            idx[1] = np.arange(naive_start + naive_split, naive_start + w)
        k = 2

    if lp is None or lp > n_trials:
        return idx

    pre_start, pre_end = lp - w, lp - 1
    if pre_start >= 1 and pre_end <= n_trials and pre_end >= pre_start:
        idx[k] = np.arange(pre_start, pre_end + 1)

    post_start = lp if expert_starts_at == "lp" else lp + 1
    if expert_starts_at not in ("lp", "lp1"):
        raise ValueError(f"unknown expert_starts_at: {expert_starts_at}")
    post_end = post_start + w - 1
    if post_start >= 1 and post_end <= n_trials:
        idx[k + 1] = np.arange(post_start, post_end + 1)
    return idx


def cohort_learning_points(preproc_mat=None) -> dict[int, int | None]:
    """``{mouse_id: learning point}`` for the 16 task animals, from the cohort struct."""
    import h5py

    out: dict[int, int | None] = {}
    with h5py.File(preproc_mat or config.PREPROC_MAT, "r") as handle:
        P = handle["preprocessed_data"]
        for i, mouse in enumerate(config.TASK_MOUSE_IDS):
            z = np.asarray(handle[P["zscored_lick_errors"][i, 0]]).ravel()
            out[mouse] = learning_point(z)
    return out


def cohort_trial_counts(preproc_mat=None) -> dict[int, int]:
    """``{mouse_id: n_trials}`` as the unit analyses count them."""
    import h5py

    out: dict[int, int] = {}
    with h5py.File(preproc_mat or config.PREPROC_MAT, "r") as handle:
        P = handle["preprocessed_data"]
        for i, mouse in enumerate(config.TASK_MOUSE_IDS):
            out[mouse] = int(np.asarray(handle[P["n_trials"][i, 0]]).ravel()[0])
    return out


def bin_speed_cm_s(bin_start_ms: np.ndarray, bin_stop_ms: np.ndarray) -> np.ndarray:
    """Running speed (cm/s) implied by each spatial bin's traversal time.

    A 5 cm bin crossed in ``stop - start`` milliseconds. Absent bins (-1) and
    zero-duration bins return ``nan``. This is the covariate a band-power change
    across learning has to survive: animals run faster as they learn.
    """
    duration_s = (np.asarray(bin_stop_ms, float) - np.asarray(bin_start_ms, float)) / 1000.0
    bad = (np.asarray(bin_start_ms) < 0) | (duration_s <= 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = BIN_SIZE_CM / duration_s
    speed[bad] = np.nan
    return speed


def joint_zscore(corridor: np.ndarray, dark: np.ndarray):
    """Z-score each channel over corridor AND dark samples pooled.

    The convention CorridorVsDarkActivity.m adopts and explains: scoring the two
    states separately forces both means to zero and makes the corridor-vs-dark
    comparison vacuous by construction. Both inputs are ``(n_channels, n_bins,
    n_trials)``; a channel with no variance returns ``nan`` rather than infinity.
    """
    n_ch = corridor.shape[0]
    pooled = np.concatenate([corridor.reshape(n_ch, -1), dark.reshape(n_ch, -1)], axis=1)
    mean = np.nanmean(pooled, axis=1)
    sd = np.nanstd(pooled, axis=1)
    sd = np.where(sd > 0, sd, np.nan)
    shape = (n_ch,) + (1,) * (corridor.ndim - 1)
    return ((corridor - mean.reshape(shape)) / sd.reshape(shape),
            (dark - mean.reshape(shape)) / sd.reshape(shape))
