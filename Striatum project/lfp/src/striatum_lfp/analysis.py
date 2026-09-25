"""Shared analysis layer: learning points, epoch windows, area aggregation.

Every "across learning" statement in this project is plotted against the same
axis -- a per-animal learning point and the epoch windows around it. These
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


def measured_learning_points(cohort=None, preproc_mat=None) -> dict[int, int | None]:
    """``{mouse_id: learning point}`` exactly as the lick errors give it.

    ``None`` where the animal never reaches criterion. This is the raw
    measurement; use :func:`cohort_learning_points` for the map the analyses
    actually run on, which fills those gaps.
    """
    import h5py

    ch = cohort or config.TASK
    out: dict[int, int | None] = {}
    with h5py.File(preproc_mat or ch.preproc_mat, "r") as handle:
        P = handle["preprocessed_data"]
        for i, mouse in enumerate(ch.mouse_ids):
            z = np.asarray(handle[P["zscored_lick_errors"][i, 0]]).ravel()
            out[mouse] = learning_point(z)
    return out


def task_average_learning_point(preproc_mat=None) -> int | None:
    """Mean learning point over the task animals that ACTUALLY reach criterion.

    Averaged over learners only, so it stays the same number whether or not the
    non-learners have since been given it (a mean is unchanged by adding copies
    of itself, but computing it from the filled map would make the definition
    circular and impossible to reason about).
    """
    lps = [v for v in measured_learning_points(config.TASK, preproc_mat).values()
           if v is not None]
    return int(round(sum(lps) / len(lps))) if lps else None


def cohort_learning_points(cohort=None, preproc_mat=None) -> dict[int, int | None]:
    """``{mouse_id: learning point}`` as the epoch windows use it.

    Two kinds of animal have no learning point of their own, and BOTH now take
    the task cohort's average over its learners:

    * **Yoked controls** -- there is nothing for them to learn, so
      ``IntegratedAll_v1.m`` gives every control animal the task average and the
      epoch windows follow from that. A cohort declaring
      ``learning_point_source == "task_average"`` gets the same treatment here.
    * **Task animals that never reach criterion** (703 and 1206). Until
      2026-09-09 these were left at ``None``, which meant the Intermediate and
      Expert windows did not exist for them at all -- and since 1206 is one of
      only three task animals with a probe in CA1 and DG, those two areas fell
      to n = 2 in half the epochs and dropped out of the figures entirely.

    What this buys and what it costs. It buys CA1/DG coverage in every epoch and
    restores DMS to 16 and ACC to 15 animals throughout. It costs the meaning of
    the word "Expert" for those two animals: their late window is a matched TIME
    window, not a matched level of performance -- exactly the caveat the controls
    already carry. :func:`learning_point_sources` says which animals are on a
    borrowed number so a table or figure can label them.
    """
    ch = cohort or config.TASK
    if ch.learning_point_source == "task_average":
        avg = task_average_learning_point()
        return {m: avg for m in ch.mouse_ids}

    measured = measured_learning_points(ch, preproc_mat)
    avg = task_average_learning_point(preproc_mat if ch is config.TASK else None)
    return {m: (v if v is not None else avg) for m, v in measured.items()}


def learning_point_sources(cohort=None, preproc_mat=None) -> dict[int, str]:
    """``{mouse_id: "measured" | "cohort_average"}`` for the map above.

    Anything reporting an epoch result should be able to say which animals had a
    learning point of their own, because for the others "Expert" means a time
    window rather than a level of performance.
    """
    ch = cohort or config.TASK
    if ch.learning_point_source == "task_average":
        return {m: "cohort_average" for m in ch.mouse_ids}
    return {m: ("measured" if v is not None else "cohort_average")
            for m, v in measured_learning_points(ch, preproc_mat).items()}


def cohort_trial_counts(cohort=None, preproc_mat=None) -> dict[int, int]:
    """``{mouse_id: n_trials}`` as the unit analyses count them."""
    import h5py

    ch = cohort or config.TASK
    out: dict[int, int] = {}
    with h5py.File(preproc_mat or ch.preproc_mat, "r") as handle:
        P = handle["preprocessed_data"]
        for i, mouse in enumerate(ch.mouse_ids):
            out[mouse] = int(np.asarray(handle[P["n_trials"][i, 0]]).ravel()[0])
    return out


def disengagement_points(cohort=None, preproc_mat=None) -> dict[int, float]:
    """``{mouse_id: change_point_mean}`` -- the trial the animal stops engaging.

    ``IntegratedAll_v1.m`` section 2 defines this alongside the learning point and
    several MATLAB callers pass a DP-truncated ``n_trials`` downstream
    (``epoch_indices.m``). **It is not applied by ``cohort_trial_counts``, and
    ``good_trials`` in the band-power cubes is an ALIGNMENT flag, not an
    engagement one** -- it is set wherever a corridor start was found.

    Anything comparing early against late trials must clip here first. Measured
    2026-09-17: without clipping, 9 of 13 task animals had their entire "last
    fifty trials" window past DP, so an early-versus-late contrast was mostly
    measuring engagement. Returns ``nan`` where the change point is undefined
    (1212), which callers must handle rather than silently include.
    """
    import h5py

    ch = cohort or config.TASK
    out: dict[int, float] = {}
    with h5py.File(preproc_mat or ch.preproc_mat, "r") as handle:
        P = handle["preprocessed_data"]
        if "change_point_mean" not in P:
            return out
        for i, mouse in enumerate(ch.mouse_ids):
            out[mouse] = float(np.asarray(handle[P["change_point_mean"][i, 0]]).ravel()[0])
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


def log_power(x: np.ndarray) -> np.ndarray:
    """log10 of band power, with non-positive cells (empty bins) left as ``nan``.

    Band power is close to lognormal over three orders of magnitude, and the two
    export batches differ ~1000x in absolute power, so every downstream statistic
    works on the log and then standardises it per channel.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log10(x)
    out[~np.isfinite(out)] = np.nan
    return out


def read_behaviour(mouse_id: int, probe: str = "striatum", cohort=None) -> dict:
    """VR position/world/trial on the millisecond grid, plus the recording crop.

    Column order follows ``OrganiseStriatumDataIncV1.m``:225-260 -- VR_data row 2
    is position, row 5 world, row 7 trial (h5py columns 1, 4 and 6). ``crop_start0``
    /``crop_end0`` are that script's ``npx_start_frame``/``npx_end_frame`` as
    0-based offsets.
    """
    import h5py

    path = config.raw_mat(mouse_id, probe, cohort or config.TASK)
    with h5py.File(path, "r") as handle:
        vr_times_s = np.asarray(handle["VR_times_synched"]).ravel().astype(float)
        vr = np.asarray(handle["VR_data"])
        n_spike_bins = int(handle["binned_spikes"].shape[0])
    if vr.shape[0] < vr.shape[1]:            # stored (n_rows, n_frames)
        vr = vr.T
    return {
        "vr_times_s": vr_times_s,
        "position": vr[:, 1].astype(float),
        "world": vr[:, 4].astype(float),
        "trial": vr[:, 6].astype(float),
        "n_spike_bins": n_spike_bins,
        "crop_start0": max(0, int(np.ceil(vr_times_s[0] * 1000.0)) - 1),
        "crop_end0": int(np.floor(vr_times_s[-1] * 1000.0)) - 1,
    }
