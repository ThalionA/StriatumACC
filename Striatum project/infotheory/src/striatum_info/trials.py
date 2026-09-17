"""Per-trial alignment and behavioural features for the information analysis.

Mirrors the design of Lemke et al. (2024) on this task. They aligned every trial
to pellet touch -- chosen over movement onset because it lowered across-trial
kinematic variability -- and scored each trial with a handful of scalar movement
features. The corridor analogue agreed with Theo on 2026-09-17 is **reward-zone
entry**, with all the behavioural features tried rather than one picked in
advance.

THE ALIGNMENT TRAP, because it is silent and it bites. VR position keeps
increasing during the 5 s dark inter-trial period, not only in the corridor, so
"the first sample at or past 100 a.u." can fire in the dark. On task animal 1
trial 1 the naive rule returned 3185 ms and the corridor-restricted rule
10478 ms, with the corridor itself not starting until 5001 ms -- a 7 s error that
would have quietly mis-aligned a subset of trials while looking fine on the rest.
The corridor is where ``trial_world > 6`` (the convention of
``separate_dark_and_corridor_periods.m``).

Created 2026-09-17.
"""
from __future__ import annotations

import numpy as np

#: Reward-zone onset in VR units (project_cfg: reward_bin 25 x bin_size_au 4).
REWARD_ZONE_AU = 100.0
#: The corridor is where trial_world exceeds this (separate_dark_and_corridor_periods.m).
CORRIDOR_WORLD_THRESHOLD = 6.0
#: 1 VR a.u. in cm (project_cfg cfg.au_to_cm).
AU_TO_CM = 1.25

FEATURE_NAMES = (
    "max_velocity_cm_s",
    "mean_velocity_cm_s",
    "velocity_at_reward_zone_cm_s",
    "corridor_duration_ms",
    "time_to_reward_zone_ms",
    "path_length_au",
    "first_lick_position_au",
    "n_licks",
    "lick_error_z",
    "success",
)


def _corridor_mask(trial_world: np.ndarray) -> np.ndarray:
    return np.asarray(trial_world, float) > CORRIDOR_WORLD_THRESHOLD


def reward_zone_entry(trial_world: np.ndarray, trial_position: np.ndarray,
                      trial_times_ms: np.ndarray,
                      rz_au: float = REWARD_ZONE_AU) -> float | None:
    """Time (ms, trial-relative) at which the animal first enters the reward zone.

    Restricted to the corridor. Returns ``None`` when the trial has no corridor
    or never reaches the boundary, rather than a fallback -- a trial with no
    event has no alignment and must be dropped, not approximated.
    """
    world = _corridor_mask(trial_world)
    pos = np.asarray(trial_position, float)
    t = np.asarray(trial_times_ms, float)
    if not world.any():
        return None
    hit = np.flatnonzero(world & (pos >= rz_au))
    if hit.size == 0:
        return None
    return float(t[hit[0]])


def align_spikes(spikes_ms: np.ndarray, npx_times_ms: np.ndarray, event_ms: float,
                 *, window_ms: tuple[int, int] = (-1000, 500),
                 bin_ms: int = 10) -> tuple[np.ndarray, int] | None:
    """``(units x time bins)`` binarised spikes around ``event_ms``, and the centre bin.

    ``spikes_ms`` is ``(milliseconds, units)`` as the preprocessed cache stores
    it. Binarised rather than counted, following the paper: any bin containing at
    least one spike is 1. Returns ``None`` when the window runs off either end of
    the trial -- padding would invent data and shifting would break the alignment
    the whole analysis rests on.
    """
    spikes = np.asarray(spikes_ms)
    npx = np.asarray(npx_times_ms, float)
    lo, hi = window_ms
    start, stop = event_ms + lo, event_ms + hi
    if start < npx[0] or stop > npx[-1]:
        return None
    n_bins = int((hi - lo) // bin_ms)
    edges = start + np.arange(n_bins + 1) * bin_ms
    idx = np.searchsorted(npx, edges)
    out = np.zeros((spikes.shape[1], n_bins), dtype=np.uint8)
    for b in range(n_bins):
        a, z = idx[b], idx[b + 1]
        if z > a:
            out[:, b] = (spikes[a:z, :].sum(axis=0) > 0).astype(np.uint8)
    return out, int(-lo // bin_ms)


def behavioural_features(trial_world: np.ndarray, trial_position: np.ndarray,
                         trial_times_ms: np.ndarray, trial_licks: np.ndarray, *,
                         lick_error_z: float, success: float) -> dict:
    """Every per-trial scalar feature, computed over the CORRIDOR only.

    All of them are produced rather than one chosen: the paper found that which
    feature is best encoded is itself a result (maximum reaching velocity and
    total trajectory length were theirs), so the choice is left to the analysis.

    Velocity features exclude the dark period. The animals run faster in the dark
    than in the corridor on this task (2026-09-07: 27 vs 17 cm/s), so letting the
    dark in would swamp the corridor kinematics entirely.
    """
    world = _corridor_mask(trial_world)
    pos = np.asarray(trial_position, float)
    t = np.asarray(trial_times_ms, float)
    licks = np.asarray(trial_licks, float)
    out = {name: np.nan for name in FEATURE_NAMES}
    out["lick_error_z"] = float(lick_error_z)
    out["success"] = float(success)

    idx = np.flatnonzero(world)
    if idx.size < 3:
        out["n_licks"] = float(np.nansum(licks > 0))
        return out
    cp, ct = pos[idx], t[idx]
    dt_s = np.diff(ct) / 1000.0
    dpos_cm = np.abs(np.diff(cp)) * AU_TO_CM
    good = dt_s > 0
    if good.any():
        v = dpos_cm[good] / dt_s[good]
        out["max_velocity_cm_s"] = float(np.percentile(v, 99))   # robust to one jump
        out["mean_velocity_cm_s"] = float(dpos_cm.sum() / (ct[-1] - ct[0]) * 1000.0)
    out["corridor_duration_ms"] = float(ct[-1] - ct[0])
    out["path_length_au"] = float(np.abs(np.diff(cp)).sum())

    entry = reward_zone_entry(trial_world, trial_position, trial_times_ms)
    if entry is not None:
        out["time_to_reward_zone_ms"] = float(entry - ct[0])
        k = int(np.argmin(np.abs(ct - entry)))
        lo, hi = max(0, k - 25), min(cp.size - 1, k + 25)
        if hi > lo and ct[hi] > ct[lo]:
            out["velocity_at_reward_zone_cm_s"] = float(
                np.abs(cp[hi] - cp[lo]) * AU_TO_CM / ((ct[hi] - ct[lo]) / 1000.0))

    lick_idx = np.flatnonzero((licks > 0) & world)
    out["n_licks"] = float(np.sum(licks > 0))
    if lick_idx.size:
        out["first_lick_position_au"] = float(pos[lick_idx[0]])
    return out
