"""Data IO and cohort logic for the striatum *temporal* CCA pipeline.

Loads ``preprocessed_data5cm.mat`` (MATLAB v7.3 / HDF5) for the Task or
Control cohort, detects learning points, classifies the cohort, and assembles
the per-animal **time-binned running-state streams** the report pipeline runs on.

The striatum stores corridor data per traversal (``corridorData.binned_spikes``,
1 ms counts; already corridor-only, dark stripped at ``trial_world > 6``), with
**no velocity channel** -- speed is derived from ``trial_position`` /
``trial_times`` (:func:`trial_velocity`). The per-trial 1 ms spikes are Gaussian
smoothed (sigma ``cfg.gaussian_sd_ms``) and summed into ``cfg.temporal_bin_ms``
bins, then concatenated into one global timeline tagged with a per-bin trial
index (the CV/epoch grouping unit).

Trial conventions: learning points and trial indices are 1-indexed in MATLAB;
``epoch_windows`` / period ranges return 0-indexed trial arrays/ranges.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from . import config


# ---------------------------------------------------------------------------
# Raw per-animal record
# ---------------------------------------------------------------------------
@dataclass
class Animal:
    """One animal's preprocessed data, as loaded from the cohort .mat file."""

    animal_id: int                       # 1-indexed (matches MATLAB)
    neurontypes: np.ndarray              # (n_units, 5)
    area_masks: dict[str, np.ndarray]    # area -> (n_units,) bool
    change_point: float                  # disengagement trial (nan if none)
    zscored_lick_errors: np.ndarray      # (n_trials_behaviour,)
    n_trials: int
    corridor_path: Path                  # .mat path (for the 1 ms read)
    group: str = "task"                  # "task" | "control"

    @property
    def n_units(self) -> int:
        return self.neurontypes.shape[0]


def load_animals(path: str | Path | None = None, group: str = "task") -> list[Animal]:
    """Load every animal from a cohort ``.mat`` (Task or Control).

    ``group`` selects the default path (``config.DATA_BY_GROUP``); an explicit
    ``path`` overrides it. The 1 ms corridor spikes are *not* read here -- only
    the metadata; the heavy read is deferred to :func:`load_temporal_streams`.
    """
    if path is None:
        path = config.DATA_BY_GROUP[group]
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"preprocessed data not found: {path}")

    animals: list[Animal] = []
    with h5py.File(path, "r") as f:
        pd = f["preprocessed_data"]
        n_animals = pd["n_trials"].shape[0]
        for i in range(n_animals):

            def deref(field_name: str):
                return f[pd[field_name][i, 0]]

            neurontypes = np.asarray(deref("final_neurontypes"), dtype=float).T
            masks = {
                area: np.asarray(deref(fld)).ravel().astype(bool)
                for area, fld in config.AREA_FIELD.items()
            }
            change_point = float(np.asarray(deref("change_point_mean")).ravel()[0])
            z = np.asarray(deref("zscored_lick_errors"), dtype=float).ravel()
            n_trials = int(np.asarray(deref("n_trials")).ravel()[0])

            _validate_masks(i + 1, neurontypes, masks)
            animals.append(
                Animal(
                    animal_id=i + 1,
                    neurontypes=neurontypes,
                    area_masks=masks,
                    change_point=change_point,
                    zscored_lick_errors=z,
                    n_trials=n_trials,
                    corridor_path=path,
                    group=group,
                )
            )
    return animals


def _validate_masks(animal_id, neurontypes, masks) -> None:
    n_units = neurontypes.shape[0]
    for area, mask in masks.items():
        if mask.size != n_units:
            raise ValueError(
                f"animal {animal_id}: mask {area} length {mask.size} != "
                f"{n_units} units"
            )


# ---------------------------------------------------------------------------
# Learning point + cohort classification (project rule)
# ---------------------------------------------------------------------------
def find_learning_point(zscored_lick_errors: np.ndarray, cfg) -> int | None:
    """Detect the learning point: the trial sustained learning begins from.

    The learning point is the first trial ``t`` that is **itself** below
    threshold (z-scored lick error <= ``lp_z_threshold``) *and* from which
    performance is sustained -- the window ``[t, t + lp_window)`` holds at least
    ``lp_min_consecutive`` below-threshold trials. (Matches the corrected
    ``find_learning_points.m`` rule -- returns a genuinely sub-threshold trial,
    not the window start.) Returns a 1-indexed trial or None.
    """
    z = np.asarray(zscored_lick_errors, dtype=float).ravel()
    below = z <= cfg.lp_z_threshold          # NaN trials -> False
    w = cfg.lp_window
    for t in range(len(z) - w + 1):
        if below[t] and np.sum(below[t : t + w]) >= cfg.lp_min_consecutive:
            return t + 1
    return None


@dataclass
class CohortEntry:
    """Cohort role and learning point for one animal."""

    animal_id: int
    role: str             # "learner" | "nonlearner"
    lp: int               # learning point used for epochs (real or yoked)
    raw_lp: int | None    # detected learning point (None if none detected)


def classify_cohort(animals: list[Animal], cfg) -> tuple[dict[int, CohortEntry], int]:
    """Split the cohort into learners and yoked non-learners.

    A learner has a detected learning point and is not in
    ``cfg.manual_nonlearners``. Non-learners receive the cohort-mean ("yoked")
    learning point. Returns ``(entries, yoked_lp)`` keyed by 1-indexed id.
    """
    raw = {a.animal_id: find_learning_point(a.zscored_lick_errors, cfg) for a in animals}
    learner_ids = [
        aid
        for aid, lp in raw.items()
        if lp is not None and aid not in cfg.manual_nonlearners
    ]
    learner_lps = [raw[aid] for aid in learner_ids]
    yoked_lp = int(round(float(np.mean(learner_lps)))) if learner_lps else 0

    entries: dict[int, CohortEntry] = {}
    for a in animals:
        aid = a.animal_id
        lp = raw[aid]
        if aid in learner_ids:
            entries[aid] = CohortEntry(aid, "learner", lp, lp)
        else:
            entries[aid] = CohortEntry(aid, "nonlearner", yoked_lp, lp)
    return entries, yoked_lp


# ---------------------------------------------------------------------------
# Trial windows: learning epochs + engagement periods
# ---------------------------------------------------------------------------
def n_usable_trials(animal: Animal) -> int:
    """Trial count after disengagement truncation at ``change_point_mean``."""
    if np.isnan(animal.change_point):
        return animal.n_trials
    return int(min(round(animal.change_point), animal.n_trials))


def period_trial_range(animal: Animal, period: str) -> tuple[int, int]:
    """0-indexed ``[start, stop)`` trial range for an engagement period.

    ``engaged``    -- trials before disengagement (``0 .. n_usable``);
    ``disengaged`` -- trials from disengagement to the end (empty if no
                      ``change_point``);
    ``all``        -- every recorded trial.
    """
    n_use = n_usable_trials(animal)
    if period == "engaged":
        return 0, n_use
    if period == "disengaged":
        if np.isnan(animal.change_point) or n_use >= animal.n_trials:
            return n_use, n_use            # empty -- animal never disengaged
        return n_use, animal.n_trials
    if period == "all":
        return 0, animal.n_trials
    raise ValueError(f"unknown period {period!r}")


def epoch_windows(lp: int, n_usable: int, cfg) -> dict[str, np.ndarray] | None:
    """0-indexed trial indices of the three learning epochs (within engaged).

    naive        = trials 1..10              (0-indexed 0..9)
    intermediate = the 10 trials ending at lp (0-indexed lp-10 .. lp-1)
    expert       = the 10 trials after lp     (0-indexed lp .. lp+9)

    Returns None if the windows would overlap (lp < 2*trials_per_epoch) or the
    expert window exceeds the usable trials.
    """
    e = cfg.trials_per_epoch
    if lp < 2 * e:
        return None
    if lp + e > n_usable:
        return None
    return {
        "naive": np.arange(0, e),
        "intermediate": np.arange(lp - e, lp),
        "expert": np.arange(lp, lp + e),
    }


# ---------------------------------------------------------------------------
# Unit selection
# ---------------------------------------------------------------------------
def select_units(animal: Animal, area: str, cfg) -> np.ndarray:
    """Indices of units in ``area`` that survive selection.

    Keeps units flagged for the area; drops fast-spiking units (type 2, in every
    area) when ``cfg.exclude_fast_spiking`` is set.
    """
    keep = animal.area_masks[area].copy()
    if cfg.exclude_fast_spiking:
        types = animal.neurontypes[:, config.NTYPE_COL]
        keep &= types != config.FS_TYPE_CODE
    return np.where(keep)[0]


# ---------------------------------------------------------------------------
# Spike preprocessing: smooth + time-bin (report Section 2.2)
# ---------------------------------------------------------------------------
def rebin_trial(spikes_1ms: np.ndarray, bin_ms: int,
                gaussian_sd_ms: float = 0.0) -> np.ndarray:
    """Smooth then sum 1 ms spike counts into ``bin_ms`` bins.

    If ``gaussian_sd_ms > 0`` each unit's 1 ms train is convolved with a Gaussian
    of that s.d. (mass-conserving) before summation -- the Gonzalez & Buzsaki
    smoothed spike-density (report Section 2.2). With ``gaussian_sd_ms == 0`` this
    reduces to raw summed counts. ``(n_1ms, n_units) -> (n_1ms // bin_ms,
    n_units)`` float32; an incomplete final bin is dropped.
    """
    n_1ms, n_units = spikes_1ms.shape
    n_bins = n_1ms // bin_ms
    if n_bins == 0:
        return np.zeros((0, n_units), dtype=np.float32)
    arr = np.asarray(spikes_1ms[: n_bins * bin_ms], dtype=np.float32)
    if gaussian_sd_ms and gaussian_sd_ms > 0:
        from scipy.ndimage import gaussian_filter1d
        arr = gaussian_filter1d(arr, sigma=float(gaussian_sd_ms), axis=0,
                                mode="nearest")
    return arr.reshape(n_bins, bin_ms, n_units).sum(axis=1)


def trial_velocity(position_au, times_ms, n_bins: int, cfg) -> np.ndarray:
    """Running speed (cm/s) per time bin, derived from VR position/time.

    ``position_au`` (a.u.) and ``times_ms`` (ms) are the corridor-period
    position/time streams. Times are re-zeroed to the corridor onset so they line
    up with the 0-based 1 ms spike columns (as ``spatial_binning.m`` does).
    Position is interpolated onto the bin-centre times, converted a.u.->cm
    (``config.AU_TO_CM``) and differentiated. Returns unsigned speed
    ``(n_bins,)``; bins with no usable position record are 0 (fail the gate).
    """
    bin_ms = int(cfg.temporal_bin_ms)
    out = np.zeros(int(n_bins), dtype=float)
    if n_bins <= 1:
        return out
    pos = np.asarray(position_au, dtype=float).ravel()
    tim = np.asarray(times_ms, dtype=float).ravel()
    if pos.size < 2 or tim.size != pos.size:
        return out                      # cannot form a velocity -> all zero
    t_rel = tim - tim[0]                 # 0-based ms, matches spike columns
    order = np.argsort(t_rel)            # np.interp needs increasing xp
    t_rel, pos = t_rel[order], pos[order]
    centres = bin_ms * np.arange(n_bins) + bin_ms / 2.0
    pos_cm = np.interp(centres, t_rel, pos) * config.AU_TO_CM
    return np.abs(np.gradient(pos_cm, bin_ms / 1000.0))


@dataclass
class TemporalStreams:
    """Global time-binned streams for one animal, concatenated over a period."""

    spikes: np.ndarray         # (n_bins, n_units) float32  -- all units
    vel: np.ndarray            # (n_bins,) cm/s
    trial_idx: np.ndarray      # (n_bins,) float; 0-indexed traversal id


def build_temporal_streams(trial_iter, n_units: int, cfg,
                           start_index: int = 0) -> TemporalStreams:
    """Assemble the global time-binned streams from a per-trial iterator.

    ``trial_iter`` yields ``(spikes_1ms, position_au, times_ms)`` per trial **in
    order**; ``spikes_1ms`` is ``(n_1ms, n_units)``. Each trial is Gaussian
    smoothed + time-binned (:func:`rebin_trial`). A malformed, empty, or
    over-long (``> temporal_max_trial_ms``) trial contributes no bins but still
    consumes its traversal index, so ``trial_idx`` stays aligned (0-indexed,
    offset by ``start_index``). Pure -- no I/O -- so it is unit-testable.
    """
    bin_ms = int(cfg.temporal_bin_ms)
    max_bins = cfg.temporal_max_trial_ms // bin_ms
    spikes_blocks, vel_blocks, tidx_blocks = [], [], []
    for t, (arr, pos, tim) in enumerate(trial_iter):
        arr = np.asarray(arr)
        if arr.ndim != 2 or arr.shape[1] != n_units:
            continue                    # malformed; index t still consumed
        binned = rebin_trial(arr, bin_ms, cfg.gaussian_sd_ms)
        nb = binned.shape[0]
        if nb == 0 or nb > max_bins:
            continue                    # empty or disengaged (over-long)
        spikes_blocks.append(binned)
        vel_blocks.append(trial_velocity(pos, tim, nb, cfg))
        tidx_blocks.append(np.full(nb, start_index + t, dtype=float))
    if spikes_blocks:
        return TemporalStreams(
            spikes=np.concatenate(spikes_blocks, axis=0),
            vel=np.concatenate(vel_blocks, axis=0),
            trial_idx=np.concatenate(tidx_blocks, axis=0),
        )
    return TemporalStreams(
        spikes=np.zeros((0, n_units), dtype=np.float32),
        vel=np.zeros(0, dtype=float),
        trial_idx=np.zeros(0, dtype=float),
    )


# One-animal cache (the 1 ms read is many MB); dropped when the next
# (animal, bin, period) is requested.
_TEMPORAL_CACHE: dict = {}


def load_temporal_streams(animal: Animal, cfg,
                          period: str = "engaged") -> TemporalStreams:
    """Read + smooth + time-bin the corridor streams for ``animal``'s ``period``.

    Cached per ``(animal, bin_ms, gaussian_sd_ms, period)``.
    """
    key = (animal.animal_id, animal.group, int(cfg.temporal_bin_ms),
           float(cfg.gaussian_sd_ms), period)
    cached = _TEMPORAL_CACHE.get(key)
    if cached is not None:
        return cached
    _TEMPORAL_CACHE.clear()                       # 1-slot -- bound memory
    n_units = animal.n_units
    start, stop = period_trial_range(animal, period)

    def _trials():
        with h5py.File(animal.corridor_path, "r") as fh:
            cd = fh[fh["preprocessed_data"]["corridorData"][animal.animal_id - 1, 0]]
            spk, pos, tim = cd["binned_spikes"], cd["trial_position"], cd["trial_times"]
            for t in range(start, stop):
                yield (np.asarray(fh[spk[t, 0]]),
                       np.asarray(fh[pos[t, 0]]).ravel(),
                       np.asarray(fh[tim[t, 0]]).ravel())

    streams = build_temporal_streams(_trials(), n_units, cfg, start_index=start)
    _TEMPORAL_CACHE[key] = streams
    return streams


def running_mask(streams: TemporalStreams, cfg) -> np.ndarray:
    """Boolean mask of in-trial, running bins (the report's engaged reference)."""
    return ~np.isnan(streams.trial_idx) & (streams.vel >= cfg.velocity_thresh_cm_s)


def zscore_over_reference(spikes_area: np.ndarray,
                          ref_mask: np.ndarray) -> np.ndarray:
    """Per-unit z-score using the rows selected by ``ref_mask`` as reference."""
    if not np.any(ref_mask):
        return spikes_area
    ref = spikes_area[ref_mask]
    sd = ref.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (spikes_area - ref.mean(axis=0)) / sd


def area_activity(animal: Animal, area: str, cfg, period: str = "engaged"
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-area time-binned activity for one ``period``.

    Returns ``(spikes_area, vel, trial_idx, unit_indices)`` on the period's
    global timeline. When ``cfg.zscore_units`` the activity is per-unit z-scored
    over the period's running+in-trial reference (report Section 2.2). The
    analysis layer applies the running mask and trial-grouped CV itself.
    """
    streams = load_temporal_streams(animal, cfg, period)
    idx = select_units(animal, area, cfg)
    spikes_area = streams.spikes[:, idx]
    if cfg.zscore_units:
        spikes_area = zscore_over_reference(spikes_area, running_mask(streams, cfg))
    return spikes_area, streams.vel, streams.trial_idx, idx
