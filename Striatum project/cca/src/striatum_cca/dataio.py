"""Data IO and cohort logic for the striatum CCA pipeline.

Loads ``preprocessed_data.mat`` (MATLAB v7.3 / HDF5), detects learning points,
classifies the cohort into learners and yoked non-learners, and builds the
per-area activity tensors with fast-spiking units excluded.

Trial / unit conventions
------------------------
``spatial_binned_fr_all`` is stored in MATLAB as (n_units, n_bins, n_trials);
h5py reads it transposed to ``(n_trials, n_bins, n_units)`` — which is exactly
the layout the CCA pipeline wants, so no transpose is applied on load.
Learning points are 1-indexed trial numbers (matching the MATLAB convention).
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
    """One animal's preprocessed data, as loaded from the .mat file."""

    animal_id: int                       # 1-indexed (matches MATLAB)
    spatial_fr: np.ndarray               # (n_trials, n_bins, n_units)
    neurontypes: np.ndarray              # (n_units, 5)
    area_masks: dict[str, np.ndarray]    # area -> (n_units,) bool
    change_point: float                  # disengagement trial (nan if none)
    zscored_lick_errors: np.ndarray      # (n_trials_behaviour,)
    n_trials: int
    corridor_path: Path | None = None    # .mat path (for temporal binning)

    @property
    def n_units(self) -> int:
        return self.spatial_fr.shape[2]

    @property
    def n_bins(self) -> int:
        return self.spatial_fr.shape[1]


def load_animals(path: str | Path | None = None) -> list[Animal]:
    """Load every animal from ``preprocessed_data.mat``."""
    path = Path(path) if path is not None else config.PREPROCESSED_DATA
    if not path.is_file():
        raise FileNotFoundError(f"preprocessed data not found: {path}")

    animals: list[Animal] = []
    with h5py.File(path, "r") as f:
        pd = f["preprocessed_data"]
        n_animals = pd["n_trials"].shape[0]
        for i in range(n_animals):

            def deref(field_name: str):
                return f[pd[field_name][i, 0]]

            spatial_fr = np.asarray(deref("spatial_binned_fr_all"), dtype=float)
            neurontypes = np.asarray(deref("final_neurontypes"), dtype=float).T
            masks = {
                area: np.asarray(deref(fld)).ravel().astype(bool)
                for area, fld in config.AREA_FIELD.items()
            }
            change_point = float(np.asarray(deref("change_point_mean")).ravel()[0])
            z = np.asarray(deref("zscored_lick_errors"), dtype=float).ravel()
            n_trials = int(np.asarray(deref("n_trials")).ravel()[0])

            _validate_shapes(i + 1, spatial_fr, neurontypes, masks)
            if animals and spatial_fr.shape[1] != animals[0].spatial_fr.shape[1]:
                raise ValueError(
                    f"animal {i + 1} has {spatial_fr.shape[1]} spatial bins "
                    f"but animal 1 has {animals[0].spatial_fr.shape[1]}"
                )
            animals.append(
                Animal(
                    animal_id=i + 1,
                    spatial_fr=spatial_fr,
                    neurontypes=neurontypes,
                    area_masks=masks,
                    change_point=change_point,
                    zscored_lick_errors=z,
                    n_trials=n_trials,
                    corridor_path=path,
                )
            )
    return animals


def _validate_shapes(animal_id, spatial_fr, neurontypes, masks) -> None:
    n_units = spatial_fr.shape[2]
    if neurontypes.shape[0] != n_units:
        raise ValueError(
            f"animal {animal_id}: neurontypes has {neurontypes.shape[0]} rows "
            f"but spatial_fr has {n_units} units"
        )
    for area, mask in masks.items():
        if mask.size != n_units:
            raise ValueError(
                f"animal {animal_id}: mask {area} length {mask.size} != "
                f"{n_units} units"
            )


# ---------------------------------------------------------------------------
# Learning point + cohort classification (D1)
# ---------------------------------------------------------------------------
def find_learning_point(zscored_lick_errors: np.ndarray, cfg) -> int | None:
    """Detect the learning point: the trial sustained learning begins from.

    The learning point is the first trial ``t`` that is **itself** below
    threshold (z-scored lick error <= ``lp_z_threshold``) *and* from which
    performance is sustained -- the window ``[t, t + lp_window)`` contains at
    least ``lp_min_consecutive`` below-threshold trials.

    The ``below[t]`` requirement is the correction noted by Theo: the old
    rule (and the MATLAB ``movsum`` it reproduced) returned the window *start*,
    which can be an above-threshold trial -- so the learning point could land
    on a trial that had not itself reached z <= -2.

    Returns
    -------
    int (1-indexed) or None if no learning point.
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
    """Split the cohort into learners and yoked non-learners (D1).

    A learner has a detected learning point and is not in
    ``cfg.manual_nonlearners``. Non-learners receive the cohort-mean ("yoked")
    learning point so their early/mid/late trial windows remain analysable.

    Returns
    -------
    (entries, yoked_lp) — ``entries`` keyed by 1-indexed animal id.
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
# Epochs (D-self-made)
# ---------------------------------------------------------------------------
def n_usable_trials(animal: Animal) -> int:
    """Trial count after disengagement truncation at ``change_point_mean``."""
    if np.isnan(animal.change_point):
        return animal.n_trials
    return int(min(round(animal.change_point), animal.n_trials))


def epoch_windows(lp: int, n_usable: int, cfg) -> dict[str, np.ndarray] | None:
    """Return the 0-indexed trial indices of the three learning epochs.

    naive        = trials 1..10                  (0-indexed 0..9)
    intermediate = the 10 trials ending at lp     (0-indexed lp-10 .. lp-1)
    expert       = the 10 trials after lp         (0-indexed lp .. lp+9)

    The intermediate epoch was re-added in round 10. Returns None if the
    windows would overlap (lp < 2 * trials_per_epoch) or the expert window
    would exceed the usable trials.
    """
    e = cfg.trials_per_epoch
    if lp < 2 * e:                 # naive and intermediate would overlap
        return None
    if lp + e > n_usable:          # expert window exceeds usable trials
        return None
    return {
        "naive": np.arange(0, e),
        "intermediate": np.arange(lp - e, lp),
        "expert": np.arange(lp, lp + e),
    }


# ---------------------------------------------------------------------------
# Per-area activity tensors
# ---------------------------------------------------------------------------
def select_units(animal: Animal, area: str, cfg) -> np.ndarray:
    """Indices of the units in ``area`` that survive selection.

    Keeps units flagged for the area; drops fast-spiking units (type 2) when
    ``cfg.exclude_fast_spiking`` is set.
    """
    keep = animal.area_masks[area].copy()
    if cfg.exclude_fast_spiking:
        types = animal.neurontypes[:, config.NTYPE_COL]
        keep &= types != config.FS_TYPE_CODE
    return np.where(keep)[0]


# ---------------------------------------------------------------------------
# Temporal binning (round 8): re-bin corridorData.binned_spikes (1 ms counts)
# ---------------------------------------------------------------------------
def rebin_trial(spikes_1ms: np.ndarray, bin_ms: int) -> np.ndarray:
    """Sum 1 ms spike counts into ``bin_ms``-wide bins.

    ``(n_1ms, n_units) -> (n_1ms // bin_ms, n_units)``; an incomplete final
    bin is dropped. Returns float32.
    """
    n_1ms, n_units = spikes_1ms.shape
    n_bins = n_1ms // bin_ms
    if n_bins == 0:
        return np.zeros((0, n_units), dtype=np.float32)
    trimmed = spikes_1ms[: n_bins * bin_ms].astype(np.float32)
    return trimmed.reshape(n_bins, bin_ms, n_units).sum(axis=1)


def pad_trials(trials: list[np.ndarray]) -> np.ndarray:
    """Stack variable-length per-trial ``(n_bins, n_units)`` arrays into a
    ``(n_trials, max_bins, n_units)`` tensor, NaN-padding the shorter trials.

    Temporal corridor traversals differ in duration; the NaN tail is dropped
    at every fit, so each trial keeps its full natural length (no truncation).
    """
    if not trials:
        return np.zeros((0, 0, 0), dtype=np.float32)
    max_bins = max(t.shape[0] for t in trials)
    n_units = trials[0].shape[1]
    out = np.full((len(trials), max_bins, n_units), np.nan, dtype=np.float32)
    for i, t in enumerate(trials):
        out[i, : t.shape[0]] = t
    return out


# One-animal cache: temporal re-binning reads many MB of 1 ms data per animal,
# so the ragged per-trial result is kept for the animal currently being
# prepared and dropped when the next animal is requested.
_CORRIDOR_CACHE: dict = {}


def _corridor_data(animal: Animal, cfg) -> list[np.ndarray]:
    """Per-trial corridor spike counts re-binned to ``cfg.temporal_bin_ms``.

    Reads ``corridorData.binned_spikes`` (1 ms counts) for the animal's
    ``n_usable`` trials and re-sums each into ``temporal_bin_ms`` bins.
    Returns a ragged list of ``(n_time_bins, n_total_units)`` arrays.
    """
    key = (animal.animal_id, cfg.temporal_bin_ms)
    cached = _CORRIDOR_CACHE.get(key)
    if cached is not None:
        return cached
    _CORRIDOR_CACHE.clear()                       # 1-slot -- bound memory
    n_use = n_usable_trials(animal)
    n_units = animal.neurontypes.shape[0]
    trials = []
    max_bins = cfg.temporal_max_trial_ms // cfg.temporal_bin_ms
    with h5py.File(animal.corridor_path, "r") as fh:
        ref = fh["preprocessed_data"]["corridorData"][animal.animal_id - 1, 0]
        binned_spikes = fh[ref]["binned_spikes"]
        for t in range(n_use):
            arr = np.asarray(fh[binned_spikes[t, 0]])
            if arr.ndim != 2 or arr.shape[1] != n_units:
                arr = np.zeros((0, n_units))      # malformed/empty trial
            binned = rebin_trial(arr, cfg.temporal_bin_ms)
            if binned.shape[0] > max_bins:        # disengaged: over-long traversal
                binned = np.zeros((0, n_units), dtype=np.float32)
            trials.append(binned)
    _CORRIDOR_CACHE[key] = trials
    return trials


def area_tensor(animal: Animal, area: str, cfg):
    """Build one area's activity for the n_usable trials.

    Returns ``(activity, unit_indices)``. For ``cfg.bin_mode == "spatial"``
    ``activity`` is a 3-D ``(n_usable, n_bins, n_units_kept)`` tensor; for
    ``"temporal"`` it is a ragged list of per-trial ``(n_time_bins, n_kept)``
    arrays -- each trial kept at its full natural length.
    """
    idx = select_units(animal, area, cfg)
    if cfg.bin_mode == "temporal":
        return [trial[:, idx] for trial in _corridor_data(animal, cfg)], idx
    n_use = n_usable_trials(animal)
    return animal.spatial_fr[:n_use][:, :, idx], idx
