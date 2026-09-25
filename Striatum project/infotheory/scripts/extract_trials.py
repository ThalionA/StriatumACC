#!/usr/bin/env python3
"""Build per-animal caches of reward-zone-aligned spikes and per-trial features.

The data layer for the Lemke/Panzeri mirror. Reads the MATLAB preprocessed
products directly through h5py -- only the small fields, so the tens-of-GB
structs never load -- and writes one npz per animal:

    features      (n_trials, n_features)  every behavioural scalar
    spikes        (n_units, n_bins, n_trials) uint8, binarised at 10 ms
    unit_area     (n_units,)              area label per unit
    entry_ms      (n_trials,)             reward-zone entry, trial-relative
    valid         (n_trials,)             trial usable: MATLAB-good, event found, window fits

Every array is on the RAW trial index -- the index of ``binned_spikes_trials``,
``npx_times_trials``, ``trial_metrics``, the LFP band-power cubes and the
disengagement point. ``trialData`` and ``zscored_lick_errors`` are MATLAB's
good-filtered arrays, so they are read through the good-trial map from
``striatum_lfp.trials``; a raw trial MATLAB dropped (1212's raw trial 102) keeps
its slot with NaN features and ``valid = False``. Indexing both sides by one
counter used to pair every later 1212 trial with the next trial's behaviour.

Aligned to REWARD-ZONE ENTRY, restricted to the corridor. See `trials` for why
the corridor restriction is not optional.

    /opt/anaconda3/bin/python scripts/extract_trials.py --cohort task
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lfp" / "src"))

from striatum_info import trials as T  # noqa: E402
from striatum_lfp import config as lfp_config  # noqa: E402
from striatum_lfp import trials as lfp_trials  # noqa: E402

WINDOW_MS = (-1000, 500)
BIN_MS = 10
AREA_FIELDS = {"DMS": "is_dms", "DLS": "is_dls", "ACC": "is_acc",
               "V1": "is_v1", "CA1": "is_ca1", "DG": "is_dg"}
OUT_DIR = Path(__file__).resolve().parents[1] / "results"


def _cell(h, ref):
    return np.array(h[ref])


def unit_areas(h, pd, i: int, n_units: int) -> np.ndarray:
    """One area name per unit, "" where the unit is in no labelled area."""
    labels = np.full(n_units, "", dtype=object)
    for area, field in AREA_FIELDS.items():
        if field not in pd:
            continue
        m = np.asarray(h[pd[field][i, 0]]).ravel().astype(bool)
        if m.size == n_units:
            labels[m] = area
    return labels


def run_animal(h, pd, i: int, mouse_label: str, matlab_good: np.ndarray) -> dict | None:
    t0 = time.time()
    td = h[pd["trialData"][i, 0]]
    bs = h[pd["binned_spikes_trials"][i, 0]]
    npxs = h[pd["npx_times_trials"][i, 0]]
    zle = np.asarray(h[pd["zscored_lick_errors"][i, 0]]).ravel()
    tm = h[pd["trial_metrics"][i, 0]]
    success = np.asarray(tm["trial_success"]).ravel() if "trial_success" in tm else None

    # Raw-indexed: spikes, npx times, trial_metrics. Good-indexed: trialData and
    # zscored_lick_errors. The two differ exactly by MATLAB's good-trial filter,
    # so the lengths must reconcile through it or the map is wrong.
    n_trials = int(bs.shape[0])
    good_raw = np.flatnonzero(matlab_good)
    n_good = int(td["trial_world"].shape[0])
    if matlab_good.size != n_trials or good_raw.size != n_good or zle.size != n_good:
        raise ValueError(f"{mouse_label}: raw {n_trials} / mask {matlab_good.size} / "
                         f"good {good_raw.size} vs trialData {n_good}, lick errors {zle.size}")
    good_number = np.full(n_trials, -1)
    good_number[good_raw] = np.arange(n_good)
    # Lick errors: good-indexed in products written after 2026-09-25 (they carry
    # `good_trials`); before that MATLAB computed them from the unfiltered
    # corridorData, i.e. the first n_trials RAW trials.
    lick_on_good_index = "good_trials" in pd
    feats, spikes, entries, valid = [], [], [], []
    n_units = None
    for t in range(n_trials):
        k = int(good_number[t])
        if k < 0:                                  # MATLAB dropped this trial
            feats.append([np.nan] * len(T.FEATURE_NAMES))
            entries.append(np.nan)
            valid.append(False)
            spikes.append(None)
            continue
        world = _cell(h, td["trial_world"][k, 0]).ravel()
        pos = _cell(h, td["trial_position"][k, 0]).ravel()
        tim = _cell(h, td["trial_times_zeroed"][k, 0]).ravel()
        licks = _cell(h, td["trial_licks"][k, 0]).ravel()
        f = T.behavioural_features(
            world, pos, tim, licks,
            lick_error_z=float(zle[k]) if lick_on_good_index
            else (float(zle[t]) if t < zle.size else np.nan),
            success=float(success[t]) if success is not None and t < success.size else np.nan)
        feats.append([f[k2] for k2 in T.FEATURE_NAMES])

        entry = T.reward_zone_entry(world, pos, tim)
        entries.append(np.nan if entry is None else entry)
        sp = _cell(h, bs[t, 0])
        npx = _cell(h, npxs[t, 0]).ravel()
        if n_units is None:
            n_units = sp.shape[1]
        aligned = None if entry is None else T.align_spikes(
            sp, npx, entry, window_ms=WINDOW_MS, bin_ms=BIN_MS)
        if aligned is None:
            valid.append(False)
            spikes.append(None)
        else:
            valid.append(True)
            spikes.append(aligned[0])

    valid = np.array(valid)
    if not valid.any() or n_units is None:
        print(f"  {mouse_label}: no usable trial", flush=True)
        return None
    n_bins = (WINDOW_MS[1] - WINDOW_MS[0]) // BIN_MS
    cube = np.zeros((n_units, n_bins, n_trials), dtype=np.uint8)
    for t, s in enumerate(spikes):
        if s is not None:
            cube[:, :, t] = s

    areas = unit_areas(h, pd, i, n_units)
    counts = {a: int((areas == a).sum()) for a in AREA_FIELDS if (areas == a).any()}
    print(f"  {mouse_label}: {valid.sum():3d}/{n_trials} trials aligned, "
          f"{n_units} units {counts}  ({time.time() - t0:.0f}s)", flush=True)
    return {
        "features": np.array(feats, dtype=float),
        "feature_names": np.array(T.FEATURE_NAMES),
        "spikes": cube,
        "unit_area": np.array([str(a) for a in areas]),
        "entry_ms": np.array(entries, dtype=float),
        "valid": valid,
        "window_ms": np.array(WINDOW_MS),
        "bin_ms": np.array(BIN_MS),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    lfp_config.add_cohort_argument(ap)
    args = ap.parse_args()

    ch = lfp_config.get_cohort(args.cohort)
    OUT_DIR.mkdir(exist_ok=True)
    print(f"[extract] cohort={args.cohort}: {ch.preproc_mat.name}, "
          f"window {WINDOW_MS} ms at {BIN_MS} ms, aligned to reward-zone entry")

    with h5py.File(ch.preproc_mat, "r") as h:
        pd = h["preprocessed_data"]
        n = pd["trialData"].shape[0]
        ids = ch.mouse_ids
        masks = lfp_trials.matlab_good_masks(ch)
        for i in range(n):
            label = str(ids[i])
            out = run_animal(h, pd, i, label, masks[ids[i]])
            if out is None:
                continue
            path = OUT_DIR / f"trials_{args.cohort}_{label}.npz"
            np.savez_compressed(path, **out)
    print(f"[extract] wrote {len(list(OUT_DIR.glob(f'trials_{args.cohort}_*.npz')))} caches")


if __name__ == "__main__":
    main()
