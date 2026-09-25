"""Per-area signals read from the voltage exports, one trial at a time.

Hoisted out of ``scripts/run_lfp_psi.py`` on 2026-09-17 when the coupling
analysis needed the same thing. Both drivers read the same corridor samples and
reduce them to the same two per-area signals; keeping one copy means a change to
the bipolar derivation or the channel selection reaches both.

Two referencing schemes come back for every area, because on this probe they
bracket the answer (see the distance-control and phase-slope results of the same
day):

monopolar
    Mean of the area's (z-scored) channels. Maximum sensitivity, maximum
    exposure to the far field.
bipolar
    Mean of non-overlapping VERTICAL pairs (``geometry.vertical_pairs``: each site
    minus the site one row up, deep minus shallow), on raw voltage. This cancels
    the far field to first order and keeps the local depth gradient.

What changed on 2026-09-25, and why each matters for coupling and PSI:

* Every read is mains-notched (:func:`read_notched`), as the band-power cubes
  always were. 50 Hz was 55-96 % of low-gamma power in 1105's area signals.
* Bipolar pairs are one row apart. Depth-sorting put the two same-row channels
  next to each other, so every earlier pair was 0 um vertical / ~32 um lateral
  and cancelled the local signal along with the field.
* The pair difference is taken on raw voltage. Z-scoring each channel first
  scales the common field by a different factor per channel whenever the local
  signal differs, and the field then leaks into the difference.
* The probe's reference site (``config.REFERENCE_CHANNELS``) is never used.
"""
from __future__ import annotations

import time
from pathlib import Path

import h5py
import numpy as np

from . import bandpower, config, geometry, psi, trials
from .cohort import discover_lfp_files
from .reader import DATASET

#: The project-wide floor for an area (``config.Config.min_sites``); the arms,
#: coupling, PSI and the LFP information arm all use this one number.
MIN_CHANNELS = config.DEFAULT.min_sites
#: Samples read on each side of a trial so the notch's transient stays outside it.
NOTCH_PAD = 1_000


def area_channels(z, min_channels: int = MIN_CHANNELS) -> dict[str, np.ndarray]:
    """``{area: channel indices}``, ordered by depth so bipolar pairs are adjacent."""
    depths = z["channel_depth_um"]
    out: dict[str, np.ndarray] = {}
    for a in config.AREAS:
        key = f"is_{a.lower()}"
        if key not in z.files:
            continue
        idx = np.flatnonzero(z[key])
        idx = idx[~np.isin(idx, config.REFERENCE_CHANNELS)]   # caches built before 09-25
        if idx.size >= min_channels:
            out[a] = idx[np.argsort(depths[idx])]
    return out


def reduce_block(block: np.ndarray, chans: dict[str, np.ndarray]) -> dict:
    """One monopolar and one bipolar signal per area, from a ``(samples, 384)`` read.

    Monopolar: channels are z-scored before averaging, so one high-variance site
    cannot dominate the mean. Bipolar: vertical pairs on raw voltage (see the
    module docstring for why not z-scored); NaN where the area has no pair.
    """
    out: dict[tuple[str, str], np.ndarray] = {}
    for area, idx in chans.items():
        sub = block[:, idx]
        centred = sub - sub.mean(axis=0, keepdims=True)
        sd = centred.std(axis=0, keepdims=True)
        out[(area, "monopolar")] = np.nanmean(centred / np.where(sd > 0, sd, np.nan), axis=1)
        pairs = geometry.vertical_pairs(idx)
        if pairs.size:
            column = {int(c): k for k, c in enumerate(idx)}
            order = [column[int(c)] for pair in pairs for c in pair]
            out[(area, "bipolar")] = psi.bipolar_derivation(centred[:, order])
        else:
            out[(area, "bipolar")] = np.full(block.shape[0], np.nan)
    return out


def read_notched(dset, start: int, stop: int, pad: int = NOTCH_PAD) -> np.ndarray:
    """``dset[start:stop]`` as float64, mains-notched on a padded read, pad removed."""
    lo, hi = max(0, start - pad), min(int(dset.shape[0]), stop + pad)
    raw = bandpower.apply_notches(np.asarray(dset[lo:hi], dtype=np.float64))
    return raw[start - lo: stop - lo]


def lfp_file_for(mouse: int, probe: str, cohort) -> Path:
    """The voltage export for one probe of one animal."""
    return discover_lfp_files(cohort.lfp_dir, cohort.mouse_ids)[(mouse, probe)]


def read_trial_signals(z, cohort, *, min_samples: int,
                       min_channels: int = MIN_CHANNELS):
    """``(chans, {trial index: {(area, reference): signal}}, seconds)``.

    Only the corridor samples of each usable trial (good, engaged and covered:
    ``trials.SessionTrials.usable``) are read, and each read is
    reduced to per-area signals immediately, so a whole session never sits in
    memory. Trials shorter than ``min_samples`` are skipped rather than padded.
    Returns an empty dict when the export is missing or nothing is long enough.
    """
    t0 = time.time()
    mouse, probe = int(z["mouse_id"]), str(z["probe"])
    chans = area_channels(z, min_channels)
    if len(chans) < 2:
        return chans, {}, time.time() - t0

    path = lfp_file_for(mouse, probe, cohort)
    if not path.exists():
        return chans, {}, time.time() - t0

    usable = trials.sessions_for(cohort.name)[mouse].with_data(z["good_trials"]).usable()
    starts, stops = z["corridor_start_sample"], z["trial_stop_sample"]
    per_trial: dict[int, dict] = {}
    with h5py.File(path, "r") as fh:
        dset = fh[DATASET]
        n_total = int(dset.shape[0])
        for t in usable:
            a, b = int(starts[t]), min(int(stops[t]), n_total)
            if b - a < min_samples:
                continue
            per_trial[int(t)] = reduce_block(read_notched(dset, a, b), chans)
    return chans, per_trial, time.time() - t0
