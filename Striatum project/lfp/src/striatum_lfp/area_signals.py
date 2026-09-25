"""Per-area signals read from the voltage exports, one trial at a time.

Hoisted out of ``scripts/run_lfp_psi.py`` on 2026-09-17 when the coupling
analysis needed the same thing. Both drivers read the same corridor samples and
reduce them to the same two per-area signals; keeping one copy means a change to
the bipolar derivation or the channel selection reaches both.

Two referencing schemes come back for every area, because on this probe they
bracket the answer (see the distance-control and phase-slope results of the same
day):

monopolar
    Mean of the area's channels. Maximum sensitivity, maximum exposure to the
    far field.
bipolar
    Mean of NON-OVERLAPPING adjacent-channel differences, which cancels the far
    field to first order.
"""
from __future__ import annotations

import time
from pathlib import Path

import h5py
import numpy as np

from . import config, psi, trials
from .cohort import discover_lfp_files
from .reader import DATASET

#: An area with fewer channels than this cannot give a bipolar derivation.
MIN_CHANNELS = 4


def area_channels(z, min_channels: int = MIN_CHANNELS) -> dict[str, np.ndarray]:
    """``{area: channel indices}``, ordered by depth so bipolar pairs are adjacent."""
    depths = z["channel_depth_um"]
    out: dict[str, np.ndarray] = {}
    for a in config.AREAS:
        key = f"is_{a.lower()}"
        if key not in z.files:
            continue
        idx = np.flatnonzero(z[key])
        if idx.size >= min_channels:
            out[a] = idx[np.argsort(depths[idx])]
    return out


def reduce_block(block: np.ndarray, chans: dict[str, np.ndarray]) -> dict:
    """One monopolar and one bipolar signal per area, from a ``(samples, 384)`` read.

    Channels are z-scored before combining so that a gain mismatch between two
    electrodes cannot leave residual common signal in the bipolar difference.
    """
    out: dict[tuple[str, str], np.ndarray] = {}
    for area, idx in chans.items():
        sub = block[:, idx]
        sub = sub - sub.mean(axis=0, keepdims=True)
        sd = sub.std(axis=0, keepdims=True)
        sub = sub / np.where(sd > 0, sd, np.nan)     # a dead channel drops out
        out[(area, "monopolar")] = np.nanmean(sub, axis=1)
        out[(area, "bipolar")] = psi.bipolar_derivation(sub)
    return out


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
            per_trial[int(t)] = reduce_block(
                np.asarray(dset[a:b, :], dtype=np.float64), chans)
    return chans, per_trial, time.time() - t0
