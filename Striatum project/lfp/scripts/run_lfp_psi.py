#!/usr/bin/env python3
"""Direction of communication between areas, from the LFP phase-slope index.

The third question from the 2026-09-09 meeting. It has to be asked with a measure
that cannot be fooled by the shared field the distance control found on
2026-09-17, so it is asked with the phase-slope index, which is blind to
instantaneous mixing by construction (see `striatum_lfp.psi`).

Two referencing schemes are computed for every pair, because they bracket the
answer:

monopolar
    Mean of the area's channels. Maximum sensitivity, maximum exposure to the
    far field -- the signal the rest of this package uses.
bipolar
    Mean of adjacent-channel differences within the area. Differencing two nearby
    electrodes cancels the far field to first order, which is the re-referencing
    the standing caveat in NOTES has been asking for since August. Less signal,
    far less volume conduction.

A direction that appears monopolar and vanishes bipolar is the field. One that
survives both is worth talking about.

Reads the voltage exports once per animal, keeping only the corridor samples and
reducing each read immediately to per-area signals, so the whole session never
sits in memory. Writes `results/lfp_psi_<cohort>.csv`, one row per
(animal, probe, epoch, area pair, band, reference).

    /opt/anaconda3/bin/python scripts/run_lfp_psi.py --cohort task
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from striatum_lfp import analysis, config, psi  # noqa: E402
from striatum_lfp.reader import DATASET  # noqa: E402

#: 2.048 s at 1 kHz -> 0.49 Hz bins, so theta (4-8 Hz) still holds ~9 of them.
NPERSEG = 2048
MIN_SEGMENTS = 8
MIN_CHANNELS = 4          # an area with fewer cannot give a bipolar derivation
BANDS = {"theta": (4.0, 8.0), "beta": (15.0, 30.0),
         "low_gamma": (30.0, 80.0), "high_gamma": (80.0, 150.0)}
EPOCHS = ("All", "Trials 1-3", "Trials 4-10", "Intermediate", "Expert")


def area_channels(z) -> dict[str, np.ndarray]:
    """{area: channel indices}, ordered by depth so bipolar pairs are adjacent."""
    depths = z["channel_depth_um"]
    out = {}
    for a in config.AREAS:
        key = f"is_{a.lower()}"
        if key not in z.files:
            continue
        idx = np.flatnonzero(z[key])
        if idx.size >= MIN_CHANNELS:
            out[a] = idx[np.argsort(depths[idx])]
    return out


def reduce_block(block: np.ndarray, chans: dict[str, np.ndarray]) -> dict[tuple[str, str], np.ndarray]:
    """One monopolar and one bipolar signal per area, from a (samples, 384) read."""
    out: dict[tuple[str, str], np.ndarray] = {}
    for area, idx in chans.items():
        sub = block[:, idx]
        sub = sub - sub.mean(axis=0, keepdims=True)
        sd = sub.std(axis=0, keepdims=True)
        sub = sub / np.where(sd > 0, sd, np.nan)     # a dead channel drops out
        out[(area, "monopolar")] = np.nanmean(sub, axis=1)
        # Non-overlapping adjacent pairs; see psi.bipolar_derivation for why an
        # averaged np.diff would silently collapse to a single wide pair.
        out[(area, "bipolar")] = psi.bipolar_derivation(sub)
    return out


def run_one(cache: Path, cohort_name: str) -> list[dict]:
    t0 = time.time()
    z = np.load(cache, allow_pickle=False)
    mouse, probe = int(z["mouse_id"]), str(z["probe"])
    chans = area_channels(z)
    if len(chans) < 2:
        print(f"[psi] {mouse}/{probe}: fewer than two usable areas, skipped", flush=True)
        return []

    ch = config.get_cohort(cohort_name)
    lp = analysis.cohort_learning_points(ch).get(mouse)
    good = np.flatnonzero(z["good_trials"].astype(bool))
    starts = z["corridor_start_sample"]
    stops = z["trial_stop_sample"]

    lfp_file = config.lfp_path(mouse, probe) if cohort_name == "task" else \
        ch.lfp_dir / f"{mouse}{'_v1' if probe == 'visual' else ''}_voltage_data_384ch.mat"
    if not lfp_file.exists():
        print(f"[psi] {mouse}/{probe}: {lfp_file.name} missing, skipped", flush=True)
        return []

    # Read each trial's corridor once, reduce to per-area signals, keep those.
    per_trial: dict[int, dict[tuple[str, str], np.ndarray]] = {}
    with h5py.File(lfp_file, "r") as fh:
        dset = fh[DATASET]
        n_total = int(dset.shape[0])
        for t in good:
            a, b = int(starts[t]), int(stops[t])
            if b > n_total:
                b = n_total
            if b - a < NPERSEG:
                continue
            block = np.asarray(dset[a:b, :], dtype=np.float64)
            per_trial[int(t)] = reduce_block(block, chans)
    if not per_trial:
        print(f"[psi] {mouse}/{probe}: no trial long enough for a {NPERSEG}-sample window",
              flush=True)
        return []

    n_trials = int(min(analysis.cohort_trial_counts(ch).get(mouse, len(good)), len(good)))
    windows = {"All": np.array(sorted(per_trial))}
    idx4 = analysis.epoch_indices(lp, n_trials, naive_split=analysis.NAIVE_SPLIT)
    for name, tr in zip(analysis.EPOCH_NAMES, idx4):
        windows[name] = np.array([t for t in (np.asarray(tr, int) - 1) if t in per_trial])

    rows: list[dict] = []
    for epoch in EPOCHS:
        trials = windows.get(epoch, np.array([], int))
        if trials.size == 0:
            continue
        for (a1, a2) in combinations(sorted(chans), 2):
            for ref in ("monopolar", "bipolar"):
                # Pass the trials as SNIPPETS, not joined: a spectral window is
                # never allowed to span two trials, whose phase relationship
                # would be arbitrary.
                x = [per_trial[t][(a1, ref)] for t in trials]
                y = [per_trial[t][(a2, ref)] for t in trials]
                if psi.count_segments(x, NPERSEG) < MIN_SEGMENTS:
                    continue
                # The reported z is the JACKKNIFE one. A mismatched-trial
                # surrogate was tried as a null and measured against synthetic
                # no-interaction data: it is anti-conservative by ~2x in z,
                # because every trial of a session shares the same slow field so
                # a mismatched pair stays coupled. See psi.trial_shuffled_nulls.
                for band, edges in BANDS.items():
                    try:
                        out = psi.phase_slope_index(
                            x, y, fs=config.FS, band=edges,
                            nperseg=NPERSEG, min_segments=MIN_SEGMENTS)
                    except ValueError:
                        continue
                    rows.append({
                        "cohort": cohort_name, "mouse_id": mouse, "probe": probe,
                        "learning_point": lp, "epoch": epoch, "n_trials": int(trials.size),
                        "area_a": a1, "area_b": a2, "reference": ref, "band": band,
                        "n_ch_a": int(chans[a1].size), "n_ch_b": int(chans[a2].size),
                        **out,
                    })
    print(f"[psi] {cohort_name[:4]:<4} {mouse}/{probe:9s} "
          f"{len(chans)} areas {sorted(chans)}, {len(per_trial):3d} trials, "
          f"{len(rows):4d} rows  {time.time() - t0:5.0f}s", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="task", choices=sorted(config.COHORTS))
    ap.add_argument("--only", default="", help="comma-separated mouse ids")
    args = ap.parse_args()

    in_dir = config.RESULTS_DIR / f"lfp_band_trials_{args.cohort}"
    files = sorted(in_dir.glob("*.npz"))
    if args.only:
        want = {m.strip() for m in args.only.split(",")}
        files = [f for f in files if f.name.split("_")[0] in want]
    if not files:
        print(f"[psi] no caches in {in_dir}")
        return
    print(f"[psi] cohort={args.cohort}: {len(files)} files, nperseg={NPERSEG} "
          f"({NPERSEG / config.FS:.2f} s, {config.FS / NPERSEG:.2f} Hz bins)")

    rows: list[dict] = []
    for f in files:
        rows += run_one(f, args.cohort)

    if not rows:
        print("[psi] nothing written")
        return
    out = config.RESULTS_DIR / f"lfp_psi_{args.cohort}.csv"
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[psi] wrote {out.name} ({len(rows)} rows)")

    for ref in ("monopolar", "bipolar"):
        sel = [r for r in rows if r["reference"] == ref and r["epoch"] == "All"
               and np.isfinite(r["z"])]
        if not sel:
            continue
        zz = np.array([r["z"] for r in sel])
        print(f"[psi] {ref:10s} epoch=All: {len(sel)} cells, "
              f"|z| > 2 in {int((abs(zz) > 2).sum())} ({(abs(zz) > 2).mean():.0%}, "
              f"calibrated chance ~5-10%), mean |z| {np.abs(zz).mean():.2f}")


if __name__ == "__main__":
    main()
