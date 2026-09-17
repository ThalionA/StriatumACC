#!/usr/bin/env python3
"""Mutual information between single-unit spiking and each behavioural feature.

The first arm of the Lemke/Panzeri mirror: how much each unit tells you about
each behavioural feature, as a function of time around reward-zone entry, and how
that changes across learning epochs.

Following the paper where it transfers:

* spikes binarised at 10 ms; each behavioural feature into **3 equipopulated
  bins**, recomputed WITHIN each epoch so the comparison across epochs is not
  contaminated by the feature's own drift;
* responses **pooled over a moving window** of time points to increase samples --
  the paper used 5 points shifted by 2, here 5 points (50 ms) shifted by 5, which
  keeps the windows non-overlapping so neighbouring values are independent;
* bias removed by **shuffle subtraction**: the mean of a permuted distribution is
  subtracted, with the permutation applied across TRIALS so a unit's own temporal
  structure survives it;
* trial counts **matched across epochs** within an animal, because mutual
  information is biased by sample size and an unmatched comparison between a
  10-trial and a 200-trial epoch measures the trial count.

Two tables:

``mi_timecourse_<cohort>.csv``   per (animal, area, epoch, feature, window)
``mi_units_<cohort>.csv``        per (animal, unit, area, epoch, feature), peak over time

    /opt/anaconda3/bin/python scripts/run_mi.py --cohort task
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lfp" / "src"))

from striatum_info import estimators as est  # noqa: E402
from striatum_lfp import analysis, config as lfp_config  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
N_FEATURE_BINS = 3
POOL_WIN = 5        # time bins pooled per window (50 ms at 10 ms bins)
POOL_SHIFT = 5      # non-overlapping, so neighbouring windows are independent
N_SHUFFLES = 50
MIN_TRIALS = 20     # below this an epoch cannot support a 3-bin estimate


def epoch_windows(valid: np.ndarray, n_third: int | None = None) -> dict[str, np.ndarray]:
    """``{epoch: trial indices}`` — All, Naive (first third), Expert (last third).

    NOT the project's usual learning-point epochs. Those are 10-trial windows
    (and 3 for "Trials 1-3"), which is far too few for a 3-bin information
    estimate: the first run of this script computed nothing but "All" because
    every learning epoch fell below the minimum. Lemke et al. faced the same
    constraint and solved it by pooling DAYS; we have one session, so the
    equivalent is a larger fraction of it.

    A thirds split is also closer to the paper than an LP split would be: their
    naive and skilled are the first 3-4 and last 2-4 DAYS of training, a
    time-based division, not a performance-based one.
    """
    usable = np.flatnonzero(valid)
    out = {"All": usable}
    if usable.size < 3:
        return out
    k = n_third or max(1, usable.size // 3)
    out["Naive"] = usable[:k]
    out["Expert"] = usable[-k:]
    return out


def run_animal(path: Path, cohort: str, rng_seed: int) -> tuple[list, list]:
    t0 = time.time()
    z = np.load(path, allow_pickle=False)
    spikes = z["spikes"]                      # (units, time bins, trials)
    feats = z["features"]                     # (trials, features)
    names = [str(s) for s in z["feature_names"]]
    areas = np.array([str(a) for a in z["unit_area"]])
    valid = z["valid"].astype(bool)
    mouse = int(path.stem.split("_")[-1]) if path.stem.split("_")[-1].isdigit() else 0

    windows = epoch_windows(valid)
    # Naive and Expert are the same size by construction; "All" is descriptive
    # only and is left unmatched. Matching matters because mutual information is
    # biased by sample size, so an unmatched epoch comparison partly measures the
    # trial count rather than the coding.
    n_match = min((windows[e].size for e in ("Naive", "Expert") if e in windows),
                  default=0)

    n_bins = spikes.shape[1]
    starts = np.arange(0, n_bins - POOL_WIN + 1, POOL_SHIFT)
    rng = np.random.default_rng(rng_seed)
    tc_rows, unit_rows = [], []

    for epoch, trials in windows.items():
        if trials.size < MIN_TRIALS:
            continue
        for fi, fname in enumerate(names):
            # Drop only the trials whose feature value is missing, not the whole
            # feature: zscored_lick_errors is NaN on trial 1 for every animal
            # (it needs a preceding trial), and an `isfinite().all()` guard threw
            # the entire lick-error feature away on the first run.
            ok = np.isfinite(feats[trials, fi])
            use = trials[ok]
            if use.size < MIN_TRIALS:
                continue
            if epoch != "All" and n_match >= MIN_TRIALS and use.size > n_match:
                use = np.sort(rng.choice(use, size=n_match, replace=False))
            fv = feats[use, fi]
            if np.unique(fv).size < N_FEATURE_BINS:
                continue
            n_tr = use.size
            codes = est.equipopulated_bins(fv, N_FEATURE_BINS)
            variants = np.empty((n_tr, 1 + N_SHUFFLES), dtype=int)
            variants[:, 0] = codes
            for sh in range(N_SHUFFLES):
                variants[:, sh + 1] = rng.permutation(codes)
            rep = np.tile(variants, (POOL_WIN, 1))

            # (units, windows, 1 + shuffles)
            mi = np.empty((spikes.shape[0], starts.size, 1 + N_SHUFFLES))
            for wi, w in enumerate(starts):
                block = spikes[:, w:w + POOL_WIN, :][:, :, use]
                mi[:, wi, :] = est.mi_binary_vs_categorical(
                    block.reshape(block.shape[0], -1), rep, N_FEATURE_BINS)

            # Shuffle subtraction per window, then a MAX-STATISTIC test over
            # windows. Taking the peak across 30 windows and comparing it with a
            # single window's null is circular -- the maximum of 30 draws beats a
            # one-draw threshold far more than 5% of the time, which is why the
            # first run called 63-78% of cells significant. Each shuffle's own
            # peak is therefore built the same way and the observed peak is
            # compared with THAT distribution (the paper's cluster permutation
            # uses the same max-per-shuffle idea).
            null_mean = mi[:, :, 1:].mean(axis=2)
            corrected = mi[:, :, 0] - null_mean
            obs_peak = corrected.max(axis=1)
            leave_one_out = ((mi[:, :, 1:].sum(axis=2)[:, :, None] - mi[:, :, 1:])
                             / (N_SHUFFLES - 1))
            null_peak = (mi[:, :, 1:] - leave_one_out).max(axis=1)   # (units, shuffles)
            peak_p = (1.0 + (null_peak >= obs_peak[:, None]).sum(axis=1)) / (N_SHUFFLES + 1.0)
            peak_ms = np.array([float(z["window_ms"][0]
                                      + (starts[k] + POOL_WIN / 2) * z["bin_ms"])
                                for k in corrected.argmax(axis=1)])

            for wi, w in enumerate(starts):
                centre_ms = float(z["window_ms"][0] + (w + POOL_WIN / 2) * z["bin_ms"])
                for area in sorted(set(areas[areas != ""])):
                    m = areas == area
                    tc_rows.append({
                        "cohort": cohort, "mouse_id": mouse, "epoch": epoch,
                        "n_trials": int(n_tr), "feature": fname, "area": area,
                        "time_ms": centre_ms, "n_units": int(m.sum()),
                        "mi_corrected_mean": float(np.nanmean(corrected[m, wi])),
                    })
            for u in range(spikes.shape[0]):
                if areas[u] == "":
                    continue
                unit_rows.append({
                    "cohort": cohort, "mouse_id": mouse, "epoch": epoch,
                    "n_trials": int(n_tr), "feature": fname, "area": areas[u],
                    "unit": u, "peak_mi_corrected": float(obs_peak[u]),
                    "peak_time_ms": float(peak_ms[u]), "peak_p": float(peak_p[u]),
                })

    used = sorted({r["epoch"] for r in unit_rows})
    print(f"[mi] {cohort[:4]:<4} {mouse:>5}: epochs {used}, matched n={n_match}, "
          f"{len(starts)} windows -> {len(unit_rows):5d} unit rows "
          f"({time.time() - t0:.0f}s)", flush=True)
    return tc_rows, unit_rows


def write(rows, path: Path) -> None:
    if not rows:
        print(f"[mi] nothing to write to {path.name}")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[mi] wrote {path.name} ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="task", choices=("task", "control"))
    args = ap.parse_args()
    files = sorted(RESULTS.glob(f"trials_{args.cohort}_*.npz"))
    if not files:
        print(f"[mi] no caches; run scripts/extract_trials.py --cohort {args.cohort}")
        return
    print(f"[mi] cohort={args.cohort}: {len(files)} animals, "
          f"{N_FEATURE_BINS} feature bins, pool {POOL_WIN}x{POOL_SHIFT}, "
          f"{N_SHUFFLES} shuffles")
    tc, units = [], []
    for k, f in enumerate(files):
        a, b = run_animal(f, args.cohort, rng_seed=1000 + k)
        tc += a
        units += b
    write(tc, RESULTS / f"mi_timecourse_{args.cohort}.csv")
    write(units, RESULTS / f"mi_units_{args.cohort}.csv")

    sig = [r for r in units if r["epoch"] == "All" and r["peak_p"] < 0.05]
    print(f"[mi] epoch=All: {len(sig)}/{len([r for r in units if r['epoch'] == 'All'])} "
          f"unit x feature cells peak at p < 0.05")


if __name__ == "__main__":
    main()
