#!/usr/bin/env python3
"""Mutual information between single-unit spiking and each behavioural feature.

The first arm of the Lemke/Panzeri mirror: how much each unit tells you about
each behavioural feature, as a function of time around reward-zone entry, and how
that changes across learning epochs.

Following the paper where it transfers:

* spikes binarised at 10 ms; each behavioural feature split at its **median**,
  recomputed WITHIN each epoch so the comparison across epochs is not
  contaminated by the feature's own drift. The paper used three equipopulated
  bins on sessions of hundreds of trials; this project's learning epochs are ten
  trials, so two bins (five trials each) is the workable split, and it is what
  the repo's existing MATLAB MI arm already uses (``cfg.mi_behav_bins = 2``);
* responses **pooled over a moving window** of time points to increase samples --
  the paper used 5 points shifted by 2, here 5 points (50 ms) shifted by 5, which
  keeps the windows non-overlapping so neighbouring values are independent;
* bias removed by **shuffle subtraction**: the mean of a permuted distribution is
  subtracted, with the permutation applied across TRIALS so a unit's own temporal
  structure survives it;
* epochs are the project's standard **Naive / Intermediate / Expert**, ten
  trials each, so they are trial-count matched by construction — which matters
  because mutual information is biased by sample size and an unmatched
  comparison between epochs would partly measure the trial count.

Two tables:

``mi_timecourse_<cohort>.csv``   per (animal, area, epoch, feature, window)
``mi_units_<cohort>.csv``        per (animal, unit, area, epoch, feature):
                                ``mean_mi_corrected`` (the unbiased summary, used
                                for every across-epoch contrast) and
                                ``peak_mi_corrected`` + ``peak_p`` (biased upward
                                by the max over windows -- read only as "does
                                this unit carry information anywhere")

    /opt/anaconda3/bin/python scripts/run_mi.py --cohort task
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lfp" / "src"))

from striatum_info import estimators as est  # noqa: E402
from striatum_lfp import config as lfp_config  # noqa: E402
from striatum_lfp import trials as lfp_trials  # noqa: E402
from striatum_lfp import results_io  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
N_FEATURE_BINS = 2   # median split, matching MutualInformationStriatum_v2's mi_behav_bins
POOL_WIN = 5        # time bins pooled per window (50 ms at 10 ms bins)
POOL_SHIFT = 5      # non-overlapping, so neighbouring windows are independent
N_SHUFFLES = 50
BLOCK_TRIALS = 5    # within-block label shuffles: slow drift stays in the null
MIN_TRIALS = 8      # a 10-trial epoch gives 5 per bin at 2 bins


def epoch_windows(mouse: int, cohort: str, valid: np.ndarray) -> dict[str, np.ndarray]:
    """``{epoch: raw trial indices}`` — the project's standard THREE-epoch scheme.

    Naive / Intermediate / Expert, ten trials each, learning-point relative
    (``project_cfg`` ``epoch_names``). Ten trials is thin for an information
    estimate, which is why the feature is split at its median rather than into
    three bins: two bins give five trials per bin instead of three, and it is the
    convention the repo's existing MATLAB MI arm already uses
    (``cfg.mi_behav_bins = 2``).

    The three windows are the same size by construction, so no trial-count
    matching is needed — and each epoch's bias is removed against its own
    shuffles anyway, which is what makes the comparison across epochs fair.
    """
    session = lfp_trials.sessions_for(cohort)[mouse].with_data(valid)
    return {"All": session.usable(), **session.epochs()}


def run_animal(path: Path, cohort: str, rng_seed: int) -> tuple[list, list]:
    t0 = time.time()
    z = np.load(path, allow_pickle=False)
    spikes = z["spikes"]                      # (units, time bins, trials)
    feats = z["features"]                     # (trials, features)
    names = [str(s) for s in z["feature_names"]]
    areas = np.array([str(a) for a in z["unit_area"]])
    valid = z["valid"].astype(bool)
    mouse = int(path.stem.split("_")[-1]) if path.stem.split("_")[-1].isdigit() else 0

    windows = epoch_windows(mouse, cohort, valid)
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
            fv = feats[use, fi]
            if np.unique(fv).size < N_FEATURE_BINS:
                continue
            n_tr = use.size
            # A split at a real value change: an equipopulated split breaks
            # ties by trial order ('success' is 98 % tied), which is how the
            # retracted 2026-09-18 success result was made. Same fix as the
            # LFP arm; this arm had kept the old binning.
            codes = est.value_boundary_split(fv)
            if codes is None:
                continue
            variants = np.empty((n_tr, 1 + N_SHUFFLES), dtype=int)
            variants[:, 0] = codes
            for sh in range(N_SHUFFLES):
                variants[:, sh + 1] = est.within_block_permutation(codes, BLOCK_TRIALS, rng)
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
            # The PEAK is what the paper reports and what peak_p tests, but it
            # is useless as an effect size to compare BETWEEN epochs: the max of
            # 30 noisy windows is positively biased, and the bias grows as the
            # sample shrinks. Measured on independent spikes and features
            # (2026-09-17): peak = +0.054 bits at 10 trials and +0.054 at 9,
            # a spurious -0.003 bit contrast purely from one missing trial --
            # which is exactly the lick_error_z case, since its NaN sits on
            # trial 1 and trial 1 lives in Naive. The MEAN over windows is
            # unbiased on the same data (-0.0001 vs +0.0000), so every
            # across-epoch comparison uses it.
            obs_mean = corrected.mean(axis=1)
            # Spikes each unit fired inside the analysed windows of these
            # trials. A silent unit has zero information by construction, and
            # 34 % of unit rows were exactly 0 -- enough to pin a per-animal
            # MEDIAN at 0 in 13-15 of 16 animals. Aggregation uses active units.
            n_spikes = spikes[:, starts[0]:starts[-1] + POOL_WIN, :][:, :, use].sum(axis=(1, 2))
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
                    "unit": u, "n_spikes": int(n_spikes[u]),
                    "mean_mi_corrected": float(obs_mean[u]),
                    "peak_mi_corrected": float(obs_peak[u]),
                    "peak_time_ms": float(peak_ms[u]), "peak_p": float(peak_p[u]),
                })

    used = sorted({r["epoch"] for r in unit_rows})
    print(f"[mi] {cohort[:4]:<4} {mouse:>5}: epochs {used}, lp={lfp_trials.sessions_for(cohort)[mouse].lp}, "
          f"{len(starts)} windows -> {len(unit_rows):5d} unit rows "
          f"({time.time() - t0:.0f}s)", flush=True)
    return tc_rows, unit_rows


def write(rows, path: Path) -> None:
    results_io.write_rows(rows, path, tag="mi")


def main() -> None:
    ap = argparse.ArgumentParser()
    lfp_config.add_cohort_argument(ap)
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
