"""Check the LFP bin map against the MATLAB unit pipeline, bin for bin.

The claim the whole product rests on is that an LFP band-power array indexes
like ``spatial_binned_fr_all``. That is testable without re-running MATLAB:
``spatial_binned_data.durations`` records, for every (trial, bin),
``(bin_times(end) - bin_times(1)) / 1000`` -- exactly the span this pipeline
integrates band power over. If the two disagree, the bins are not the same bins.

Run from ``Striatum project/lfp``::

    /opt/anaconda3/bin/python scripts/validate_lfp_bandpower.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from striatum_lfp import config  # noqa: E402

# The index into preprocessed_data is the animal's POSITION in its organiser's
# list, not its mouse id (OrganiseStriatumDataIncV1.m:9 /
# OrganiseStriatumDataControlIncV1.m:20).


def matlab_animal(handle, index: int):
    P = handle["preprocessed_data"]
    sbd = handle[P["spatial_binned_data"][index, 0]]
    # MATLAB stores durations as (n_trials, 50); h5py transposes it to (50, n_trials),
    # which is already the orientation of this pipeline's corridor_counts.
    durations = np.asarray(sbd["durations"])
    n_trials = int(np.asarray(handle[P["n_trials"][index, 0]]).ravel()[0])
    return durations, n_trials


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=str, default="task",
                        choices=sorted(config.COHORTS))
    args = parser.parse_args()
    ch = config.get_cohort(args.cohort)
    out_dir = config.RESULTS_DIR / f"lfp_band_trials_{args.cohort}"

    rows = []
    with h5py.File(ch.preproc_mat, "r") as handle:
        for path in sorted(out_dir.glob("*.npz")):
            z = np.load(path, allow_pickle=False)
            mouse = int(z["mouse_id"])
            probe = str(z["probe"])
            if mouse not in ch.mouse_ids:
                continue
            durations, n_trials_matlab = matlab_animal(handle, ch.mouse_ids.index(mouse))

            counts = z["corridor_counts"]                    # (50, n_trials_stored)
            good = z["good_trials"]
            n = min(counts.shape[1], durations.shape[1])
            got = counts[:, :n].astype(float)

            # MATLAB's `durations` is the UNCLIPPED VR span (last - first) / 1000,
            # but the spike sum it feeds uses npx indices clipped to the trial's
            # length -- so the last bin of a trial is systematically shorter in the
            # data than in the durations field. Reproduce the clip here, otherwise
            # the check flags the pipeline for copying MATLAB correctly.
            start_ms = z["corridor_bin_start_ms"][:, :n].astype(np.int64)
            stop_ms = z["corridor_bin_stop_ms"][:, :n].astype(np.int64)
            cor0 = z["corridor_start_sample"][:n].astype(np.int64)[None, :]
            trial_stop = z["trial_stop_sample"][:n].astype(np.int64)[None, :]
            clipped_stop = np.minimum(cor0 + stop_ms + 1, trial_stop)
            expected_ms = np.where(start_ms >= 0,
                                   clipped_stop - (cor0 + start_ms), np.nan).astype(float)
            unclipped_ms = durations[:, :n] * 1000.0 + 1.0
            n_clipped = int(np.sum(np.isfinite(expected_ms) & np.isfinite(unclipped_ms)
                                   & (np.abs(expected_ms - unclipped_ms) > 1)
                                   & good[:n][None, :]))

            both = np.isfinite(expected_ms) & (got > 0) & good[:n][None, :]
            diff = np.abs(got - expected_ms)[both]
            rows.append({
                "cohort": args.cohort, "mouse_id": mouse, "probe": probe,
                "n_trials_matlab": n_trials_matlab,
                "n_good_trials_lfp": int(good.sum()),
                "n_compared_cells": int(both.sum()),
                "median_abs_diff_ms": float(np.median(diff)) if diff.size else np.nan,
                "p99_abs_diff_ms": float(np.percentile(diff, 99)) if diff.size else np.nan,
                "max_abs_diff_ms": float(diff.max()) if diff.size else np.nan,
                "frac_within_1ms": float((diff <= 1).mean()) if diff.size else np.nan,
                # A bin MATLAB used but this pipeline dropped (or vice versa).
                "n_matlab_only": int((np.isfinite(expected_ms) & (got == 0)
                                      & good[:n][None, :]).sum()),
                "n_lfp_only": int((~np.isfinite(expected_ms) & (got > 0)
                                   & good[:n][None, :]).sum()),
                "n_trial_end_clipped": n_clipped,
            })
            r = rows[-1]
            flag = "" if r["frac_within_1ms"] > 0.99 and r["n_matlab_only"] == 0 else "  <-- CHECK"
            print(f"{mouse:>5}/{probe:9s} trials {r['n_good_trials_lfp']:>4} vs MATLAB "
                  f"{n_trials_matlab:>4} | {r['n_compared_cells']:>6} cells | "
                  f"median {r['median_abs_diff_ms']:.1f} ms  p99 {r['p99_abs_diff_ms']:.1f}  "
                  f"max {r['max_abs_diff_ms']:.0f} | within 1 ms {r['frac_within_1ms']:.4f} | "
                  f"matlab-only {r['n_matlab_only']} lfp-only {r['n_lfp_only']} "
                  f"clipped {r['n_trial_end_clipped']}{flag}", flush=True)

    out = config.RESULTS_DIR / f"lfp_bandpower_validation_{args.cohort}.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    ok = sum(r["frac_within_1ms"] > 0.99 and r["n_matlab_only"] == 0 for r in rows)
    print(f"\n{ok}/{len(rows)} files reproduce the MATLAB bin map to within 1 ms")


if __name__ == "__main__":
    main()
