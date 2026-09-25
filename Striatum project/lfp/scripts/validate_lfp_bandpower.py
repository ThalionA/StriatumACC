"""Check the LFP trial and bin map against the MATLAB unit pipeline.

The claim the whole product rests on is that an LFP band-power array indexes
like ``spatial_binned_fr_all``. Two checks are independent of this pipeline:

1. **Trials.** The cube's good trials against MATLAB's own good mask
   (``trials.matlab_good_masks``). A good LFP trial MATLAB dropped is an error;
   a MATLAB trial the LFP lacks is either a short export (407) or an error.
2. **Bin spans.** ``spatial_binned_data.durations`` records, for every good
   (trial, bin), ``(bin_times(end) - bin_times(1)) / 1000``. The same span is
   recomputed here from the VR frames (``corridor_bin_stop_ms - _start_ms``) and
   compared on MATLAB-good trials. ``spatial_binned_data`` is on the RAW trial
   index (156 columns for 1212, whose ``n_trials`` is 155): ``ProcessStriatumTask.m``
   bins ``corridorData`` before its good-trial filter.

What neither can see is a constant offset between the VR clock and the voltage
samples: both sides share it. The pre-2026-09-25 "0.0 ms" compared this
pipeline's accumulator counts with spans built from its own stored boundaries,
i.e. Python with Python; that comparison is kept only as an internal check.

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

from striatum_lfp import config, trials  # noqa: E402

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

    masks = trials.matlab_good_masks(ch)
    rows = []
    with h5py.File(ch.preproc_mat, "r") as handle:
        for path in sorted(out_dir.glob("*.npz")):
            z = np.load(path, allow_pickle=False)
            mouse = int(z["mouse_id"])
            probe = str(z["probe"])
            if mouse not in ch.mouse_ids:
                continue
            durations, n_trials_matlab = matlab_animal(handle, ch.mouse_ids.index(mouse))
            good = z["good_trials"].astype(bool)
            n_stored = good.size
            mask = masks[mouse][:n_stored]

            # 1. trials, against MATLAB's own mask
            lfp_only_trials = int((good & ~mask).sum())
            matlab_only_trials = int((mask & ~good).sum())

            # 2. bin spans. `durations` is raw-indexed (see the module docstring);
            # compare on the trials MATLAB calls good.
            k = int(min(durations.shape[1], n_stored))
            raw_cols = np.flatnonzero(masks[mouse][:k])
            start_ms = z["corridor_bin_start_ms"][:, raw_cols].astype(float)
            stop_ms = z["corridor_bin_stop_ms"][:, raw_cols].astype(float)
            span_py = np.where(start_ms >= 0, stop_ms - start_ms, np.nan)
            span_matlab = durations[:, raw_cols] * 1000.0
            both = (np.isfinite(span_py) & np.isfinite(span_matlab) & (span_matlab > 0)
                    & good[raw_cols][None, :])
            diff = np.abs(span_py - span_matlab)[both]

            # internal only: accumulator counts vs the stored boundaries
            counts = z["corridor_counts"][:, raw_cols].astype(float)
            cor0 = z["corridor_start_sample"][raw_cols].astype(np.int64)[None, :]
            trial_stop = z["trial_stop_sample"][raw_cols].astype(np.int64)[None, :]
            expected = np.minimum(cor0 + stop_ms + 1, trial_stop) - (cor0 + start_ms)
            internal = np.abs(counts - expected)[both & (counts > 0)]

            rows.append({
                "cohort": args.cohort, "mouse_id": mouse, "probe": probe,
                "n_trials_matlab": n_trials_matlab,
                "n_good_trials_lfp": int(good.sum()),
                "lfp_good_not_matlab_good": lfp_only_trials,
                "matlab_good_not_lfp_good": matlab_only_trials,
                "n_compared_bins": int(both.sum()),
                "span_median_abs_diff_ms": float(np.median(diff)) if diff.size else np.nan,
                "span_p99_abs_diff_ms": float(np.percentile(diff, 99)) if diff.size else np.nan,
                "span_frac_within_1ms": float((diff <= 1).mean()) if diff.size else np.nan,
                "internal_max_abs_diff_ms": float(internal.max()) if internal.size else np.nan,
            })
            r = rows[-1]
            flag = ("" if r["span_frac_within_1ms"] > 0.99 and lfp_only_trials == 0
                    else "  <-- CHECK")
            print(f"{mouse:>5}/{probe:9s} good trials LFP {r['n_good_trials_lfp']:>4} vs "
                  f"MATLAB {n_trials_matlab:>4} (LFP-only {lfp_only_trials}, MATLAB-only "
                  f"{matlab_only_trials}) | {r['n_compared_bins']:>6} bins, span within 1 ms "
                  f"{r['span_frac_within_1ms']:.4f}, p99 {r['span_p99_abs_diff_ms']:.1f} ms"
                  f"{flag}", flush=True)

    out = config.RESULTS_DIR / f"lfp_bandpower_validation_{args.cohort}.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    ok = sum(r["span_frac_within_1ms"] > 0.99 and r["lfp_good_not_matlab_good"] == 0
             for r in rows)
    print(f"\n{ok}/{len(rows)} files: no good LFP trial MATLAB dropped, and >99 % of bin "
          f"spans within 1 ms of MATLAB's durations (a shared clock offset is invisible here)")


if __name__ == "__main__":
    main()
