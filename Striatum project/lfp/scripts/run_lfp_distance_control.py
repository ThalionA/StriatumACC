#!/usr/bin/env python3
"""Is the cross-area LFP coupling anything more than distance along the shank?

The control asked for at the 2026-09-09 meeting. DMS, DLS and ACC sit on one
shank, so "different area" and "further apart" are the same axis, and every
cross-area coupling number in this package is confounded with separation. The
test that separates them: compare pairs of channels the SAME distance apart,
within one area against across an area boundary.

Reads the band-power caches (`results/lfp_band_trials_<cohort>/*.npz`) -- no
re-extraction from the voltage exports -- and writes:

    results/lfp_distance_control_<cohort>.csv         coupling by separation bin
    results/lfp_distance_matched_<cohort>.csv         the matched-distance contrast

The second table is the answer in one row per (animal, probe, band): mean
coupling within and across an area boundary, over the separation range where
BOTH classes exist, plus the mean separation of each class so a reader can check
they really were matched.

    /opt/anaconda3/bin/python scripts/run_lfp_distance_control.py --cohort task
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from striatum_lfp import config, distance, trials  # noqa: E402

SEP_BIN_UM = 100.0
#: A class needs at least this many pairs in the matched range to be reported.
MIN_PAIRS = 20


def area_labels(z) -> np.ndarray:
    """One area name per channel, "" where the channel is in no area.

    Depth bands in the CSVs can touch (1206's probe 2 has DG ending and CA1
    starting at 1160 um), so a channel can carry two masks. `geometry` resolves
    that with a fixed precedence and the cache stores the resolved masks; the
    assertion below makes sure the file being read really is resolved, rather
    than silently taking whichever area is checked last.
    """
    masks = {a: z[f"is_{a.lower()}"] for a in config.AREAS if f"is_{a.lower()}" in z.files}
    stack = np.vstack([masks[a] for a in masks]) if masks else np.zeros((0, 0), bool)
    if stack.size:
        overlap = stack.sum(axis=0) > 1
        if overlap.any():
            raise ValueError(
                f"{int(overlap.sum())} channel(s) carry more than one area mask; "
                "the cache was written before geometry resolved touching bands")
    labels = np.full(z["channel_depth_um"].shape, "", dtype=object)
    for a, m in masks.items():
        labels[m] = a
    return labels


def run_one(path: Path, cohort_name: str) -> tuple[list[dict], list[dict]]:
    t0 = time.time()
    z = np.load(path, allow_pickle=False)
    mouse, probe = int(z["mouse_id"]), str(z["probe"])
    labels = area_labels(z)
    # Good, engaged (<= DP) and covered trials: `good_trials` alone is an
    # alignment flag and would keep the disengaged tail of the session.
    usable = trials.sessions_for(cohort_name)[mouse].with_data(z["good_trials"]).usable()
    depths = z["channel_depth_um"]
    bands = [str(b) for b in z["bands"]]

    by_bin: list[dict] = []
    matched: list[dict] = []
    for bi, band in enumerate(bands):
        cube = z["corridor"][bi][:, :, usable].astype(np.float64)
        res = distance.pairwise_coupling(cube, depths, labels)
        if res.separation_um.size == 0:
            continue
        tag = {"cohort": cohort_name, "mouse_id": mouse, "probe": probe, "band": band}
        by_bin += distance.summarise(res, bin_um=SEP_BIN_UM, extra=tag)

        # The result: within minus across at IDENTICAL separation. A shared
        # separation RANGE is not enough -- see exact_matched_contrast's note.
        row = {**tag}
        row.update(distance.exact_matched_contrast(res, min_pairs=MIN_PAIRS))
        # Kept alongside so the confound stays visible rather than being quietly
        # corrected away: the range-restricted numbers and how mismatched the
        # two classes' separations are inside that range.
        lo, hi = distance.matched_separation_range(res)
        row["range_lo_um"], row["range_hi_um"] = lo, hi
        if np.isfinite(lo):
            in_range = (res.separation_um >= lo) & (res.separation_um <= hi)
            for cls, mask in (("within", res.same_area), ("across", ~res.same_area)):
                sel = in_range & mask
                if sel.sum() >= MIN_PAIRS:
                    row[f"range_mean_sep_{cls}_um"] = float(res.separation_um[sel].mean())
                    row[f"range_r_raw_{cls}"] = float(np.nanmean(res.r_raw[sel]))
        if "range_r_raw_within" in row and "range_r_raw_across" in row:
            row["range_d_raw"] = row["range_r_raw_within"] - row["range_r_raw_across"]
            row["range_sep_imbalance_um"] = (row["range_mean_sep_within_um"]
                                             - row["range_mean_sep_across_um"])
        matched.append(row)

    n_lab = int(sum(1 for a in labels if a))
    print(f"[dist] {cohort_name[:4]:<4} {mouse}/{probe:9s} "
          f"{n_lab:3d} labelled ch, {usable.size:3d} trials, "
          f"{len(bands)} bands, {len(by_bin):4d} bins  {time.time() - t0:5.1f}s",
          flush=True)
    return by_bin, matched


def write(rows: list[dict], path: Path) -> None:
    if not rows:
        print(f"[dist] nothing to write to {path.name}")
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[dist] wrote {path.name} ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="task", choices=sorted(config.COHORTS))
    args = ap.parse_args()

    in_dir = config.RESULTS_DIR / f"lfp_band_trials_{args.cohort}"
    files = sorted(in_dir.glob("*.npz"))
    if not files:
        print(f"[dist] no caches in {in_dir}; run the bandpower step first")
        return
    print(f"[dist] cohort={args.cohort}: {len(files)} cached files")

    by_bin: list[dict] = []
    matched: list[dict] = []
    for p in files:
        b, m = run_one(p, args.cohort)
        by_bin += b
        matched += m

    write(by_bin, config.RESULTS_DIR / f"lfp_distance_control_{args.cohort}.csv")
    write(matched, config.RESULTS_DIR / f"lfp_distance_matched_{args.cohort}.csv")

    usable = [r for r in matched if r.get("n_separations", 0) > 0]
    if not usable:
        return
    d_raw = np.array([r["d_raw"] for r in usable])
    d_res = np.array([r["d_residual"] for r in usable])
    print(f"\n[dist] EXACT separation matching, {len(usable)} (animal, probe, band) cells, "
          f"median {int(np.median([r['n_separations'] for r in usable]))} separations each:")
    print(f"[dist]   within - across, raw log power : {d_raw.mean():+.4f} "
          f"(median {np.median(d_raw):+.4f}, {int((d_raw > 0).sum())}/{d_raw.size} positive)")
    print(f"[dist]   within - across, residual      : {d_res.mean():+.4f} "
          f"(median {np.median(d_res):+.4f}, {int((d_res > 0).sum())}/{d_res.size} positive)")
    biased = [r for r in usable if "range_d_raw" in r]
    if biased:
        rd = np.array([r["range_d_raw"] for r in biased])
        imb = np.array([r["range_sep_imbalance_um"] for r in biased])
        print(f"[dist] for comparison, the RANGE-restricted contrast on the same cells: "
              f"{rd.mean():+.4f}")
        print(f"[dist]   its classes differ in mean separation by {imb.mean():+.0f} um, "
              f"and that imbalance correlates with the contrast at "
              f"r = {np.corrcoef(imb, rd)[0, 1]:+.2f} — which is why it is not the result.")


if __name__ == "__main__":
    main()
