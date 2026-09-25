"""Full inventory of every named LFP voltage export: structure, integrity, spectra.

One full pass over every stored value per file, plus windowed Welch spectra and
a content fingerprint. Files are processed in parallel (one process per file).

Run from ``Striatum project/lfp``::

    /opt/anaconda3/bin/python scripts/run_lfp_inventory.py [--jobs N] [--only 727,1105]

Writes ``results/lfp_inventory.csv`` (one row per file), ``results/lfp_inventory.json``
(same plus the per-channel arrays), and ``results/lfp_psd.npz`` (the spectra).
Nothing under ``RawData/`` is written.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from striatum_lfp import cohort, config, geometry, inventory  # noqa: E402
from striatum_lfp.analysis import read_behaviour  # noqa: E402

FS = config.FS
SPECTRAL_WINDOWS = 12
SPECTRAL_WINDOW_S = 10
DEAD_CHANNEL_ZERO_FRACTION = 0.5   # a channel that is zero for most of the session


def behaviour_bounds(mouse_id: int, probe: str, ch):
    """First/last VR timestamp (s) and ``binned_spikes`` bin count for one probe."""
    if not config.raw_mat(mouse_id, probe, ch).exists():
        return None, None, None
    beh = read_behaviour(mouse_id, probe, ch)
    vr = beh["vr_times_s"]
    return float(vr.min()), float(vr.max()), beh["n_spike_bins"]


def area_counts(mouse_id: int, probe: str, n_channels: int, ch):
    """Channels per area, using the same depth boundaries the sorted units use."""
    try:
        bounds = geometry.load_area_boundaries(mouse_id, probe=probe, cohort=ch)
    except KeyError:
        return {}, [f"mouse {mouse_id} absent from the {probe} depth CSV"]
    # Unit (Kilosort) convention, not the shipped 0-3820 um: the boundaries were
    # drawn against unit depths, one row above the export's.
    depths = geometry.channel_depths(n_channels)
    masks = geometry.channel_area_masks(depths, bounds)
    counts = {area: int(mask.sum()) for area, mask in masks.items()}
    notes = [f"{area}: 0 channels in [{bounds[area][0]:g}, {bounds[area][1]:g}] um"
             for area, n in counts.items() if n == 0]
    return counts, notes


def inventory_one(item) -> dict:
    (mouse_id, probe), path, cohort_name = item
    ch = config.get_cohort(cohort_name)
    t0 = time.time()
    path = Path(path)
    notes: list[str] = []

    struct = inventory.read_structure(path)
    depth_ok, depth_err = inventory.check_depth_against_geometry(
        struct["depth"], struct["n_channels"]
    )
    if not depth_ok:
        notes.append(f"depth_to_save deviates from NP1.0 geometry by {depth_err:g} um")
    channels_ok = bool(
        struct["channels"].size == struct["n_channels"]
        and np.array_equal(struct["channels"], np.arange(1, struct["n_channels"] + 1))
    )
    if not channels_ok:
        notes.append("channels_to_save is not 1..n")

    integ = inventory.scan_integrity(path, fs=FS)
    pad_s = inventory.padding_onset_s(integ["zero_fraction_per_s"])
    dead = int((integ["channel_zero_fraction"] > DEAD_CHANNEL_ZERO_FRACTION).sum())
    if dead:
        notes.append(f"{dead} channels are exactly zero for >50% of the session")
    if integ["nonfinite_fraction"] > 0:
        notes.append(f"non-finite values present ({integ['nonfinite_fraction']:.2e})")

    vr_first, vr_last, n_bins = behaviour_bounds(mouse_id, probe, ch)
    grid_ok = None if n_bins is None else bool(struct["n_samples"] == n_bins)
    if grid_ok is False:
        notes.append(
            f"GRID MISMATCH: LFP {struct['n_samples']} samples vs binned_spikes {n_bins} bins"
        )

    # Spectral windows are placed inside behaviour and clear of terminal padding.
    first = int((vr_first or 0) * FS)
    last = struct["n_samples"] if pad_s is None else int(pad_s * FS)
    if vr_last is not None:
        last = min(last, int(vr_last * FS))
    win = SPECTRAL_WINDOW_S * FS
    starts = inventory.window_starts(struct["n_samples"], SPECTRAL_WINDOWS, win,
                                     first=first, last=last)
    spec = inventory.spectral_profile(path, starts, win, fs=FS)

    counts, area_notes = area_counts(mouse_id, probe, struct["n_channels"], ch)
    notes += area_notes

    row = {
        "cohort": cohort_name,
        "mouse_id": mouse_id,
        "probe": probe,
        "file": path.name,
        "size_gb": path.stat().st_size / 1e9,
        "n_samples": struct["n_samples"],
        "duration_min": struct["n_samples"] / FS / 60,
        "n_channels": struct["n_channels"],
        "dtype": struct["dtype"],
        "chunks": str(struct["chunks"]),
        "compression": struct["compression"],
        "depth_matches_geometry": depth_ok,
        "depth_max_abs_error_um": depth_err,
        "channels_are_1_to_n": channels_ok,
        "spike_n_bins": n_bins,
        "grid_compatible": grid_ok,
        "vr_first_s": vr_first,
        "vr_last_s": vr_last,
        "exact_zero_fraction": integ["exact_zero_fraction"],
        "nonfinite_fraction": integ["nonfinite_fraction"],
        "padding_start_s": pad_s,
        "dead_channel_count": dead,
        "median_channel_rms": float(np.median(integ["channel_rms"])),
        "lf_hf_ratio": spec["lf_hf_ratio"],
        "loglog_slope_2_40hz": spec["loglog_slope_2_40hz"],
        "adjacent_r": spec["adjacent_r"],
        "distant_r": spec["distant_r"],
        "common_mean_residual": spec["common_mean_residual"],
        "common_median_residual": spec["common_median_residual"],
        "fingerprint": inventory.fingerprint(path),
        "elapsed_s": None,
        "notes": "; ".join(notes),
    }
    for hz, ratio in spec["line_ratios"].items():
        row[f"line_ratio_{hz}"] = ratio
    for area in config.AREAS:
        row[f"n_ch_{area}"] = counts.get(area, 0)
    row["elapsed_s"] = time.time() - t0

    arrays = {
        "freqs": spec["freqs"],
        "psd": spec["psd"].astype(np.float32),
        "channel_rms": integ["channel_rms"],
        "channel_zero_fraction": integ["channel_zero_fraction"],
        "rms_per_s": integ["rms_per_s"].astype(np.float32),
        "zero_fraction_per_s": integ["zero_fraction_per_s"].astype(np.float32),
        "depth": struct["depth"],
        "window_starts": starts,
    }
    print(f"[inventory] {cohort_name[:4]:<4} {mouse_id}/{probe:9s} done in {row['elapsed_s']:6.1f}s"
          f"  LF/HF={row['lf_hf_ratio']:8.1f}  slope={row['loglog_slope_2_40hz']:6.2f}"
          f"  adj/dist r={row['adjacent_r']:.2f}/{row['distant_r']:.2f}"
          + (f"  !! {row['notes']}" if row["notes"] else ""), flush=True)
    return {"row": row, "arrays": arrays}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--only", type=str, default="",
                        help="comma-separated mouse ids to restrict to")
    parser.add_argument("--cohort", type=str, default="task",
                        choices=sorted(config.COHORTS))
    args = parser.parse_args()
    ch = config.get_cohort(args.cohort)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    found, skipped = cohort.discover_lfp_files(ch.lfp_dir, ch.mouse_ids,
                                               return_skipped=True)
    if args.only:
        keep = {int(x) for x in args.only.split(",")}
        found = {k: v for k, v in found.items() if k[0] in keep}

    print(f"[inventory] cohort={args.cohort}: {len(found)} named exports; "
          f"skipped (not in this cohort's analysis list): {skipped}")
    items = [(k, str(v), args.cohort) for k, v in sorted(found.items())]

    t0 = time.time()
    with mp.Pool(min(args.jobs, len(items))) as pool:
        results = pool.map(inventory_one, items)
    print(f"[inventory] all files in {(time.time() - t0) / 60:.1f} min")

    rows = [r["row"] for r in results]
    fields = list(rows[0].keys())
    with (config.RESULTS_DIR / f"lfp_inventory_{args.cohort}.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (config.RESULTS_DIR / f"lfp_inventory_{args.cohort}.json").write_text(
        json.dumps(rows, indent=2, default=str))

    bundle = {}
    for r in results:
        tag = f"{r['row']['mouse_id']}_{r['row']['probe']}"
        for name, arr in r["arrays"].items():
            bundle[f"{tag}__{name}"] = arr
    np.savez_compressed(config.RESULTS_DIR / f"lfp_psd_{args.cohort}.npz", **bundle)

    # Duplicate detection: two names for one recording is the failure mode that
    # cost the July audit a mouse (614/731), so it is checked, not assumed away.
    seen: dict[str, list[str]] = {}
    for row in rows:
        seen.setdefault(row["fingerprint"], []).append(f"{row['mouse_id']}/{row['probe']}")
    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    print(f"[inventory] duplicate fingerprints: {dupes if dupes else 'none'}")
    print(f"[inventory] wrote lfp_inventory_{args.cohort}.csv")


if __name__ == "__main__":
    main()
