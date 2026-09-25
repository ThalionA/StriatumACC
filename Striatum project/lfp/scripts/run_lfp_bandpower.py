"""Extract per-trial, per-bin LFP band power -- the drop-in analogue of firing rate.

For every named export, one streaming pass produces, per band and per channel:
mean band power in each of the 50 corridor spatial bins and each of the 50 dark
100 ms bins, for each trial. The bin-to-sample map is the one
``spatial_binning.m`` uses for spikes, so the arrays index like
``spatial_binned_fr_all`` and ``temp_binned_dark_fr``.

Run from ``Striatum project/lfp``::

    /opt/anaconda3/bin/python scripts/run_lfp_bandpower.py [--jobs N] [--only 727,1105]

Writes ``results/lfp_band_trials/<mouse>_<probe>.npz`` (~150 MB each) and
``results/lfp_bandpower_summary.csv``.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from striatum_lfp import bandpower, cohort, config, geometry  # noqa: E402
from striatum_lfp.analysis import read_behaviour  # noqa: E402
from striatum_lfp.reader import DATASET  # noqa: E402

FS = config.FS
# Cap on stored trials. The largest learning point in the cohort is 84, so the
# Expert window ends by trial 93; 200 leaves better than 2x headroom for the
# decoding / reliability / CCA arms while keeping each file near 150 MB.
MAX_TRIALS = 200
BLOCK_SAMPLES = 210_000          # 5000 x 42-row HDF5 chunks
PAD_SAMPLES = 3_000              # >= 3 s: covers the 1 Hz filter transient
def out_dir(cohort_name: str) -> Path:
    return config.RESULTS_DIR / f"lfp_band_trials_{cohort_name}"


def build_segments(beh: dict, n_lfp_samples: int) -> dict:
    """Absolute LFP sample ranges for every (trial, bin) cell of both states.

    Returns the corridor and dark segment lists as ``(cell, start, stop)`` with
    ``stop`` exclusive, plus the per-trial bookkeeping the caller stores.
    """
    corrected_vr_ms = (beh["vr_times_s"] - beh["vr_times_s"][0]) * 1000.0
    starts_vr, ends_vr = bandpower.trial_boundaries(beh["trial"])
    n_npx = min(beh["crop_end0"], n_lfp_samples - 1) - beh["crop_start0"] + 1

    npx_start0 = bandpower.npx_index(corrected_vr_ms[starts_vr], n_npx)
    npx_end0 = bandpower.npx_index(corrected_vr_ms[ends_vr], n_npx)
    truncated = bandpower.truncated_trials(corrected_vr_ms[ends_vr], n_lfp_samples,
                                           beh["crop_start0"])
    edges = bandpower.spatial_bin_edges()

    corridor_segs: list[tuple[int, int, int]] = []
    dark_segs: list[tuple[int, int, int]] = []
    # Corridor-relative, UNCLIPPED first/last VR-frame millisecond of each bin.
    # These are what spatial_binning.m turns into its `durations` field, and they
    # give the per-bin traversal time -- i.e. running speed, the covariate every
    # learning claim on band power has to be checked against.
    bin_start_ms = np.full((bandpower.N_SPATIAL_BINS, starts_vr.size), -1, dtype=np.int32)
    bin_stop_ms = np.full((bandpower.N_SPATIAL_BINS, starts_vr.size), -1, dtype=np.int32)
    good = np.zeros(starts_vr.size, dtype=bool)
    trial_start_sample = np.full(starts_vr.size, -1, dtype=np.int64)
    trial_stop_sample = np.full(starts_vr.size, -1, dtype=np.int64)
    corridor_start_sample = np.full(starts_vr.size, -1, dtype=np.int64)
    dark_len = np.zeros(starts_vr.size, dtype=np.int64)

    n_keep = min(starts_vr.size, MAX_TRIALS)
    for i in range(n_keep):
        s, e = starts_vr[i], ends_vr[i]
        times_zeroed = corrected_vr_ms[s:e + 1] - corrected_vr_ms[s]
        world = beh["world"][s:e + 1]
        position = beh["position"][s:e + 1]
        trial_len_npx = int(npx_end0[i] - npx_start0[i] + 1)
        if trial_len_npx <= 1:
            continue
        trial_abs0 = beh["crop_start0"] + int(npx_start0[i])
        trial_stop = min(beh["crop_start0"] + int(npx_end0[i]) + 1, n_lfp_samples)
        trial_start_sample[i] = trial_abs0
        trial_stop_sample[i] = trial_stop
        if truncated[i]:
            # 1212's export stops 41 min before its session does, so one trial
            # straddles the end of the file. A half-trial is not a trial: bin
            # whatever is there for the record, but never mark it good.
            continue

        onset = bandpower.corridor_start(world, times_zeroed)
        if onset is None:
            continue                       # the try/catch case: not a good trial
        vr_idx, onset_ms = onset
        corridor_rel0 = int(np.clip(round(onset_ms), 0, trial_len_npx - 1))
        if corridor_rel0 <= 0 or vr_idx >= position.size - 1:
            continue
        corridor_abs0 = trial_abs0 + corridor_rel0
        corridor_start_sample[i] = corridor_abs0
        dark_len[i] = corridor_rel0
        good[i] = True

        for b, seg in enumerate(bandpower.dark_bin_segments(corridor_rel0)):
            if seg is None:
                continue
            cell = i * bandpower.N_DARK_BINS + b
            dark_segs.append((cell, trial_abs0 + seg[0], trial_abs0 + seg[1] + 1))

        segs = bandpower.spatial_bin_segments(position[vr_idx:], times_zeroed[vr_idx:], edges)
        for b, seg in enumerate(segs):
            if seg is None:
                continue
            bin_start_ms[b, i], bin_stop_ms[b, i] = seg
            lo = corridor_abs0 + seg[0]
            hi = min(corridor_abs0 + seg[1] + 1, trial_stop, n_lfp_samples)
            if hi <= lo:
                continue
            corridor_segs.append((i * bandpower.N_SPATIAL_BINS + b, lo, hi))

    return {
        "corridor_segs": corridor_segs, "dark_segs": dark_segs,
        "good_trials": good[:n_keep], "n_trials_stored": n_keep,
        "n_trials_total": int(starts_vr.size),
        "trial_start_sample": trial_start_sample[:n_keep],
        "trial_stop_sample": trial_stop_sample[:n_keep],
        "corridor_bin_start_ms": bin_start_ms[:, :n_keep],
        "corridor_bin_stop_ms": bin_stop_ms[:, :n_keep],
        "corridor_start_sample": corridor_start_sample[:n_keep],
        "dark_n_samples": dark_len[:n_keep],
    }


def extract_one(item) -> dict:
    (mouse_id, probe), path, cohort_name = item
    ch = config.get_cohort(cohort_name)
    t0 = time.time()
    path = Path(path)
    band_names = list(bandpower.ANALYSIS_BANDS)

    with h5py.File(path, "r") as handle:
        n_samples, n_channels = map(int, handle[DATASET].shape)
    beh = read_behaviour(mouse_id, probe, ch)
    geo = build_segments(beh, n_samples)

    n_trials = geo["n_trials_stored"]
    n_cor_cells = n_trials * bandpower.N_SPATIAL_BINS
    n_dark_cells = n_trials * bandpower.N_DARK_BINS
    accs = {
        (b, "corridor"): bandpower.SegmentAccumulator(n_cor_cells, n_channels)
        for b in band_names
    }
    accs.update({
        (b, "dark"): bandpower.SegmentAccumulator(n_dark_cells, n_channels)
        for b in band_names
    })

    all_segs = geo["corridor_segs"] + geo["dark_segs"]
    if not all_segs:
        raise RuntimeError(f"{mouse_id}/{probe}: no usable trials")
    span_lo = max(0, min(s[1] for s in all_segs) - PAD_SAMPLES)
    span_hi = min(n_samples, max(s[2] for s in all_segs) + PAD_SAMPLES)

    with h5py.File(path, "r") as handle:
        dset = handle[DATASET]
        for core0 in range(span_lo, span_hi, BLOCK_SAMPLES):
            core1 = min(core0 + BLOCK_SAMPLES, span_hi)
            read0 = max(span_lo, core0 - PAD_SAMPLES)
            read1 = min(span_hi, core1 + PAD_SAMPLES)
            raw = np.asarray(dset[read0:read1, :], dtype=np.float32)
            raw = bandpower.apply_notches(raw, fs=FS)
            for band in band_names:
                power = bandpower.band_power_series(
                    raw, bandpower.ANALYSIS_BANDS[band], fs=FS
                )
                core = power[core0 - read0:core1 - read0]
                accs[(band, "corridor")].add_block(core, core0, geo["corridor_segs"])
                accs[(band, "dark")].add_block(core, core0, geo["dark_segs"])

    def stack(state: str, n_bins: int) -> np.ndarray:
        # (band, channel, bin, trial), mirroring MATLAB's (unit, bin, trial).
        out = np.stack([
            accs[(b, state)].result().reshape(n_trials, n_bins, n_channels)
            for b in band_names
        ])
        return np.ascontiguousarray(out.transpose(0, 3, 2, 1), dtype=np.float32)

    corridor = stack("corridor", bandpower.N_SPATIAL_BINS)
    dark = stack("dark", bandpower.N_DARK_BINS)
    cor_counts = accs[(band_names[0], "corridor")].counts.reshape(
        n_trials, bandpower.N_SPATIAL_BINS).T.astype(np.int32)
    dark_counts = accs[(band_names[0], "dark")].counts.reshape(
        n_trials, bandpower.N_DARK_BINS).T.astype(np.int32)

    depths = geometry.channel_depths(n_channels)
    try:
        bounds = geometry.load_area_boundaries(mouse_id, probe=probe, cohort=ch)
        masks = geometry.channel_area_masks(depths, bounds)
    except KeyError:
        masks = {}

    target = out_dir(cohort_name)
    target.mkdir(parents=True, exist_ok=True)
    out_path = target / f"{mouse_id}_{probe}.npz"
    np.savez_compressed(
        out_path,
        corridor=corridor, dark=dark,
        corridor_counts=cor_counts, dark_counts=dark_counts,
        bands=np.array(band_names), band_edges=np.array(
            [bandpower.ANALYSIS_BANDS[b] for b in band_names]),
        mouse_id=mouse_id, probe=probe, cohort=cohort_name,
        channel_depth_um=depths,
        good_trials=geo["good_trials"],
        n_trials_total=geo["n_trials_total"],
        trial_start_sample=geo["trial_start_sample"],
        trial_stop_sample=geo["trial_stop_sample"],
        corridor_bin_start_ms=geo["corridor_bin_start_ms"],
        corridor_bin_stop_ms=geo["corridor_bin_stop_ms"],
        corridor_start_sample=geo["corridor_start_sample"],
        dark_n_samples=geo["dark_n_samples"],
        **{f"is_{a.lower()}": masks.get(a, np.zeros(n_channels, bool)) for a in config.AREAS},
    )

    n_good = int(geo["good_trials"].sum())
    filled = float(np.isfinite(corridor[0, :, :, geo["good_trials"]]).mean())
    row = {
        "cohort": cohort_name, "mouse_id": mouse_id, "probe": probe,
        "n_trials_total": geo["n_trials_total"], "n_trials_stored": n_trials,
        "n_good_trials": n_good,
        "truncated_at_max": geo["n_trials_total"] > MAX_TRIALS,
        "corridor_cells_filled": filled,
        "median_samples_per_spatial_bin": float(
            np.median(cor_counts[:, geo["good_trials"]][cor_counts[:, geo["good_trials"]] > 0])),
        "file_mb": out_path.stat().st_size / 1e6,
        "elapsed_s": time.time() - t0,
    }
    print(f"[bandpower] {cohort_name[:4]:<4} {mouse_id}/{probe:9s} {n_good:4d}/{n_trials:4d} good trials  "
          f"{filled:5.1%} cells filled  median {row['median_samples_per_spatial_bin']:.0f} ms/bin  "
          f"{row['file_mb']:5.0f} MB  {row['elapsed_s']:5.0f}s"
          + ("  [TRUNCATED to 200 trials]" if row["truncated_at_max"] else ""), flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--only", type=str, default="")
    parser.add_argument("--cohort", type=str, default="task",
                        choices=sorted(config.COHORTS))
    args = parser.parse_args()
    ch = config.get_cohort(args.cohort)

    found = cohort.discover_lfp_files(ch.lfp_dir, ch.mouse_ids)
    if args.only:
        keep = {int(x) for x in args.only.split(",")}
        found = {k: v for k, v in found.items() if k[0] in keep}
    items = [(k, str(v), args.cohort) for k, v in sorted(found.items())]
    print(f"[bandpower] cohort={args.cohort}: {len(items)} files, bands {list(bandpower.ANALYSIS_BANDS)}, "
          f"notch {bandpower.NOTCH_HZ} Hz", flush=True)

    t0 = time.time()
    with mp.Pool(min(args.jobs, len(items))) as pool:
        rows = pool.map(extract_one, items)
    print(f"[bandpower] all files in {(time.time() - t0) / 60:.1f} min")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / f"lfp_bandpower_summary_{args.cohort}.csv"
    # Merge on (mouse, probe) rather than overwrite: a `--only` rerun must not
    # wipe the rows for the files it did not touch.
    merged = {}
    if out.exists():
        with out.open() as fh:
            for old in csv.DictReader(fh):
                merged[(old["mouse_id"], old["probe"])] = old
    for r in rows:
        merged[(str(r["mouse_id"]), r["probe"])] = {k: str(v) for k, v in r.items()}
    ordered = sorted(merged.values(), key=lambda r: (int(r["mouse_id"]), r["probe"]))
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(ordered)
    print(f"[bandpower] wrote {out.name}")


if __name__ == "__main__":
    main()
