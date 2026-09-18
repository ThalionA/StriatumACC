#!/usr/bin/env python3
"""What LFP band power says about behaviour, and whether it survives running speed.

The Lemke/Panzeri mirror run on the FIELD rather than on single units. The spike
arm found real but tiny information (~0.004 bits) that did not move with
learning, and it had no positive control -- a null there was uninterpretable.
The field has one: band power must carry information about running speed. That
anchor is computed first, and every feature result is read against it.

Two arms, both per (animal, probe, area, band, epoch):

**speed**    ``I(power; speed)`` computed WITHIN each spatial bin across trials,
             then averaged over bins. Within-bin is the whole point: power and
             speed are both functions of position, so pooling bins would let
             that shared dependence masquerade as coupling.

Epochs are the project's ten-trial Naive / Intermediate / Expert, plus ``All``
and a wide ``EarlyHalf`` / ``LateHalf`` split. The ten-trial epochs are kept for
comparability with every other arm, but they are far too small to answer the
learning question -- their contrast noise exceeds the effect -- so the wide split
is what any statement about learning is read off.

**features** ``I(power; feature)`` and ``I(power; feature | speed)`` per window
             of ``POOL_BINS`` adjacent spatial bins, averaged over windows.
             Animals run faster as they learn and speed alone moves band power,
             so the unconditioned value is not interpretable on its own -- the
             conditional one is the result.

Design decisions carried over from the spike arm's failures:

* the summary over windows is the MEAN, never the peak. A max over windows is a
  selection statistic: it returns +0.05 bits on data with no information and its
  bias grows as the sample shrinks, so it manufactures contrasts between epochs
  of unequal size (``test_the_peak_over_windows_is_biased_...``).
* power and speed are ranked WITHIN each spatial bin before pooling, so a bin's
  baseline cannot survive into the pooled estimate.
* every value is shuffle-subtracted, with the permutation applied across TRIALS.

All five bands are reported, each with a measured ``band_status`` -- see
``BAND_STATUS`` below for the spectra and spike-coupling measurements behind it.

    /opt/anaconda3/bin/python scripts/run_lfp_mi.py --cohort task
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
LFP_RESULTS = Path(__file__).resolve().parents[2] / "lfp" / "results"

MAX_BIN = 30            # project spatial truncation (config.max_bin); reward zone is bin 25
POOL_BINS = 5           # adjacent spatial bins pooled per window -> 6 windows
N_POWER_BINS = 3
N_FEATURE_BINS = 2      # two bins; a ten-trial epoch cannot support more
N_SPEED_BINS = 2
N_SHUFFLES = 50
MIN_TRIALS = 8
WIDE_TRIALS = 25        # minimum half-size for the wide contrast; see epoch construction
MIN_CHANNELS = 4
# Band status, MEASURED rather than inherited (2026-09-18):
#
# * Mains is notched at 50/100/150 Hz unconditionally when the cubes are built
#   (run_lfp_bandpower.py:173). The dominant narrow peak in the RAW spectra is
#   50 Hz -- line noise -- at up to 31 dB above the 1/f background in 20 of 21
#   animal-probes, with a 150 Hz harmonic in 6. config.py's note about a "~75 Hz
#   narrow peak" is not what the spectra show.
# * What notching cannot remove is spike bleed-through. Measured as
#   Spearman(area firing rate, area band power) across trials, median over 50
#   area-cells: theta +0.03, beta +0.06, low_gamma +0.11, high_gamma +0.19,
#   total +0.16 -- monotonic in frequency, exactly the bleed-through signature.
#   High gamma shares about 3.5% of its variance with firing rate.
#
# So every band is reported. The gamma bands are usable but partly a spike-rate
# readout, and anything claimed from them must say so.
BAND_STATUS = {"theta": "clean", "beta": "clean", "low_gamma": "spike_coupled",
               "high_gamma": "spike_coupled", "total": "spike_coupled"}


def area_power(block: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``(spatial bin, trial)`` log power, averaged over an area's channels.

    ``block`` is one band's ``(channel, spatial bin, trial)`` slice.
    """
    return np.log10(np.maximum(block[mask], 1e-30)).mean(axis=0)


def variants_for(codes: np.ndarray, rng) -> np.ndarray:
    """Observed labels in column 0, ``N_SHUFFLES`` trial permutations after it."""
    out = np.empty((codes.size, 1 + N_SHUFFLES), int)
    out[:, 0] = codes
    for s in range(N_SHUFFLES):
        out[:, s + 1] = rng.permutation(codes)
    return out


def pooled_codes(power, speed, trials, bins, variants):
    """Stack ``bins`` worth of within-bin ranks into one pooled sample set."""
    xs, zs, vs = [], [], []
    for b in bins:
        ok = np.isfinite(power[b, trials]) & np.isfinite(speed[b, trials])
        if ok.sum() < MIN_TRIALS:
            continue
        xs.append(est.equipopulated_bins(power[b, trials][ok], N_POWER_BINS))
        zs.append(est.equipopulated_bins(speed[b, trials][ok], N_SPEED_BINS))
        vs.append(variants[ok])
    if not xs:
        return None
    return np.concatenate(xs), np.concatenate(zs), np.concatenate(vs, axis=0)


def run_one(path: Path, cohort: str, seed: int, dp_clip: bool = True) -> tuple[list, list]:
    t0 = time.time()
    z = np.load(path, allow_pickle=False)
    mouse, probe = int(z["mouse_id"]), str(z["probe"])
    ch = lfp_config.get_cohort(cohort)

    cache = RESULTS / f"trials_{cohort}_{mouse}.npz"
    if not cache.exists():
        print(f"[lfpmi] {mouse} {probe}: no behavioural cache, skipped", flush=True)
        return [], []
    zc = np.load(cache, allow_pickle=False)
    names = [str(s) for s in zc["feature_names"]]

    bands = [str(b) for b in z["bands"]]
    corridor = z["corridor"].astype(np.float64)[:, :, :MAX_BIN, :]
    speed = analysis.bin_speed_cm_s(z["corridor_bin_start_ms"],
                                    z["corridor_bin_stop_ms"])[:MAX_BIN]
    n_stored = corridor.shape[3]
    feats = zc["features"][:n_stored]
    good = z["good_trials"].astype(bool)[:n_stored]

    lp = analysis.cohort_learning_points(ch).get(mouse)
    lp_source = analysis.learning_point_sources(ch).get(mouse, "unknown")
    n_matlab = min(analysis.cohort_trial_counts(ch).get(mouse, n_stored), n_stored)
    # CLIP AT THE DISENGAGEMENT POINT. good_trials is an ALIGNMENT flag, not an
    # engagement one. Measured 2026-09-17 before this clip existed: 9 of 13 task
    # animals had their entire late window past DP, so an early-versus-late
    # contrast was largely measuring whether the animal was still doing the task.
    dp = analysis.disengagement_points(ch).get(mouse, np.nan) if dp_clip else np.nan
    usable = np.flatnonzero(good)
    if np.isfinite(dp):
        usable = usable[usable + 1 <= dp]        # trial numbers are 1-based
    windows = {"All": usable}
    for name, tr in zip(("Naive", "Intermediate", "Expert"),
                        analysis.epoch_indices(lp, n_matlab)):
        idx = np.asarray(tr, int) - 1
        keep = np.array([t for t in idx if t < n_stored and good[t]])
        # An LP-relative epoch can run past DP when the two are close (418: LP=26,
        # DP=27). Drop the epoch rather than compare engaged with disengaged.
        windows[name] = (np.array([], int) if keep.size and np.isfinite(dp)
                         and (keep + 1 > dp).any() else keep)

    # EarlyHalf / LateHalf split the ENGAGED period in two, count-matched within
    # each animal so the information bias cancels in the paired difference. Their
    # size therefore varies between animals, which is fine for a paired contrast
    # and would not be for a level comparison.
    #
    # The project's ten-trial epochs cannot answer the learning question here:
    # measured on the task cohort (2026-09-17) their Expert - Naive standard
    # error is 0.003-0.005 bits against an information LEVEL of 0.004, so only a
    # change larger than the whole effect would be detectable. WIDE is the same
    # contrast with enough trials to see a change -- first versus last fifty
    # stored trials, a session-time split like the paper's naive and skilled
    # DAYS, which is a time contrast rather than a performance-locked one.
    half = usable.size // 2
    if half >= WIDE_TRIALS:
        windows["EarlyHalf"] = usable[:half]
        windows["LateHalf"] = usable[-half:]

    areas = {a: z[f"is_{a.lower()}"] for a in lfp_config.AREAS}
    areas = {a: m for a, m in areas.items() if m.sum() >= MIN_CHANNELS}
    starts = np.arange(0, MAX_BIN - POOL_BINS + 1, POOL_BINS)
    rng = np.random.default_rng(seed)
    speed_rows, feat_rows = [], []
    base = {"cohort": cohort, "mouse_id": mouse, "probe": probe,
            "learning_point": lp, "lp_source": lp_source}

    for area, mask in areas.items():
        for bi, band in enumerate(bands):
            power = area_power(corridor[bi], mask)
            tag = BAND_STATUS.get(band, "unknown")
            for epoch, trials in windows.items():
                if trials.size < MIN_TRIALS:
                    continue

                # --- anchor: power vs speed, within bin, averaged over bins ---
                vals = []
                for b in range(MAX_BIN):
                    ok = np.isfinite(power[b, trials]) & np.isfinite(speed[b, trials])
                    if ok.sum() < MIN_TRIALS:
                        continue
                    pc = est.equipopulated_bins(power[b, trials][ok], N_POWER_BINS)
                    sc = est.equipopulated_bins(speed[b, trials][ok], N_SPEED_BINS)
                    m = est.mi_codes_vs_variants(pc, variants_for(sc, rng),
                                                 N_POWER_BINS, N_SPEED_BINS)
                    vals.append(m[0] - m[1:].mean())
                if vals:
                    speed_rows.append({**base, "area": area, "band": band,
                                       "band_status": tag, "epoch": epoch,
                                       "n_trials": int(trials.size),
                                       "n_bins": len(vals),
                                       "mi_speed": float(np.mean(vals))})

                # --- features, raw and conditioned on speed ------------------
                for fi, fname in enumerate(names):
                    fv = feats[trials, fi]
                    keep = np.isfinite(fv)
                    use = trials[keep]
                    if use.size < MIN_TRIALS or np.unique(fv[keep]).size < N_FEATURE_BINS:
                        continue
                    # The split must fall at a REAL value change. An
                    # equipopulated split breaks ties by array position, and the
                    # array is ordered by trial, so a near-constant feature turns
                    # silently into trial order. `success` is 97.8% tied and
                    # `n_licks` 47%; both produced perfectly balanced splits that
                    # encoded drift (2026-09-18).
                    codes = est.value_boundary_split(fv[keep])
                    if codes is None:
                        continue
                    variants = variants_for(codes, rng)
                    mis, cmis = [], []
                    for w in starts:
                        got = pooled_codes(power, speed, use,
                                           range(w, w + POOL_BINS), variants)
                        if got is None:
                            continue
                        X, Z, V = got
                        m = est.mi_codes_vs_variants(X, V, N_POWER_BINS, N_FEATURE_BINS)
                        c = est.cmi_codes_vs_variants(X, V, Z, N_POWER_BINS,
                                                      N_FEATURE_BINS)
                        mis.append(m[0] - m[1:].mean())
                        cmis.append(c[0] - c[1:].mean())
                    if mis:
                        feat_rows.append({**base, "area": area, "band": band,
                                          "band_status": tag, "epoch": epoch,
                                          "feature": fname,
                                          "n_trials": int(use.size),
                                          "n_windows": len(mis),
                                          "mi": float(np.mean(mis)),
                                          "cmi_given_speed": float(np.mean(cmis))})

    print(f"[lfpmi] {cohort[:4]:<4} {mouse:>5} {probe:<9}: {len(areas)} areas "
          f"{sorted(areas)}, lp={lp} ({lp_source}), {len(speed_rows):3d} speed rows, "
          f"{len(feat_rows):5d} feature rows ({time.time() - t0:.0f}s)", flush=True)
    return speed_rows, feat_rows


def write(rows, path: Path) -> None:
    if not rows:
        print(f"[lfpmi] nothing to write to {path.name}")
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[lfpmi] wrote {path.name} ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="task", choices=("task", "control"))
    # For the audit figure only. Without the clip, "late" trials are largely past
    # the disengagement point (9 of 13 task animals, entirely), so the contrast
    # measures whether the animal is still doing the task. Never for inference.
    ap.add_argument("--no-dp-clip", action="store_true",
                    help="AUDIT ONLY: do not clip at the disengagement point")
    args = ap.parse_args()
    files = sorted((LFP_RESULTS / f"lfp_band_trials_{args.cohort}").glob("*.npz"))
    if not files:
        print(f"[lfpmi] no band cubes for cohort={args.cohort}")
        return
    print(f"[lfpmi] cohort={args.cohort}: {len(files)} animal-probes, "
          f"{N_POWER_BINS} power bins x {N_FEATURE_BINS} feature bins, "
          f"pool {POOL_BINS} spatial bins, {N_SHUFFLES} shuffles")
    speed_rows, feat_rows = [], []
    for k, f in enumerate(files):
        a, b = run_one(f, args.cohort, seed=2000 + k, dp_clip=not args.no_dp_clip)
        speed_rows += a
        feat_rows += b
    tag = "_NODPCLIP" if args.no_dp_clip else ""
    write(speed_rows, RESULTS / f"lfp_mi_speed_{args.cohort}{tag}.csv")
    write(feat_rows, RESULTS / f"lfp_mi_features_{args.cohort}{tag}.csv")


if __name__ == "__main__":
    main()
