"""Four analyses on the binned LFP band power, mirroring the unit pipeline.

1. **Evolution across learning** -- band power per area, per band, in the four
   epoch windows, for corridor and dark. Carries the band/total ratio and the
   per-epoch running speed alongside, because an aperiodic-slope change or a
   speed change would otherwise be read as a band change.
2. **Spatial decoding** -- corridor position from band power, ridge with folds
   split by trial, against a trial-shuffled target null.
3. **Trial-to-trial reliability** -- split-half (interleaved, Spearman-Brown)
   and mean pairwise correlation of each channel's spatial profile.
4. **Cross-area CCA** -- held-out top canonical correlation, bracketed by a
   trial-permutation null and the within-area volume-conduction ceiling.

Run from ``Striatum project/lfp``::

    /opt/anaconda3/bin/python scripts/run_lfp_arms.py [--jobs N]

Writes ``results/lfp_arms_{evolution,decoding,reliability,cca}.csv``.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from striatum_lfp import analysis, arms, config  # noqa: E402
from striatum_lfp.decode import ridge_cv_decode  # noqa: E402

IN_DIR = config.RESULTS_DIR / "lfp_band_trials"
MIN_SITES = config.DEFAULT.min_sites          # 5 channels per area
N_CCA_SHUFFLES = 20
BIN_CM = analysis.BIN_SIZE_CM


def log_power(x: np.ndarray) -> np.ndarray:
    """log10 power, with non-positive cells (empty bins) left as nan.

    Band power is close to lognormal over three orders of magnitude, and the two
    export batches differ ~1000x in absolute power, so every downstream statistic
    works on the log and then z-scores it per channel.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log10(x)
    out[~np.isfinite(out)] = np.nan
    return out


def zscore_channels(x: np.ndarray) -> np.ndarray:
    """Z-score each channel over all its (bin, trial) cells."""
    flat = x.reshape(x.shape[0], -1)
    mean = np.nanmean(flat, axis=1)
    sd = np.nanstd(flat, axis=1)
    sd = np.where(sd > 0, sd, np.nan)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    return (x - mean.reshape(shape)) / sd.reshape(shape)


def analyse_one(path_str: str) -> dict[str, list[dict]]:
    t0 = time.time()
    z = np.load(path_str, allow_pickle=False)
    mouse = int(z["mouse_id"])
    probe = str(z["probe"])
    band_names = [str(b) for b in z["bands"]]
    corridor = z["corridor"].astype(np.float64)      # (band, ch, 50, trial)
    dark = z["dark"].astype(np.float64)
    n_stored = corridor.shape[3]

    lp = analysis.cohort_learning_points().get(mouse)
    n_trials_matlab = min(analysis.cohort_trial_counts().get(mouse, n_stored), n_stored)
    epochs = analysis.epoch_indices(lp, n_trials_matlab, naive_split=analysis.NAIVE_SPLIT)
    speed = analysis.bin_speed_cm_s(z["corridor_bin_start_ms"], z["corridor_bin_stop_ms"])

    areas = {a: z[f"is_{a.lower()}"] for a in config.AREAS}
    areas = {a: m for a, m in areas.items() if m.sum() >= MIN_SITES}
    depths = z["channel_depth_um"]
    centre = {a: float(np.median(depths[m])) for a, m in areas.items()}

    total_idx = band_names.index("total")
    rows = {"evolution": [], "decoding": [], "reliability": [], "cca": []}
    base = {"mouse_id": mouse, "probe": probe, "learning_point": lp,
            "n_trials": n_trials_matlab}

    # Precompute the z-scored log cubes once per (band, area).
    cubes: dict[tuple[str, str], np.ndarray] = {}
    for area, mask in areas.items():
        for bi, band in enumerate(band_names):
            zc, _ = analysis.joint_zscore(log_power(corridor[bi, mask]),
                                          log_power(dark[bi, mask]))
            cubes[(band, area)] = zc

    for area, mask in areas.items():
        n_ch = int(mask.sum())
        for bi, band in enumerate(band_names):
            cor_lin = corridor[bi, mask]
            dark_lin = dark[bi, mask]
            ratio_cor = cor_lin / corridor[total_idx, mask]
            ratio_dark = dark_lin / dark[total_idx, mask]
            zc, zd = analysis.joint_zscore(log_power(cor_lin), log_power(dark_lin))
            # Running speed rises ~34% from the first trials to expert, and band
            # power tracks speed, so the epoch effect is reported twice: raw, and
            # with the linear speed component removed per channel.
            with np.errstate(divide="ignore", invalid="ignore"):
                log_speed = np.log10(speed)
            log_speed[~np.isfinite(log_speed)] = np.nan
            zc_resid = arms.residualise_on(zc, log_speed)

            # --- 1. evolution ------------------------------------------------
            for ei, name in enumerate(analysis.EPOCH_NAMES):
                tr = epochs[ei] - 1
                tr = tr[(tr >= 0) & (tr < n_stored)]
                if tr.size == 0:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    rows["evolution"].append({
                        **base, "area": area, "band": band, "epoch": name,
                        "n_epoch_trials": int(tr.size), "n_channels": n_ch,
                        "z_corridor": float(np.nanmedian(np.nanmean(zc[:, :, tr], axis=(1, 2)))),
                        "z_dark": float(np.nanmedian(np.nanmean(zd[:, :, tr], axis=(1, 2)))),
                        "z_corridor_speed_resid": float(np.nanmedian(
                            np.nanmean(zc_resid[:, :, tr], axis=(1, 2)))),
                        "log_corridor": float(np.nanmedian(
                            np.nanmean(log_power(cor_lin)[:, :, tr], axis=(1, 2)))),
                        "log_dark": float(np.nanmedian(
                            np.nanmean(log_power(dark_lin)[:, :, tr], axis=(1, 2)))),
                        "frac_of_total_corridor": float(np.nanmedian(
                            np.nanmean(ratio_cor[:, :, tr], axis=(1, 2)))),
                        "frac_of_total_dark": float(np.nanmedian(
                            np.nanmean(ratio_dark[:, :, tr], axis=(1, 2)))),
                        "mean_speed_cm_s": float(np.nanmean(speed[:, tr])),
                    })

            # --- 2. decoding + 3. reliability --------------------------------
            windows = [("All", np.arange(n_trials_matlab))] + [
                (name, epochs[ei]) for ei, name in enumerate(analysis.EPOCH_NAMES)
            ]
            for name, trials in windows:
                tr = np.asarray(trials, int) - 1 if name != "All" else np.asarray(trials, int)
                tr = tr[(tr >= 0) & (tr < n_stored)]
                if tr.size < 4:
                    continue
                sub = zc[:, :, tr]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    half = arms.split_half_reliability(sub)
                    pair = arms.mean_pairwise_trial_r(sub)
                # A reliable SPATIAL profile is not automatically position coding:
                # if band power tracks running speed, and the animal is reliably
                # slow at the same places, the profile is a speed profile. Report
                # the correlation so the reader can tell the two apart.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    speed_profile = np.nanmean(speed[:, tr], axis=1)
                    power_profile = np.nanmean(log_power(cor_lin)[:, :, tr], axis=2)
                ok = np.isfinite(speed_profile)
                r_speed = [np.corrcoef(pp[ok], speed_profile[ok])[0, 1]
                           for pp in power_profile
                           if np.isfinite(pp[ok]).all() and np.nanstd(pp[ok]) > 0]
                rows["reliability"].append({
                    **base, "area": area, "band": band, "window": name,
                    "n_window_trials": int(tr.size), "n_channels": n_ch,
                    "r_profile_vs_speed": float(np.median(r_speed)) if r_speed else np.nan,
                    "split_half_r": float(np.nanmedian(half)),
                    "split_half_r_p25": float(np.nanpercentile(half, 25)),
                    "split_half_r_p75": float(np.nanpercentile(half, 75)),
                    "mean_pairwise_r": float(np.nanmedian(pair)),
                })

                X, y, groups = arms.design_matrix(sub, trials=np.arange(tr.size))
                if X.shape[0] < 40 or np.unique(groups).size < 5:
                    continue
                r2, mae, _ = ridge_cv_decode(X, y.astype(float), groups)
                rng = np.random.default_rng(0)
                # Null: rotate the position labels within each trial. A trial
                # PERMUTATION does nothing here -- every trial carries the same
                # 0..49 sequence, so it leaves y bit-identical and silently
                # re-runs the real decoder (the defect this replaces).
                null_r2 = [
                    ridge_cv_decode(X, arms.circular_shift_targets(y, groups, rng).astype(float),
                                    groups)[0]
                    for _ in range(5)
                ]
                rows["decoding"].append({
                    **base, "area": area, "band": band, "window": name,
                    "n_window_trials": int(tr.size), "n_channels": n_ch,
                    "n_samples": int(X.shape[0]),
                    "r2": r2, "mae_bins": mae, "mae_cm": mae * BIN_CM,
                    "null_r2_median": float(np.nanmedian(null_r2)),
                    "chance_mae_bins": float(np.mean(np.abs(y - np.mean(y)))),
                })

    # --- 4. cross-area CCA ---------------------------------------------------
    for band in band_names:
        for a, b in itertools.combinations(sorted(areas), 2):
            Xa, _, ga = arms.design_matrix(cubes[(band, a)][:, :, :n_trials_matlab])
            Xb, _, gb = arms.design_matrix(cubes[(band, b)][:, :, :n_trials_matlab])
            n = min(Xa.shape[0], Xb.shape[0])
            if n < 100 or not np.array_equal(ga[:n], gb[:n]):
                continue
            Xa, Xb, g = Xa[:n], Xb[:n], ga[:n]
            real = arms.heldout_cca_grouped(Xa, Xb, g)
            null = arms.trial_shuffle_cca_null(Xa, Xb, g, n_shuffles=N_CCA_SHUFFLES)
            rows["cca"].append({
                **base, "band": band, "area_a": a, "area_b": b,
                "n_ch_a": Xa.shape[1], "n_ch_b": Xb.shape[1], "n_samples": n,
                # Distance between the two areas' channel centres along the shank.
                # If coupling is a shared field rather than area-specific, it should
                # fall off with this and with nothing else.
                "separation_um": abs(centre[a] - centre[b]),
                "heldout_cc1": real,
                "null_median": float(np.nanmedian(null)),
                "null_p95": float(np.nanpercentile(null, 95)),
                "ceiling_a": arms.within_area_ceiling(Xa, g),
                "ceiling_b": arms.within_area_ceiling(Xb, g),
            })

    print(f"[arms] {mouse}/{probe:9s} lp={lp} areas={sorted(areas)} "
          f"{len(rows['evolution'])}+{len(rows['decoding'])}+{len(rows['reliability'])}"
          f"+{len(rows['cca'])} rows  {time.time() - t0:5.0f}s", flush=True)
    return rows


def write_evolution_stats(rows: list[dict]) -> None:
    """Paired naive-to-expert test per area x band, BH-corrected over that family.

    The family is declared here and nowhere else: area x band, one test each,
    animals as n. Everything else in the evolution CSV is a sensitivity check
    and carries no stars. Each metric is corrected within its own family, since
    the raw z-power and the speed-residualised version answer different questions.
    """
    from scipy import stats

    out_rows = []
    for metric, naive_epoch in (("z_corridor", "Trials 4-10"),
                                ("z_corridor_speed_resid", "Trials 4-10"),
                                ("frac_of_total_corridor", "Trials 4-10")):
        cells = []
        for area in config.AREAS:
            for band in [b for b in bandpower_bands(rows) if b != "total"]:
                naive, expert = {}, {}
                for r in rows:
                    if r["area"] != area or r["band"] != band:
                        continue
                    if r["epoch"] == naive_epoch and np.isfinite(r[metric]):
                        naive[int(r["mouse_id"])] = r[metric]
                    if r["epoch"] == "Expert" and np.isfinite(r[metric]):
                        expert[int(r["mouse_id"])] = r[metric]
                common = sorted(set(naive) & set(expert))
                if len(common) < 3:
                    continue
                delta = np.array([expert[m] - naive[m] for m in common])
                t, p = stats.ttest_1samp(delta, 0.0)
                cells.append({"metric": metric, "area": area, "band": band,
                              "n_animals": len(common), "mean_delta": float(delta.mean()),
                              "sem_delta": float(delta.std(ddof=1) / np.sqrt(len(common))),
                              "t": float(t), "p_raw": float(p)})
        if not cells:
            continue
        adjusted, reject = arms.fdr_bh(np.array([c["p_raw"] for c in cells]), q=0.05)
        for c, a, r in zip(cells, adjusted, reject):
            c["p_fdr"] = float(a)
            c["survives_fdr"] = bool(r)
            c["family_size"] = len(cells)
        out_rows += cells

    out = config.RESULTS_DIR / "lfp_arms_evolution_stats.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    n_sig = sum(r["survives_fdr"] for r in out_rows)
    print(f"[arms] wrote {out.name}: {n_sig}/{len(out_rows)} cells survive BH-FDR at q=0.05")


def bandpower_bands(rows: list[dict]) -> list[str]:
    seen, order = set(), []
    for r in rows:
        if r["band"] not in seen:
            seen.add(r["band"])
            order.append(r["band"])
    return order


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    paths = sorted(str(p) for p in IN_DIR.glob("*.npz"))
    print(f"[arms] {len(paths)} band-power files", flush=True)

    t0 = time.time()
    with mp.Pool(min(args.jobs, len(paths))) as pool:
        results = pool.map(analyse_one, paths)
    print(f"[arms] all files in {(time.time() - t0) / 60:.1f} min")

    write_evolution_stats([r for res in results for r in res["evolution"]])

    for key in ("evolution", "decoding", "reliability", "cca"):
        rows = [r for res in results for r in res[key]]
        if not rows:
            continue
        out = config.RESULTS_DIR / f"lfp_arms_{key}.csv"
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[arms] wrote {out.name} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
