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
from striatum_lfp.analysis import log_power  # noqa: E402
from striatum_lfp.decode import ridge_cv_decode  # noqa: E402


MIN_SITES = config.DEFAULT.min_sites          # 5 channels per area
N_CCA_SHUFFLES = 20
BIN_CM = analysis.BIN_SIZE_CM


def analyse_one(item) -> dict[str, list[dict]]:
    path_str, cohort_name = item
    ch = config.get_cohort(cohort_name)
    t0 = time.time()
    z = np.load(path_str, allow_pickle=False)
    mouse = int(z["mouse_id"])
    probe = str(z["probe"])
    band_names = [str(b) for b in z["bands"]]
    corridor = z["corridor"].astype(np.float64)      # (band, ch, 50, trial)
    dark = z["dark"].astype(np.float64)
    n_stored = corridor.shape[3]

    lp = analysis.cohort_learning_points(ch).get(mouse)
    n_trials_matlab = min(analysis.cohort_trial_counts(ch).get(mouse, n_stored), n_stored)
    epochs = analysis.epoch_indices(lp, n_trials_matlab, naive_split=analysis.NAIVE_SPLIT)
    speed = analysis.bin_speed_cm_s(z["corridor_bin_start_ms"], z["corridor_bin_stop_ms"])

    areas = {a: z[f"is_{a.lower()}"] for a in config.AREAS}
    areas = {a: m for a, m in areas.items() if m.sum() >= MIN_SITES}
    depths = z["channel_depth_um"]
    centre = {a: float(np.median(depths[m])) for a, m in areas.items()}

    total_idx = band_names.index("total")
    rows = {"evolution": [], "decoding": [], "reliability": [], "cca": [],
            "moving_reliability": [], "moving_reliability_epochs": [], "behaviour": []}
    # The single-unit stability figures roll the same moving metric up over the
    # THREE-window epoch convention (IntegratedAll_v1.m:554 "the neural analyses
    # use the three-window convention"), not the four-window one the corridor-vs-
    # dark figures use. Match it, so figures/stability_by_animal.csv and the LFP
    # table are the same statistic on the same windows and can sit side by side.
    epochs3 = analysis.epoch_indices(lp, n_trials_matlab)
    EPOCH3_NAMES = ("Naive", "Intermediate", "Expert")
    base = {"cohort": cohort_name, "mouse_id": mouse, "probe": probe,
            "learning_point": lp, "n_trials": n_trials_matlab}

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

            # --- 3b. moving-window reliability -------------------------------
            # The project's own trial-to-trial stability metric, on the project's
            # own window: mean pairwise correlation of the spatial profiles in a
            # 5-trial window centred on each trial and clipped at the edges
            # (IntegratedAll_v1.m:565-630 via batch_triu_corr_mean.m).
            #
            # ONE DELIBERATE DIFFERENCE. MATLAB substitutes 0 for a missing bin
            # before z-scoring, because 0 Hz is a meaningful firing rate. Log
            # power has no zero, so the NaN is left to reach the z-score step
            # inside batch_triu_corr_mean, where it becomes that trial's own mean
            # -- the neutral fill. It affects 0.2-3% of cells.
            keep = np.arange(min(n_trials_matlab, n_stored))
            cube_all = zc[:, :, keep]
            if cube_all.shape[2] >= 2:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    moving = arms.moving_window_reliability(cube_all)
                    shuffled = arms.moving_window_reliability(
                        arms.shuffle_trials(cube_all, np.random.default_rng(0)))
                for ei, ename in enumerate(EPOCH3_NAMES):
                    idx = epochs3[ei] - 1
                    idx = idx[(idx >= 0) & (idx < cube_all.shape[2])]
                    if idx.size == 0:
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        # Mean over channels then over the epoch's trials, as
                        # IntegratedAll_v1.m:648 averages over units.
                        obs = float(np.nanmean(np.nanmean(moving[:, idx], axis=0)))
                        shf = float(np.nanmean(np.nanmean(shuffled[:, idx], axis=0)))
                    rows["moving_reliability_epochs"].append({
                        "group": f"{cohort_name.title()} (LFP)",
                        "area": area, "band": band,
                        "epoch": ename, "animal": mouse, "probe": probe,
                        "n_channels": n_ch, "n_epoch_trials": int(idx.size),
                        "reliability": obs, "shuffle": shf,
                        "obs_minus_shuffle": obs - shf,
                    })
                for ti in range(cube_all.shape[2]):
                    rows["moving_reliability"].append({
                        **base, "area": area, "band": band,
                        "trial": ti + 1,
                        "trial_rel_lp": (ti + 1 - lp) if lp else "",
                        "n_channels": n_ch,
                        "reliability": float(np.nanmedian(moving[:, ti])),
                        "reliability_shuffled": float(np.nanmedian(shuffled[:, ti])),
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
                    pair = arms.batch_triu_corr_mean(sub)
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
                    "mean_pairwise_r": float(np.nanmedian(pair)),   # batch_triu_corr_mean
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

    # --- 0. behaviour: how stereotyped is the traversal itself? --------------
    # Load-bearing, not decorative. The LFP spatial profile largely tracks the
    # speed profile (population-profile r ~ -0.9 for beta), so a group difference
    # in LFP spatial reliability is only a neural claim if the two groups run the
    # corridor the same way. Recorded once per animal, from the striatum probe,
    # since both probes share one behavioural record.
    if probe == "striatum":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            tr_all = np.arange(n_trials_matlab)[z["good_trials"][:n_trials_matlab]]
            sp = speed[:, tr_all]
            rows["behaviour"].append({
                **base, "n_trials_used": int(tr_all.size),
                "speed_profile_split_half_r": float(
                    arms.split_half_reliability(sp[None, :, :])[0]),
                "mean_speed_cm_s": float(np.nanmean(sp)),
                "speed_bin_cv": float(np.nanmean(np.nanstd(sp, axis=1)
                                                 / np.nanmean(sp, axis=1))),
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

    print(f"[arms] {cohort_name[:4]:<4} {mouse}/{probe:9s} lp={lp} areas={sorted(areas)} "
          f"{len(rows['evolution'])}+{len(rows['decoding'])}+{len(rows['reliability'])}"
          f"+{len(rows['cca'])}+{len(rows['moving_reliability'])}"
          f"+{len(rows['moving_reliability_epochs'])}+{len(rows['behaviour'])} rows  "
          f"{time.time() - t0:5.0f}s", flush=True)
    return rows


def write_evolution_stats(rows: list[dict], cohort_name: str = "task") -> None:
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
            c["cohort"] = cohort_name
            c["p_fdr"] = float(a)
            c["survives_fdr"] = bool(r)
            c["family_size"] = len(cells)
        out_rows += cells

    out = config.RESULTS_DIR / f"lfp_arms_evolution_stats_{cohort_name}.csv"
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
    parser.add_argument("--cohort", type=str, default="task",
                        choices=sorted(config.COHORTS))
    args = parser.parse_args()
    in_dir = config.RESULTS_DIR / f"lfp_band_trials_{args.cohort}"
    items = [(str(p), args.cohort) for p in sorted(in_dir.glob("*.npz"))]
    print(f"[arms] cohort={args.cohort}: {len(items)} band-power files", flush=True)

    t0 = time.time()
    with mp.Pool(min(args.jobs, len(items))) as pool:
        results = pool.map(analyse_one, items)
    print(f"[arms] all files in {(time.time() - t0) / 60:.1f} min")

    write_evolution_stats([r for res in results for r in res["evolution"]], args.cohort)

    for key in ("evolution", "decoding", "reliability", "cca", "moving_reliability",
                "moving_reliability_epochs", "behaviour"):
        rows = [r for res in results for r in res[key]]
        if not rows:
            continue
        out = config.RESULTS_DIR / f"lfp_arms_{key}_{args.cohort}.csv"
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[arms] wrote {out.name} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
