"""Four analyses on the binned LFP band power, mirroring the unit pipeline.

1. **Evolution across learning** -- band power per area, per band, in the three
   epoch windows (``trials.EPOCHS``), for corridor and dark. Carries the band/total ratio and the
   per-epoch running speed alongside, because an aperiodic-slope change or a
   speed change would otherwise be read as a band change.
2. **Spatial decoding** -- corridor position from band power, ridge with folds
   split by trial, against a trial-shuffled target null.
3. **Trial-to-trial reliability** -- split-half (interleaved, Spearman-Brown)
   and mean pairwise correlation of each channel's spatial profile.
4. **Cross-area CCA** -- held-out top canonical correlation against a
   trial-permutation null (shared field and communication are not separable
   here; see the distance control).

Run from ``Striatum project/lfp``::

    /opt/anaconda3/bin/python scripts/run_lfp_arms.py [--jobs N]

Writes ``results/lfp_arms_{evolution,decoding,reliability,cca}.csv``.

Every window -- "All", "First 20" and the three epochs -- comes from
``trials.SessionTrials``: good, engaged (<= DP) and covered trials only.
"""

from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from striatum_lfp import analysis, arms, config, results_io, stats, trials  # noqa: E402
from striatum_lfp.analysis import log_power  # noqa: E402

# Length of the unaligned early-session window (see `windows` in run_one).
FIRST_N_TRIALS = 20


MIN_SITES = config.DEFAULT.min_sites          # 5 channels per area
DEPTH_PANEL_BAND = "low_gamma"               # channel x trial reliability image
DEPTH_PANEL_TRIALS = 100


def write_depth_panel(z, usable, lp, cohort_name: str) -> None:
    """Channel x trial moving reliability for one file, for the depth figure.

    The LFP analogue of the neurons x trials ``imagesc(avg_corrs)`` panel in
    ProcessStriatumTask.m:997. Computed here, not in the plotting script, and
    saved as results/lfp_arms_moving_depth_<cohort>/<mouse>_<probe>.npz.
    """
    bands = [str(b) for b in z["bands"]]
    keep = usable[:DEPTH_PANEL_TRIALS]              # x axis = good-trial number
    cube = log_power(z["corridor"][bands.index(DEPTH_PANEL_BAND)][:, :, keep]
                     .astype(np.float64))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rel = arms.moving_window_reliability(cube)
    out = config.RESULTS_DIR / f"lfp_arms_moving_depth_{cohort_name}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"{int(z['mouse_id'])}_{z['probe']}.npz", reliability=rel,
        channel_depth_um=z["channel_depth_um"], band=DEPTH_PANEL_BAND,
        learning_point=-1 if lp is None else lp, n_trials=keep.size,
        **{f"is_{a.lower()}": z[f"is_{a.lower()}"] for a in config.AREAS})
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

    session = trials.sessions_for(ch.name)[mouse].with_data(z["good_trials"])
    usable = session.usable()
    epochs = session.epochs()
    lp = session.lp
    speed = analysis.bin_speed_cm_s(z["corridor_bin_start_ms"], z["corridor_bin_stop_ms"])

    areas = {a: z[f"is_{a.lower()}"] for a in config.AREAS}
    areas = {a: m for a, m in areas.items() if m.sum() >= MIN_SITES}
    depths = z["channel_depth_um"]
    centre = {a: float(np.median(depths[m])) for a, m in areas.items()}

    total_idx = band_names.index("total")
    rows = {"evolution": [], "decoding": [], "reliability": [], "cca": [],
            "moving_reliability": [], "moving_reliability_epochs": [], "behaviour": []}
    # "measured" or "cohort_average": for a borrowed learning point the late
    # window is a matched TIME window, not a matched level of performance, and
    # anything quoting an epoch result should be able to say so.
    base = {"cohort": cohort_name, "mouse_id": mouse, "probe": probe,
            "learning_point": lp, "lp_source": session.lp_source,
            "disengagement_point": session.dp, "n_trials": int(usable.size)}

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
            for name, tr in epochs.items():
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
                        # log of the MEAN linear power, not the mean of per-bin
                        # logs: a short bin's log is biased down by an amount
                        # that depends on its duration, i.e. on running speed
                        # (-0.21 log10 at 110 ms vs -0.17 at 200 ms for theta).
                        "log_corridor": float(np.nanmedian(
                            log_power(np.nanmean(cor_lin[:, :, tr], axis=(1, 2))))),
                        "log_dark": float(np.nanmedian(
                            log_power(np.nanmean(dark_lin[:, :, tr], axis=(1, 2))))),
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
            #
            # Neighbours are the usable trials in order, so a non-good or
            # disengaged trial never sits inside anyone's window; position k in
            # that sequence is good-trial number k + 1.
            cube_all = zc[:, :, usable]
            if cube_all.shape[2] >= 2:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    moving = arms.moving_window_reliability(cube_all)
                    shuffled = arms.moving_window_reliability(
                        arms.shuffle_trials(cube_all, np.random.default_rng(0)))
                for ename, raw in epochs.items():
                    if raw.size == 0:
                        continue
                    idx = np.searchsorted(usable, raw)
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
                        "raw_trial": int(usable[ti]) + 1,
                        "trial_rel_lp": (ti + 1 - lp) if lp else "",
                        "n_channels": n_ch,
                        "reliability": float(np.nanmedian(moving[:, ti])),
                        "reliability_shuffled": float(np.nanmedian(shuffled[:, ti])),
                    })

            # --- 2. decoding + 3. reliability --------------------------------
            # All windows are raw 0-based trial indices from the trial layer.
            # "First 20" is deliberately NOT learning-point aligned: it is the first
            # 20 usable trials of the session for every animal, so task and yoked
            # control are compared over the same stretch of exposure rather than
            # over windows defined by a learning point the controls do not have.
            windows = [("All", usable), ("First 20", usable[:FIRST_N_TRIALS]),
                       *epochs.items()]
            for name, tr in windows:
                if tr.size < 4:
                    continue
                sub = zc[:, :, tr]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    half = arms.split_half_reliability(sub)
                    # Same statistic after removing each channel's linear speed
                    # component: if a group gap in reliability is behavioural
                    # (stereotyped running), it should shrink here.
                    half_resid = arms.split_half_reliability(zc_resid[:, :, tr])
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
                    "split_half_r_speed_resid": float(np.nanmedian(half_resid)),
                    "mean_pairwise_r": float(np.nanmedian(pair)),   # batch_triu_corr_mean
                })

                X, y, groups = arms.design_matrix(sub, trials=np.arange(tr.size))
                if X.shape[0] < 40 or np.unique(groups).size < 5:
                    continue
                r2, mae, _ = arms.ridge_cv_decode(X, y.astype(float), groups)
                rng = np.random.default_rng(0)
                # Null: rotate the position labels within each trial. A trial
                # PERMUTATION does nothing here -- every trial carries the same
                # 0..49 sequence, so it leaves y bit-identical and silently
                # re-runs the real decoder (the defect this replaces).
                null_r2 = [
                    arms.ridge_cv_decode(X, arms.circular_shift_targets(y, groups, rng).astype(float),
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

    write_depth_panel(z, usable, lp, cohort_name)

    # --- 0. behaviour: how stereotyped is the traversal itself? --------------
    # Load-bearing, not decorative. The LFP spatial profile largely tracks the
    # speed profile (population-profile r ~ -0.9 for beta), so a group difference
    # in LFP spatial reliability is only a neural claim if the two groups run the
    # corridor the same way. Recorded once per animal, from the striatum probe,
    # since both probes share one behavioural record.
    if probe == "striatum":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sp = speed[:, usable]
            rows["behaviour"].append({
                **base, "n_trials_used": int(usable.size),
                "speed_profile_split_half_r": float(
                    arms.split_half_reliability(sp[None, :, :])[0]),
                "mean_speed_cm_s": float(np.nanmean(sp)),
                "speed_bin_cv": float(np.nanmean(np.nanstd(sp, axis=1)
                                                 / np.nanmean(sp, axis=1))),
            })

    # --- 4. cross-area CCA ---------------------------------------------------
    for band in band_names:
        for a, b in itertools.combinations(sorted(areas), 2):
            cube_a, cube_b = cubes[(band, a)], cubes[(band, b)]
            Xa, Xb, g = arms.paired_design(cube_a, cube_b, usable)
            n = Xa.shape[0]
            if n < 100:
                continue
            real = arms.heldout_cca_grouped(Xa, Xb, g)
            null = arms.trial_shuffle_cca_null(cube_a, cube_b, usable,
                                               n_shuffles=N_CCA_SHUFFLES)
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
            })

    print(f"[arms] {cohort_name[:4]:<4} {mouse}/{probe:9s} lp={lp} areas={sorted(areas)} "
          f"{len(rows['evolution'])}+{len(rows['decoding'])}+{len(rows['reliability'])}"
          f"+{len(rows['cca'])}+{len(rows['moving_reliability'])}"
          f"+{len(rows['moving_reliability_epochs'])}+{len(rows['behaviour'])} rows  "
          f"{time.time() - t0:5.0f}s", flush=True)
    return rows


EVOLUTION_METRICS = ("log_corridor", "z_corridor", "z_corridor_speed_resid",
                     "frac_of_total_corridor")


def _write(rows: list[dict], name: str) -> None:
    results_io.write_rows(rows, config.RESULTS_DIR / name)


def _test_cells(cells: list[dict]) -> list[dict]:
    """Exact sign-flip per cell (animals as n), then BH over the cells given."""
    for c in cells:
        c["p_raw"] = stats.sign_flip_test(c.pop("values"))
        c["p_floor"] = stats.sign_flip_floor(c["n_animals"])
        c["reachable"] = stats.can_reach(c["p_floor"])
    adjusted, reject = stats.fdr_bh(np.array([c["p_raw"] for c in cells]), q=0.05)
    for c, a, r in zip(cells, adjusted, reject):
        c["p_fdr"] = float(a)
        c["survives_fdr"] = bool(r)
        c["family_size"] = len(cells)
    return cells


def write_evolution_stats(rows: list[dict], cohort_name: str = "task") -> None:
    """Paired Naive-to-Expert test per area x band, BH-corrected within each metric.

    The family is declared here and nowhere else: area x band (``total``
    included), one exact sign-flip test each, animals as n. ``log_corridor`` is
    the primary metric; the others are sensitivity checks.
    """
    out_rows = []
    for metric in EVOLUTION_METRICS:
        cells = []
        for area in config.AREAS:
            for band in bandpower_bands(rows):
                naive, expert = {}, {}
                for r in rows:
                    if r["area"] != area or r["band"] != band or not np.isfinite(r[metric]):
                        continue
                    if r["epoch"] == trials.EPOCHS[0]:
                        naive[int(r["mouse_id"])] = r[metric]
                    elif r["epoch"] == trials.EPOCHS[-1]:
                        expert[int(r["mouse_id"])] = r[metric]
                common = sorted(set(naive) & set(expert))
                if not common:
                    continue
                delta = np.array([expert[m] - naive[m] for m in common])
                cells.append({"cohort": cohort_name, "metric": metric, "area": area,
                              "band": band, "n_animals": len(common),
                              "mean_delta": float(delta.mean()),
                              "sem_delta": float(delta.std(ddof=1) / np.sqrt(len(common)))
                              if len(common) > 1 else np.nan,
                              "values": delta})
        if cells:
            out_rows += _test_cells(cells)
    _write(out_rows, f"lfp_arms_evolution_stats_{cohort_name}.csv")
    n_sig = sum(r["survives_fdr"] for r in out_rows if r["metric"] == EVOLUTION_METRICS[0])
    print(f"[arms] evolution: {n_sig} {EVOLUTION_METRICS[0]} cells survive BH-FDR at q=0.05")


def write_decoding_stats(rows: list[dict], cohort_name: str = "task") -> None:
    """Does band power decode position beyond the rotated-label null?

    Per animal, held-out R2 minus the median of its own null, on the "All"
    window; exact sign-flip across animals; BH over area x band. This is the
    test the "beats its null in every cell" figure title used to state without
    anything computing it.
    """
    cells = []
    for area in config.AREAS:
        for band in bandpower_bands(rows):
            sel = [r for r in rows if r["area"] == area and r["band"] == band
                   and r["window"] == "All"]
            vals = np.array([r["r2"] - r["null_r2_median"] for r in sel], float)
            vals = vals[np.isfinite(vals)]
            if not vals.size:
                continue
            cells.append({"cohort": cohort_name, "metric": "r2_minus_null", "area": area,
                          "band": band, "n_animals": int(vals.size),
                          "mean_delta": float(vals.mean()),
                          "sem_delta": float(vals.std(ddof=1) / np.sqrt(vals.size))
                          if vals.size > 1 else np.nan,
                          "values": vals})
    if not cells:
        return
    out_rows = _test_cells(cells)
    _write(out_rows, f"lfp_arms_decoding_stats_{cohort_name}.csv")
    print(f"[arms] decoding: {sum(r['survives_fdr'] for r in out_rows)}/{len(out_rows)} "
          f"area x band cells beat their null (BH q=0.05)")


def write_cca_distance_stats(rows: list[dict], cohort_name: str = "task") -> None:
    """Does CC1 fall with separation WITHIN an animal? Slope per probe, then animal.

    Pooling every area pair of every animal into one Spearman treats three
    pairs of one probe as independent and lets separation and pair identity
    stand in for each other. Here each probe gives one least-squares slope of
    CC1 on separation (mm) over its area pairs, an animal's probes are averaged,
    and the slopes are tested across animals (exact sign-flip, BH over bands).
    """
    cells = []
    for band in bandpower_bands(rows):
        slopes: dict[int, list[float]] = {}
        for (mouse, probe) in {(int(r["mouse_id"]), r["probe"]) for r in rows}:
            sel = [r for r in rows if r["band"] == band and int(r["mouse_id"]) == mouse
                   and r["probe"] == probe and np.isfinite(r["heldout_cc1"])]
            sep = np.array([r["separation_um"] for r in sel], float) / 1000.0
            if sep.size < 3 or np.unique(sep).size < 2:
                continue
            cc = np.array([r["heldout_cc1"] for r in sel], float)
            slopes.setdefault(mouse, []).append(float(np.polyfit(sep, cc, 1)[0]))
        vals = np.array([np.mean(v) for v in slopes.values()])
        if not vals.size:
            continue
        cells.append({"cohort": cohort_name, "metric": "cc1_slope_per_mm", "area": "all",
                      "band": band, "n_animals": int(vals.size),
                      "mean_delta": float(vals.mean()),
                      "sem_delta": float(vals.std(ddof=1) / np.sqrt(vals.size))
                      if vals.size > 1 else np.nan,
                      "values": vals})
    if cells:
        _write(_test_cells(cells), f"lfp_arms_cca_distance_stats_{cohort_name}.csv")


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
    config.add_cohort_argument(parser)
    args = parser.parse_args()
    in_dir = config.RESULTS_DIR / f"lfp_band_trials_{args.cohort}"
    items = [(str(p), args.cohort) for p in sorted(in_dir.glob("*.npz"))]
    print(f"[arms] cohort={args.cohort}: {len(items)} band-power files", flush=True)

    t0 = time.time()
    with mp.Pool(min(args.jobs, len(items))) as pool:
        results = pool.map(analyse_one, items)
    print(f"[arms] all files in {(time.time() - t0) / 60:.1f} min")

    write_evolution_stats([r for res in results for r in res["evolution"]], args.cohort)
    write_decoding_stats([r for res in results for r in res["decoding"]], args.cohort)
    write_cca_distance_stats([r for res in results for r in res["cca"]], args.cohort)

    for key in ("evolution", "decoding", "reliability", "cca", "moving_reliability",
                "moving_reliability_epochs", "behaviour"):
        rows = [r for res in results for r in res[key]]
        if not rows:
            continue
        _write(rows, f"lfp_arms_{key}_{args.cohort}.csv")


if __name__ == "__main__":
    main()
