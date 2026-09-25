"""Figures for the four LFP band-power arms.

Hierarchical throughout: the animal is the unit of analysis, error bars are the
across-animal SEM, and the animal count is printed on every panel. Channel-level
pooling is not plotted at all -- channels within an area correlate at r = 0.83-0.96
(measured, lfp_inventory.csv), so a channel-level error bar would be a
near-meaningless fraction of the real uncertainty.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from striatum_lfp import config, trials  # noqa: E402
from striatum_lfp.results_io import hierarchical, load_arms as load  # noqa: E402
from striatum_lfp.figstyle import (  # noqa: E402
    AREA_COLOUR, AREA_ORDER, BAND_LABEL, PLOT_BANDS, save_pair,
)
EPOCHS = list(trials.EPOCHS)
WINDOWS = ["All"] + EPOCHS


def _epoch_axis(ax):
    ax.set_xticks(range(len(EPOCHS)))
    ax.set_xticklabels(["Naive", "Inter", "Expert"], fontsize=7)


def load_stats(cohort_name: str = "task"):
    """{(metric, area, band): (mean_delta, p_fdr, survives)} from the declared family."""
    path = config.RESULTS_DIR / f"lfp_arms_evolution_stats_{cohort_name}.csv"
    if not path.exists():
        return {}
    with path.open() as fh:
        return {(r["metric"], r["area"], r["band"]):
                (float(r["mean_delta"]), float(r["p_fdr"]), r["survives_fdr"] == "True")
                for r in csv.DictReader(fh)}


def plot_evolution(rows, value_c, value_d, ylabel, stem, suptitle, stats=None):
    areas = [a for a in AREA_ORDER if any(r["area"] == a for r in rows)]
    fig, axes = plt.subplots(len(PLOT_BANDS), len(areas),
                             figsize=(2.5 * len(areas), 2.3 * len(PLOT_BANDS)),
                             squeeze=False, sharex=True)
    agg_c = hierarchical(rows, ("area", "band", "epoch"), value_c)
    agg_d = hierarchical(rows, ("area", "band", "epoch"), value_d)
    for bi, band in enumerate(PLOT_BANDS):
        for ai, area in enumerate(areas):
            ax = axes[bi][ai]
            for agg, colour, label in ((agg_c, AREA_COLOUR[area], "corridor"),
                                       (agg_d, "0.45", "dark (ITI)")):
                x, m, e, ns = [], [], [], []
                for ei, ep in enumerate(EPOCHS):
                    if (area, band, ep) in agg:
                        mu, sem, n = agg[(area, band, ep)]
                        x.append(ei)
                        m.append(mu)
                        e.append(sem)
                        ns.append(n)
                if x:
                    ax.errorbar(x, m, yerr=e, marker="o", ms=3.5, lw=1.3,
                                capsize=2, color=colour, label=label)
            key = (area, band, EPOCHS[0])
            n_txt = "/".join(str(agg_c[(area, band, ep)][2]) if (area, band, ep) in agg_c
                             else "0" for ep in EPOCHS)
            ax.text(0.02, 0.03, f"N = {n_txt}", transform=ax.transAxes, fontsize=6,
                    color="0.35")
            st = (stats or {}).get((value_c, area, band))
            if st is not None:
                delta, p_fdr, survives = st
                lo, hi = ax.get_ylim()
                ax.set_ylim(lo, hi + 0.42 * (hi - lo))     # headroom for the label
                ax.text(0.97, 0.97, f"Δ={delta:+.3f}  p(FDR)={p_fdr:.3f}"
                        + ("  ✱" if survives else ""),
                        transform=ax.transAxes, fontsize=6, ha="right", va="top",
                        color="#b3001b" if survives else "0.45",
                        fontweight="bold" if survives else "normal")
            ax.axhline(0, color="k", lw=0.5, ls=":")
            _epoch_axis(ax)
            ax.tick_params(labelsize=7)
            if bi == 0:
                ax.set_title(area, fontsize=10, color=AREA_COLOUR[area], fontweight="bold")
            if ai == 0:
                ax.set_ylabel(f"{BAND_LABEL[band]}\n{ylabel}", fontsize=7)
            if bi == 0 and ai == len(areas) - 1:
                ax.legend(fontsize=6, loc="upper right")
            if key not in agg_c:
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center",
                        color="0.7", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("epoch (trials relative to learning point)", fontsize=7)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_pair(fig, stem)


def plot_speed(rows, stem="lfp_evolution_speed"):
    agg = hierarchical(rows, ("area", "epoch"), "mean_speed_cm_s")
    fig, ax = plt.subplots(figsize=(6, 4))
    for area in [a for a in AREA_ORDER if any(k[0] == a for k in agg)]:
        x, m, e = [], [], []
        for ei, ep in enumerate(EPOCHS):
            if (area, ep) in agg:
                mu, sem, _ = agg[(area, ep)]
                x.append(ei)
                m.append(mu)
                e.append(sem)
        ax.errorbar(x, m, yerr=e, marker="o", ms=4, lw=1.4, capsize=2,
                    color=AREA_COLOUR[area], label=area)
    _epoch_axis(ax)
    ax.set_xlabel("epoch (trials relative to learning point)")
    ax.set_ylabel("running speed (cm/s), mean over corridor bins")
    ax.set_title("The covariate: running speed across learning\n"
                 "any band-power change over these epochs has to be read against this",
                 fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    save_pair(fig, stem)


def plot_decoding(rows, stem="lfp_decoding", stats_rows=()):
    areas = [a for a in AREA_ORDER if any(r["area"] == a for r in rows)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    agg = hierarchical([r for r in rows if r["window"] == "All"],
                       ("area", "band"), "r2")
    null = hierarchical([r for r in rows if r["window"] == "All"],
                        ("area", "band"), "null_r2_median")
    width = 0.8 / len(PLOT_BANDS)
    for bi, band in enumerate(PLOT_BANDS):
        x = np.arange(len(areas)) + bi * width - 0.4 + width / 2
        m = [agg.get((a, band), (np.nan,) * 3)[0] for a in areas]
        e = [agg.get((a, band), (np.nan,) * 3)[1] for a in areas]
        ax.bar(x, m, width, yerr=e, capsize=2, label=BAND_LABEL[band])
        nm = [null.get((a, band), (np.nan,) * 3)[0] for a in areas]
        ax.plot(x, nm, "k_", ms=6, mew=1.2)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(areas)))
    ax.set_xticklabels([f"{a}\nN={agg.get((a, 'theta'), (0, 0, 0))[2]}"
                        + ("\n(not tested)" if agg.get((a, 'theta'), (0, 0, 0))[2] < 3 else "")
                        for a in areas], fontsize=8)
    ax.set_ylabel("cross-validated R² for corridor position")
    ax.set_title("(a) Position decoded from band power, folds split by trial\n"
                 "black dashes = trial-shuffled target null", fontsize=9)
    ax.legend(fontsize=7)

    # Panel (b): the actual evidence, one point per animal. Per-EPOCH decoding is
    # not plotted because it is not estimable -- a 10-trial window gives the ridge
    # ~500 samples for 30-140 channels and every animal comes back with a negative
    # R2, i.e. worse than predicting the mean. That is a limit of the window, not
    # a result about learning.
    ax = axes[1]
    sub = [r for r in rows if r["window"] == "All" and r["band"] == "high_gamma"]
    for area in areas:
        pts = [(r["null_r2_median"], r["r2"]) for r in sub if r["area"] == area]
        if pts:
            ax.scatter(*zip(*pts), s=26, alpha=0.85, color=AREA_COLOUR[area], label=area)
    lim = [-0.06, 0.16]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("R² with position labels rotated within each trial (null)")
    ax.set_ylabel("R² with the true position labels")
    ax.set_title("(b) One point per animal, high gamma\n"
                 "points above the dashed line decode better than the null", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")

    n_sig = sum(r["survives_fdr"] == "True" for r in stats_rows)
    verdict = (f"{n_sig}/{len(stats_rows)} area × band cells beat their null "
               "(exact sign-flip across animals, BH q=0.05)" if stats_rows
               else "no decoding stats table — run run_lfp_arms.py")
    fig.suptitle(f"Spatial decoding from LFP band power — {verdict}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_pair(fig, stem)


def plot_reliability(rows, stem="lfp_reliability"):
    areas = [a for a in AREA_ORDER if any(r["area"] == a for r in rows)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    agg = hierarchical([r for r in rows if r["window"] == "All"],
                       ("area", "band"), "split_half_r")
    width = 0.8 / len(PLOT_BANDS)
    for bi, band in enumerate(PLOT_BANDS):
        x = np.arange(len(areas)) + bi * width - 0.4 + width / 2
        m = [agg.get((a, band), (np.nan,) * 3)[0] for a in areas]
        e = [agg.get((a, band), (np.nan,) * 3)[1] for a in areas]
        ax.bar(x, m, width, yerr=e, capsize=2, label=BAND_LABEL[band])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(areas)))
    ax.set_xticklabels([f"{a}\nN={agg.get((a, 'theta'), (0, 0, 0))[2]}" for a in areas],
                       fontsize=8)
    ax.set_ylabel("split-half r of the spatial profile (Spearman-Brown)")
    ax.set_title("(a) Trial-to-trial reliability, all trials\n"
                 "interleaved halves, so session drift is not counted as noise",
                 fontsize=9)
    ax.legend(fontsize=7)

    # Panel (b): the qualifier. A reliable spatial profile is only position coding
    # if it is not simply tracking where the animal runs slowly.
    ax = axes[1]
    agg = hierarchical([r for r in rows if r["window"] == "All"],
                       ("area", "band"), "r_profile_vs_speed")
    for bi, band in enumerate(PLOT_BANDS):
        x = np.arange(len(areas)) + bi * width - 0.4 + width / 2
        m = [agg.get((a, band), (np.nan,) * 3)[0] for a in areas]
        e = [agg.get((a, band), (np.nan,) * 3)[1] for a in areas]
        ax.bar(x, m, width, yerr=e, capsize=2, label=BAND_LABEL[band])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(areas)))
    ax.set_xticklabels(areas, fontsize=8)
    ax.set_ylabel("r between the spatial power profile and the speed profile")
    ax.set_title("(b) Is the spatial profile just a speed profile?\n"
                 "beta says largely yes (r ≈ −0.5 everywhere); theta says no",
                 fontsize=9)
    ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("Reliability of the LFP spatial profile — high, but read (b) before "
                 "calling it position coding", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_pair(fig, stem)


def plot_cca(rows, stem="lfp_cca"):
    pairs = sorted({(r["area_a"], r["area_b"]) for r in rows})
    fig, axes = plt.subplots(1, len(PLOT_BANDS), figsize=(3.4 * len(PLOT_BANDS), 4.8),
                             sharey=True, squeeze=False)
    for bi, band in enumerate(PLOT_BANDS):
        ax = axes[0][bi]
        sub = [r for r in rows if r["band"] == band]
        real = hierarchical(sub, ("area_a", "area_b"), "heldout_cc1")
        null = hierarchical(sub, ("area_a", "area_b"), "null_p95")
        x = np.arange(len(pairs))
        for i, p in enumerate(pairs):
            if p not in real:
                continue
            lo = null.get(p, (np.nan,) * 3)[0]
            ax.plot([i - 0.42, i + 0.42], [lo, lo], color="0.35", lw=1.2, ls="--")
            mu, sem, n = real[p]
            ax.errorbar([i], [mu], yerr=[sem], marker="o", ms=6, capsize=3,
                        color="#d95319", zorder=3)
            ax.text(i, 0.02, f"N={n}", ha="center", fontsize=6, color="0.35")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a}–{b}" for a, b in pairs], rotation=45, fontsize=7,
                           ha="right")
        ax.set_title(BAND_LABEL[band], fontsize=9)
        ax.set_ylim(0, 1.02)
        if bi == 0:
            ax.set_ylabel("held-out top canonical correlation")
    fig.suptitle("Cross-area co-fluctuation of band-power profiles\n"
                 "orange = held-out CC1 (mean ± SEM over animals) · dashed = "
                 "trial-permutation null (95th pct)\n"
                 "above the null says the pairing of trials matters; it does not separate "
                 "a shared field from communication (see the distance control)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_pair(fig, stem)


def plot_cca_distance(rows, stem="lfp_cca_vs_distance", stats_rows=()):
    """Does cross-area coupling fall off with distance along the shank?

    A shared volume-conducted field decays with electrode separation. The test
    is the WITHIN-animal slope of CC1 on separation, tested across animals
    (``lfp_arms_cca_distance_stats``); pooling area pairs from the same animal
    as independent points is pseudoreplication.
    """
    fig, axes = plt.subplots(1, len(PLOT_BANDS), figsize=(3.3 * len(PLOT_BANDS), 3.9),
                             sharey=True, squeeze=False)
    for bi, band in enumerate(PLOT_BANDS):
        ax = axes[0][bi]
        sub = [r for r in rows if r["band"] == band
               and np.isfinite(r["heldout_cc1"]) and np.isfinite(r["separation_um"])]
        x = np.array([r["separation_um"] for r in sub]) / 1000.0
        y = np.array([r["heldout_cc1"] for r in sub])
        pairs = np.array([f"{r['area_a']}-{r['area_b']}" for r in sub])
        animals = np.array([f"{r['mouse_id']}/{r['probe']}" for r in sub])
        for animal in np.unique(animals):
            m = animals == animal
            order = np.argsort(x[m])
            ax.plot(x[m][order], y[m][order], color="0.75", lw=0.7, zorder=1)
        for pair in np.unique(pairs):
            m = pairs == pair
            ax.scatter(x[m], y[m], s=16, alpha=0.75, label=pair, zorder=2)
        st = next((r for r in stats_rows if r["band"] == band), None)
        if st is not None:
            ax.text(0.96, 0.95, f"within-animal slope {float(st['mean_delta']):+.2f} /mm\n"
                    f"p = {float(st['p_raw']):.3f}, p_FDR = {float(st['p_fdr']):.3f}\n"
                    f"n = {int(st['n_animals'])} animals",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7)
        ax.set_xlabel("separation between area centres (mm)")
        ax.set_title(BAND_LABEL[band], fontsize=9)
        if bi == 0:
            ax.set_ylabel("held-out top canonical correlation")
        if bi == len(PLOT_BANDS) - 1:
            ax.legend(fontsize=6, loc="lower left")
    fig.suptitle("Cross-area coupling falls off with distance along the shank\n"
                 "grey lines join one animal's area pairs; the test is the within-animal "
                 "slope, across animals (exact sign-flip, BH over bands)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save_pair(fig, stem)


# --- moving-window reliability (the unit pipeline's stability metric) --------

MIN_ANIMALS_PER_OFFSET = 3      # do not draw a mean+-SEM that rests on <3 animals


def _moving_by_offset(rows, area, band, field, xkey):
    """{offset: array of per-animal values} for one area x band."""
    by_offset = defaultdict(dict)
    for r in rows:
        if r["area"] != area or r["band"] != band:
            continue
        x = r[xkey]
        if x == "" or not np.isfinite(r[field]):
            continue
        by_offset[int(x)][int(r["mouse_id"])] = r[field]
    return by_offset


def _draw_moving(ax, rows, area, band, xkey, xlim) -> int:
    """Draw the observed and shuffled traces; return how many were drawable."""
    drawn = 0
    for field, colour, label, style in (
        ("reliability", AREA_COLOUR[area], "observed", "-"),
        ("reliability_shuffled", "0.55", "trial-shuffled", "--"),
    ):
        by_offset = _moving_by_offset(rows, area, band, field, xkey)
        xs = sorted(o for o, d in by_offset.items()
                    if len(d) >= MIN_ANIMALS_PER_OFFSET and xlim[0] <= o <= xlim[1])
        if not xs:
            continue
        m = np.array([np.mean(list(by_offset[o].values())) for o in xs])
        e = np.array([np.std(list(by_offset[o].values()), ddof=1)
                      / np.sqrt(len(by_offset[o])) for o in xs])
        ax.plot(xs, m, style, color=colour, lw=1.4, label=label)
        ax.fill_between(xs, m - e, m + e, color=colour, alpha=0.2, lw=0)
        drawn += 1
    ax.axhline(0, color="k", lw=0.5, ls=":")
    return drawn


def plot_moving_reliability(rows, stem="lfp_reliability_moving"):
    areas = [a for a in AREA_ORDER if any(r["area"] == a for r in rows)]
    fig, axes = plt.subplots(len(PLOT_BANDS), len(areas),
                             figsize=(2.6 * len(areas), 2.3 * len(PLOT_BANDS)),
                             squeeze=False, sharex=True)
    xlim = (-30, 40)
    for bi, band in enumerate(PLOT_BANDS):
        for ai, area in enumerate(areas):
            ax = axes[bi][ai]
            drawn = _draw_moving(ax, rows, area, band, "trial_rel_lp", xlim)
            ax.axvline(0, color="#b3001b", lw=0.9)
            ax.set_xlim(*xlim)
            ax.tick_params(labelsize=7)
            n_animals = len({int(r["mouse_id"]) for r in rows
                             if r["area"] == area and r["band"] == band
                             and r["trial_rel_lp"] != ""})
            ax.text(0.02, 0.04, f"N = {n_animals}", transform=ax.transAxes,
                    fontsize=6, color="0.35")
            if bi == 0:
                ax.set_title(area, fontsize=10, color=AREA_COLOUR[area],
                             fontweight="bold")
            if ai == 0:
                ax.set_ylabel(f"{BAND_LABEL[band]}\nreliability", fontsize=7)
            if not drawn:
                ax.text(0.5, 0.5, f"n = {n_animals} learners\n"
                        f"(< {MIN_ANIMALS_PER_OFFSET}, not plotted)",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=7, color="0.6")
            if bi == 0 and ai == 0:
                ax.legend(fontsize=6, loc="upper left")
    for ax in axes[-1]:
        ax.set_xlabel("trial relative to learning point", fontsize=7)
    fig.suptitle("Moving trial-to-trial reliability of the LFP spatial profile\n"
                 "5-trial window centred on each trial, clipped at the edges; mean "
                 "pairwise correlation across the window\n"
                 "same window and same statistic as the single-unit stability figures "
                 "(IntegratedAll_v1 via batch_triu_corr_mean); red line = learning point",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_pair(fig, stem)


def plot_moving_reliability_absolute(rows, stem="lfp_reliability_moving_session"):
    """The same trace against absolute trial number, which keeps the non-learners."""
    areas = [a for a in AREA_ORDER if any(r["area"] == a for r in rows)]
    fig, axes = plt.subplots(1, len(PLOT_BANDS), figsize=(3.4 * len(PLOT_BANDS), 3.8),
                             sharey=True, squeeze=False)
    for bi, band in enumerate(PLOT_BANDS):
        ax = axes[0][bi]
        for area in areas:
            _draw_moving(ax, rows, area, band, "trial", (1, 100))
        ax.set_xlim(1, 100)
        ax.set_xlabel("trial from session start")
        ax.set_title(BAND_LABEL[band], fontsize=9)
        if bi == 0:
            ax.set_ylabel("moving reliability (5-trial window)")
    handles = [plt.Line2D([], [], color=AREA_COLOUR[a], label=a) for a in areas]
    handles.append(plt.Line2D([], [], color="0.55", ls="--", label="trial-shuffled"))
    axes[0][-1].legend(handles=handles, fontsize=6, loc="upper right")
    fig.suptitle("Moving reliability against absolute trial — includes the two "
                 "non-learners, which have no learning point to align to", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_pair(fig, stem)


def plot_moving_reliability_depth(cohort_name="task", stem="lfp_reliability_moving_depth"):
    """Per-file depth x trial reliability -- the LFP analogue of the neurons x trials
    ``imagesc(avg_corrs)`` panel in ProcessStriatumTask.m:997. Draws the matrices
    run_lfp_arms.py saves; computes nothing."""
    files = sorted((config.RESULTS_DIR / f"lfp_arms_moving_depth_{cohort_name}").glob("*.npz"))
    if not files:
        return
    ncol = 6
    nrow = int(np.ceil(len(files) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.7 * nrow), squeeze=False)
    for k, path in enumerate(files):
        ax = axes[k // ncol][k % ncol]
        z = np.load(path, allow_pickle=False)
        mouse, probe = path.stem.split("_", 1)
        rel = z["reliability"]
        n_keep = int(z["n_trials"])
        band = str(z["band"])
        im = ax.imshow(rel, aspect="auto", cmap="magma", vmin=-0.2, vmax=0.8,
                       extent=(0.5, n_keep + 0.5, z["channel_depth_um"].max(), 0),
                       interpolation="nearest")
        for area in config.AREAS:
            m = z[f"is_{area.lower()}"]
            if m.sum() < 5:
                continue
            d = z["channel_depth_um"][m]
            ax.plot([0.6, 0.6], [d.min(), d.max()], lw=4, solid_capstyle="butt",
                    color=AREA_COLOUR[area])
            ax.text(n_keep * 0.03, (d.min() + d.max()) / 2, area, fontsize=5.5,
                    color=AREA_COLOUR[area], va="center", fontweight="bold")
        lp = int(z["learning_point"]) if int(z["learning_point"]) > 0 else None
        if lp and lp <= n_keep:
            ax.axvline(lp, color="#39ff14", lw=1.0)
        ax.set_title(f"{mouse}{'·v1' if probe == 'visual' else ''}", fontsize=8)
        ax.tick_params(labelsize=6)
        if k % ncol == 0:
            ax.set_ylabel("depth from tip (µm)", fontsize=7)
        if k // ncol == nrow - 1:
            ax.set_xlabel("trial", fontsize=7)
    for k in range(len(files), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    cb = fig.colorbar(im, ax=axes, fraction=0.014, pad=0.01)
    cb.set_label("moving reliability (5-trial window)")
    fig.suptitle(f"Moving reliability per channel — {BAND_LABEL[band]}\n"
                 "the LFP analogue of the neurons × trials stability image; "
                 "green line = learning point", fontsize=12)
    save_pair(fig, stem)


def plot_moving_vs_units(cohort_name="task", stem="lfp_reliability_moving_vs_units"):
    """The LFP moving metric beside the single-unit one, same window, same epochs.

    ``figures/stability_by_animal.csv`` is written by IntegratedAll_v1 from this
    statistic on this 5-trial window. Like for like means the same ANIMALS too:
    its ``animal`` column is the position in the organiser's list, so it is
    mapped to the mouse id and only (mouse, area, epoch) cells present in BOTH
    tables are compared (IntegratedAll_v1 skips non-learners; the LFP arm does
    not). Raw reliability is plotted with each signal's own trial shuffle: the
    difference measures drift, not single-trial reliability, and a stationary,
    perfectly reliable profile scores ~0 on it.
    """
    lfp_path = config.RESULTS_DIR / f"lfp_arms_moving_reliability_epochs_{cohort_name}.csv"
    unit_path = config.STABILITY_BY_ANIMAL_CSV
    if not lfp_path.exists() or not unit_path.exists():
        print("[plot] moving-vs-units needs both tables; skipping")
        return
    ids = config.get_cohort(cohort_name).mouse_ids
    with lfp_path.open() as fh:
        lfp = list(csv.DictReader(fh))
    with unit_path.open() as fh:
        want = "Task" if cohort_name == "task" else "Control 1"
        units = [dict(r, mouse=str(ids[int(r["animal"]) - 1]))
                 for r in csv.DictReader(fh) if r["group"].startswith(want)]
    for r in lfp:
        r["mouse"] = str(r["animal"])
    unit_cells = {(r["mouse"], r["area"], r["epoch"]) for r in units}

    areas = [a for a in AREA_ORDER if any(r["area"] == a for r in lfp)
             and any(c[1] == a for c in unit_cells)]
    fig, axes = plt.subplots(1, len(areas), figsize=(2.6 * len(areas), 4.4),
                             sharey=True, squeeze=False)

    def agg(rows, pred, field):
        out = {}
        for ep in EPOCHS:
            vals = [float(r[field]) for r in rows
                    if r["epoch"] == ep and pred(r) and r[field] not in ("", "None")
                    and (r["mouse"], r["area"], ep) in unit_cells]
            vals = [v for v in vals if np.isfinite(v)]
            if len(vals) >= 2:
                out[ep] = (float(np.mean(vals)),
                           float(np.std(vals, ddof=1) / np.sqrt(len(vals))), len(vals))
        return out

    def draw(ax, stats_by_epoch, **kw):
        ax.errorbar(x, [stats_by_epoch.get(e, (np.nan,) * 3)[0] for e in EPOCHS],
                    yerr=[stats_by_epoch.get(e, (np.nan,) * 3)[1] for e in EPOCHS], **kw)

    x = np.arange(len(EPOCHS))
    for ai, area in enumerate(areas):
        ax = axes[0][ai]
        u = agg(units, lambda r: r["area"] == area, "reliability")
        if u:
            n = max(v[2] for v in u.values())
            draw(ax, u, marker="s", ms=6, lw=2.2, capsize=3, color="k",
                 label=f"single units (N={n})")
            draw(ax, agg(units, lambda r: r["area"] == area, "shuffle"),
                 lw=1.0, ls="--", color="k", label="units, trial shuffle")
        for band in PLOT_BANDS:
            def same(r, band=band):
                return r["area"] == area and r["band"] == band
            b = agg(lfp, same, "reliability")
            if not b:
                continue
            draw(ax, b, marker="o", ms=4, lw=1.2, capsize=2, alpha=0.9, label=f"LFP {band}")
        ax.axhline(0, color="0.5", lw=0.7, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels(EPOCHS, rotation=30, fontsize=7, ha="right")
        ax.set_title(area, fontsize=10, color=AREA_COLOUR[area], fontweight="bold")
        if ai == 0:
            ax.set_ylabel("moving reliability (raw)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=7,
               frameon=False)
    fig.suptitle("Single-trial spatial reliability: LFP band power vs single units\n"
                 "same statistic, same 5-trial window and epochs, only (mouse, area, epoch) "
                 "cells present in both tables; dashed = units' trial shuffle",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.9))
    save_pair(fig, stem)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=str, default="task",
                        choices=sorted(config.COHORTS))
    args = parser.parse_args()
    c = args.cohort
    tag = f"_{c}"

    evo = load("evolution", c)
    if evo:
        st = load_stats(c)
        plot_evolution(evo, "log_corridor", "log_dark", "log10 band power",
                       f"lfp_evolution_log{tag}",
                       f"LFP band power across learning, per area — {c.upper()} cohort "
                       "(PRIMARY: log10 of mean power)\n"
                       "Δ and p(FDR) are the Naive → Expert exact sign-flip test, "
                       "BH-corrected over the area × band family; ✱ = survives",
                       stats=st)
        plot_evolution(evo, "z_corridor", "z_dark", "z log power",
                       f"lfp_evolution_z{tag}",
                       f"Sensitivity — {c.upper()} cohort: z-scored log power "
                       "(divides by a session SD that differs between animals)",
                       stats=st)
        plot_evolution(evo, "z_corridor_speed_resid", "z_dark",
                       "z log power, speed removed",
                       f"lfp_evolution_speed_residual{tag}",
                       f"The speed control — {c.upper()} cohort: the same effect after the "
                       "linear log-speed component is removed per channel", stats=st)
        plot_evolution(evo, "frac_of_total_corridor", "frac_of_total_dark",
                       "band / total power", f"lfp_evolution_fraction{tag}",
                       f"The aperiodic guard — {c.upper()} cohort: band power as a FRACTION "
                       "of 1–150 Hz total", stats=st)
        plot_speed(evo, stem=f"lfp_evolution_speed{tag}")
    rows = load("decoding", c)
    if rows:
        plot_decoding(rows, stem=f"lfp_decoding{tag}", stats_rows=load("decoding_stats", c))
    rows = load("cca", c)
    if rows:
        plot_cca_distance(rows, stem=f"lfp_cca_vs_distance{tag}",
                          stats_rows=load("cca_distance_stats", c))
    for name, fn, stem in (
        ("reliability", plot_reliability, "lfp_reliability"),
        ("cca", plot_cca, "lfp_cca"),
        ("moving_reliability", plot_moving_reliability, "lfp_reliability_moving"),
        ("moving_reliability", plot_moving_reliability_absolute,
         "lfp_reliability_moving_session"),
    ):
        rows = load(name, c)
        if rows:
            fn(rows, stem=f"{stem}{tag}")
    plot_moving_reliability_depth(c, stem=f"lfp_reliability_moving_depth{tag}")
    plot_moving_vs_units(c, stem=f"lfp_reliability_moving_vs_units{tag}")


if __name__ == "__main__":
    main()
