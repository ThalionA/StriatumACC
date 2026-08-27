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

from striatum_lfp import analysis, config  # noqa: E402

MAX_PNG_PX = 1600
AREA_ORDER = ("DMS", "DLS", "ACC", "V1", "CA1", "DG")
AREA_COLOUR = {"DMS": "#0072b2", "DLS": "#77ac30", "ACC": "#d95319",
               "V1": "#7e2f8e", "CA1": "#cc1a33", "DG": "#33b3b3"}
PLOT_BANDS = ("theta", "beta", "low_gamma", "high_gamma")
BAND_LABEL = {"theta": "theta 4–8 Hz", "beta": "beta 15–30 Hz",
              "low_gamma": "low gamma 30–80 Hz", "high_gamma": "high gamma 80–150 Hz",
              "total": "total 1–150 Hz"}
EPOCHS = list(analysis.EPOCH_NAMES)
WINDOWS = ["All"] + EPOCHS


def save_pair(fig, stem: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / f"{stem}.svg")
    fig.savefig(config.FIGURES_DIR / f"{stem}.png",
                dpi=min(150, MAX_PNG_PX / max(fig.get_size_inches())))
    plt.close(fig)
    print(f"[plot] {stem}.svg + .png", flush=True)


def load(name: str) -> list[dict]:
    path = config.RESULTS_DIR / f"lfp_arms_{name}.csv"
    if not path.exists():
        return []
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            if k in ("probe", "area", "band", "epoch", "window", "area_a", "area_b"):
                continue
            r[k] = float(v) if v not in ("", "None") else np.nan
    return rows


def hierarchical(rows, key_fields, value_field):
    """{key: (mean, sem, n_animals)} with the ANIMAL as the unit of analysis."""
    by_key = defaultdict(dict)
    for r in rows:
        v = r[value_field]
        if np.isfinite(v):
            by_key[tuple(r[f] for f in key_fields)][int(r["mouse_id"])] = v
    out = {}
    for key, per_animal in by_key.items():
        vals = np.array(list(per_animal.values()))
        n = vals.size
        out[key] = (float(vals.mean()),
                    float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan, n)
    return out


def _epoch_axis(ax):
    ax.set_xticks(range(len(EPOCHS)))
    ax.set_xticklabels(["1–3", "4–10", "Inter", "Expert"], fontsize=7)


def load_stats():
    """{(metric, area, band): (mean_delta, p_fdr, survives)} from the declared family."""
    path = config.RESULTS_DIR / "lfp_arms_evolution_stats.csv"
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
                        x.append(ei); m.append(mu); e.append(sem); ns.append(n)
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
                x.append(ei); m.append(mu); e.append(sem)
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


def plot_decoding(rows, stem="lfp_decoding"):
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
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("R² with position labels rotated within each trial (null)")
    ax.set_ylabel("R² with the true position labels")
    ax.set_title("(b) One point per animal, high gamma\n"
                 "points above the dashed line decode better than the null", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("Spatial decoding from LFP band power — reliable but small: the decoder "
                 "beats its null in every striatal/ACC cell (BH-FDR q=0.05),\n"
                 "yet median error improves only ~1–2.5 cm on a 250 cm corridor",
                 fontsize=11)
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
        ceil_a = hierarchical(sub, ("area_a", "area_b"), "ceiling_a")
        ceil_b = hierarchical(sub, ("area_a", "area_b"), "ceiling_b")
        x = np.arange(len(pairs))
        for i, p in enumerate(pairs):
            if p not in real:
                continue
            lo = null.get(p, (np.nan,) * 3)[0]
            hi = np.nanmean([ceil_a.get(p, (np.nan,) * 3)[0],
                             ceil_b.get(p, (np.nan,) * 3)[0]])
            ax.fill_between([i - 0.42, i + 0.42], lo, hi, color="0.85", zorder=0)
            ax.plot([i - 0.42, i + 0.42], [hi, hi], color="0.35", lw=1.2)
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
    fig.suptitle("Cross-area coupling, bracketed by what it must beat\n"
                 "orange = cross-area held-out CC1 · dashed = trial-permutation null (95th pct) · "
                 "solid = within-area split-half ceiling\n"
                 "a value inside the grey band is consistent with a shared field "
                 "(one shank, 1.2–1.9 mm apart), not with area-specific coupling",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_pair(fig, stem)


def plot_cca_distance(rows, stem="lfp_cca_vs_distance"):
    """Does cross-area coupling fall off with distance along the shank?

    The cleanest discriminator available without re-referencing: a shared
    volume-conducted field decays with electrode separation, whereas
    area-specific coupling has no reason to.
    """
    fig, axes = plt.subplots(1, len(PLOT_BANDS), figsize=(3.3 * len(PLOT_BANDS), 3.9),
                             sharey=True, squeeze=False)
    from scipy import stats as sps
    for bi, band in enumerate(PLOT_BANDS):
        ax = axes[0][bi]
        sub = [r for r in rows if r["band"] == band
               and np.isfinite(r["heldout_cc1"]) and np.isfinite(r["separation_um"])]
        x = np.array([r["separation_um"] for r in sub]) / 1000.0
        y = np.array([r["heldout_cc1"] for r in sub])
        pairs = np.array([f"{r['area_a']}-{r['area_b']}" for r in sub])
        for pair in np.unique(pairs):
            m = pairs == pair
            ax.scatter(x[m], y[m], s=16, alpha=0.75, label=pair)
        if x.size > 3:
            rho, pv = sps.spearmanr(x, y)
            fit = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 20)
            ax.plot(xs, np.polyval(fit, xs), "k--", lw=1)
            ax.text(0.96, 0.95, f"Spearman ρ = {rho:+.2f}\np = {pv:.1e}\nn = {x.size} pairs",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7)
        ax.set_xlabel("separation between area centres (mm)")
        ax.set_title(BAND_LABEL[band], fontsize=9)
        if bi == 0:
            ax.set_ylabel("held-out top canonical correlation")
        if bi == len(PLOT_BANDS) - 1:
            ax.legend(fontsize=6, loc="lower left")
    fig.suptitle("Cross-area coupling falls off with distance along the shank\n"
                 "each point is one animal-pair; the decay is the signature of a shared "
                 "field, not of area-specific communication", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save_pair(fig, stem)


def main() -> None:
    evo = load("evolution")
    if evo:
        st = load_stats()
        plot_evolution(evo, "z_corridor", "z_dark", "z log power", "lfp_evolution_z",
                       "LFP band power across learning, per area (z-scored log power)\n"
                       "Δ and p(FDR) are the trials 4–10 → Expert paired test, "
                       "BH-corrected over the 24-cell area × band family; ✱ = survives",
                       stats=st)
        plot_evolution(evo, "z_corridor_speed_resid", "z_dark",
                       "z log power, speed removed", "lfp_evolution_speed_residual",
                       "The speed control: the same effect after the linear log-speed "
                       "component is removed per channel\n"
                       "running speed rises ~34% from the first trials to expert, "
                       "so anything that vanishes here was speed",
                       stats=st)
        plot_evolution(evo, "frac_of_total_corridor", "frac_of_total_dark",
                       "band / total power", "lfp_evolution_fraction",
                       "The aperiodic guard: band power as a FRACTION of 1–150 Hz total\n"
                       "a change here is a change in spectral shape, not in overall power",
                       stats=st)
        plot_speed(evo)
    for name, fn in (("decoding", plot_decoding), ("reliability", plot_reliability),
                     ("cca", plot_cca), ("cca", plot_cca_distance)):
        rows = load(name)
        if rows:
            fn(rows)


if __name__ == "__main__":
    main()
