"""Epoch ANOVA for communication strength (CC) and the Information Flow Index.

Tests whether held-out CC and the IFI (|lag| <= 10 bins) change across the
three epochs (naive / intermediate / expert), per area pair, two ways:

  * per significant dimension -- one-way ANOVA across epochs (scipy f_oneway)
    + Tukey HSD post-hoc + a linear trend (value vs epoch index 0/1/2);
  * per learner animal -- one-way repeated-measures ANOVA
    (epoch_stats.rm_anova) + paired-t post-hoc with Holm correction
    + a per-animal linear trend (one-sample t on the per-animal slopes).

Reads the committed Stage-2 pkl (partial CCA by default; --variant plain for
the plain pipeline). Writes figures/epoch_stats_<variant>.csv (the full table)
and figures/epoch_anova_{cc,ifi}_<variant>.png.

Run:  python scripts/epoch_anova.py [--variant plain|partial]
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

from striatum_cca import config, epoch_stats  # noqa: E402

EPOCHS = config.EPOCH_NAMES
EPOCH_LABEL = ["naive", "inter", "expert"]
EPOCH_IDX = np.arange(3, dtype=float)
EPOCH_COLOUR = config.EPOCH_COLOURS
ALPHA = 0.05
IFI_WINDOW = 10
METRICS = {"cc": "held-out CC", "ifi": f"IFI (|lag| <= {IFI_WINDOW} bins)"}

VARIANT_PKL = {"plain": "stage2_committed_circshift.pkl",
               "partial": "stage2_committed_circshift_partial.pkl"}


# --- data access -------------------------------------------------------------
def learner_pairs(results, area_x, area_y):
    return [r for r in results
            if (r.area_x, r.area_y) == (area_x, area_y) and r.role == "learner"]


def _sig(ea):
    return np.where(ea.p_per_dim < ALPHA)[0]


def dim_values(r, epoch, metric):
    """Significant-dim CC or IFI(w10) values for one pair x epoch."""
    ea = r.epochs[epoch]
    js = _sig(ea)
    if metric == "cc":
        return ea.held_out_cc[js]
    return ea.ifi_windows[js, IFI_WINDOW - 1]


def per_dim_groups(learners, metric):
    """Three arrays -- significant-dim values per epoch, pooled over animals."""
    groups = []
    for epoch in EPOCHS:
        vals = (np.concatenate([dim_values(r, epoch, metric) for r in learners])
                if learners else np.array([]))
        groups.append(vals[np.isfinite(vals)])
    return groups


def per_animal_matrix(learners, metric):
    """(n_complete_animals, 3) of per-animal mean over significant dims."""
    rows = []
    for r in learners:
        row = []
        for epoch in EPOCHS:
            v = dim_values(r, epoch, metric)
            row.append(float(np.nanmean(v)) if v.size else np.nan)
        rows.append(row)
    mat = np.array(rows, dtype=float) if rows else np.empty((0, 3))
    return mat[np.all(np.isfinite(mat), axis=1)] if mat.size else mat


# --- statistics --------------------------------------------------------------
def per_dim_stats(groups):
    """One-way ANOVA + Tukey HSD + linear trend over significant dimensions."""
    out = {"n": [g.size for g in groups], "F": np.nan, "p": np.nan,
           "tukey": (np.nan, np.nan, np.nan),
           "slope": np.nan, "slope_p": np.nan}
    if min(g.size for g in groups) < 2:
        return out
    out["F"], out["p"] = (float(v) for v in stats.f_oneway(*groups))
    th = stats.tukey_hsd(*groups).pvalue
    out["tukey"] = (float(th[0, 1]), float(th[1, 2]), float(th[0, 2]))
    x = np.concatenate([np.full(g.size, i) for i, g in enumerate(groups)])
    lr = stats.linregress(x, np.concatenate(groups))
    out["slope"], out["slope_p"] = float(lr.slope), float(lr.pvalue)
    return out


def per_animal_stats(mat):
    """RM-ANOVA + paired-t/Holm post-hoc + per-animal linear trend."""
    n = mat.shape[0]
    out = {"n": n, "F": np.nan, "p": np.nan,
           "posthoc": (np.nan, np.nan, np.nan),
           "slope": np.nan, "slope_p": np.nan}
    if n < 2:
        return out
    out["F"], out["p"] = epoch_stats.rm_anova(mat)
    raw = np.array([
        stats.ttest_rel(mat[:, a], mat[:, b]).pvalue
        if np.any(mat[:, a] != mat[:, b]) else np.nan
        for a, b in ((0, 1), (1, 2), (0, 2))])
    adj = np.full(3, np.nan)
    if np.isfinite(raw).any():
        adj[np.isfinite(raw)] = epoch_stats.holm(raw[np.isfinite(raw)])
    out["posthoc"] = tuple(float(v) for v in adj)
    slopes = np.array([np.polyfit(EPOCH_IDX, mat[i], 1)[0] for i in range(n)])
    out["slope"] = float(np.mean(slopes))
    if np.ptp(slopes) > 0:
        out["slope_p"] = float(stats.ttest_1samp(slopes, 0.0).pvalue)
    return out


def _fmt(p):
    return f"{p:.3f}" if np.isfinite(p) else "n/a"


# --- figure ------------------------------------------------------------------
def _bracket(ax, x0, x1, y, p):
    """Significance bracket between two epochs when post-hoc p < ALPHA."""
    if not (np.isfinite(p) and p < ALPHA):
        return
    ax.plot([x0, x0, x1, x1], [y, y + 0.02, y + 0.02, y], color="k", lw=0.8,
            transform=ax.get_xaxis_transform(), clip_on=False)
    ax.text((x0 + x1) / 2, y + 0.03, "*", ha="center", va="bottom", fontsize=11,
            transform=ax.get_xaxis_transform(), clip_on=False)


def plot_metric(results, metric, variant):
    """2 x 5 pair grid: per-animal epoch trajectories + trend + ANOVA p."""
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.8))
    axes = axes.ravel()
    for ax, (area_x, area_y) in zip(axes, config.PAIRS):
        learners = learner_pairs(results, area_x, area_y)
        mat = per_animal_matrix(learners, metric)
        ds = per_dim_stats(per_dim_groups(learners, metric))
        as_ = per_animal_stats(mat)
        for i in range(mat.shape[0]):                       # per-animal lines
            ax.plot(EPOCH_IDX, mat[i], "-", color="0.8", lw=0.7, zorder=1)
        if mat.shape[0]:
            mean = mat.mean(axis=0)
            sem = mat.std(axis=0) / np.sqrt(mat.shape[0])
            ax.plot(EPOCH_IDX, mean, "-", color="0.45", lw=1.4, zorder=2)
            for j, epoch in enumerate(EPOCHS):
                ax.errorbar(j, mean[j], yerr=sem[j], fmt="o", ms=8,
                            color=EPOCH_COLOUR[epoch], capsize=3, zorder=3)
            mid = mean.mean()                               # per-animal trend
            if np.isfinite(as_["slope"]):
                ax.plot(EPOCH_IDX, mid + as_["slope"] * (EPOCH_IDX - 1),
                        "--", color="k", lw=1.0, zorder=2)
            for k, (a, b) in enumerate(((0, 1), (1, 2), (0, 2))):
                _bracket(ax, a, b, 0.84 + 0.06 * k, ds["tukey"][k])
        ax.set_xticks(EPOCH_IDX)
        ax.set_xticklabels(EPOCH_LABEL)
        ax.set_xlim(-0.4, 2.4)
        ax.set_title(f"{area_x}-{area_y}  n={as_['n']}\n"
                     f"dim p={_fmt(ds['p'])}  RM p={_fmt(as_['p'])}  "
                     f"trend p={_fmt(as_['slope_p'])}", fontsize=7.5)
    for ax in axes[::5]:
        ax.set_ylabel(METRICS[metric])
    fig.suptitle(f"Epoch ANOVA -- {METRICS[metric]} across learning "
                 f"({variant} CCA, committed config; per-animal trajectories, "
                 f"mean +/- SEM; '*' Tukey p<0.05 over dimensions)")
    fig.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / f"epoch_anova_{metric}_{variant}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


# --- csv ---------------------------------------------------------------------
COLS = ["pair", "metric", "n_dim_naive", "n_dim_inter", "n_dim_expert",
        "n_animals", "anova_dim_F", "anova_dim_p", "tukey_ni_p", "tukey_ie_p",
        "tukey_ne_p", "trend_dim_slope", "trend_dim_p", "rm_anova_F",
        "rm_anova_p", "posthoc_ni_p", "posthoc_ie_p", "posthoc_ne_p",
        "trend_animal_slope", "trend_animal_p"]


def _round(v):
    if isinstance(v, float):
        if not np.isfinite(v):
            return "" if np.isnan(v) else ("inf" if v > 0 else "-inf")
        return round(v, 5)
    return v


def write_csv(results, variant):
    rows = []
    for area_x, area_y in config.PAIRS:
        learners = learner_pairs(results, area_x, area_y)
        for metric in METRICS:
            ds = per_dim_stats(per_dim_groups(learners, metric))
            as_ = per_animal_stats(per_animal_matrix(learners, metric))
            rows.append([
                f"{area_x}-{area_y}", metric, *ds["n"], as_["n"],
                ds["F"], ds["p"], *ds["tukey"], ds["slope"], ds["slope_p"],
                as_["F"], as_["p"], *as_["posthoc"], as_["slope"],
                as_["slope_p"]])
    path = config.FIGURES_DIR / f"epoch_stats_{variant}.csv"
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(COLS)
        for r in rows:
            wr.writerow([_round(v) for v in r])
    print(f"saved {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=("plain", "partial"), default="partial")
    variant = p.parse_args().variant
    path = config.RESULTS_DIR / VARIANT_PKL[variant]
    if not path.exists():
        sys.exit(f"missing {path.name} -- run run_committed.py --stage 2")
    with open(path, "rb") as fh:
        results = pickle.load(fh)["results"]
    write_csv(results, variant)
    for metric in METRICS:
        plot_metric(results, metric, variant)
    print("epoch-ANOVA outputs done.")


if __name__ == "__main__":
    main()
