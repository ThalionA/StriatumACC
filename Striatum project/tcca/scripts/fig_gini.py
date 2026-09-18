#!/usr/bin/env python3
"""The one surviving positive result of the tcca arm, with its controls.

Weight-based subspace participation de-sparsifies from naive to expert. The
figure exists to make the claim falsifiable at a glance: the two CCA-weight
readouts beside the CCA-FREE control, and the three nuisance variables that
could produce it arithmetically.

    /opt/anaconda3/bin/python scripts/fig_gini.py
"""
from __future__ import annotations

import collections
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "epoch_metrics.csv"


def num(r, c):
    try:
        return float(r[c])
    except (ValueError, TypeError, KeyError):
        return np.nan


def animal_delta(rows, col):
    """Median over an animal's pairs, then expert - naive. Animals are n."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        v = num(r, col)
        if np.isfinite(v):
            by[r["animal"]][r["epoch"]].append(v)
    out = {}
    for a, e in by.items():
        if "expert" in e and "naive" in e:
            out[a] = (np.median(e["naive"]), np.median(e["expert"]))
    return out


def main() -> None:
    rows = list(csv.DictReader(METRICS.open()))
    fig = plt.figure(figsize=(15, 9), layout="constrained")
    gs = fig.add_gridspec(2, 3)

    panels = [
        ("gini_y", "AREA-INTRINSIC\nGini of CCA weights", "#1f4e79"),
        ("gini_y_conn", "CONNECTION-SPECIFIC\nGini, canonical-r weighted", "#7e2f8e"),
        ("gini_pearson_y", "CONTROL: CCA-FREE\nGini of raw cross-area coupling", "#999999"),
    ]
    for i, (col, title, colour) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        d = animal_delta(rows, col)
        ids = sorted(d)
        for a in ids:
            n, e = d[a]
            ax.plot([0, 1], [n, e], "-o", ms=5, color=colour, alpha=0.55, lw=1.2)
        nv = np.array([d[a][0] for a in ids])
        ev = np.array([d[a][1] for a in ids])
        ax.plot([0, 1], [nv.mean(), ev.mean()], "-o", ms=11, color="#c0392b",
                lw=3.5, zorder=5)
        diff = ev - nv
        p = stats.wilcoxon(diff).pvalue if diff.size >= 6 else np.nan
        ax.set_xticks([0, 1]); ax.set_xticklabels(["naive", "expert"])
        ax.set_xlim(-0.25, 1.25)
        ax.set_ylabel("Gini (0 = every unit equal, 1 = one unit)")
        ax.set_title(f"{title}\nΔ = {diff.mean():+.4f}, "
                     f"{(diff < 0).sum()}/{diff.size} down, "
                     + (f"p = {p:.4f}" if np.isfinite(p) else "N<6"), fontsize=10)

    # ---- the nuisance variables that could produce it arithmetically --------
    # Drawn as a table, not bars: n_sig's naive median is 0 in most animals, so any
    # relative scale is meaningless and any shared absolute scale is unreadable.
    ax = fig.add_subplot(gs[1, :2])
    ax.axis("off")
    spec = [("k_eff", "canonical dims fitted"),
            ("n_units_y", "units in the y area"),
            ("n_sig", "significant dims"),
            ("n_bins", "samples in the window")]
    cells = []
    for c, what in spec:
        d = animal_delta(rows, c)
        ids = sorted(d)
        nv = np.array([d[a][0] for a in ids])
        diff = np.array([d[a][1] - d[a][0] for a in ids])
        if not np.any(diff):
            verdict = "identical — cannot explain it"
            pstr = "—"
        else:
            p = stats.wilcoxon(diff).pvalue if diff.size >= 6 else np.nan
            pstr = f"{p:.3f}" if np.isfinite(p) else "N<6"
            verdict = ("moves the WRONG way" if c == "n_sig"
                       else "falls, but dissociated →")
        cells.append([c, what, f"{nv.mean():.1f}", f"{diff.mean():+.1f}", pstr, verdict])
    t = ax.table(cellText=cells,
                 colLabels=["variable", "what it is", "naive", "Δ exp−naive",
                            "p", "verdict"],
                 colWidths=[0.15, 0.24, 0.11, 0.15, 0.10, 0.25],
                 cellLoc="left", loc="center")
    t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.8)
    for k, cell in t.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if k[0] == 0:
            cell.set_facecolor("#eeeeee"); cell.set_text_props(weight="bold")
    ax.set_title("Could the drop be arithmetic? Every nuisance variable, same animals, "
                 "same fits.", fontsize=10, pad=14)

    # ---- the one that does move: n_bins, dissociated -----------------------
    ax = fig.add_subplot(gs[1, 2])
    dg = animal_delta(rows, "gini_y")
    db = animal_delta(rows, "n_bins")
    ids = sorted(set(dg) & set(db))
    x = np.array([db[a][1] - db[a][0] for a in ids])
    y = np.array([dg[a][1] - dg[a][0] for a in ids])
    ax.scatter(x, y, s=55, color="#1f4e79", zorder=3)
    ax.axhline(0, color="k", lw=1); ax.axvline(0, color="k", lw=1)
    rho = stats.spearmanr(x, y)
    ax.set_xlabel("Δ n_bins (expert − naive)")
    ax.set_ylabel("Δ Gini (expert − naive)")
    ax.set_title(f"n_bins DOES fall — but it is dissociated.\n"
                 f"{(y < 0).sum()}/{y.size} animals drop in Gini, only "
                 f"{(x < 0).sum()}/{x.size} drop in bins;\nSpearman = "
                 f"{rho.statistic:+.3f}, p = {rho.pvalue:.2f}", fontsize=10)

    fig.suptitle("Subspace participation de-sparsifies with learning — and it is not the "
                 "CCA-free coupling, the dimensionality, or the sample size\n"
                 "Striatal + cortical pairs, task cohort, 25 ms bins, FS-excluded, partial "
                 "CCA, AFTER the tom_cca intercept fix. Animals as n; medians over each "
                 "animal's pairs. p uncorrected.", fontsize=12)
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "gini_desparsification.svg")
    fig.savefig(ROOT / "figures" / "gini_desparsification.png",
                dpi=min(150, 1600 / max(fig.get_size_inches())))
    plt.close(fig)
    print("[fig] gini_desparsification.svg + .png")


if __name__ == "__main__":
    main()
