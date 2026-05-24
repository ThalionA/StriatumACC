"""Per-pair parallel-coordinate plots + enrichment over the spatial sweep.

`pair` is the unit of observation, not a hyperparameter -- so every figure
shows all 10 area-pairs (a 2x5 grid, one parcoords panel per pair) and
enrichment is computed *within* each pair, over the genuine hyperparameters
(fs, k_rule, ... -- whatever is left free by FOCUS).

Reads figures/sweep_summary_spatial.csv; for each chosen p-value column writes
a per-pair parcoords grid and a per-pair enrichment table, and prints the
per-pair significance rate (the all-pairs headline).

Run:  python scripts/plot_parcoords.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from striatum_cca import config  # noqa: E402

ALPHA = 0.05
# Genuine analysis hyperparameters -- NOT pair.
HYPER = ["bin", "cca", "fs", "z", "k_rule", "min_units", "lp_consec"]
PAIR_ORDER = [f"{ax}-{ay}" for ax, ay in config.PAIRS]
ORDER = {
    "bin": ["2.5cm", "5cm"],
    "cca": ["residual", "signal"],
    "fs": ["excl", "incl"],
    "z": ["on", "off"],
    "k_rule": ["samples15", "samples25", "samples40", "fixed3", "fixed5",
               "fixed10", "fixed20", "fixed30", "var75", "var85", "var95"],
    "min_units": [4, 6, 10],
    "lp_consec": [7, 8],
}
METRICS = {
    "p_ifi_w3": "IFI |lag|<=3 -- naive/expert directionality p-value",
    "p_naive_vs_expert": "delta-CC -- naive-vs-expert strength p-value",
}
# Committed hyperparameter region (round 8). Only fs stays free.
FOCUS = {"bin": "2.5cm", "cca": "residual", "z": "on",
         "min_units": 6, "lp_consec": 7, "k_rule": "samples15"}

RESULT_COLS = ["pair", "fs", "n_dims_naive", "n_dims_expert", "cc_naive",
               "cc_expert", "d_cc", "p_naive_vs_expert", "ifi_w3", "p_ifi_w3",
               "angle_minus_floor"]


def _positions(series, order):
    idx = {v: i for i, v in enumerate(order)}
    return series.map(idx).to_numpy(float) / max(1, len(order) - 1)


def _draw_panel(ax, d, metric, axes):
    """One pair's parcoords over `axes` + the -log10(p) metric axis."""
    if len(d) == 0:
        ax.set_axis_off()
        return
    nlp = -np.log10(d[metric].clip(lower=1e-4))
    sig = d[metric].to_numpy() < ALPHA
    span = nlp.max() - nlp.min() + 1e-9
    nlp_norm = ((nlp - nlp.min()) / span).to_numpy()
    Y = np.column_stack([_positions(d[h], ORDER[h]) for h in axes]
                        + [nlp_norm])
    x = np.arange(Y.shape[1])
    lines = [np.column_stack([x, Y[i]]) for i in range(Y.shape[0])]
    ax.add_collection(LineCollection([lines[i] for i in np.where(~sig)[0]],
                                     colors="0.7", lw=0.5, alpha=0.35))
    hot = np.where(sig)[0]
    if hot.size:
        ax.add_collection(LineCollection(
            [lines[i] for i in hot],
            colors=plt.cm.plasma(nlp_norm[hot]), lw=1.3, alpha=0.85))
    for xi in x:
        ax.axvline(xi, color="0.6", lw=0.7, zorder=0)
    for xi, h in zip(x, axes):
        order = ORDER[h]
        n = max(1, len(order) - 1)
        for j, v in enumerate(order):
            ax.text(xi - 0.06, j / n, str(v), ha="right", va="center",
                    fontsize=5)
    thr = (-np.log10(ALPHA) - nlp.min()) / span
    ax.plot([x[-1] - 0.12, x[-1] + 0.12], [thr, thr], color="red", lw=1.3)
    ax.set_xticks(x)
    ax.set_xticklabels(axes + ["-log10 p"], fontsize=6)
    ax.set_yticks([])
    ax.set_ylim(-0.08, 1.08)
    ax.set_xlim(-1.0, x[-1] + 0.5)


def parcoords_grid(df, metric, label, path, axes):
    fig, axarr = plt.subplots(2, 5, figsize=(18, 8))
    for ax, pair in zip(axarr.ravel(), PAIR_ORDER):
        d = df[(df["pair"] == pair) & df[metric].notna()]
        _draw_panel(ax, d, metric, axes)
        n_sig = int((d[metric] < ALPHA).sum())
        ax.set_title(f"{pair}  ({n_sig}/{len(d)} configs p<.05)", fontsize=9)
    fig.suptitle(f"Per-pair parallel coordinates -- {label}\n"
                 f"red line = p=0.05; coloured lines = significant configs",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def enrichment(df, metric, path, axes):
    """Enrichment computed WITHIN each pair (pair is never a hyperparameter)."""
    rows = []
    for pair in PAIR_ORDER:
        d = df[(df["pair"] == pair) & df[metric].notna()]
        if len(d) == 0:
            continue
        base = float((d[metric] < ALPHA).mean())
        for h in axes:
            for v, g in d.groupby(h):
                rate = float((g[metric] < ALPHA).mean())
                rows.append({"pair": pair, "hyperparam": h, "value": v,
                             "n_cells": len(g), "frac_p<0.05": round(rate, 4),
                             "pair_baseline": round(base, 4),
                             "enrichment_vs_pair": round(rate / base, 2)
                             if base else np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    print(f"saved {path}")
    return out


def committed_results(df, path_csv, path_fig, focus_txt):
    """Per-pair learner result at the committed config (only FS still free)."""
    sub = df[RESULT_COLS].sort_values(["pair", "fs"]).round(4)
    sub.to_csv(path_csv, index=False)
    print(f"saved {path_csv}")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    x = np.arange(len(PAIR_ORDER))
    panels = [("d_cc", "p_naive_vs_expert", "delta-CC (expert - naive)"),
              ("ifi_w3", "p_ifi_w3", "IFI (|lag| <= 3 bins)")]
    for ax, (mkey, pkey, label) in zip(axes, panels):
        for i, (fs, col) in enumerate([("excl", "tab:blue"),
                                       ("incl", "tab:orange")]):
            d = df[df["fs"] == fs].set_index("pair")
            vals = [d[mkey].get(p, np.nan) for p in PAIR_ORDER]
            ps = [d[pkey].get(p, np.nan) for p in PAIR_ORDER]
            ax.bar(x + (i - 0.5) * 0.4, vals, 0.38, color=col,
                   label=f"FS-{fs}", alpha=0.85)
            for xi, v, pv in zip(x, vals, ps):
                if np.isfinite(pv) and pv < ALPHA and np.isfinite(v):
                    ax.text(xi + (i - 0.5) * 0.4,
                            v + (0.015 if v >= 0 else -0.05), "*",
                            ha="center", fontsize=14)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(PAIR_ORDER, rotation=45, ha="right", fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"Committed config [{focus_txt}] -- per-pair learner result; "
                 f"'*' = p<0.05 (n = significant subspace dims)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path_fig, dpi=150)
    plt.close(fig)
    print(f"saved {path_fig}")


def main():
    csv = config.FIGURES_DIR / "sweep_summary_spatial.csv"
    if not csv.exists():
        print(f"{csv} not found -- run summarise_sweep.py first")
        return
    df = pd.read_csv(csv)
    full = len(df)
    for col, val in FOCUS.items():
        df = df[df[col] == val]
    axes = [h for h in HYPER if h not in FOCUS]
    focus_txt = ", ".join(f"{k}={v}" for k, v in FOCUS.items())
    print(f"loaded {full} rows; focus [{focus_txt}] -> {df['tag'].nunique()} "
          f"configs. Free hyperparameters: {axes}\n")

    if len(axes) >= 2:
        for metric, label in METRICS.items():
            parcoords_grid(df, metric, f"{label}  [focus: {focus_txt}]",
                           config.FIGURES_DIR
                           / f"sweep_parcoords_focus_{metric}.png", axes)
            enrichment(df, metric, config.FIGURES_DIR
                       / f"sweep_enrichment_focus_{metric}.csv", axes)
        return

    print("Focus has converged to a single config (only FS free) -- "
          "reporting the committed-config result directly.\n")
    committed_results(df, config.FIGURES_DIR / "sweep_committed_results.csv",
                      config.FIGURES_DIR / "sweep_committed_results.png",
                      focus_txt)
    for metric, label in METRICS.items():
        print(f"\n  [{metric}]  per pair  (FS-excl / FS-incl):")
        for pair in PAIR_ORDER:
            d = df[df["pair"] == pair].set_index("fs")
            cells = []
            for fs in ("excl", "incl"):
                v = d[metric].get(fs, np.nan) if fs in d.index else np.nan
                cells.append("  n/a " if not np.isfinite(v) else f"{v:.3f}")
            flag = " <--" if any(c not in ("  n/a ",)
                                 and float(c) < ALPHA for c in cells) else ""
            print(f"    {pair:>9}   {cells[0]} / {cells[1]}{flag}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
