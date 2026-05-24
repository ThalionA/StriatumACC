"""Parallel-coordinate plots over the spatial sweep.

Reads figures/sweep_summary_spatial.csv and, for each chosen p-value column,
draws a parallel-coordinate plot (one line per config x pair, significant
lines highlighted) and writes a per-hyperparameter enrichment table -- the
fraction of cells reaching p<0.05 for each hyperparameter value, against the
~5 % chance baseline. A value whose enrichment clearly exceeds 1.0 is one
where significance genuinely concentrates; scattered significance is noise.

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
HYPER = ["pair", "bin", "cca", "fs", "z", "k_rule", "min_units", "lp_consec"]
PAIR_ORDER = [f"{ax}-{ay}" for ax, ay in config.PAIRS]
ORDER = {
    "pair": PAIR_ORDER,
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

# Committed hyperparameter region (round 8): focus the analysis here. The
# remaining axes (pair, fs, k_rule, min_units, lp_consec) stay free.
FOCUS = {"bin": "2.5cm", "cca": "residual", "z": "on"}


def _positions(series, order):
    idx = {v: i for i, v in enumerate(order)}
    return series.map(idx).to_numpy(float) / max(1, len(order) - 1)


def parcoords(df, metric, label, path, axes):
    d = df[df[metric].notna()].copy()
    nlp = -np.log10(d[metric].clip(lower=1e-4))
    sig = d[metric].to_numpy() < ALPHA
    ycols = [_positions(d[h], ORDER[h]) for h in axes]
    span = nlp.max() - nlp.min() + 1e-9
    nlp_norm = ((nlp - nlp.min()) / span).to_numpy()
    Y = np.column_stack(ycols + [nlp_norm])
    x = np.arange(Y.shape[1])
    lines = [np.column_stack([x, Y[i]]) for i in range(Y.shape[0])]

    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.add_collection(LineCollection([lines[i] for i in np.where(~sig)[0]],
                                     colors="0.75", lw=0.3, alpha=0.06))
    hot = np.where(sig)[0]
    if hot.size:
        cmap = plt.cm.plasma
        lc = LineCollection([lines[i] for i in hot],
                            colors=cmap(nlp_norm[hot]), lw=0.8, alpha=0.6)
        ax.add_collection(lc)
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array(nlp.to_numpy())
        fig.colorbar(sm, ax=ax, label="-log10(p)", fraction=0.025, pad=0.02)
    for xi in x:
        ax.axvline(xi, color="0.6", lw=0.8, zorder=0)
    for xi, h in zip(x, axes):
        order = ORDER[h]
        n = max(1, len(order) - 1)
        for j, v in enumerate(order):
            ax.text(xi - 0.05, j / n, str(v), ha="right", va="center",
                    fontsize=7)
    thr = (-np.log10(ALPHA) - nlp.min()) / span
    ax.plot([x[-1] - 0.12, x[-1] + 0.12], [thr, thr], color="red", lw=1.6)
    ax.text(x[-1] + 0.15, thr, "p=0.05", color="red", fontsize=7, va="center")
    ax.set_xticks(x)
    ax.set_xticklabels(axes + ["-log10(p)"], fontsize=9)
    ax.set_yticks([])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-1.0, x[-1] + 0.9)
    fig.suptitle(f"Spatial sweep parallel coordinates -- {label}\n"
                 f"{len(d)} (config x pair) cells; {int(sig.sum())} reach "
                 f"p<0.05 (chance ~{0.05 * len(d):.0f}); coloured = significant",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def enrichment(df, metric, path, axes):
    d = df[df[metric].notna()]
    base = float((d[metric] < ALPHA).mean())
    rows = []
    for h in axes:
        for v, g in d.groupby(h):
            rate = float((g[metric] < ALPHA).mean())
            rows.append({"hyperparam": h, "value": v, "n_cells": len(g),
                         "frac_p<0.05": round(rate, 4),
                         "enrichment_vs_chance": round(rate / base, 2)
                         if base else np.nan})
    out = pd.DataFrame(rows).sort_values("frac_p<0.05", ascending=False)
    out.to_csv(path, index=False)
    print(f"saved {path}  (baseline p<0.05 rate = {base:.3f})")
    return out, base


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
    print(f"loaded {full} rows; focused on [{focus_txt}] -> {len(df)} rows "
          f"({df['tag'].nunique()} configs). Free axes: {axes}\n")
    for metric, label in METRICS.items():
        parcoords(df, metric, f"{label}  [focus: {focus_txt}]",
                  config.FIGURES_DIR / f"sweep_parcoords_focus_{metric}.png",
                  axes)
        out, base = enrichment(
            df, metric,
            config.FIGURES_DIR / f"sweep_enrichment_focus_{metric}.csv", axes)
        top = out[out["enrichment_vs_chance"] > 1.0].head(8)
        print(f"  [{metric}] most-enriched values (baseline {base:.3f}):")
        for _, r in top.iterrows():
            print(f"    {r['hyperparam']:>10} = {str(r['value']):<10} "
                  f"frac={r['frac_p<0.05']:.3f}  "
                  f"{r['enrichment_vs_chance']:.2f}x chance  (n={r['n_cells']})")
        print()


if __name__ == "__main__":
    main()
