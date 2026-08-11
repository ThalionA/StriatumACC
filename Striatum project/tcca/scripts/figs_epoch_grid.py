"""Figures for the 2026-08-11 tcca epoch grid (see analyze_epoch_grid.py).

Six figures into figures/ as svg+png pairs (PNG longest side <= 1600 px):

  grid_strength_b25 / _b10 -- held-out CC1 by epoch, per config x pair
                              (grey per-animal lines, coloured median, Wilcoxon p)
  grid_ifi_windows         -- pooled IFI vs integration window, 4 configs per panel
  grid_contrasts           -- matched-cell scatters: plain vs partial, FS, bin size
  grid_gini                -- corrected (partner-dependent) vs retracted Gini by epoch
  grid_lagcurves           -- cohort-median CC1 lag curves per pair x epoch

Data: the tracked epoch_*_{b25,b10}.csv / grid_ifi_windows.csv tables.
Run:  python3 scripts/figs_epoch_grid.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import analyze_epoch_grid as g  # noqa: E402  (inserts src/ on path itself)
from striatum_tcca import lagged, paired_stats  # noqa: E402

FIGDIR = SCRIPTS.parent / "figures"
FIGDIR.mkdir(exist_ok=True)
EPOCHS = ["naive", "intermediate", "expert"]
# Okabe-Ito hues, fixed assignment (pair identity / config identity)
PAIR_COLOR = {"DMS-DLS": "#0072B2", "DMS-ACC": "#D55E00", "DLS-ACC": "#009E73"}
CONFIG_COLOR = {"fsexcl/partial": "#0072B2", "fsincl/partial": "#56B4E9",
                "fsexcl/plain": "#D55E00", "fsincl/plain": "#E69F00"}
EPOCH_COLOR = {"naive": "#9ecae1", "intermediate": "#4292c6", "expert": "#08519c"}


def save(fig, name: str) -> None:
    fig.savefig(FIGDIR / f"{name}.svg", bbox_inches="tight")
    fig.savefig(FIGDIR / f"{name}.png", dpi=115, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{name}.{{svg,png}}")


def fig_strength(tag: str) -> None:
    confs = [c for c in g._configs() if c["tag"] == tag]
    fig, axes = plt.subplots(3, 4, figsize=(12.5, 8.2), sharey="row")
    for j, c in enumerate(confs):
        m = g.load_metrics(c["suffix"])
        for i, pair in enumerate(g.TRIANGLE):
            ax = axes[i, j]
            pa = g.per_animal_cc1(m, pair)
            xs = np.arange(3)
            for _, v in sorted(pa.items()):
                ys = [v.get(e, np.nan) for e in EPOCHS]
                ax.plot(xs, ys, color="0.78", lw=0.8, marker="o", ms=2.2, zorder=1)
            med = [float(np.nanmedian([v[e] for v in pa.values() if e in v]))
                   if any(e in v for v in pa.values()) else np.nan for e in EPOCHS]
            ax.plot(xs, med, color=PAIR_COLOR[pair], lw=2.4, marker="o", ms=5,
                    zorder=3, label="median")
            deltas = [v["expert"] - v["naive"] for v in pa.values()
                      if "expert" in v and "naive" in v]
            _, _, _, p = paired_stats.wilcoxon_signed(np.array(deltas))
            ax.axhline(0, color="0.88", lw=0.8, zorder=0)
            ax.text(0.04, 0.95, f"p={p:.2f}\nn={len(deltas)}",
                    transform=ax.transAxes, fontsize=7.5, va="top",
                    color="0.25")
            ax.set_xticks(xs)
            ax.set_xticklabels(["N", "I", "E"], fontsize=8)
            ax.tick_params(labelsize=8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if i == 0:
                ax.set_title(c["name"].split("/", 1)[1], fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{pair}\nheld-out CC1", fontsize=9)
    bin_ms = g.BINS[tag]
    fig.suptitle(f"Held-out CC1 across learning epochs — {bin_ms} ms bins  "
                 f"(grey: animals; colour: cohort median; p: Wilcoxon naive→expert, "
                 f"animals-as-n)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, f"grid_strength_b{bin_ms}")


def _ifi_window_profiles(suffix: str) -> dict[str, dict[str, np.ndarray]]:
    """{pair: {animal: epoch-mean IFI-by-window vector}}."""
    curves = g.load_curves(suffix)
    out: dict[str, dict[str, np.ndarray]] = {}
    for pair in g.TRIANGLE:
        an_vecs = defaultdict(list)
        for (animal, p, _), (lags, curve) in curves.items():
            if p == pair:
                an_vecs[animal].append(lagged.ifi_by_window(lags, curve))
        out[pair] = {an: np.nanmean(np.vstack(v), axis=0)
                     for an, v in an_vecs.items()}
    return out


def fig_ifi_windows() -> None:
    bh = defaultdict(list)  # (config, pair) -> [window_ms with a BH hit]
    for r in g._rows(g.RESULTS / "grid_ifi_windows.csv"):
        if int(r["bh_exist"]) or int(r["bh_change"]):
            bh[(r["config"], r["pair"])].append(int(r["window_ms"]))
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 8.2), sharey=True)
    for j, tag in enumerate(g.BINS):
        bin_ms = g.BINS[tag]
        for cname, color in CONFIG_COLOR.items():
            fs = cname.startswith("fsincl")
            plain = cname.endswith("plain")
            suffix = g._suffix(fs, plain, tag)
            prof = _ifi_window_profiles(suffix)
            for i, pair in enumerate(g.TRIANGLE):
                ax = axes[i, j]
                vecs = np.vstack(list(prof[pair].values()))
                w_ms = (np.arange(vecs.shape[1]) + 1) * bin_ms
                med = np.nanmedian(vecs, axis=0)
                ax.plot(w_ms, med, color=color, lw=1.8, label=cname)
                if cname == "fsexcl/partial":     # IQR band on the primary config only
                    q1, q3 = np.nanpercentile(vecs, [25, 75], axis=0)
                    ax.fill_between(w_ms, q1, q3, color=color, alpha=0.14, lw=0)
                hits = bh.get((f"b{bin_ms}/{cname}", pair), [])
                if hits:
                    hy = med[[int(h / bin_ms) - 1 for h in hits]]
                    ax.plot(hits, hy, "v", color=color, ms=7, mec="white",
                            mew=0.8, zorder=5)
    for i, pair in enumerate(g.TRIANGLE):
        for j, bin_ms in enumerate(g.BINS.values()):
            ax = axes[i, j]
            ax.axhline(0, color="0.8", lw=0.9, zorder=0)
            ax.set_xlim(0, 260)
            ax.tick_params(labelsize=8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if i == 0:
                ax.set_title(f"{bin_ms} ms bins", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{pair}\nIFI (pooled epochs)", fontsize=9)
            if i == 2:
                ax.set_xlabel("integration half-window (ms)", fontsize=9)
    axes[0, 0].legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
    fig.suptitle("Information-flow index vs integration window — cohort median per "
                 "config (band: IQR, primary config; ▼: BH-surviving cells — only "
                 "DLS-ACC b10/fsincl/plain)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "grid_ifi_windows")


def _scatter(ax, m_a, m_b, xl, yl, title):
    for pair in g.TRIANGLE:
        keys = [k for k in m_a if k in m_b and k[1] == pair]
        xs = [float(m_a[k]["cc1"]) for k in keys]
        ys = [float(m_b[k]["cc1"]) for k in keys]
        ax.plot(xs, ys, "o", ms=4, color=PAIR_COLOR[pair], alpha=0.75,
                mec="white", mew=0.4, label=pair)
    lim = (-0.15, 0.75)
    ax.plot(lim, lim, color="0.8", lw=1, zorder=0)
    ax.set_xlim(lim); ax.set_ylim(lim)
    keys = [k for k in m_a if k in m_b and k[1] in g.TRIANGLE]
    d = np.array([float(m_b[k]["cc1"]) - float(m_a[k]["cc1"]) for k in keys])
    by_an = defaultdict(list)
    for k, v in zip(keys, d):
        by_an[k[0]].append(v)
    _, _, _, p = paired_stats.wilcoxon_signed(
        np.array([np.median(v) for v in by_an.values()]))
    ax.text(0.03, 0.97, f"median Δ {np.median(d):+.3f}\np={p:.3f} "
            f"(n={len(by_an)} animals)", transform=ax.transAxes, fontsize=8,
            va="top", color="0.25")
    ax.set_xlabel(xl, fontsize=9); ax.set_ylabel(yl, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_contrasts() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 9.4))
    panels = [
        (("_b25", False, False), ("_b25", False, True),
         "partial CC1", "plain CC1", "plain vs partial — 25 ms, FS-excl"),
        (("_b10", False, False), ("_b10", False, True),
         "partial CC1", "plain CC1", "plain vs partial — 10 ms, FS-excl"),
        (("_b25", False, False), ("_b25", True, False),
         "FS-excluded CC1", "FS-included CC1", "FS contrast — 25 ms, partial"),
        (("_b25", False, False), ("_b10", False, False),
         "25 ms CC1", "10 ms CC1", "bin size — FS-excl, partial"),
    ]
    for ax, (ka, kb, xl, yl, title) in zip(axes.flat, panels):
        m_a = g.load_metrics(g._suffix(ka[1], ka[2], ka[0]))
        m_b = g.load_metrics(g._suffix(kb[1], kb[2], kb[0]))
        _scatter(ax, m_a, m_b, xl, yl, title)
    axes[0, 0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("Matched-cell CC1 contrasts (points: animal×pair×epoch cells; "
                 "p: Wilcoxon on per-animal medians)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "grid_contrasts")


def fig_gini() -> None:
    rows = g._rows(g.RESULTS / "epoch_metrics_b25.csv")
    cols = [("gini_pearson_x", "partner-DEPENDENT, area X\n(corrected metric)"),
            ("gini_pearson_y", "partner-DEPENDENT, area Y\n(corrected metric)"),
            ("gini_x", "partner-INVARIANT, area X\n(retracted metric — shown for comparison)")]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), sharey=True)
    for ax, (col, label) in zip(axes, cols):
        per = defaultdict(dict)
        for r in rows:
            if r["pair"] in g.TRIANGLE:
                per[(r["animal"], r["pair"])][r["epoch"]] = float(r[col])
        xs = np.arange(3)
        for v in per.values():
            ax.plot(xs, [v.get(e, np.nan) for e in EPOCHS], color="0.8",
                    lw=0.7, zorder=1)
        med = [float(np.nanmedian([v[e] for v in per.values() if e in v]))
               for e in EPOCHS]
        ax.plot(xs, med, color="#0072B2", lw=2.4, marker="o", ms=5, zorder=3)
        by_an = defaultdict(list)
        for (an, _), v in per.items():
            if "naive" in v and "expert" in v:
                by_an[an].append(v["expert"] - v["naive"])
        _, _, _, p = paired_stats.wilcoxon_signed(
            np.array([np.median(v) for v in by_an.values()]))
        ax.text(0.04, 0.95, f"naive→expert p={p:.2f}", transform=ax.transAxes,
                fontsize=8.5, va="top", color="0.25")
        ax.set_xticks(xs)
        ax.set_xticklabels(["naive", "interm.", "expert"], fontsize=9)
        ax.set_title(label, fontsize=9.5)
        ax.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Gini coefficient", fontsize=9)
    fig.suptitle("Subspace-sparsity (Gini) across epochs — 25 ms committed config, "
                 "striatal triangle (grey: animal×pair; blue: median)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "grid_gini")


def fig_lagcurves() -> None:
    curves = g.load_curves("_b25")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), sharey=True)
    for ax, pair in zip(axes, g.TRIANGLE):
        for epoch in EPOCHS:
            mats = [c for (an, p, e), (lg, c) in curves.items()
                    if p == pair and e == epoch]
            lags_ms = None
            for (an, p, e), (lg, c) in curves.items():
                if p == pair and e == epoch:
                    lags_ms = lg * 25
                    break
            if not mats or lags_ms is None:
                continue
            med = np.nanmedian(np.vstack(mats), axis=0)
            ax.plot(lags_ms, med, color=EPOCH_COLOR[epoch], lw=1.9, label=epoch)
        ax.axvline(0, color="0.85", lw=0.9, zorder=0)
        ax.axhline(0, color="0.9", lw=0.8, zorder=0)
        ax.set_title(pair, fontsize=10, color=PAIR_COLOR[pair])
        ax.set_xlabel("lag (ms; +ve = first area leads)", fontsize=9)
        ax.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("in-sample CC1 at lag", fontsize=9)
    axes[2].legend(fontsize=8.5, frameon=False)
    fig.suptitle("CC1 lag curves — cohort median, 25 ms partial FS-excl "
                 "(symmetric, zero-lag peaked: no directional flow)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "grid_lagcurves")


if __name__ == "__main__":
    fig_strength("_b25")
    fig_strength("_b10")
    fig_ifi_windows()
    fig_contrasts()
    fig_gini()
    fig_lagcurves()
