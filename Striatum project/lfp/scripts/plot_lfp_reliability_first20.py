#!/usr/bin/env python3
"""Trial-to-trial reliability over the FIRST 20 TRIALS, task versus Control 1.

One panel per area, every frequency band within a panel, both cohorts side by
side on a single shared y-scale.

Why this window. Every other reliability figure in this package reports
learning-point-aligned epochs, which the yoked controls do not really have --
they inherit the task cohort's average learning point, so a control "Expert"
window is a matched time window, not a matched level of performance. The first
20 trials of the session are the same stretch of exposure for every animal in
both cohorts, so the comparison needs no alignment assumption at all.

What is plotted. `split_half_r` from the reliability arm: the correlation
between a channel's spatial band-power profile built from odd trials and the
same profile from even trials, Spearman-Brown corrected, median over the area's
channels, then averaged across animals with the ANIMAL as the unit of analysis.
No shuffle or null series is drawn, by request -- the reference here is the
task-versus-control difference, not a distance from chance.

Run from `lfp/` after `run_lfp_arms.py` has been run for both cohorts:
    /opt/anaconda3/bin/python scripts/plot_lfp_reliability_first20.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from striatum_lfp import config, figstyle  # noqa: E402
from striatum_lfp.results_io import hierarchical, load_arms  # noqa: E402

WINDOW = "First 20"
BANDS: tuple[str, ...] = ("theta", "beta", "low_gamma", "high_gamma", "total")
BAND_SHORT = {"theta": "θ", "beta": "β", "low_gamma": "γ low",
              "high_gamma": "γ high", "total": "total"}
COHORTS = [("task", "Task", "#1f4e79"), ("control", "Control 1", "#e69f00")]
VALUE = "split_half_r"


def main() -> None:
    per_cohort = {}
    for key, _, _ in COHORTS:
        rows = [r for r in load_arms("reliability", key) if r.get("window") == WINDOW]
        if not rows:
            print(f"[plot] no '{WINDOW}' rows for cohort {key}. Re-run "
                  f"scripts/run_lfp_arms.py --cohort {key} (the window was added 2026-09-08).")
            return
        per_cohort[key] = hierarchical(rows, ("area", "band"), VALUE)
        n_trials = {int(r["n_window_trials"]) for r in rows}
        print(f"[plot] {key}: {len(rows)} rows, window length {sorted(n_trials)} trials")

    areas = [a for a in figstyle.AREA_ORDER
             if any((a, b) in m for m in per_cohort.values() for b in BANDS)]
    ncol = 3
    nrow = int(np.ceil(len(areas) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.6 * nrow), squeeze=False)

    x = np.arange(len(BANDS))
    width = 0.38
    for i, area in enumerate(areas):
        ax = axes[i // ncol][i % ncol]
        counts = []
        for j, (key, label, colour) in enumerate(COHORTS):
            m = per_cohort[key]
            mu = [m.get((area, b), (np.nan,) * 3)[0] for b in BANDS]
            se = [m.get((area, b), (np.nan,) * 3)[1] for b in BANDS]
            ns = [m.get((area, b), (np.nan, np.nan, 0))[2] for b in BANDS]
            counts.append(f"{label.split()[0].lower()} {max(ns) if ns else 0}")
            ax.bar(x + (j - 0.5) * width, mu, width, yerr=se, capsize=3,
                   color=colour, label=label, edgecolor="none")
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([BAND_SHORT[b] for b in BANDS], fontsize=9)
        ax.set_title(f"{area}  (N: {', '.join(counts)} mice)", fontsize=11,
                     color=figstyle.AREA_COLOUR.get(area, "black"))
        if i % ncol == 0:
            ax.set_ylabel("split-half r of the spatial profile")
        if i == 0:
            ax.legend(fontsize=9, frameon=False, loc="upper left")

    for i in range(len(areas), nrow * ncol):
        axes[i // ncol][i % ncol].axis("off")

    # Match the scale across every panel: the point of the figure is that areas
    # differ, which is unreadable if each panel picks its own limits.
    drawn = [axes[i // ncol][i % ncol] for i in range(len(areas))]
    lo = min(a.get_ylim()[0] for a in drawn)
    hi = max(a.get_ylim()[1] for a in drawn)
    for a in drawn:
        a.set_ylim(lo, hi)

    fig.suptitle(
        "Trial-to-trial reliability of the LFP spatial profile, first 20 trials\n"
        "split-half (interleaved, Spearman-Brown), animal means ± SEM; "
        "no learning-point alignment, shared y-scale", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    figstyle.save_pair(fig, "lfp_reliability_first20_task_vs_control")

    print(f"[plot] y-scale {lo:.2f} to {hi:.2f} on all {len(drawn)} panels")
    for area in areas:
        for b in BANDS:
            t = per_cohort["task"].get((area, b))
            c = per_cohort["control"].get((area, b))
            if t and c:
                print(f"    {area:4s} {b:11s} task {t[0]:+.3f} (n={t[2]})   "
                      f"control {c[0]:+.3f} (n={c[2]})")


if __name__ == "__main__":
    main()
