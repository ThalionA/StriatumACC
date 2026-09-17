#!/usr/bin/env python3
"""What single units tell you about behaviour, and how it changes with learning.

Three panels, mirroring the first arm of Lemke et al. (2024):

(a) Information time course around reward-zone entry, per area, for each feature.
(b) Naive versus Expert, per feature — the learning comparison, on the project's
    standard ten-trial learning epochs.
(c) The same contrast per area, task against yoked control.

Across-epoch contrasts use the MEAN over time windows. The peak is biased upward
by the max over windows (+0.054 bits on independent data) by an amount that grows
as the sample shrinks, so it cannot be compared between epochs.

All values are shuffle-subtracted, so zero means "no more than the bias this bin
count produces at this sample size". Shuffle subtraction removes the sample-size bias
per window; it does not remove the bias in a max TAKEN OVER windows, which is why
the contrasts use the mean. Peaks are tested with a max-statistic
permutation over time windows, not against a single window's null.

    /opt/anaconda3/bin/python scripts/plot_mi.py
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
from scipy import stats  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGURES = Path(__file__).resolve().parents[1] / "figures"
AREAS = ("DMS", "DLS", "ACC", "V1", "CA1", "DG")
AREA_COLOUR = {"DMS": "#0072b2", "DLS": "#77ac30", "ACC": "#d95319",
               "V1": "#7e2f8e", "CA1": "#cc1a33", "DG": "#33b3b3"}
COHORTS = (("task", "Task", "#1f4e79"), ("control", "Control 1", "#e69f00"))
MIN_MICE = 3


def load(name: str, cohort: str) -> list[dict]:
    path = RESULTS / f"{name}_{cohort}.csv"
    if not path.exists():
        return []
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            if k in ("cohort", "epoch", "feature", "area"):
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                r[k] = np.nan
    return rows


def per_animal(rows, *, value: str = "mean_mi_corrected", **where) -> dict[int, float]:
    """{mouse: median over its units}, for one cell of the design.

    Defaults to the MEAN over time windows, not the peak. The peak is the max of
    30 noisy windows, so it is biased upward by an amount that grows as the
    sample shrinks -- +0.054 bits on data with no information at all, and about
    -0.003 bits of spurious contrast for a single missing trial. Comparing peaks
    across epochs measures the trial count.
    """
    by = defaultdict(list)
    for r in rows:
        if any(r.get(k) != v for k, v in where.items()):
            continue
        val = r.get(value, np.nan)
        if np.isfinite(val):
            by[int(r["mouse_id"])].append(val)
    return {m: float(np.median(v)) for m, v in by.items() if v}


def main() -> None:
    units = {c: load("mi_units", c) for c, _, _ in COHORTS}
    tc = {c: load("mi_timecourse", c) for c, _, _ in COHORTS}
    if not units["task"]:
        print("[plot] no MI tables; run scripts/run_mi.py first")
        return
    features = sorted({r["feature"] for r in units["task"]})

    fig = plt.figure(figsize=(17, 12), layout="constrained")
    gs = fig.add_gridspec(3, 1, height_ratios=[1.1, 1.0, 1.0])

    # ---- (a) time course, task, per area, averaged over features ------------
    axa = fig.add_subplot(gs[0])
    for area in AREAS:
        by_t = defaultdict(list)
        for r in tc["task"]:
            if r["area"] != area or r["epoch"] != "All":
                continue
            by_t[r["time_ms"]].append(r["mi_corrected_mean"])
        if len(by_t) < 5:
            continue
        ts = np.array(sorted(by_t))
        m = np.array([np.nanmean(by_t[t]) for t in ts])
        e = np.array([np.nanstd(by_t[t]) / np.sqrt(len(by_t[t])) for t in ts])
        axa.plot(ts, m, "-", color=AREA_COLOUR[area], lw=1.8, label=area)
        axa.fill_between(ts, m - e, m + e, color=AREA_COLOUR[area], alpha=0.18, lw=0)
    axa.axvline(0, color="k", lw=1.2, ls="--")
    axa.axhline(0, color="0.6", lw=0.8, ls=":")
    axa.set_xlabel("time from reward-zone entry (ms)")
    axa.set_ylabel("shuffle-subtracted MI (bits)")
    axa.set_title("(a) Information about behaviour around reward-zone entry — task cohort, "
                  "pooled over all ten features. Dashed line = zone entry.", fontsize=11)
    axa.legend(fontsize=9, frameon=False, ncol=6)

    # ---- (b) naive vs expert per feature ------------------------------------
    axb = fig.add_subplot(gs[1])
    x = np.arange(len(features))
    for key, label, colour in COHORTS:
        if not units[key]:
            continue
        m, e, ns = [], [], []
        for f in features:
            n = per_animal(units[key], feature=f, epoch="Naive")
            ex = per_animal(units[key], feature=f, epoch="Expert")
            shared = sorted(set(n) & set(ex))
            d = np.array([ex[k] - n[k] for k in shared])
            if d.size < MIN_MICE:
                m.append(np.nan); e.append(np.nan); ns.append(0); continue
            m.append(d.mean()); e.append(d.std(ddof=1) / np.sqrt(d.size)); ns.append(d.size)
        off = -0.15 if key == "task" else 0.15
        axb.errorbar(x + off, m, yerr=e, fmt="o", ms=7, capsize=4, lw=0,
                     elinewidth=2, color=colour, label=f"{label} (N up to {max(ns)} mice)")
    axb.axhline(0, color="k", lw=1.0)
    axb.set_xticks(x)
    axb.set_xticklabels([f.replace("_", " ") for f in features], rotation=25,
                        ha="right", fontsize=8)
    axb.set_ylabel("Expert − Naive\nshuffle-subtracted MI (bits)")
    axb.set_title("(b) Does the information change with learning? Animal medians over the "
                  "project's ten-trial Naive and Expert epochs, mean over time windows.",
                  fontsize=11)
    axb.legend(fontsize=9, frameon=False)

    # ---- (c) the same contrast per area -------------------------------------
    axc = fig.add_subplot(gs[2])
    xa = np.arange(len(AREAS))
    marks = []
    for key, label, colour in COHORTS:
        if not units[key]:
            continue
        m, e = [], []
        for ai, area in enumerate(AREAS):
            n = per_animal(units[key], area=area, epoch="Naive")
            ex = per_animal(units[key], area=area, epoch="Expert")
            shared = sorted(set(n) & set(ex))
            d = np.array([ex[k] - n[k] for k in shared])
            if d.size < MIN_MICE:
                m.append(np.nan); e.append(np.nan); continue
            m.append(d.mean()); e.append(d.std(ddof=1) / np.sqrt(d.size))
            p = stats.wilcoxon(d).pvalue if d.size >= 6 else np.nan
            mark = ("*" if np.isfinite(p) and p < 0.05
                    else ("n.s." if np.isfinite(p) else f"n={d.size}"))
            marks.append((ai + (-0.15 if key == "task" else 0.15),
                          m[-1] + e[-1], mark, colour))
        off = -0.15 if key == "task" else 0.15
        axc.errorbar(xa + off, m, yerr=e, fmt="s", ms=7, capsize=4, lw=0,
                     elinewidth=2, color=colour, label=label)
    axc.axhline(0, color="k", lw=1.0)
    lo, hi = axc.get_ylim()
    axc.set_ylim(lo, hi + 0.12 * (hi - lo))
    pad = 0.03 * (axc.get_ylim()[1] - axc.get_ylim()[0])
    for x, y, mark, colour in marks:
        axc.text(x, y + pad, mark, ha="center", fontsize=7, color=colour)
    axc.set_xticks(xa)
    axc.set_xticklabels(AREAS)
    axc.set_ylabel("Expert − Naive\nshuffle-subtracted MI (bits)")
    axc.set_title("(c) The same contrast by area. Wilcoxon against zero where N ≥ 6 mice; "
                  "below that the test cannot reach p < 0.05 and the N is printed instead.",
                  fontsize=11)
    axc.legend(fontsize=9, frameon=False)

    fig.suptitle("Single-unit information about behaviour, mirroring Lemke et al. (2024)\n"
                 "Aligned to reward-zone entry; spikes binarised at 10 ms; features split at "
                 "the median; shuffle-subtracted. Ten-trial learning epochs; p-values "
                 "uncorrected across features.", fontsize=13)
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "mi_overview.svg")
    fig.savefig(FIGURES / "mi_overview.png", dpi=min(150, 1600 / max(fig.get_size_inches())))
    plt.close(fig)
    print("[plot] mi_overview.svg + .png")

    # Uncorrected across features, by request. Ten features were tried, so the
    # smallest p here is not a 5% claim -- read the effect sizes, not the stars.
    print("\n   Expert - Naive, per feature (task cohort, UNCORRECTED p):")
    for f in features:
        n = per_animal(units["task"], feature=f, epoch="Naive")
        ex = per_animal(units["task"], feature=f, epoch="Expert")
        shared = sorted(set(n) & set(ex))
        if len(shared) < MIN_MICE:
            print(f"   {f:30s} too few mice ({len(shared)})")
            continue
        d = np.array([ex[k] - n[k] for k in shared])
        pv = stats.wilcoxon(d).pvalue if d.size >= 6 else np.nan
        verdict = f"p = {pv:.3f}" if np.isfinite(pv) else f"UNDERPOWERED (n={d.size})"
        print(f"   {f:30s} {d.mean():+.5f} +/- "
              f"{d.std(ddof=1) / np.sqrt(d.size):.5f} bits (N={d.size} mice, {verdict})")


if __name__ == "__main__":
    main()
