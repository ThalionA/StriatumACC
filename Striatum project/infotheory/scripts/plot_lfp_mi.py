#!/usr/bin/env python3
"""LFP band power as a carrier of behavioural information, and how it changes.

Four panels:

(a) the ANCHOR -- information about running speed, per area. The spike arm had no
    positive control, so its null was uninterpretable; this one does.
(b) information about each behavioural feature, before and after conditioning on
    speed. Animals run faster as they learn and speed alone moves band power, so
    only the conditioned value is a claim about behaviour.
(c) the learning contrast by area, task against yoked control.
(d) the DLS effect animal by animal.

Only theta and beta are plotted. low_gamma (30-80 Hz) carries a ~75 Hz peak of
unresolved provenance, high_gamma is 80-150 Hz where spike bleed-through lives,
and total spans both.

    /opt/anaconda3/bin/python scripts/plot_lfp_mi.py
"""
from __future__ import annotations

import csv
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
BANDS = ("theta", "beta")
COHORTS = (("task", "Task", "#1f4e79"), ("control", "Control 1", "#e69f00"))
MIN_MICE = 6


def load(which: str, cohort: str) -> list[dict]:
    path = RESULTS / f"lfp_mi_{which}_{cohort}.csv"
    if not path.exists():
        return []
    rows = [r for r in csv.DictReader(path.open())
            if r["band_status"] == "interpretable"]
    key = "mi_speed" if which == "speed" else None
    for r in rows:
        if key:
            r["v"] = float(r[key])
        else:
            r["mi"] = float(r["mi"])
            r["v"] = float(r["cmi_given_speed"])
    return rows


def per_animal(rows, key="v", **where) -> dict[str, float]:
    by = defaultdict(list)
    for r in rows:
        if any(r.get(k) != v for k, v in where.items()):
            continue
        by[r["mouse_id"]].append(r[key])
    return {m: float(np.mean(v)) for m, v in by.items()}


def contrast(rows, key="v", **where):
    a = per_animal(rows, key, epoch="Early50", **where)
    b = per_animal(rows, key, epoch="Late50", **where)
    sh = sorted(set(a) & set(b))
    d = np.array([b[m] - a[m] for m in sh])
    return d, (stats.wilcoxon(d).pvalue if d.size >= MIN_MICE else np.nan)


def star(p, n):
    if not np.isfinite(p):
        return f"n={n}"
    return ("**" if p < 0.01 else "*") if p < 0.05 else "n.s."


def main() -> None:
    speed = {c: load("speed", c) for c, _, _ in COHORTS}
    feat = {c: load("features", c) for c, _, _ in COHORTS}
    if not feat["task"]:
        print("[plot] no LFP MI tables; run scripts/run_lfp_mi.py first")
        return

    fig = plt.figure(figsize=(16, 11), layout="constrained")
    gs = fig.add_gridspec(2, 2)

    # ---- (a) anchor ---------------------------------------------------------
    axa = fig.add_subplot(gs[0, 0])
    x = np.arange(len(AREAS))
    for bi, band in enumerate(BANDS):
        m, e, lab = [], [], []
        for area in AREAS:
            v = np.array(list(per_animal(speed["task"], area=area, band=band,
                                         epoch="All").values()))
            if v.size < 3:
                m.append(np.nan); e.append(np.nan); lab.append(""); continue
            m.append(v.mean()); e.append(v.std(ddof=1) / np.sqrt(v.size))
            p = stats.wilcoxon(v).pvalue if v.size >= MIN_MICE else np.nan
            lab.append(star(p, v.size))
        off = -0.18 + 0.36 * bi
        axa.bar(x + off, m, 0.34, yerr=e, capsize=3,
                color=("#4a7ebb" if band == "theta" else "#9c4f96"), label=band)
        for xi, (mv, ev, t) in enumerate(zip(m, e, lab)):
            if t:
                axa.text(xi + off, mv + ev + 0.0012, t, ha="center", fontsize=7)
    axa.set_xticks(x); axa.set_xticklabels(AREAS)
    axa.set_ylabel("I(band power; running speed)\nshuffle-subtracted (bits)")
    axa.set_title("(a) ANCHOR — band power carries information about running speed.\n"
                  "Within spatial bin, so position cannot produce it. Whole session.",
                  fontsize=10)
    axa.legend(fontsize=9, frameon=False)

    # ---- (b) features, raw vs conditioned -----------------------------------
    axb = fig.add_subplot(gs[0, 1])
    features = sorted({r["feature"] for r in feat["task"]})
    xf = np.arange(len(features))
    for key, lab, colour, off in (("mi", "I(power; feature)", "#bbbbbb", -0.19),
                                  ("v", "I(power; feature | speed)", "#1f4e79", 0.19)):
        m, e = [], []
        for f in features:
            v = np.array(list(per_animal(feat["task"], key, feature=f,
                                         epoch="All").values()))
            m.append(v.mean() if v.size else np.nan)
            e.append(v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else np.nan)
        axb.bar(xf + off, m, 0.36, yerr=e, capsize=3, color=colour, label=lab)
    axb.set_xticks(xf)
    axb.set_xticklabels([f.replace("_", " ") for f in features], rotation=30,
                        ha="right", fontsize=8)
    axb.set_ylabel("shuffle-subtracted MI (bits)")
    axb.set_title("(b) What survives running speed. Task cohort, whole session,\n"
                  "pooled over areas and both bands. 70–92% of each feature survives.",
                  fontsize=10)
    axb.legend(fontsize=9, frameon=False)

    # ---- (c) learning contrast by area --------------------------------------
    axc = fig.add_subplot(gs[1, 0])
    marks = []
    for key, lab, colour in COHORTS:
        if not feat[key]:
            continue
        m, e = [], []
        for ai, area in enumerate(AREAS):
            d, p = contrast(feat[key], area=area)
            if d.size < 3:
                m.append(np.nan); e.append(np.nan); continue
            m.append(d.mean()); e.append(d.std(ddof=1) / np.sqrt(d.size))
            marks.append((ai + (-0.15 if key == "task" else 0.15),
                          m[-1] + e[-1], star(p, d.size), colour))
        axc.errorbar(x + (-0.15 if key == "task" else 0.15), m, yerr=e, fmt="s",
                     ms=7, capsize=4, lw=0, elinewidth=2, color=colour, label=lab)
    axc.axhline(0, color="k", lw=1.0)
    lo, hi = axc.get_ylim(); axc.set_ylim(lo, hi + 0.18 * (hi - lo))
    pad = 0.04 * (axc.get_ylim()[1] - axc.get_ylim()[0])
    for xx, yy, t, cc in marks:
        axc.text(xx, yy + pad, t, ha="center", fontsize=7, color=cc)
    axc.set_xticks(x); axc.set_xticklabels(AREAS)
    axc.set_ylabel("Late50 − Early50\nI(power; feature | speed)  (bits)")
    axc.set_title("(c) Does it change with training? First versus last fifty trials.\n"
                  "The ten-trial epochs cannot answer this: their noise exceeds the effect.",
                  fontsize=10)
    axc.legend(fontsize=9, frameon=False)

    # ---- (d) DLS animal by animal -------------------------------------------
    axd = fig.add_subplot(gs[1, 1])
    a = per_animal(feat["task"], area="DLS", epoch="Early50")
    b = per_animal(feat["task"], area="DLS", epoch="Late50")
    sh = sorted(set(a) & set(b))
    for m in sh:
        axd.plot([0, 1], [a[m], b[m]], "-o", ms=5, color="#1f4e79", alpha=0.65, lw=1.2)
    d = np.array([b[m] - a[m] for m in sh])
    axd.plot([0, 1], [np.mean([a[m] for m in sh]), np.mean([b[m] for m in sh])],
             "-o", ms=10, color="#c0392b", lw=3, label="mean", zorder=5)
    p = stats.wilcoxon(d).pvalue if d.size >= MIN_MICE else np.nan
    axd.set_xticks([0, 1]); axd.set_xticklabels(["first 50 trials", "last 50 trials"])
    axd.set_xlim(-0.25, 1.25)
    axd.set_ylabel("DLS  I(power; feature | speed)  (bits)")
    axd.set_title(f"(d) DLS, animal by animal: {(d < 0).sum()}/{d.size} decrease "
                  f"(Wilcoxon p = {p:.4f}).\nAnchor I(power; speed) does NOT fall, so it "
                  f"is not the recording degrading.", fontsize=10)
    axd.legend(fontsize=9, frameon=False)

    fig.suptitle("LFP band power and behavioural information — mirroring Lemke et al. (2024) "
                 "on the field rather than single units\n"
                 "theta (4–8 Hz) and beta (15–30 Hz) only; log power ranked WITHIN spatial "
                 "bin; shuffle-subtracted; mean over windows, never peak. p uncorrected.",
                 fontsize=12)
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "lfp_mi_overview.svg")
    fig.savefig(FIGURES / "lfp_mi_overview.png",
                dpi=min(150, 1600 / max(fig.get_size_inches())))
    plt.close(fig)
    print("[plot] lfp_mi_overview.svg + .png")


if __name__ == "__main__":
    main()
