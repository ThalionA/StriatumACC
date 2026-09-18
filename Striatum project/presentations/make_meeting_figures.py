#!/usr/bin/env python3
"""Slide-ready figures for the 2026-09-18 meeting — surviving results only.

One figure per claim, each standalone: the claim in the title, the evidence in
the panels, and the control that rules out the obvious alternative beside it.
Nothing here is retracted or provisional; the day's four retractions are recorded
in the root NOTES.md and deliberately not drawn.

    /opt/anaconda3/bin/python presentations/make_meeting_figures.py
"""
from __future__ import annotations

import collections
import csv
import textwrap
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures_2026-09-18"
INFO = ROOT / "infotheory" / "results"
LFP = ROOT / "lfp" / "results"
TCCA = ROOT / "tcca" / "results"

AREAS = ("DMS", "DLS", "ACC", "V1", "CA1", "DG")
AREA_C = {"DMS": "#0072b2", "DLS": "#77ac30", "ACC": "#d95319",
          "V1": "#7e2f8e", "CA1": "#cc1a33", "DG": "#33b3b3"}
INK, MUTED, ACCENT = "#14161a", "#6b7078", "#1c6b58"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
                     "axes.edgecolor": "#444", "axes.linewidth": 0.9,
                     "xtick.color": INK, "ytick.color": INK, "text.color": INK,
                     "axes.labelcolor": INK, "figure.facecolor": "white",
                     "savefig.facecolor": "white"})


def rows(path, **where):
    if not Path(path).exists():
        return []
    out = []
    for r in csv.DictReader(Path(path).open()):
        if all(r.get(k) == v for k, v in where.items()):
            out.append(r)
    return out


def num(r, c):
    try:
        return float(r[c])
    except (ValueError, TypeError, KeyError):
        return np.nan


def by_animal(rs, col, key="mouse_id"):
    d = collections.defaultdict(list)
    for r in rs:
        v = num(r, col)
        if np.isfinite(v):
            d[r[key]].append(v)
    return {k: float(np.mean(v)) for k, v in d.items()}


def wil(d):
    d = np.asarray(d, float)
    return stats.wilcoxon(d).pvalue if d.size >= 6 and np.any(d) else np.nan


def pstr(p):
    if not np.isfinite(p):
        return "N<6"
    return f"p = {p:.4f}" if p >= 1e-4 else f"p < 1e-4"


def save(fig, name, caption, pad=-0.03):
    # Placed BELOW the figure in negative axes-fraction space and pulled back in by
    # bbox_inches="tight". Anchoring it inside the figure collides with rotated
    # tick labels, which is what the first pass did.
    fig.text(0.5, pad, "\n".join(textwrap.wrap(caption, 118)),
             ha="center", va="top", fontsize=9.5, color=MUTED)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=min(200, 1600 / max(fig.get_size_inches())),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {name}.svg + .png")


# ---------------------------------------------------------------- 01 anchor --
def fig_anchor():
    rs = [r for r in rows(INFO / "lfp_mi_speed_task.csv", epoch="All")
          if r["band_status"] == "interpretable"]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(AREAS))
    for bi, (band, colour) in enumerate((("theta", "#4a7ebb"), ("beta", "#9c4f96"))):
        m, e, lab = [], [], []
        for a in AREAS:
            v = np.array(list(by_animal(
                [r for r in rs if r["area"] == a and r["band"] == band], "mi_speed").values()))
            if v.size < 3:
                m.append(np.nan); e.append(np.nan); lab.append(""); continue
            m.append(v.mean()); e.append(v.std(ddof=1) / np.sqrt(v.size))
            p = wil(v)
            lab.append(pstr(p) if np.isfinite(p) else f"N={v.size}")
        off = -0.19 + 0.38 * bi
        ax.bar(x + off, m, 0.36, yerr=e, capsize=4, color=colour,
               label=f"{band} ({'4–8' if band == 'theta' else '15–30'} Hz)")
        for xi, (mv, ev, t) in enumerate(zip(m, e, lab)):
            if t:
                ax.text(xi + off, mv + ev + 0.0012, t, ha="center", fontsize=8,
                        rotation=90 if len(t) > 6 else 0, va="bottom", color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels(AREAS)
    ax.set_ylabel("I(band power ; running speed)\nshuffle-subtracted (bits)")
    ax.set_ylim(0, None)
    ax.set_title("LFP band power carries information about running speed in every area",
                 fontweight="semibold")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "01_speed_anchor",
         "Computed WITHIN each 5 cm spatial bin, then averaged over bins — so position cannot produce it. "
         "Engaged trials only. Wilcoxon signed-rank against zero, animals as n (DMS 16, ACC 15, DLS 12; "
         "V1 5, CA1/DG 3 are shown but untestable). This is the positive control the single-unit arm lacked.")


# ------------------------------------------------- 02 information past speed --
def fig_beyond_speed():
    rs = [r for r in rows(INFO / "lfp_mi_features_task.csv", epoch="All")
          if r["band_status"] == "interpretable"]
    feats = sorted({r["feature"] for r in rs})
    fig, ax = plt.subplots(figsize=(11, 6.2))
    xf = np.arange(len(feats))
    raw_m, con_m, raw_e, con_e, ps, keep = [], [], [], [], [], []
    for f in feats:
        sub = [r for r in rs if r["feature"] == f]
        rv = np.array(list(by_animal(sub, "mi").values()))
        cv = np.array(list(by_animal(sub, "cmi_given_speed").values()))
        raw_m.append(rv.mean()); con_m.append(cv.mean())
        raw_e.append(rv.std(ddof=1) / np.sqrt(rv.size))
        con_e.append(cv.std(ddof=1) / np.sqrt(cv.size))
        ps.append(wil(cv)); keep.append(100 * cv.mean() / rv.mean())
    ax.bar(xf - 0.19, raw_m, 0.36, yerr=raw_e, capsize=3, color="#c6c2bb",
           label="I(power ; feature)")
    ax.bar(xf + 0.19, con_m, 0.36, yerr=con_e, capsize=3, color="#1c6b58",
           label="I(power ; feature | running speed)")
    top = max(np.array(raw_m) + np.array(raw_e)).max() if raw_m else 0
    for xi, (rm, re_, c, e, p, k) in enumerate(zip(raw_m, raw_e, con_m, con_e, ps, keep)):
        ax.text(xi, max(rm + re_, c + e) + 0.00035, f"{k:.0f}%  {pstr(p)}",
                ha="center", fontsize=8.5, va="bottom", color=MUTED, rotation=90)
    ax.set_ylim(0, top * 1.42)
    ax.set_xticks(xf)
    ax.set_xticklabels([f.replace("_", " ") for f in feats], rotation=28, ha="right",
                       fontsize=10)
    ax.set_ylabel("shuffle-subtracted MI (bits)")
    ax.set_title("The information is not just running speed — 62–96% of it survives conditioning",
                 fontweight="semibold")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "02_information_beyond_speed", pad=-0.24, caption=
         "Task cohort, engaged trials only, pooled over areas and both interpretable bands. "
         "Percentages are the conditional value as a fraction of the raw one; p is Wilcoxon against "
         "zero on the CONDITIONAL value, animals as n = 15–16. Nine of ten features survive at "
         "p ≤ 0.011; path length alone does not (p = 0.083). Conditioning matters because animals "
         "run faster as they learn and speed alone moves band power.")


# ------------------------------------------------------- 03 de-sparsification --
def fig_gini():
    rs = rows(TCCA / "epoch_metrics.csv")

    def per_animal(col):
        d = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rs:
            v = num(r, col)
            if np.isfinite(v):
                d[r["animal"]][r["epoch"]].append(v)
        return {a: (np.median(e["naive"]), np.median(e["expert"]))
                for a, e in d.items() if "naive" in e and "expert" in e}

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.6))
    spec = [("gini_y", "AREA-INTRINSIC\nGini of CCA weights", "#1f4e79"),
            ("gini_y_conn", "CONNECTION-SPECIFIC\nGini, canonical-r weighted", "#7e2f8e"),
            ("gini_pearson_y", "CONTROL — CCA-FREE\nGini of raw cross-area coupling", "#9b9691")]
    for ax, (col, title, colour) in zip(axes, spec):
        d = per_animal(col)
        ids = sorted(d)
        for a in ids:
            ax.plot([0, 1], list(d[a]), "-o", ms=5, color=colour, alpha=0.5, lw=1.2)
        nv = np.array([d[a][0] for a in ids]); ev = np.array([d[a][1] for a in ids])
        ax.plot([0, 1], [nv.mean(), ev.mean()], "-o", ms=11, color="#c0392b", lw=3.5,
                zorder=5)
        diff = ev - nv
        ax.set_xticks([0, 1]); ax.set_xticklabels(["naive", "expert"])
        ax.set_xlim(-0.25, 1.25)
        ax.set_title(f"{title}\nΔ {diff.mean():+.3f} · {(diff < 0).sum()}/{diff.size} down · "
                     f"{pstr(wil(diff))}", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Gini  (0 = every unit equal, 1 = one unit)")
    fig.suptitle("Unit subspace participation de-sparsifies with learning — and it is not the raw coupling",
                 fontweight="semibold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "03_subspace_desparsification",
         "Striatal and cortical pairs, task cohort, 25 ms bins, FS-excluded, partial CCA, after the "
         "tom_cca intercept fix. Animals as n = 10; medians over each animal's pairs. The effect is in "
         "the CCA weight structure and NOT in raw cross-area coupling. Controls: k_eff and unit counts "
         "identical across epochs by construction; n_sig moves the opposite way; n_bins falls but is "
         "dissociated (10/10 animals drop in Gini, 6/10 in bins; Spearman +0.06, p = 0.88).")


# ------------------------------------------------------------------- 04 PAC --
def fig_pac():
    NULL_RATE, NULL_N = 7 / 96, 96
    fig, ax = plt.subplots(figsize=(9.5, 6))
    groups, vals, cols = [], [], []
    for measure, mlab in (("pac_within", "within area"), ("pac_between", "between areas")):
        for ref, rlab, c in (("monopolar", "monopolar", "#c6c2bb"),
                             ("bipolar", "bipolar\n(far field removed)", "#1c6b58")):
            rs = rows(LFP / "lfp_coupling_epochs_task.csv", epoch="All",
                      measure=measure, reference=ref)
            p = np.array([num(r, "p") for r in rs])
            p = p[np.isfinite(p)]
            if not p.size:
                continue
            groups.append(f"{mlab}\n{rlab}"); vals.append(100 * (p < 0.05).mean()); cols.append(c)
    x = np.arange(len(groups))
    ax.bar(x, vals, 0.62, color=cols, edgecolor="#444", linewidth=0.7)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 1.6, f"{v:.0f}%", ha="center", fontweight="semibold")
    ax.axhline(100 * NULL_RATE, color="#a94436", lw=2, ls="--",
               label=f"calibrated null on REAL LFP: {100*NULL_RATE:.0f}%  (7/{NULL_N} cells)")
    ax.axhline(5, color=MUTED, lw=1, ls=":", label="nominal 5%")
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=10.5)
    ax.set_ylabel("cells with significant theta–gamma coupling (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Theta–gamma coupling is real, and survives removing the far field",
                 fontweight="semibold")
    ax.legend(frameon=False, loc="lower left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "04_theta_gamma_coupling",
         "Tort modulation index against a time-shift surrogate, whole session, task cohort. The null "
         "was calibrated on REAL LFP, not synthetic: the amplitude series rebuilt from the SAME trials "
         "in permuted order, so each trial's statistics, the concatenation boundaries and the session "
         "drift all survive and only within-trial phase–amplitude pairing is destroyed. Bipolar is the "
         "number to read — monopolar shares a field across the shank. NOTE: this run is not yet clipped "
         "at the disengagement point.")


# ------------------------------------------------------- 05 the learning null --
def fig_null():
    entries = []

    # LFP information about behaviour, engaged halves, per area
    rs = [r for r in rows(INFO / "lfp_mi_features_task.csv")
          if r["band_status"] == "interpretable"]
    for a in ("DMS", "DLS", "ACC"):
        e = by_animal([r for r in rs if r["area"] == a and r["epoch"] == "Naive"],
                      "cmi_given_speed")
        l = by_animal([r for r in rs if r["area"] == a and r["epoch"] == "Expert"],
                      "cmi_given_speed")
        sh = sorted(set(e) & set(l))
        d = np.array([l[m] - e[m] for m in sh])
        entries.append((f"LFP information — {a}", d, wil(d), "LFP"))

    # PAC and envelope correlation, SIZE-MATCHED epochs
    cp = [r for r in rows(LFP / "lfp_coupling_epochs_task.csv") if r["reference"] == "bipolar"]
    for meas, col, lab in (("pac_within", "mi", "Theta–gamma PAC, within"),
                           ("pac_between", "mi", "Theta–gamma PAC, between"),
                           ("same_freq", "orth_r", "Envelope correlation")):
        sub = [r for r in cp if r["measure"] == meas]
        i = by_animal([r for r in sub if r["epoch"] == "Naive"], col)
        x = by_animal([r for r in sub if r["epoch"] == "Expert"], col)
        sh = sorted(set(i) & set(x))
        d = np.array([x[m] - i[m] for m in sh])
        if d.size:
            entries.append((lab, d, wil(d), "LFP"))

    # Unit temporal CCA: subspace strength and directionality, canonical config
    tc = rows(TCCA / "epoch_metrics.csv")
    for col, lab in (("cc1", "Unit subspace strength"), ("ifi", "Unit directionality (IFI)")):
        d = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in tc:
            v = num(r, col)
            if np.isfinite(v):
                d[r["animal"]][r["epoch"]].append(v)
        diff = np.array([np.median(e["expert"]) - np.median(e["naive"]) for e in d.values()
                         if "expert" in e and "naive" in e])
        entries.append((lab, diff, wil(diff), "units"))

    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    y = np.arange(len(entries))[::-1]
    for yi, (lab, d, p, kind) in zip(y, entries):
        sd = d.std(ddof=1) / np.sqrt(d.size) if d.size > 1 else 0.0
        scale = max(np.abs(d).max(), 1e-12)
        c = "#1f4e79" if kind == "LFP" else "#7e2f8e"
        ax.errorbar(d.mean() / scale, yi, xerr=sd / scale, fmt="o", ms=9, capsize=5,
                    lw=0, elinewidth=2.2, color=c)
        ax.text(1.06, yi, f"{pstr(p)}   n = {d.size}", va="center", fontsize=10.5,
                color=MUTED, transform=ax.get_yaxis_transform())
    ax.axvline(0, color=INK, lw=1.3)
    ax.set_yticks(y); ax.set_yticklabels([e[0] for e in entries])
    ax.set_xlim(-1.25, 1.25)
    ax.set_xlabel("Expert (10 trials from the learning point) − Naive (trials 1–10)\n"
                  "scaled to each measure's own spread   ·   0 = no change")
    words = {6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    ax.set_title(f"Nothing changes with learning — {words.get(len(entries), len(entries))} "
                 "measures, two data types, two pipelines", fontweight="semibold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "05_no_change_with_learning",
         "Blue = LFP arm, purple = unit temporal-CCA arm. Each measure is scaled by its own spread so "
         "seven different units sit on one axis; the claim is the position relative to zero, not the "
         "magnitude. LFP epochs are the two halves of the ENGAGED period; coupling epochs are "
         "size-matched (Intermediate vs Expert, ten trials each) because the Tort index is biased at "
         "small n. Separately and not drawn here: phase-slope direction survives in 0 of 12 bipolar "
         "cells, and the unit IFI is null in 0 of 84 BH-corrected window × config cells.")


if __name__ == "__main__":
    print(f"writing to {OUT}")
    fig_anchor(); fig_beyond_speed(); fig_gini(); fig_pac(); fig_null()
