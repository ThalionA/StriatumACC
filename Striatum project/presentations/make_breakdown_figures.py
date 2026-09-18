#!/usr/bin/env python3
"""The same five claims, cut by AREA x COHORT x BAND.

Bands exist only for the LFP measures (theta 4-8, beta 15-30); the unit
temporal-CCA arm has none, and its "area" is an area PAIR. Every panel keeps
task and control side by side.

    /opt/anaconda3/bin/python presentations/make_breakdown_figures.py
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
INFO, LFP, TCCA = ROOT / "infotheory/results", ROOT / "lfp/results", ROOT / "tcca/results"
AREAS = ("DMS", "DLS", "ACC", "V1", "CA1", "DG")
BANDS = ("theta", "beta")
TASK_C, CTRL_C = "#1f4e79", "#e69f00"
MUTED = "#6b7078"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})


def rows(path, **w):
    if not Path(path).exists():
        return []
    return [r for r in csv.DictReader(Path(path).open())
            if all(r.get(k) == v for k, v in w.items())]


def num(r, c):
    try:
        return float(r[c])
    except (ValueError, TypeError, KeyError):
        return np.nan


def animals(rs, col, key="mouse_id"):
    d = collections.defaultdict(list)
    for r in rs:
        v = num(r, col)
        if np.isfinite(v):
            d[r[key]].append(v)
    return np.array([np.mean(v) for v in d.values()])


def between(a, b):
    return (stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
            if min(a.size, b.size) >= 3 else np.nan)


def pstr(p):
    return f"p={p:.3f}" if np.isfinite(p) else "n/a"


def save(fig, name, caption, pad=-0.03):
    fig.text(0.5, pad, "\n".join(textwrap.wrap(caption, 130)), ha="center", va="top",
             fontsize=9, color=MUTED)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=min(200, 1600 / max(fig.get_size_inches())),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {name}")


def paired_area_panel(ax, get, title, ylabel):
    """One panel: x = area, a task bar and a control bar at each, with n and p."""
    x = np.arange(len(AREAS))
    for k, (coh, colour) in enumerate((("task", TASK_C), ("control", CTRL_C))):
        m, e = [], []
        for a in AREAS:
            v = get(coh, a)
            m.append(v.mean() if v.size else np.nan)
            e.append(v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else np.nan)
        ax.bar(x + (-0.2 + 0.4 * k), m, 0.38, yerr=e, capsize=3, color=colour,
               label=coh.capitalize())
    for xi, a in enumerate(AREAS):
        ta, ca = get("task", a), get("control", a)
        if not ta.size or not ca.size:
            continue
        top = max(np.nanmax([ta.mean(), ca.mean()]), 0)
        ax.text(xi, top * 1.12 + 1e-9, f"{pstr(between(ta, ca))}\n{ta.size}v{ca.size}",
                ha="center", fontsize=7.5, color=MUTED, va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(AREAS)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)


# ------------------------------------------------------- B1 anchor by band --
def b1_anchor():
    data = {c: rows(INFO / f"lfp_mi_speed_{c}.csv", epoch="All") for c in ("task", "control")}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    for ax, band in zip(axes, BANDS):
        paired_area_panel(
            ax,
            lambda coh, a, band=band: animals(
                [r for r in data[coh] if r["area"] == a and r["band"] == band
                 and r["band_status"] == "interpretable"], "mi_speed"),
            f"{band} ({'4–8' if band == 'theta' else '15–30'} Hz)",
            "I(band power ; running speed)\nshuffle-subtracted (bits)" if band == "theta" else "")
    axes[0].legend(frameon=False)
    fig.suptitle("ANCHOR — band power vs running speed, by area, band and cohort",
                 fontweight="semibold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "B1_anchor_by_area_band_cohort",
         "Within spatial bin, engaged trials only. Bars are animal means ± s.e.m.; p is a two-sided "
         "Mann-Whitney between cohorts at that area and band; counts under it are task vs control "
         "animals. Beta exceeds theta in every area of both cohorts.")


# ------------------------------------- B2 information beyond speed by band --
def b2_information():
    data = {c: [r for r in rows(INFO / f"lfp_mi_features_{c}.csv", epoch="All")
                if r["band_status"] == "interpretable"] for c in ("task", "control")}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), sharey="row")
    for col, band in enumerate(BANDS):
        for row_i, (key, what) in enumerate((("mi", "RAW  I(power ; feature)"),
                                             ("cmi_given_speed", "CONDITIONED on speed"))):
            paired_area_panel(
                axes[row_i][col],
                lambda coh, a, band=band, key=key: animals(
                    [r for r in data[coh] if r["area"] == a and r["band"] == band], key),
                f"{band} — {what}",
                "shuffle-subtracted MI (bits)" if col == 0 else "")
    axes[0][0].legend(frameon=False)
    fig.suptitle("Information about behaviour, by area, band and cohort — raw (top) and "
                 "speed-conditioned (bottom)", fontweight="semibold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "B2_information_by_area_band_cohort",
         "Pooled over the nine usable behavioural features ('success' is excluded: it is 98% "
         "constant within the engaged period, so no split of it carries information). The BOTTOM "
         "row is the one to read — animals run faster as they learn and speed alone moves band "
         "power. Engaged trials only; animal means ± s.e.m.; Mann-Whitney between cohorts.")


# ----------------------------------------------- B3 Gini by pair and cohort --
def b3_gini():
    t = rows(TCCA / "epoch_metrics.csv")
    c = rows(TCCA / "epoch_metrics_control.csv")
    cols = [("gini_y", "AREA-INTRINSIC"), ("gini_y_conn", "CONNECTION-SPECIFIC"),
            ("gini_pearson_y", "CONTROL — CCA-FREE")]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.4), sharey=True)

    def delta(rs, col, pair=None):
        d = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rs:
            if pair and r["pair"] != pair:
                continue
            v = num(r, col)
            if np.isfinite(v):
                d[r["animal"]][r["epoch"]].append(v)
        return np.array([np.median(e["expert"]) - np.median(e["naive"])
                         for e in d.values() if "naive" in e and "expert" in e])

    pairs = sorted(set(r["pair"] for r in t) & set(r["pair"] for r in c))
    x = np.arange(len(pairs))
    for ax, (col, lab) in zip(axes, cols):
        for k, (rs, coh, colour) in enumerate(((t, "Task", TASK_C), (c, "Control", CTRL_C))):
            m, e = [], []
            for p in pairs:
                d = delta(rs, col, p)
                m.append(d.mean() if d.size else np.nan)
                e.append(d.std(ddof=1) / np.sqrt(d.size) if d.size > 1 else np.nan)
            ax.bar(x + (-0.2 + 0.4 * k), m, 0.38, yerr=e, capsize=3, color=colour, label=coh)
        ta, ca = delta(t, col), delta(c, col)
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(pairs, rotation=35, ha="right", fontsize=9)
        ax.set_title(f"{lab}\nall pairs: task {ta.mean():+.3f} (N={ta.size}) vs "
                     f"control {ca.mean():+.3f} (N={ca.size}), {pstr(between(ta, ca))}",
                     fontsize=10)
    axes[0].set_ylabel("Expert − Naive   Gini")
    axes[0].legend(frameon=False)
    fig.suptitle("Subspace de-sparsification is NOT task-specific — controls do the same",
                 fontweight="semibold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "B3_gini_by_pair_cohort", pad=-0.10, caption=
         "Unit temporal CCA; no band dimension (spikes). Control animals carry an IMPOSED learning "
         "point of 46, the task cohort's mean, so both cohorts contrast trials 1–10 against roughly "
         "46–55 — the same time window. Yoked animals that do not learn de-sparsify by the same "
         "amount, which makes this a function of trial number rather than of learning. In the "
         "control cohort the CCA-free metric drops too, so the specificity argument does not hold "
         "there either. N = 4 control animals: no within-cohort test is possible.")


# ------------------------------------------------ B4 coupling by pair, band --
def b4_coupling():
    data = {c: [r for r in rows(LFP / f"lfp_coupling_epochs_{c}.csv",
                                epoch="All", reference="bipolar")]
            for c in ("task", "control")}
    for rs in data.values():
        for r in rs:
            r["mi_corrected"] = num(r, "mi") - num(r, "mi_surrogate_mean")
    gamma = sorted({r["band"] for r in data["task"] if r["measure"] == "pac_between"})
    fig, axes = plt.subplots(1, len(gamma) + 1, figsize=(15, 5.4))
    pairs = sorted({f"{r['area_a']}-{r['area_b']}" for r in data["task"]
                    if r["measure"] == "pac_between"}
                   & {f"{r['area_a']}-{r['area_b']}" for r in data["control"]
                      if r["measure"] == "pac_between"})

    def get(coh, measure, band, pair, col):
        return animals([r for r in data[coh] if r["measure"] == measure
                        and r["band"] == band
                        and f"{r['area_a']}-{r['area_b']}" == pair], col)

    for ax, band in zip(axes, gamma):
        x = np.arange(len(pairs))
        for k, (coh, colour) in enumerate((("task", TASK_C), ("control", CTRL_C))):
            m = [get(coh, "pac_between", band, p, "mi_corrected") for p in pairs]
            ax.bar(x + (-0.2 + 0.4 * k), [v.mean() if v.size else np.nan for v in m], 0.38,
                   yerr=[v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else np.nan for v in m],
                   capsize=3, color=colour, label=coh.capitalize())
        ax.axhline(0, color="k", lw=0.9)
        ax.set_xticks(x); ax.set_xticklabels(pairs, rotation=40, ha="right", fontsize=8)
        ax.set_title(f"theta–{band} PAC, between areas", fontsize=10)
    axes[0].set_ylabel("shuffle-subtracted Tort MI")
    axes[0].legend(frameon=False, fontsize=9)

    ax = axes[-1]
    sf = sorted({r["band"] for r in data["task"] if r["measure"] == "same_freq"})
    x = np.arange(len(sf))
    for k, (coh, colour) in enumerate((("task", TASK_C), ("control", CTRL_C))):
        m = [animals([r for r in data[coh] if r["measure"] == "same_freq" and r["band"] == b],
                     "orth_r") for b in sf]
        ax.bar(x + (-0.2 + 0.4 * k), [v.mean() if v.size else np.nan for v in m], 0.38,
               yerr=[v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else np.nan for v in m],
               capsize=3, color=colour)
    ax.set_xticks(x); ax.set_xticklabels(sf, rotation=40, ha="right", fontsize=8)
    ax.set_title("orthogonalised envelope corr,\nby band", fontsize=10)
    ax.set_ylabel("orth. r")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Coupling by area pair, band and cohort — bipolar (far field removed)",
                 fontweight="semibold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "B4_coupling_by_pair_band_cohort", pad=-0.13, caption=
         "Shuffle-subtracted Tort index, so the values are comparable across cohorts with different "
         "trial counts; the RAW index is not. The envelope correlation is NOT bias-corrected and "
         "does depend on trial count (Spearman with n_trials = −0.35, p = 3e-7) — control sessions "
         "are shorter (131 vs 200 trials), which is why its bars sit higher. Read that panel with "
         "that in mind.")


# ------------------------------- B5 learning contrast by area, band, cohort --
def b5_learning():
    data = {c: [r for r in rows(INFO / f"lfp_mi_features_{c}.csv")
                if r["band_status"] == "interpretable"] for c in ("task", "control")}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)

    def contrast(coh, area, band):
        sub = [r for r in data[coh] if r["area"] == area and r["band"] == band]
        n = {r["mouse_id"]: [] for r in sub}
        e = {r["mouse_id"]: [] for r in sub}
        for r in sub:
            v = num(r, "cmi_given_speed")
            if not np.isfinite(v):
                continue
            (n if r["epoch"] == "Naive" else e if r["epoch"] == "Expert" else {}).setdefault(
                r["mouse_id"], []).append(v)
        sh = [m for m in n if n[m] and e.get(m)]
        return np.array([np.mean(e[m]) - np.mean(n[m]) for m in sh])

    for ax, band in zip(axes, BANDS):
        paired_area_panel(ax, lambda coh, a, band=band: contrast(coh, a, band),
                          f"{band} ({'4–8' if band == 'theta' else '15–30'} Hz)",
                          "Expert − Naive\nI(power ; feature | speed)  (bits)"
                          if band == "theta" else "")
        ax.axhline(0, color="k", lw=1)
    axes[0].legend(frameon=False)
    fig.suptitle("Learning contrast by area, band and cohort — Naive (trials 1–10) vs "
                 "Expert (10 trials from LP)", fontweight="semibold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "B5_learning_by_area_band_cohort",
         "Ten trials per epoch, count-matched by construction. For this arm the unit of observation "
         "is the TRIAL, so ten trials is ten samples: the contrast noise (~0.0037 bits) exceeds the "
         "information level (~0.0033), and only a change larger than the whole effect would be "
         "detectable. Read these as unpowered, not as null.")





# --------------------------------- B6 which band carries it, task vs control --
def b6_band_preference():
    """POST-HOC. Found after the band-pooled comparison returned nothing, because
    theta and beta move in OPPOSITE directions in DMS and cancelled. Stated as
    post-hoc because it is: the areas were chosen after seeing B2."""
    data = {c: [r for r in rows(INFO / f"lfp_mi_features_{c}.csv", epoch="All")
                if r["band_status"] == "interpretable"] for c in ("task", "control")}

    def per_animal(coh, area, band):
        d = collections.defaultdict(list)
        for r in data[coh]:
            if r["area"] == area and r["band"] == band:
                v = num(r, "cmi_given_speed")
                if np.isfinite(v):
                    d[r["mouse_id"]].append(v)
        return {k: float(np.mean(v)) for k, v in d.items()}

    areas = ("DMS", "DLS", "ACC")
    fig, axes = plt.subplots(1, len(areas), figsize=(13.5, 5.4), sharey=True)
    for ax, a in zip(axes, areas):
        vals, labels, colours = [], [], []
        for coh, colour in (("task", TASK_C), ("control", CTRL_C)):
            th, be = per_animal(coh, a, "theta"), per_animal(coh, a, "beta")
            d = np.array([th[m] - be[m] for m in sorted(set(th) & set(be))])
            vals.append(d); labels.append(coh.capitalize()); colours.append(colour)
        for xi, (d, colour) in enumerate(zip(vals, colours)):
            ax.scatter(np.full(d.size, xi) + np.random.default_rng(0).normal(0, 0.045, d.size),
                       d, s=48, color=colour, zorder=3, edgecolor="white", linewidth=0.8)
            ax.plot([xi - 0.22, xi + 0.22], [d.mean()] * 2, color="#c0392b", lw=3.5, zorder=4)
        p = between(vals[0], vals[1])
        ax.axhline(0, color="k", lw=1.1)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([f"{l}\nn={v.size}" for l, v in zip(labels, vals)])
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_title(f"{a}\n{int((vals[0] > 0).sum())}/{vals[0].size} vs "
                     f"{int((vals[1] > 0).sum())}/{vals[1].size} theta-dominant · {pstr(p)}",
                     fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("theta − beta\nI(power ; feature | speed)   (bits)\n"
                       "above 0 = theta carries more")
    fig.suptitle("Which band carries the behavioural information — task and control differ in DMS",
                 fontweight="semibold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "B6_band_preference_by_cohort", pad=-0.05, caption=
         "POST-HOC, and stated as such: the band-pooled comparison found nothing because these two "
         "bands move in OPPOSITE directions and cancelled, and these three areas were chosen after "
         "seeing that. One test per area rather than two, so three tests; DMS at p = 0.0029 "
         "survives Bonferroni over them. Every one of the five control animals is beta-dominant in "
         "DMS, against 12 of 16 task animals theta-dominant. This is a LEVEL difference over the "
         "whole engaged session, not a change with learning. N = 5 controls — it needs confirming.")


if __name__ == "__main__":
    print(f"writing to {OUT}")
    b1_anchor(); b2_information(); b3_gini(); b4_coupling(); b5_learning(); b6_band_preference()
