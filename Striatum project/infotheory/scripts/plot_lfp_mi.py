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
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lfp" / "src"))
from striatum_lfp import stats  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGURES = Path(__file__).resolve().parents[1] / "figures"
AREAS = ("DMS", "DLS", "ACC", "V1", "CA1", "DG")
BANDS = ("theta", "beta")
COHORTS = (("task", "Task", "#1f4e79"), ("control", "Control 1", "#e69f00"))
#: run_lfp_mi.py measures each band's status; only spike-free bands are plotted.
PLOTTED_STATUS = "clean"


def load(which: str, cohort: str) -> list[dict]:
    path = RESULTS / f"lfp_mi_{which}_{cohort}.csv"
    if not path.exists():
        return []
    rows = [r for r in csv.DictReader(path.open())
            if r["band_status"] == PLOTTED_STATUS]
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


def p_across_animals(d: np.ndarray) -> float:
    """Exact sign-flip p, or nan when n animals cannot reach 0.05 at all."""
    d = np.asarray(d, float)
    return stats.sign_flip_test(d) if stats.can_reach(stats.sign_flip_floor(d.size)) \
        else np.nan


def contrast(rows, key="v", **where):
    """LateHalf - EarlyHalf per animal, over the features present in BOTH halves.

    Averaging each half over whichever features survived its own split made the
    two halves means over different feature sets (70 of 102 cells differed).
    """
    by = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if any(r.get(k) != v for k, v in where.items()):
            continue
        if r["epoch"] in ("EarlyHalf", "LateHalf"):
            by[r["mouse_id"]][r.get("feature", "")].setdefault(r["epoch"], []).append(r[key])
    d = []
    for feats in by.values():
        shared = [f for f, e in feats.items() if "EarlyHalf" in e and "LateHalf" in e]
        if shared:
            d.append(np.mean([np.mean(feats[f]["LateHalf"]) - np.mean(feats[f]["EarlyHalf"])
                              for f in shared]))
    d = np.array(d)
    return d, p_across_animals(d)


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
                m.append(np.nan)
                e.append(np.nan)
                lab.append("")
                continue
            m.append(v.mean())
            e.append(v.std(ddof=1) / np.sqrt(v.size))
            p = p_across_animals(v)
            lab.append(star(p, v.size))
        off = -0.18 + 0.36 * bi
        axa.bar(x + off, m, 0.34, yerr=e, capsize=3,
                color=("#4a7ebb" if band == "theta" else "#9c4f96"), label=band)
        for xi, (mv, ev, t) in enumerate(zip(m, e, lab)):
            if t:
                axa.text(xi + off, mv + ev + 0.0012, t, ha="center", fontsize=7)
    axa.set_xticks(x)
    axa.set_xticklabels(AREAS)
    axa.set_ylabel("I(band power; running speed)\nshuffle-subtracted (bits)")
    axa.set_title("(a) ANCHOR — I(band power; running speed), within spatial bin, "
                  "whole engaged session.\nStars: exact sign-flip across animals; "
                  "n=… where n animals cannot reach 0.05.", fontsize=10)
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
    # The residual-confound floor: speed conditioned on a two-level speed code
    # still leaves speed information behind. The trial's own mean velocity,
    # conditioned the same way, measures how much -- a feature is only
    # "beyond speed" by as much as it clears this line, not zero.
    floor = np.array(list(per_animal(feat["task"], "v", feature="mean_velocity_cm_s",
                                     epoch="All").values()))
    if floor.size:
        axb.axhline(floor.mean(), color="#c0392b", ls="--", lw=1.2,
                    label="residual speed floor (mean velocity | speed)")
    axb.set_xticks(xf)
    axb.set_xticklabels([f.replace("_", " ") for f in features], rotation=30,
                        ha="right", fontsize=8)
    axb.set_ylabel("shuffle-subtracted MI (bits)")
    axb.set_title("(b) Raw vs speed-conditioned information. Task cohort, whole engaged "
                  "session,\npooled over areas and both bands; read against the dashed "
                  "floor, not zero.", fontsize=10)
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
                m.append(np.nan)
                e.append(np.nan)
                continue
            m.append(d.mean())
            e.append(d.std(ddof=1) / np.sqrt(d.size))
            marks.append((ai + (-0.15 if key == "task" else 0.15),
                          m[-1] + e[-1], star(p, d.size), colour))
        axc.errorbar(x + (-0.15 if key == "task" else 0.15), m, yerr=e, fmt="s",
                     ms=7, capsize=4, lw=0, elinewidth=2, color=colour, label=lab)
    axc.axhline(0, color="k", lw=1.0)
    lo, hi = axc.get_ylim()
    axc.set_ylim(lo, hi + 0.18 * (hi - lo))
    pad = 0.04 * (axc.get_ylim()[1] - axc.get_ylim()[0])
    for xx, yy, t, cc in marks:
        axc.text(xx, yy + pad, t, ha="center", fontsize=7, color=cc)
    axc.set_xticks(x)
    axc.set_xticklabels(AREAS)
    axc.set_ylabel("LateHalf − EarlyHalf\nI(power; feature | speed)  (bits)")
    axc.set_title("(c) Does it change with training? Two halves of the ENGAGED period\n"
                  "(clipped at the disengagement point), features shared by both halves.",
                  fontsize=10)
    axc.legend(fontsize=9, frameon=False)

    # ---- (d) DLS animal by animal -------------------------------------------
    axd = fig.add_subplot(gs[1, 1])
    d_all, p = contrast(feat["task"], area="DLS")
    a = per_animal(feat["task"], area="DLS", epoch="EarlyHalf")
    b = per_animal(feat["task"], area="DLS", epoch="LateHalf")
    sh = sorted(set(a) & set(b))
    for m in sh:
        axd.plot([0, 1], [a[m], b[m]], "-o", ms=5, color="#1f4e79", alpha=0.65, lw=1.2)
    axd.plot([0, 1], [np.mean([a[m] for m in sh]), np.mean([b[m] for m in sh])],
             "-o", ms=10, color="#c0392b", lw=3, label="mean", zorder=5)
    axd.set_xticks([0, 1])
    axd.set_xticklabels(["first half\n(engaged)", "second half\n(engaged)"])
    axd.set_xlim(-0.25, 1.25)
    axd.set_ylabel("DLS  I(power; feature | speed)  (bits)")
    axd.set_title(f"(d) DLS, animal by animal: {(d_all < 0).sum()}/{d_all.size} decrease "
                  f"(sign-flip p = {p:.4f}, shared features).\nBefore clipping at disengagement this read 10/10, p = 0.0020 "
                  f"— it was disengagement, not learning.", fontsize=10)
    axd.legend(fontsize=9, frameon=False)

    fig.suptitle("LFP band power and behavioural information — mirroring Lemke et al. (2024) "
                 "on the field rather than single units\n"
                 "theta (4–8 Hz) and beta (15–30 Hz) only; log power ranked WITHIN spatial "
                 "bin; shuffle-subtracted; mean over windows, never peak.\nALL trials clipped at the disengagement point (change_point_mean); the wide contrast is the two halves of what remains. p uncorrected.",
                 fontsize=12)
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "lfp_mi_overview.svg")
    fig.savefig(FIGURES / "lfp_mi_overview.png",
                dpi=min(150, 1600 / max(fig.get_size_inches())))
    plt.close(fig)
    print("[plot] lfp_mi_overview.svg + .png")


if __name__ == "__main__":
    main()
