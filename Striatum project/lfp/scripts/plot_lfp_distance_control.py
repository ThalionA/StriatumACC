#!/usr/bin/env python3
"""Does crossing an area boundary change LFP coupling, once distance is matched?

Two panels per cohort:

(a) Coupling against electrode separation, within-area pairs against across-area
    pairs. If the two curves lie on top of each other, coupling is a function of
    distance along the shank and the area labels add nothing.

(b) The matched contrast per animal and band: within minus across at IDENTICAL
    separation, so the classes cannot differ in distance at all. The animal is
    the unit of analysis; a Wilcoxon signed-rank test against zero per band,
    BH-FDR across bands.

Panel (a) also carries the trap that makes (b) necessary: within-area pairs are
systematically closer than across-area ones, because an area is only a few
hundred microns thick, so simply restricting both classes to a shared separation
range leaves them badly mismatched and manufactures a positive difference.

    /opt/anaconda3/bin/python scripts/plot_lfp_distance_control.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from striatum_lfp import arms, config, figstyle  # noqa: E402

BANDS = ("theta", "beta", "low_gamma", "high_gamma", "total")
COHORTS = (("task", "Task", "#1f4e79"), ("control", "Control 1", "#e69f00"))
CLASS_STYLE = {"within": ("#2d6a4f", "-", "within an area"),
               "across": ("#9d0208", "--", "across an area boundary")}
MAX_SEP_UM = 1600.0        # beyond this, within-area pairs are too rare to plot


def _load(name: str, cohort: str) -> list[dict]:
    path = config.RESULTS_DIR / f"lfp_{name}_{cohort}.csv"
    if not path.exists():
        return []
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            if k in ("cohort", "probe", "band", "pair_class"):
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                r[k] = np.nan
    return rows


def _decay_curve(rows, band, cls):
    """{separation bin: (mean over animals, sem, n_animals)} for one band and class."""
    per_animal = defaultdict(dict)
    for r in rows:
        if r["band"] != band or r["pair_class"] != cls:
            continue
        if not np.isfinite(r["mean_r_raw"]) or r["sep_bin_lo_um"] > MAX_SEP_UM:
            continue
        # One value per animal per bin; a second probe overwrites, which is what
        # we want when an animal contributes two probes to the same separation.
        per_animal[r["sep_bin_lo_um"]][int(r["mouse_id"])] = r["mean_r_raw"]
    out = {}
    for sep, d in sorted(per_animal.items()):
        v = np.array(list(d.values()))
        if v.size < 2:
            continue
        out[sep] = (v.mean(), v.std(ddof=1) / np.sqrt(v.size), v.size)
    return out


def _per_animal_contrast(matched, band, field="d_raw"):
    """{mouse_id: contrast} for one band, averaging an animal's probes."""
    by_mouse = defaultdict(list)
    for r in matched:
        if r["band"] != band or not np.isfinite(r.get(field, np.nan)):
            continue
        if not r.get("n_separations", 0):
            continue
        by_mouse[int(r["mouse_id"])].append(r[field])
    return {m: float(np.mean(v)) for m, v in by_mouse.items()}


def main() -> None:
    fig, axes = plt.subplots(2, len(COHORTS), figsize=(7.2 * len(COHORTS), 9.0),
                             squeeze=False)
    summary_lines: list[str] = []

    for ci, (key, label, _) in enumerate(COHORTS):
        by_bin = _load("distance_control", key)
        matched = _load("distance_matched", key)
        if not by_bin:
            axes[0][ci].axis("off")
            axes[1][ci].axis("off")
            continue

        # --- (a) decay curves, total band (the broadest measure) --------------
        ax = axes[0][ci]
        for cls, (colour, ls, leg) in CLASS_STYLE.items():
            curve = _decay_curve(by_bin, "total", cls)
            if not curve:
                continue
            xs = np.array(list(curve))
            m = np.array([curve[x][0] for x in xs])
            e = np.array([curve[x][1] for x in xs])
            ax.plot(xs + 50, m, ls, color=colour, lw=2.0, marker="o", ms=4, label=leg)
            ax.fill_between(xs + 50, m - e, m + e, color=colour, alpha=0.20, lw=0)
        ax.set_xlabel("electrode separation along the shank (µm)")
        ax.set_ylabel("coupling, r of log band power")
        ax.set_title(f"{label} — coupling falls with distance, and the area\n"
                     f"boundary does not move the curve (total 1–150 Hz)", fontsize=11)
        ax.legend(fontsize=9, frameon=False)
        ax.axhline(0, color="0.6", lw=0.8, ls=":")

        # --- (b) exact-matched contrast per band -----------------------------
        ax = axes[1][ci]
        xs, means, sems, ps, ns, floors = [], [], [], [], [], []
        for bi, band in enumerate(BANDS):
            vals = _per_animal_contrast(matched, band)
            if len(vals) < 3:
                continue
            v = np.array(list(vals.values()))
            xs.append(bi)
            means.append(v.mean())
            sems.append(v.std(ddof=1) / np.sqrt(v.size))
            ns.append(v.size)
            # A two-sided signed-rank test on n animals cannot return a p below
            # 2 / 2**n, so at n = 5 the smallest attainable p is 0.0625 and the
            # test CANNOT reject at 0.05 however large the effect. Marking such a
            # band "n.s." would read as evidence of no effect when it is only
            # evidence of no power, so the floor is carried and labelled.
            ps.append(stats.wilcoxon(v).pvalue if v.size >= 6 else np.nan)
            floors.append(2.0 / 2 ** v.size)
            ax.scatter(np.full(v.size, bi) + np.linspace(-0.12, 0.12, v.size),
                       v, s=16, color="0.55", zorder=2, alpha=0.8)
        if xs:
            _, sig = arms.fdr_bh(np.array(ps, dtype=float))
            ax.errorbar(xs, means, yerr=sems, fmt="s", ms=8, lw=0, elinewidth=2,
                        capsize=4, color="#1f4e79", zorder=3)
            for k, bi in enumerate(xs):
                if not np.isfinite(ps[k]):
                    mark = "underpowered"
                elif sig[k]:
                    mark = "*"
                else:
                    mark = "n.s."
                ax.text(bi, max(means[k] + sems[k], 0) + 0.012, mark, ha="center",
                        fontsize=9 if mark == "*" else 6.5, color="0.25")
            summary_lines.append(f"{label}:")
            for k, bi in enumerate(xs):
                if np.isfinite(ps[k]):
                    verdict = (f"Wilcoxon p = {ps[k]:.3f}"
                               f"{', survives BH-FDR' if sig[k] else ''}")
                else:
                    verdict = (f"UNDERPOWERED: n = {ns[k]}, the smallest attainable "
                               f"two-sided p is {floors[k]:.3f}")
                summary_lines.append(
                    f"   {BANDS[bi]:11s} within − across = {means[k]:+.4f} ± {sems[k]:.4f} "
                    f"(N = {ns[k]} mice, {verdict})")
        ax.axhline(0, color="k", lw=1.0)
        ax.set_xticks(range(len(BANDS)))
        ax.set_xticklabels(["θ", "β", "γ low", "γ high", "total"], rotation=0)
        ax.set_ylabel("within − across, at IDENTICAL separation")
        ax.set_title(f"{label} — with distance matched exactly, the boundary\n"
                     f"makes no difference (each dot is one mouse)", fontsize=11)

    lo = min(a.get_ylim()[0] for a in axes[1] if a.has_data())
    hi = max(a.get_ylim()[1] for a in axes[1] if a.has_data())
    for a in axes[1]:
        if a.has_data():
            a.set_ylim(lo, hi)

    fig.suptitle(
        "Is the cross-area LFP coupling anything more than distance along the shank?\n"
        "Top: coupling vs separation, within-area pairs vs across-area pairs, animal means ± SEM.\n"
        "Bottom: the same comparison at EXACTLY equal separation — a shared separation RANGE is not "
        "enough, because an area is only a few hundred µm thick,\nso within-area pairs inside that "
        "range sit ~220 µm closer and manufacture a positive difference.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    figstyle.save_pair(fig, "lfp_distance_control")

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
