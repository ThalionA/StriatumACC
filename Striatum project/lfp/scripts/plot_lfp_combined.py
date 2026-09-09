#!/usr/bin/env python3
"""Task and Control 1 on the same axes, one figure per result instead of two.

Every result figure in this package used to be written once per cohort, so the
comparison that matters -- does the task cohort differ from the yoked control --
had to be made by holding two files side by side and trusting that their y-axes
matched. They often did not. These figures overlay the two cohorts on one shared
scale, which is the only way the difference is legible.

Three figures:

1. `lfp_reliability_moving_session_task_vs_control`
   Moving-window reliability against trial number FROM SESSION START. This is the
   learning-point-aligned evolution figure re-cut on an axis both cohorts share:
   yoked controls have no learning point of their own (they inherit the task
   cohort's average, 41), so an aligned control curve is an artefact of that
   borrowed number. Trial 1 is trial 1 in both cohorts.

2. `lfp_reliability_moving_lp_task_vs_control`
   The same statistic on the learning-point axis, both cohorts overlaid, kept as
   the direct counterpart of (1) so the two alignments can be compared. The
   control curve here carries the borrowed-learning-point caveat.

3. `lfp_evolution_z_task_vs_control`
   Band power across the four epochs, both cohorts, per area and band -- the
   learning result itself, previously split across two 6x4 grids.

No trial-shuffled series is drawn: the reference in these figures is the other
cohort, not a distance from chance. The per-cohort figures from
`plot_lfp_arms.py` keep the shuffle.

Run from `lfp/` (no recomputation; reads the arm tables):
    /opt/anaconda3/bin/python scripts/plot_lfp_combined.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from striatum_lfp import figstyle  # noqa: E402
from striatum_lfp.results_io import load_arms  # noqa: E402

# Cohort key -> (legend label, colour). Colours are the ones the task-vs-control
# summary already uses, so a reader moving between figures keeps the mapping.
COHORTS = (("task", "Task", "#1f4e79"), ("control", "Control 1", "#e69f00"))
BANDS = ("theta", "beta", "low_gamma", "high_gamma", "total")
BAND_ROW = {b: figstyle.BAND_LABEL[b] for b in BANDS}
FIRST_N = 20              # trials from session start (figure 1)
MIN_ANIMALS = 3           # never draw a mean +- SEM resting on fewer animals


def _by_offset(rows, area, band, xkey, field="reliability"):
    """{x offset: {mouse: value}} for one area x band."""
    out = defaultdict(dict)
    for r in rows:
        if r["area"] != area or r["band"] != band:
            continue
        x = r[xkey]
        if x == "" or x is None or not np.isfinite(r.get(field, np.nan)):
            continue
        out[int(float(x))][int(r["mouse_id"])] = r[field]
    return out


def _trace(ax, rows, area, band, xkey, xlim, colour, label):
    """Mean +- SEM across animals at each offset. Returns True if anything drew."""
    by_off = _by_offset(rows, area, band, xkey)
    xs = sorted(o for o, d in by_off.items()
                if len(d) >= MIN_ANIMALS and xlim[0] <= o <= xlim[1])
    if not xs:
        return False
    m = np.array([np.mean(list(by_off[o].values())) for o in xs])
    e = np.array([np.std(list(by_off[o].values()), ddof=1) / np.sqrt(len(by_off[o]))
                  for o in xs])
    ax.plot(xs, m, "-", color=colour, lw=1.6, label=label)
    ax.fill_between(xs, m - e, m + e, color=colour, alpha=0.20, lw=0)
    return True


def _share_y(axes_flat):
    drawn = [a for a in axes_flat if a.lines]
    if not drawn:
        return
    lo = min(a.get_ylim()[0] for a in drawn)
    hi = max(a.get_ylim()[1] for a in drawn)
    for a in axes_flat:
        a.set_ylim(lo, hi)
    return lo, hi


def _grid(areas, nrow):
    fig, axes = plt.subplots(nrow, len(areas),
                             figsize=(2.75 * len(areas), 2.25 * nrow),
                             squeeze=False, sharex=True)
    return fig, axes


def moving_figure(xkey: str, xlim: tuple[int, int], stem: str, title: str,
                  xlabel: str, mark_zero: bool = False) -> None:
    per_cohort = {c: load_arms("moving_reliability", c) for c, _, _ in COHORTS}
    if not any(per_cohort.values()):
        print(f"[plot] no moving-reliability tables; skipping {stem}")
        return
    areas = [a for a in figstyle.AREA_ORDER
             if any(r["area"] == a for rows in per_cohort.values() for r in rows)]
    fig, axes = _grid(areas, len(BANDS))
    counts: dict[str, set] = defaultdict(set)
    for bi, band in enumerate(BANDS):
        for ai, area in enumerate(areas):
            ax = axes[bi][ai]
            for key, label, colour in COHORTS:
                rows = per_cohort[key]
                if _trace(ax, rows, area, band, xkey, xlim, colour, label):
                    counts[key].update(
                        int(r["mouse_id"]) for r in rows if r["area"] == area)
            ax.axhline(0, color="k", lw=0.5, ls=":")
            if mark_zero:
                ax.axvline(0, color="0.3", lw=1.0)
            ax.set_xlim(*xlim)
            if bi == 0:
                ax.set_title(area, fontsize=11, fontweight="bold",
                             color=figstyle.AREA_COLOUR[area])
            if ai == 0:
                ax.set_ylabel(BAND_ROW[band], fontsize=8)
            if bi == len(BANDS) - 1:
                ax.set_xlabel(xlabel, fontsize=8)
    lims = _share_y([a for row in axes for a in row])
    axes[0][0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    figstyle.save_pair(fig, stem)
    if lims:
        print(f"[plot]   shared y {lims[0]:.3f} to {lims[1]:.3f}; "
              f"animals task {len(counts['task'])}, control {len(counts['control'])}")


def evolution_figure(stem="lfp_evolution_z_task_vs_control") -> None:
    epochs = ("Trials 1-3", "Trials 4-10", "Intermediate", "Expert")
    per_cohort = {c: load_arms("evolution", c) for c, _, _ in COHORTS}
    if not any(per_cohort.values()):
        print(f"[plot] no evolution tables; skipping {stem}")
        return
    areas = [a for a in figstyle.AREA_ORDER
             if any(r["area"] == a for rows in per_cohort.values() for r in rows)]
    fig, axes = _grid(areas, len(BANDS))
    counts: dict = {}
    x = np.arange(len(epochs))
    for bi, band in enumerate(BANDS):
        for ai, area in enumerate(areas):
            ax = axes[bi][ai]
            for key, label, colour in COHORTS:
                per_epoch = defaultdict(dict)
                for r in per_cohort[key]:
                    if r["area"] != area or r["band"] != band:
                        continue
                    v = r.get("z_corridor", np.nan)
                    if np.isfinite(v):
                        per_epoch[r["epoch"]][int(r["mouse_id"])] = v
                m, e, ns = [], [], []
                for ep in epochs:
                    vals = list(per_epoch.get(ep, {}).values())
                    ns.append(len(vals))
                    if len(vals) >= MIN_ANIMALS:
                        m.append(float(np.mean(vals)))
                        e.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))))
                    else:
                        m.append(np.nan)
                        e.append(np.nan)
                if np.all(np.isnan(m)):
                    continue
                ax.errorbar(x, m, yerr=e, marker="o", ms=4, lw=1.6, capsize=3,
                            color=colour, label=label)
                # Print the per-epoch animal count. A point is dropped when fewer
                # than MIN_ANIMALS animals have it, and the reason is never the
                # data being short: the Intermediate and Expert windows are
                # LEARNING-POINT relative, so the two task non-learners (703 and
                # 1206, which never reach criterion) have no such window at all.
                # In CA1 and DG, where 1206 is one of only three task animals,
                # that takes n to 2 and the line stops after "Trials 4-10".
                counts.setdefault((area, band, key), ns)
            ax.axhline(0, color="k", lw=0.5, ls=":")
            ax.set_xticks(x)
            lines = []
            for key, label, colour in COHORTS:
                ns = counts.get((area, band, key))
                if ns:
                    lines.append(f"{label.split()[0].lower()} N={'/'.join(map(str, ns))}")
            if lines:
                ax.text(0.02, 0.03, "\n".join(lines), transform=ax.transAxes,
                        fontsize=5.5, color="0.35", va="bottom")
            if bi == 0:
                ax.set_title(area, fontsize=11, fontweight="bold",
                             color=figstyle.AREA_COLOUR[area])
            if ai == 0:
                ax.set_ylabel(BAND_ROW[band], fontsize=8)
            if bi == len(BANDS) - 1:
                ax.set_xticklabels(["1-3", "4-10", "Inter", "Expert"],
                                   rotation=30, ha="right", fontsize=7)
    _share_y([a for row in axes for a in row])
    axes[0][0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle(
        "LFP band power across learning: task versus yoked Control 1\n"
        "z-scored log power in the corridor, animal means ± SEM, shared y-scale. "
        f"N per epoch is printed in each panel; a point needs {MIN_ANIMALS} animals to be drawn.\n"
        "Intermediate and Expert are LEARNING-POINT relative, so the two task "
        "non-learners (703, 1206) have no such window — in CA1/DG, where 1206 is "
        "one of three animals, the task line therefore stops after trials 4-10.\n"
        "Control epochs are matched TIME windows: yoked controls have no learning "
        "point and inherit the task cohort's average (41).", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    figstyle.save_pair(fig, stem)


def main() -> None:
    moving_figure(
        "trial", (1, FIRST_N),
        "lfp_reliability_moving_session_task_vs_control",
        "Evolution of moving-window reliability from the START OF THE SESSION\n"
        f"first {FIRST_N} trials, 5-trial centred window, mean pairwise correlation "
        "between spatial profiles\n"
        "animal means ± SEM, shared y-scale, no learning-point alignment and no "
        "shuffled series",
        "trial from session start")
    moving_figure(
        "trial_rel_lp", (-30, 40),
        "lfp_reliability_moving_lp_task_vs_control",
        "The same statistic on the LEARNING-POINT axis, for comparison\n"
        "the control curve is aligned to a learning point it does not have "
        "(it inherits the task cohort's average, 41)",
        "trial relative to learning point", mark_zero=True)
    evolution_figure()


if __name__ == "__main__":
    main()
