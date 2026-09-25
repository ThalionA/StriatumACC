"""Task vs Control 1: which LFP measures differ between the groups.

Four panels, one arm each, plus the behaviour. Every title is computed from
``lfp_group_contrast.csv``; interpretation belongs in NOTES, not in a title
that survives a re-run unchanged.

Run from ``Striatum project/lfp``::

    /opt/anaconda3/bin/python scripts/plot_lfp_task_vs_control.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from striatum_lfp import config  # noqa: E402
from striatum_lfp.figstyle import AREA_ORDER, PLOT_BANDS, save_pair  # noqa: E402
from striatum_lfp.results_io import hierarchical, load_arms  # noqa: E402

TASK_C, CTRL_C = "#1f4e79", "#d98b00"
BAND_SHORT = {"theta": "\u03b8", "beta": "\u03b2",
              "low_gamma": "\u03b3L", "high_gamma": "\u03b3H", "total": "tot"}


def cell_labels(areas, bands):
    return [f"{a} {BAND_SHORT.get(b, b)}" for a in areas for b in bands]


def _contrast_rows():
    path = config.RESULTS_DIR / "lfp_group_contrast.csv"
    if not path.exists():
        return []
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("task_mean", "task_sem", "control_mean", "control_sem",
                  "p_raw", "p_fdr"):
            r[k] = float(r[k]) if r[k] not in ("", "None", "nan") else np.nan
    return rows


def _paired_bars(ax, cells, labels, contrast, ylabel, title):
    """Task and control side by side, with a star where the groups differ."""
    x = np.arange(len(cells))
    w = 0.38
    for off, (col, colour, name) in enumerate(((("task_mean", "task_sem"), TASK_C, "Task"),
                                               (("control_mean", "control_sem"), CTRL_C,
                                                "Control 1"))):
        m = [contrast.get(c, {}).get(col[0], np.nan) for c in cells]
        e = [contrast.get(c, {}).get(col[1], np.nan) for c in cells]
        ax.bar(x + (off - 0.5) * w, m, w, yerr=e, capsize=2, color=colour, label=name)
    for i, c in enumerate(cells):
        r = contrast.get(c)
        if r and r.get("differs") == "True":
            top = np.nanmax([r["task_mean"] + (r["task_sem"] or 0),
                             r["control_mean"] + (r["control_sem"] or 0), 0.0])
            span = np.diff(ax.get_ylim())[0] if ax.get_ylim()[1] > ax.get_ylim()[0] else 1
            ax.text(i, top + 0.03 * span, "\u2731", ha="center", fontsize=10,
                    color="#b3001b")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)


def main() -> None:
    rows = _contrast_rows()
    if not rows:
        print("[plot] no group-contrast table; run run_lfp_group_contrast.py first")
        return
    idx = {(r["arm"], r["metric"], r["area"], r["band"]): r for r in rows}
    areas = [a for a in AREA_ORDER
             if any(r["area"] == a for r in rows if r["arm"] == "reliability")]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))

    # Panel titles are computed from the table, never typed in: the 2026-08-28
    # tables carried "no decoding cell differs" and "p_FDR = 0.049" into the
    # 2026-09-07 re-run, where neither was true any more.
    def differing(cells):
        hits = [c for c in cells if idx.get(c, {}).get("differs") == "True"]
        return hits, len(cells)

    def cell_str(hits, limit=4):
        names = [f"{a} {BAND_SHORT.get(b, b)}" for _, _, a, b in hits]
        return ", ".join(names[:limit]) + (", …" if len(names) > limit else "")

    # (a) the learning claim
    cells = [("evolution", "delta_log_corridor", a, b) for a in areas for b in PLOT_BANDS]
    hits, n_cells = differing(cells)
    _paired_bars(axes[0][0], cells, cell_labels(areas, PLOT_BANDS), idx,
                 "Δ log10 band power, Naive → Expert",
                 f"(a) Evolution (pre-registered primary) — groups differ in "
                 f"{len(hits)}/{n_cells} cells{': ' + cell_str(hits) if hits else ''}")

    # (b) position information
    cells = [("decoding", "r2_minus_null", a, b) for a in areas for b in PLOT_BANDS]
    hits, n_cells = differing(cells)
    _paired_bars(axes[0][1], cells, cell_labels(areas, PLOT_BANDS), idx,
                 "R² above the rotated-label null",
                 f"(b) Spatial decoding — groups differ in {len(hits)}/{n_cells} cells"
                 f"{': ' + cell_str(hits) if hits else ''}")

    # (c) reliability, and (d) the behaviour that explains it
    cells = [("reliability", "split_half_r", a, b) for a in areas for b in PLOT_BANDS]
    hits, n_cells = differing(cells)
    resid_hits, _ = differing([("reliability", "split_half_r_speed_resid", a, b)
                               for a in areas for b in PLOT_BANDS])
    _paired_bars(axes[1][0], cells, cell_labels(areas, PLOT_BANDS), idx,
                 "split-half r of the spatial profile",
                 f"(c) Reliability — groups differ in {len(hits)}/{n_cells} cells raw, "
                 f"{len(resid_hits)}/{n_cells} after removing each channel's speed component")

    ax = axes[1][1]
    measures = [("speed_profile_split_half_r", "reliability of the\nSPEED profile"),
                ("speed_bin_cv", "within-bin\nspeed CV"),
                ("mean_speed_cm_s", "mean speed\n(cm/s ÷ 50)")]
    cells = [("behaviour", m, "behaviour", m) for m, _ in measures]
    scaled = {}
    for key in cells:
        r = dict(idx.get(key, {}))
        if r and key[1] == "mean_speed_cm_s":
            for f in ("task_mean", "task_sem", "control_mean", "control_sem"):
                r[f] = r[f] / 50.0
        scaled[key] = r
    rel = idx.get(("behaviour", "speed_profile_split_half_r", "behaviour", "speed_profile_split_half_r"), {})
    spd = idx.get(("behaviour", "mean_speed_cm_s", "behaviour", "mean_speed_cm_s"), {})
    _paired_bars(ax, cells, [lab for _, lab in measures], scaled, "value",
                 "(d) Behaviour — speed-profile split-half r "
                 f"task {rel.get('task_mean', np.nan):.2f} vs control "
                 f"{rel.get('control_mean', np.nan):.2f}; mean speed "
                 f"{spd.get('task_mean', np.nan):.0f} vs {spd.get('control_mean', np.nan):.0f} cm/s")

    axes[0][0].legend(fontsize=8, loc="lower left")
    fig.suptitle("LFP band power: task versus yoked Control 1\n"
                 "bars are animal means ± SEM; ✱ = groups differ, exact permutation test, "
                 "BH-FDR q=0.05 within each arm and metric", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_pair(fig, "lfp_task_vs_control")

    # A second figure for the one cell that differed on 2026-08-28 (DLS theta),
    # on the pre-registered primary metric, so its fate after the re-run is shown.
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for c, colour, name in (("task", TASK_C, "Task"), ("control", CTRL_C, "Control 1")):
        ev = load_arms("evolution", c)
        naive = hierarchical([r for r in ev if r["area"] == "DLS" and r["band"] == "theta"
                              and r["epoch"] == "Naive"], ("area",), "log_corridor")
        expert = hierarchical([r for r in ev if r["area"] == "DLS" and r["band"] == "theta"
                               and r["epoch"] == "Expert"], ("area",), "log_corridor")
        if ("DLS",) not in naive or ("DLS",) not in expert:
            continue
        n, e = naive[("DLS",)], expert[("DLS",)]
        ax.errorbar([0, 1], [n[0], e[0]], yerr=[n[1], e[1]], marker="o", ms=7, lw=2,
                    capsize=4, color=colour, label=f"{name} (N={n[2]})")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Naive", "Expert"])
    ax.set_ylabel("DLS theta, log10 band power (corridor)")
    q = idx.get(("evolution", "delta_log_corridor", "DLS", "theta"), {}).get("p_fdr", np.nan)
    q_z = idx.get(("evolution", "delta_z_corridor", "DLS", "theta"), {}).get("p_fdr", np.nan)
    q_speed = idx.get(("evolution", "delta_z_corridor_speed_resid", "DLS", "theta"),
                      {}).get("p_fdr", np.nan)
    ax.set_title("DLS theta, task vs yoked control (the 2026-08-28 candidate)\n"
                 f"group × epoch p_FDR: primary Δlog {q:.3f}; Δz {q_z:.3f}; "
                 f"Δz speed-residualised {q_speed:.3f}", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_pair(fig, "lfp_dls_theta_task_vs_control")


if __name__ == "__main__":
    main()
