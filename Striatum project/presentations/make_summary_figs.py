#!/usr/bin/env python3
"""Render cohort summary-count figures for the update deck.

Everything is read live from the preprocessed caches, so the numbers cannot
drift from the data. Output: figures/summary_0{1,2,3}_*.{svg,png}
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = Path("/Users/theoamvr/Desktop/Experiments/StriatumACC/Striatum project")
FIG = PROJ / "figures"
NAMES = {1: "MSN", 2: "FS", 3: "TAN", 4: "UIN", 5: "RS"}
AREAS = [("DMS", "is_dms"), ("DLS", "is_dls"), ("ACC", "is_acc"),
         ("V1", "is_v1"), ("CA1", "is_ca1"), ("DG", "is_dg")]
AREA_ORDER = [a for a, _ in AREAS]
# Okabe-Ito, fixed assignment by area identity (never cycled)
AREA_COL = dict(zip(AREA_ORDER,
                    ["#0072B2", "#56B4E9", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]))
TYPE_COL = {"MSN": "#0072B2", "FS": "#D55E00", "TAN": "#009E73",
            "UIN": "#999999", "RS": "#56B4E9"}
GROUPS = [("Task", "processed_data/preprocessed_data5cm.mat"),
          ("Control 1", "processed_data/preprocessed_data_control5cm.mat"),
          ("Control 2", "processed_data/preprocessed_data_control2.mat")]


def read(path):
    """-> (per_session_area_counts, per_area_type_counts, learning_points)"""
    f = h5py.File(PROJ / path)
    pd = f["preprocessed_data"]
    n = len(pd[list(pd.keys())[0]])
    sessions, types, lps = [], Counter(), []
    for i in range(n):
        codes = None
        if "final_neurontypes" in pd:
            nt = np.array(f[pd["final_neurontypes"][i][0]]).T
            if nt.ndim == 2 and nt.shape[1] >= 5:
                codes = nt[:, 4]
        row = {}
        for a, k in AREAS:
            if k not in pd:
                continue
            m = np.array(f[pd[k][i][0]]).ravel().astype(bool)
            if m.sum() == 0:
                continue
            row[a] = int(m.sum())
            if codes is not None and codes.size == m.size:
                good = codes[m][np.isfinite(codes[m])].astype(int)
                for c, v in Counter(good).items():
                    types[(a, NAMES.get(c, str(c)))] += v
        sessions.append(row)
        if "learning_point" in pd:
            try:
                lp = np.array(f[pd["learning_point"][i][0]]).ravel()
                lps.append(float(lp[0]) if lp.size else np.nan)
            except Exception:
                lps.append(np.nan)
    return sessions, types, lps


DATA = {g: read(p) for g, p in GROUPS}


def save(fig, name):
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=115, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- figure 1
# Cohort table: one row per group, columns = sessions, units, per-area totals.
fig, ax = plt.subplots(figsize=(12.4, 3.4))
ax.axis("off")
cols = ["sessions", "units"] + AREA_ORDER
cells, rows = [], []
for g, _ in GROUPS:
    sess, types, _ = DATA[g]
    per_area = Counter()
    for r in sess:
        per_area.update(r)
    rows.append(g)
    cells.append([str(len(sess)), str(sum(per_area.values()))]
                 + [str(per_area.get(a, 0)) if per_area.get(a, 0) else "–"
                    for a in AREA_ORDER])
tbl = ax.table(cellText=cells, rowLabels=rows, colLabels=cols,
               cellLoc="center", rowLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1, 2.0)
for j, c in enumerate(cols):
    tbl[0, j].set_text_props(weight="bold")
    if c in AREA_COL:
        tbl[0, j].set_facecolor(AREA_COL[c])
        tbl[0, j].set_text_props(color="white", weight="bold")
for i in range(len(rows)):
    tbl[i + 1, -1].set_text_props(weight="bold")
ax.set_title("Recorded units by group and area  ·  all counts read live from the "
             "preprocessed caches (2026-08-12)", fontsize=12, pad=18)
save(fig, "summary_01_cohort_by_group_and_area")

# ---------------------------------------------------------------- figure 2
# Units per session, stacked by area, one panel per group.
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3),
                         gridspec_kw={"width_ratios": [16, 5, 6]})
for ax, (g, _) in zip(axes, GROUPS):
    sess, _, _ = DATA[g]
    x = np.arange(len(sess))
    bottom = np.zeros(len(sess))
    for a in AREA_ORDER:
        v = np.array([s.get(a, 0) for s in sess], float)
        if v.sum() == 0:
            continue
        ax.bar(x, v, 0.74, bottom=bottom, color=AREA_COL[a], label=a,
               edgecolor="white", linewidth=0.6)
        bottom += v
    for xi, tot in zip(x, bottom):
        ax.text(xi, tot + max(bottom) * 0.02, int(tot), ha="center",
                fontsize=7.5, color="0.25")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in range(len(sess))], fontsize=8)
    ax.set_title(f"{g}  (n = {len(sess)})", fontsize=11)
    ax.set_xlabel("session", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_ylim(0, max(bottom) * 1.13)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("units", fontsize=9)
axes[0].legend(fontsize=8.5, frameon=False, ncol=3, loc="upper right")
fig.suptitle("Units per session, stacked by area", fontsize=12.5)
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "summary_02_units_per_session_by_area")

# ---------------------------------------------------------------- figure 3
# Cell types per area, Task vs Control 1 (control 2 has no type labels).
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.3), sharey=False)
for ax, g in zip(axes, ["Task", "Control 1"]):
    _, types, _ = DATA[g]
    present = [a for a in AREA_ORDER if any(k[0] == a for k in types)]
    x = np.arange(len(present))
    order = ["MSN", "FS", "TAN", "UIN", "RS"]
    bottom = np.zeros(len(present))
    for t in order:
        v = np.array([types.get((a, t), 0) for a in present], float)
        if v.sum() == 0:
            continue
        ax.bar(x, v, 0.66, bottom=bottom, color=TYPE_COL[t], label=t,
               edgecolor="white", linewidth=0.6)
        for xi, (b, vi) in enumerate(zip(bottom, v)):
            if vi >= max(bottom.max(), 1) * 0.06:
                ax.text(xi, b + vi / 2, int(vi), ha="center", va="center",
                        fontsize=8, color="white", weight="bold")
        bottom += v
    for xi, tot in zip(x, bottom):
        ax.text(xi, tot + bottom.max() * 0.02, int(tot), ha="center",
                fontsize=9, color="0.25", weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(present, fontsize=10)
    ax.set_title(g, fontsize=11)
    ax.set_ylim(0, bottom.max() * 1.12)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("units", fontsize=9)
axes[0].legend(fontsize=9, frameon=False, ncol=5, loc="upper left")
fig.suptitle("Cell types per area  ·  striatum MSN/FS/TAN/UIN; cortex & "
             "hippocampus FS vs RS (classification corrected 2026-08-12)",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.92))
save(fig, "summary_03_cell_types_per_area")
print("done")
