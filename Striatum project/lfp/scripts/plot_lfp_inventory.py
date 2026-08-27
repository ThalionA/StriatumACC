"""Exploration figures for the 2026-08 LFP cohort.

Reads the caches written by ``run_lfp_inventory.py`` and ``run_lfp_identity.py``
and draws five panels: cohort coverage, per-file spectra, depth x frequency
structure, session integrity in time, and the filename-identity matrix.

Run from ``Striatum project/lfp``::

    /opt/anaconda3/bin/python scripts/plot_lfp_inventory.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from striatum_lfp import config, geometry  # noqa: E402

MAX_PNG_PX = 1600
AREA_COLOUR = {"DMS": "#0072b2", "DLS": "#77ac30", "ACC": "#d95319",
               "V1": "#7e2f8e", "CA1": "#cc1a33", "DG": "#33b3b3"}
# The two gain regimes found by the inventory: the 2026-07 batch stores ~30x
# smaller voltages than the 2026-08 batch, so absolute power is not comparable.
GAIN_SPLIT_RMS = 5e-5


def save_pair(fig, stem: str) -> None:
    """Save ``stem.svg`` + ``stem.png``, PNG capped at 1600 px on its long side."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / f"{stem}.svg")
    long_in = max(fig.get_size_inches())
    fig.savefig(config.FIGURES_DIR / f"{stem}.png", dpi=min(150, MAX_PNG_PX / long_in))
    plt.close(fig)
    print(f"[plot] {stem}.svg + .png", flush=True)


def load_rows():
    with (config.RESULTS_DIR / "lfp_inventory.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["mouse_id"] = int(r["mouse_id"])
        for k in ("duration_min", "median_channel_rms", "lf_hf_ratio",
                  "loglog_slope_2_40hz", "adjacent_r", "distant_r",
                  "common_mean_residual", "common_median_residual",
                  "line_ratio_50Hz", "exact_zero_fraction", "vr_first_s", "vr_last_s"):
            r[k] = float(r[k]) if r[k] not in ("", "None") else np.nan
        r["padding_start_s"] = (float(r["padding_start_s"])
                                if r["padding_start_s"] not in ("", "None") else None)
        r["tag"] = f"{r['mouse_id']}_{r['probe']}"
        r["label"] = f"{r['mouse_id']}{'·v1' if r['probe'] == 'visual' else ''}"
    return sorted(rows, key=lambda r: (r["mouse_id"], r["probe"]))


# --- 1. coverage + per-file diagnostics --------------------------------------

def plot_overview(rows, out="lfp_cohort_overview"):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))
    new_gain = np.array([r["median_channel_rms"] > GAIN_SPLIT_RMS for r in rows])
    colours = np.where(new_gain, "#d95319", "#0072b2")

    # (a) coverage matrix over the full task cohort
    ax = axes[0, 0]
    mice = list(config.TASK_MOUSE_IDS)
    have = {(r["mouse_id"], r["probe"]) for r in rows}
    expect_visual = {int(m) for m in _v1_csv_mice()}
    grid = np.full((2, len(mice)), np.nan)
    for j, m in enumerate(mice):
        grid[0, j] = 1.0 if (m, "striatum") in have else 0.0
        grid[1, j] = (1.0 if (m, "visual") in have
                      else (0.0 if m in expect_visual else np.nan))
    ax.imshow(grid, cmap="RdYlGn", vmin=-0.3, vmax=1.3, aspect="auto")
    ax.set_xticks(range(len(mice)))
    ax.set_xticklabels(mice, rotation=90, fontsize=7)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["probe 1\n(DMS/DLS/ACC)", "probe 2\n(V1/CA1/DG)"], fontsize=7)
    ax.set_title("(a) LFP coverage of the 16-animal task cohort\n"
                 "green = downloaded, red = missing, white = no such probe", fontsize=9)
    ax.set_xlabel("mouse ID")

    # (b) amplitude scale: the two gain regimes
    ax = axes[0, 1]
    ax.bar(x, [r["median_channel_rms"] for r in rows], color=colours)
    ax.axhline(GAIN_SPLIT_RMS, ls="--", c="k", lw=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("median channel RMS (stored units)")
    ax.set_title("(b) Amplitude scale differs ~30x between export batches", fontsize=9)
    _xticks(ax, x, labels)
    ax.legend(handles=_gain_handles(), fontsize=7, loc="upper left")

    # (c) mains contamination
    ax = axes[0, 2]
    ax.bar(x, [r["line_ratio_50Hz"] for r in rows], color=colours)
    for level, style, note in [(1, "-", "no peak"), (3, "--", "notch advisable"),
                               (100, ":", "50 Hz dominates")]:
        ax.axhline(level, ls=style, c="k", lw=0.8)
        ax.text(len(rows) - 0.4, level, f" {note}", fontsize=6, va="bottom", ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("50 Hz power / ±5 Hz shoulder (ratio)")
    ax.set_title("(c) Mains contamination per file", fontsize=9)
    _xticks(ax, x, labels)

    # (d) is it LFP? 1/f slope vs low-to-high power ratio
    ax = axes[1, 0]
    ax.scatter([r["loglog_slope_2_40hz"] for r in rows],
               [r["lf_hf_ratio"] for r in rows], c=colours, s=40)
    for r in rows:
        ax.annotate(r["label"], (r["loglog_slope_2_40hz"], r["lf_hf_ratio"]),
                    fontsize=6, xytext=(3, 2), textcoords="offset points")
    ax.axhspan(0.6, 1.5, color="grey", alpha=0.25)
    ax.text(-0.7, 1.0, "broadband / scrambled\n(June 2026 export)", fontsize=6, va="center")
    ax.set_yscale("log")
    ax.set_xlabel("1/f slope, log-log fit 2–40 Hz (decades per decade)")
    ax.set_ylabel("power 1–10 Hz / power 100–200 Hz (ratio)")
    ax.set_title("(d) Spectral shape: every file is LFP-like (far above the\n"
                 "broadband band), but 614/730/523 are markedly flatter", fontsize=9)

    # (e) depth structure
    ax = axes[1, 1]
    ax.scatter([r["adjacent_r"] for r in rows], [r["distant_r"] for r in rows],
               c=colours, s=40)
    for r in rows:
        ax.annotate(r["label"], (r["adjacent_r"], r["distant_r"]),
                    fontsize=6, xytext=(3, 2), textcoords="offset points")
    ax.axhline(0, c="k", lw=0.6)
    ax.set_xlabel("median r, adjacent channels (20 µm apart)")
    ax.set_ylabel("median r, channels 100 apart (~1 mm)")
    ax.set_title("(e) Depth smoothness: local correlation high,\n"
                 "millimetre-scale correlation ~0 (correct layout)", fontsize=9)

    # (f) referencing residual
    ax = axes[1, 2]
    w = 0.4
    ax.bar(x - w / 2, [r["common_mean_residual"] for r in rows], w,
           label="across-channel mean", color="#7e2f8e")
    ax.bar(x + w / 2, [r["common_median_residual"] for r in rows], w,
           label="across-channel median", color="#33b3b3")
    ax.set_ylabel("residual SD (in units of a channel SD)")
    ax.set_title("(f) Median residual ≪ mean residual:\nconsistent with common-MEDIAN referencing", fontsize=9)
    _xticks(ax, x, labels)
    ax.legend(fontsize=7)

    fig.suptitle("New LFP cohort: coverage and signal character "
                 f"({len(rows)} files, 1 kHz, 384 ch)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_pair(fig, out)


def _xticks(ax, x, labels):
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)


def _gain_handles():
    from matplotlib.patches import Patch
    return [Patch(color="#0072b2", label="2026-07 batch (low gain)"),
            Patch(color="#d95319", label="2026-08 batch (high gain)")]


def _v1_csv_mice():
    with open(config.V1_CSV, newline="") as fh:
        return [row["Mouse ID"] for row in csv.DictReader(fh) if row.get("Mouse ID")]


# --- 2. spectra --------------------------------------------------------------

def plot_spectra(rows, z, out="lfp_spectra"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
    cmap = plt.get_cmap("viridis")
    groups = [(False, "2026-07 batch (low gain)"), (True, "2026-08 batch (high gain)")]
    for ax, (is_new, title) in zip(axes, groups):
        sel = [r for r in rows if (r["median_channel_rms"] > GAIN_SPLIT_RMS) == is_new]
        for i, r in enumerate(sel):
            freqs = z[f"{r['tag']}__freqs"]
            psd = z[f"{r['tag']}__psd"].mean(axis=1)
            ax.loglog(freqs[1:], psd[1:], lw=1.1, color=cmap(i / max(1, len(sel) - 1)),
                      label=r["label"])
        for hz in (50, 150, 250):
            ax.axvline(hz, color="r", lw=0.6, alpha=0.4)
        ax.text(50, ax.get_ylim()[1], " 50 Hz mains\n and odd harmonics",
                fontsize=6, color="r", va="top")
        ax.set_xlabel("frequency (Hz)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=0.2, which="both")
    axes[0].set_ylabel("PSD, mean over 384 channels (stored units² / Hz)")
    fig.suptitle("Mean power spectrum per file (12 × 10 s windows inside behaviour)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_pair(fig, out)


# --- 3. depth x frequency ----------------------------------------------------

def plot_depth_frequency(rows, z, out="lfp_depth_by_frequency"):
    n = len(rows)
    ncol = 6
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.9 * nrow),
                             squeeze=False, sharex=True)
    fmax = 200
    for k, r in enumerate(rows):
        ax = axes[k // ncol][k % ncol]
        freqs = z[f"{r['tag']}__freqs"]
        psd = z[f"{r['tag']}__psd"]
        depth = z[f"{r['tag']}__depth"]
        if depth.size != psd.shape[1]:
            depth = geometry.channel_depths(psd.shape[1])
        sel = (freqs >= 1) & (freqs <= fmax)
        # Divide out each file's own median power: the two gain regimes differ by
        # ~1000x in power, so only the relative depth/frequency structure is comparable.
        rel = np.log10(psd[sel].T / np.median(psd[sel]))
        im = ax.pcolormesh(freqs[sel], depth, rel, cmap="magma",
                           vmin=-2.5, vmax=2.5, shading="nearest")
        ax.set_xscale("log")
        try:
            bounds = geometry.load_area_boundaries(r["mouse_id"], probe=r["probe"])
        except KeyError:
            bounds = {}
        for area, (lo, hi) in bounds.items():
            ax.axhspan(lo, hi, xmin=0.0, xmax=0.035, color=AREA_COLOUR[area], lw=0)
            ax.text(1.15, (lo + hi) / 2, area, fontsize=5.5, va="center",
                    color=AREA_COLOUR[area], fontweight="bold")
        ax.set_title(f"{r['label']}  ({r['probe']})", fontsize=8)
        if k % ncol == 0:
            ax.set_ylabel("depth from tip (µm)")
        if k // ncol == nrow - 1:
            ax.set_xlabel("frequency (Hz)")
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("log₁₀ PSD relative to that file's median (dimensionless)")
    fig.suptitle("Depth × frequency structure per file — coloured bars mark the "
                 "depth bands the sorted units use", fontsize=12)
    save_pair(fig, out)


# --- 4. integrity in time ----------------------------------------------------

def plot_integrity(rows, z, out="lfp_session_integrity"):
    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.15 * n), sharex=True, squeeze=False)
    for k, r in enumerate(rows):
        ax = axes[k][0]
        rms = z[f"{r['tag']}__rms_per_s"]
        t = np.arange(rms.size)
        ax.plot(t, rms, lw=0.4, color="#333333")
        ax.axvspan(r["vr_first_s"], r["vr_last_s"], color="#0072b2", alpha=0.12)
        if r["vr_last_s"] > rms.size:
            ax.annotate("VR continues past the end of the export",
                        xy=(rms.size, np.nanmedian(rms)), fontsize=6, color="red",
                        xytext=(6, 0), textcoords="offset points", va="center")
        if r["padding_start_s"] is not None:
            ax.axvspan(r["padding_start_s"], t[-1], color="red", alpha=0.15)
        ax.set_yscale("log")
        ax.set_ylabel(r["label"], fontsize=7, rotation=0, ha="right", va="center")
        ax.tick_params(labelsize=6)
    axes[-1][0].set_xlabel("time from file start (s)")
    fig.suptitle("Per-second median channel RMS over the whole session\n"
                 "blue = VR behaviour window, red = terminal zero padding\n"
                 "(1212: the VR session runs 190 min but the export stops at 140 min)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_pair(fig, out)


# --- 5. identity matrix ------------------------------------------------------

def plot_identity(out="lfp_identity_matrix"):
    path = config.RESULTS_DIR / "lfp_identity_matrix.csv"
    if not path.exists():
        print("[plot] no identity matrix cached; skipping")
        return
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    files = sorted({(int(r["mouse_id"]), r["probe"]) for r in rows})
    cands = sorted({int(r["candidate"]) for r in rows})
    windows = sorted({int(r["window_start_s"]) for r in rows})
    # Median across windows of the column-normalised score: "how much more does
    # this file couple to that animal than the other files do".
    stack = np.full((len(windows), len(files), len(cands)), np.nan)
    for r in rows:
        stack[windows.index(int(r["window_start_s"])),
              files.index((int(r["mouse_id"]), r["probe"])),
              cands.index(int(r["candidate"]))] = float(r["normalised"])
    mat = np.nanmedian(stack, axis=0)

    fig, ax = plt.subplots(figsize=(1.5 + 0.45 * len(cands), 1.5 + 0.42 * len(files)))
    finite = mat[np.isfinite(mat)]
    im = ax.imshow(mat, cmap="viridis", aspect="auto",
                   norm=LogNorm(vmin=max(finite.min(), 0.1), vmax=finite.max()))
    for i, (m, p) in enumerate(files):
        if m in cands:
            ax.add_patch(plt.Rectangle((cands.index(m) - .5, i - .5), 1, 1,
                                       fill=False, edgecolor="red", lw=1.6))
    ax.set_xticks(range(len(cands)))
    ax.set_xticklabels(cands, rotation=90, fontsize=7)
    ax.set_yticks(range(len(files)))
    ax.set_yticklabels([f"{m}{chr(183) + 'v1' if p == 'visual' else ''}" for m, p in files],
                       fontsize=7)
    ax.set_xlabel("candidate animal supplying the multi-unit rate")
    ax.set_ylabel("LFP file (named animal · probe)")
    ax.set_title("Every filename verified against physiology\n"
                 "30–90 Hz LFP envelope vs MUA, 100 ms bins, median over "
                 f"{len(windows)} × 10 min windows\n"
                 "score = |r| ÷ that candidate's median |r| over all files "
                 "(1 = no better than any other file)\n"
                 "red box = the animal the filename claims; "
                 "white = that animal has no probe-2 recording", fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    cb.set_label("coupling relative to the candidate's median (×)")
    fig.tight_layout()
    save_pair(fig, out)


def main() -> None:
    rows = load_rows()
    z = np.load(config.RESULTS_DIR / "lfp_psd.npz")
    plot_overview(rows, )
    plot_spectra(rows, z)
    plot_depth_frequency(rows, z)
    plot_integrity(rows, z)
    plot_identity()


if __name__ == "__main__":
    main()
