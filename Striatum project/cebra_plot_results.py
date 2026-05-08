"""
cebra_plot_results.py
=====================

Generate summary plots from cebra_results/cebra_results.npz produced by
cebra_analysis.py. Saves PNGs alongside the .npz file.

Plots:
  1. fig_decoder_r2_heatmap.png   — per-mouse, per-area held-out decoder R^2
  2. fig_decoder_r2_summary.png   — mean +/- SEM R^2 across mice, with per-mouse dots
  3. fig_decoder_rmse_summary.png — same but RMSE
  4. fig_consistency_matrix.png   — pairwise multi-session embedding consistency
  5. fig_decoder_r2_paired.png    — per-mouse R^2, lines connecting areas

Usage:
    cd "Striatum project"
    python cebra_plot_results.py            # uses ./cebra_results/
    python cebra_plot_results.py path/to/results_dir

Created 2026-05-08.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Avoid matplotlib trying to write into a full $HOME on shared envs
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cebra")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


# Project area colours, kept in lockstep with project_cfg.m
AREA_COLORS = {
    "all": (0.30, 0.30, 0.30),
    "DMS": (0.000, 0.4470, 0.7410),
    "DLS": (0.4660, 0.6740, 0.1880),
    "ACC": (0.8500, 0.3250, 0.0980),
    "V1":  (0.4940, 0.1840, 0.5560),
}


def _resolve_results_dir(argv: list[str]) -> Path:
    if len(argv) > 1:
        return Path(argv[1]).expanduser().resolve()
    return Path("./cebra_results").resolve()


def _load(results_dir: Path) -> dict:
    npz_path = results_dir / "cebra_results.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"Could not find {npz_path}")
    print(f"Loading {npz_path}")
    d = np.load(npz_path, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    # Normalise object arrays to Python lists/strings
    out["dataset_ids"]  = [str(x) for x in out["dataset_ids"]]
    out["label_keys"]   = [str(x) for x in out["label_keys"]]
    out["area_subsets"] = [str(x) for x in out["area_subsets"]]
    return out


# ------------------------------------------------------------------ #
# Plot 1 — decoder R^2 heatmap (mouse x area)
# ------------------------------------------------------------------ #


def plot_decoder_r2_heatmap(data: dict, out_dir: Path) -> None:
    r2 = data["single_decoder_r2"]                    # (n_mice, n_areas)
    mice = data["dataset_ids"]
    areas = data["area_subsets"]

    fig, ax = plt.subplots(figsize=(1.2 * len(areas) + 2, 0.35 * len(mice) + 2))
    # Centre at zero with a diverging colormap so negatives are visible
    vmax = max(0.5, np.nanmax(np.abs(r2)))
    im = ax.imshow(r2, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, origin="upper")

    # Annotate cells
    for i in range(r2.shape[0]):
        for j in range(r2.shape[1]):
            v = r2[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center",
                        color="0.4", fontsize=9)
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if abs(v) > 0.55 * vmax else "black",
                        fontsize=8)

    ax.set_xticks(range(len(areas)))
    ax.set_xticklabels(areas, rotation=0)
    ax.set_yticks(range(len(mice)))
    ax.set_yticklabels(mice)
    ax.set_xlabel("Area subset")
    ax.set_ylabel("Mouse")
    ax.set_title(f"CEBRA decoder R²  (target = position, held-out trials)\n"
                 f"NaN = subset skipped (too few units / no V1 probe)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("R² on held-out trials")
    fig.tight_layout()
    out = out_dir / "fig_decoder_r2_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


# ------------------------------------------------------------------ #
# Plot 2 — per-area summary (mean +/- SEM bars + per-mouse strip)
# ------------------------------------------------------------------ #


def _summary_panel(ax, values: np.ndarray, areas: list[str],
                   ylabel: str, title: str) -> None:
    n_areas = values.shape[1]
    means = np.nanmean(values, axis=0)
    sems  = np.nanstd(values, axis=0, ddof=1) / np.sqrt(
        np.maximum(np.sum(~np.isnan(values), axis=0), 1))
    counts = np.sum(~np.isnan(values), axis=0)

    xpos = np.arange(n_areas)
    bar_colors = [AREA_COLORS.get(a, (0.5, 0.5, 0.5)) for a in areas]
    ax.bar(xpos, means, color=bar_colors, edgecolor="black", alpha=0.6,
           yerr=sems, capsize=4, error_kw={"linewidth": 1.2})

    # Per-mouse jittered dots
    rng = np.random.default_rng(0)
    for j in range(n_areas):
        col = values[:, j]
        ok = ~np.isnan(col)
        if ok.any():
            jitter = rng.uniform(-0.12, 0.12, size=ok.sum())
            ax.scatter(np.full(ok.sum(), xpos[j]) + jitter, col[ok],
                       s=18, color="black", alpha=0.5, zorder=3)

    # Annotate sample sizes below x labels
    ymin = ax.get_ylim()[0]
    for j in range(n_areas):
        ax.text(xpos[j], ymin, f"n={counts[j]}",
                ha="center", va="top", fontsize=8, color="0.3",
                transform=ax.get_xaxis_transform())

    ax.axhline(0, color="0.6", linewidth=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(areas)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def plot_decoder_summary(data: dict, out_dir: Path) -> None:
    areas = data["area_subsets"]

    fig, ax = plt.subplots(figsize=(1.4 * len(areas) + 2, 4.5))
    _summary_panel(ax, data["single_decoder_r2"], areas,
                   ylabel="R² (held-out)",
                   title="CEBRA position decoder — R² by area subset")
    fig.tight_layout()
    out = out_dir / "fig_decoder_r2_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")

    fig, ax = plt.subplots(figsize=(1.4 * len(areas) + 2, 4.5))
    _summary_panel(ax, data["single_decoder_rmse"], areas,
                   ylabel="RMSE (held-out, bins)",
                   title="CEBRA position decoder — RMSE by area subset")
    fig.tight_layout()
    out = out_dir / "fig_decoder_rmse_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


# ------------------------------------------------------------------ #
# Plot 3 — paired per-mouse R^2 (lines connecting same mouse across areas)
# ------------------------------------------------------------------ #


def plot_paired_per_mouse(data: dict, out_dir: Path) -> None:
    r2 = data["single_decoder_r2"]
    mice = data["dataset_ids"]
    areas = data["area_subsets"]
    n_mice, n_areas = r2.shape
    xpos = np.arange(n_areas)

    fig, ax = plt.subplots(figsize=(1.4 * len(areas) + 2, 5))
    cmap = plt.get_cmap("tab20")
    for i in range(n_mice):
        row = r2[i, :]
        ok = ~np.isnan(row)
        if ok.sum() < 2:
            # Single point; just plot it
            if ok.any():
                ax.plot(xpos[ok], row[ok], "o", color=cmap(i % 20), alpha=0.9,
                        label=mice[i], markersize=7)
            continue
        ax.plot(xpos[ok], row[ok], "-o", color=cmap(i % 20), alpha=0.7,
                label=mice[i], markersize=6, linewidth=1.2)

    means = np.nanmean(r2, axis=0)
    ax.plot(xpos, means, "k-", linewidth=2.5, label="Mean", zorder=10)
    ax.scatter(xpos, means, s=80, color="black", zorder=11)

    ax.axhline(0, color="0.6", linewidth=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(areas)
    ax.set_ylabel("R² (held-out)")
    ax.set_title("CEBRA R² per mouse, paired across area subsets")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=8, frameon=False)
    fig.tight_layout()
    out = out_dir / "fig_decoder_r2_paired.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


# ------------------------------------------------------------------ #
# Plot 4 — multi-session consistency matrix
# ------------------------------------------------------------------ #


def plot_consistency_matrix(data: dict, out_dir: Path) -> None:
    scores = data["consistency_scores"]
    pairs  = data["consistency_pairs"]   # (n_pairs, 2) of strings
    mice = data["dataset_ids"]

    if scores.size == 0 or pairs.size == 0:
        print("No consistency data; skipping consistency plot.")
        return

    n = len(mice)
    mat = np.full((n, n), np.nan)
    name_to_idx = {m: i for i, m in enumerate(mice)}
    for k, (a, b) in enumerate(pairs):
        a, b = str(a), str(b)
        if a in name_to_idx and b in name_to_idx:
            i, j = name_to_idx[a], name_to_idx[b]
            mat[i, j] = scores[k]

    # If only one direction was reported, mirror it
    if np.isnan(mat[np.tril_indices(n, -1)]).all():
        mat[np.tril_indices(n, -1)] = mat.T[np.tril_indices(n, -1)]
    np.fill_diagonal(mat, 1.0)

    fig, ax = plt.subplots(figsize=(0.4 * n + 3, 0.4 * n + 3))
    im = ax.imshow(mat, aspect="equal", cmap="viridis",
                   vmin=np.nanmin(mat), vmax=1.0, origin="upper")
    ax.set_xticks(range(n)); ax.set_xticklabels(mice, rotation=90, fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(mice, fontsize=8)
    ax.set_title("Multi-session embedding consistency\n(higher = embeddings align across mice)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Consistency score")
    fig.tight_layout()
    out = out_dir / "fig_consistency_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


# ------------------------------------------------------------------ #
# Plot 5 — consistency vs decoding scatter
# ------------------------------------------------------------------ #


def _per_mouse_mean_consistency(data: dict) -> np.ndarray:
    """Mean off-diagonal consistency for each mouse (its row of the matrix)."""
    scores = data["consistency_scores"]
    pairs  = data["consistency_pairs"]
    mice   = data["dataset_ids"]
    n = len(mice)
    name_to_idx = {m: i for i, m in enumerate(mice)}
    mat = np.full((n, n), np.nan)
    for k, (a, b) in enumerate(pairs):
        a, b = str(a), str(b)
        if a in name_to_idx and b in name_to_idx:
            mat[name_to_idx[a], name_to_idx[b]] = scores[k]
    if np.isnan(mat[np.tril_indices(n, -1)]).all():
        mat[np.tril_indices(n, -1)] = mat.T[np.tril_indices(n, -1)]
    np.fill_diagonal(mat, np.nan)
    return np.nanmean(mat, axis=1)


def plot_consistency_vs_decoding(data: dict, out_dir: Path) -> None:
    if data["consistency_scores"].size == 0:
        return
    cons_per_mouse = _per_mouse_mean_consistency(data)
    r2_all = data["single_decoder_r2"][:, 0]   # 'all' is column 0
    mice = data["dataset_ids"]

    ok = ~np.isnan(cons_per_mouse) & ~np.isnan(r2_all)
    if ok.sum() < 3:
        print("Skipping consistency-vs-decoding: too few valid mice.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(cons_per_mouse[ok], r2_all[ok], s=70, color="black", alpha=0.7,
               edgecolor="white", zorder=3)
    for i in np.where(ok)[0]:
        ax.annotate(mice[i], (cons_per_mouse[i], r2_all[i]),
                    fontsize=8, alpha=0.8, xytext=(4, 4),
                    textcoords="offset points")

    # Pearson r
    x = cons_per_mouse[ok]; y = r2_all[ok]
    if np.std(x) > 0 and np.std(y) > 0:
        r = float(np.corrcoef(x, y)[0, 1])
        # Two-sided p-value via Fisher transform (no scipy)
        n = ok.sum()
        z = 0.5 * np.log((1 + r) / (1 - r))
        se = 1.0 / np.sqrt(n - 3)
        from math import erf, sqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(z) / se / sqrt(2))))
        # Fit line
        m, b = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, m * xs + b, "k--", linewidth=1, alpha=0.6)
        ax.set_title(f"Per-mouse: cross-mouse consistency vs internal decoding\n"
                     f"Pearson r = {r:+.2f}, p = {p:.3g}, n = {n}")
    else:
        ax.set_title("Per-mouse: cross-mouse consistency vs internal decoding")

    ax.set_xlabel("Mean consistency with other mice")
    ax.set_ylabel("Decoder R² (all units)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = out_dir / "fig_consistency_vs_decoding.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


# ------------------------------------------------------------------ #
# Plot 6 — R^2 distribution as box + strip
# ------------------------------------------------------------------ #


def plot_r2_distribution(data: dict, out_dir: Path) -> None:
    r2 = data["single_decoder_r2"]
    areas = data["area_subsets"]
    n_areas = len(areas)

    # Build per-area arrays without NaNs
    per_area = [r2[~np.isnan(r2[:, j]), j] for j in range(n_areas)]

    fig, ax = plt.subplots(figsize=(1.4 * n_areas + 2, 5))

    # Box plot (no scipy needed; matplotlib computes IQR)
    bp = ax.boxplot(per_area, positions=np.arange(n_areas), widths=0.55,
                    patch_artist=True, showfliers=False, zorder=1)
    for patch, area in zip(bp["boxes"], areas):
        patch.set_facecolor(AREA_COLORS.get(area, (0.5, 0.5, 0.5)))
        patch.set_alpha(0.35)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.4)

    # Strip overlay
    rng = np.random.default_rng(1)
    for j in range(n_areas):
        vals = per_area[j]
        if vals.size == 0:
            continue
        jitter = rng.uniform(-0.15, 0.15, size=vals.size)
        ax.scatter(np.full(vals.size, j) + jitter, vals,
                   s=22, color="black", alpha=0.7, zorder=3, edgecolor="white",
                   linewidths=0.5)

    # n labels under x ticks
    for j, vals in enumerate(per_area):
        ax.text(j, ax.get_ylim()[0], f"n={vals.size}",
                ha="center", va="top", fontsize=8, color="0.3",
                transform=ax.get_xaxis_transform())

    ax.axhline(0, color="0.6", linewidth=0.8)
    ax.set_xticks(np.arange(n_areas))
    ax.set_xticklabels(areas)
    ax.set_ylabel("R² (held-out)")
    ax.set_title("CEBRA decoder R² — distribution by area subset")
    fig.tight_layout()
    out = out_dir / "fig_decoder_r2_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


# ------------------------------------------------------------------ #
# Plot 7 — composite headline figure (poster/paper-ready)
# ------------------------------------------------------------------ #


def plot_headline_composite(data: dict, out_dir: Path) -> None:
    r2 = data["single_decoder_r2"]
    areas = data["area_subsets"]
    mice = data["dataset_ids"]

    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3,
                          width_ratios=[1.3, 1.0, 1.4],
                          height_ratios=[1.0, 1.0])

    # Panel A: R^2 heatmap
    axA = fig.add_subplot(gs[:, 0])
    vmax = max(0.3, np.nanmax(np.abs(r2)))
    im = axA.imshow(r2, aspect="auto", cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax, origin="upper")
    for i in range(r2.shape[0]):
        for j in range(r2.shape[1]):
            v = r2[i, j]
            if np.isnan(v):
                axA.text(j, i, "—", ha="center", va="center",
                         color="0.4", fontsize=8)
            else:
                axA.text(j, i, f"{v:.2f}", ha="center", va="center",
                         color="white" if abs(v) > 0.55 * vmax else "black",
                         fontsize=8)
    axA.set_xticks(range(len(areas))); axA.set_xticklabels(areas)
    axA.set_yticks(range(len(mice)));  axA.set_yticklabels(mice, fontsize=8)
    axA.set_xlabel("Area subset"); axA.set_ylabel("Mouse")
    axA.set_title("A. Per-mouse, per-area decoder R²")
    fig.colorbar(im, ax=axA, fraction=0.04, pad=0.02, label="R²")

    # Panel B: per-area summary bars
    axB = fig.add_subplot(gs[0, 1])
    _summary_panel(axB, r2, areas,
                   ylabel="R² (held-out)",
                   title="B. R² across mice (mean ± SEM)")

    # Panel C: paired per-mouse
    axC = fig.add_subplot(gs[1, 1])
    cmap = plt.get_cmap("tab20")
    n_areas = r2.shape[1]
    xpos = np.arange(n_areas)
    for i in range(r2.shape[0]):
        row = r2[i, :]
        ok = ~np.isnan(row)
        if ok.sum() >= 2:
            axC.plot(xpos[ok], row[ok], "-o", color=cmap(i % 20),
                     alpha=0.55, markersize=5, linewidth=1)
        elif ok.any():
            axC.plot(xpos[ok], row[ok], "o", color=cmap(i % 20),
                     alpha=0.7, markersize=5)
    means = np.nanmean(r2, axis=0)
    axC.plot(xpos, means, "k-", linewidth=2.5, zorder=10)
    axC.scatter(xpos, means, s=70, color="black", zorder=11)
    axC.axhline(0, color="0.6", linewidth=0.8)
    axC.set_xticks(xpos); axC.set_xticklabels(areas)
    axC.set_ylabel("R²")
    axC.set_title("C. Per-mouse trajectories across areas")

    # Panel D: consistency matrix
    axD = fig.add_subplot(gs[:, 2])
    if data["consistency_scores"].size > 0:
        n = len(mice)
        mat = np.full((n, n), np.nan)
        name_to_idx = {m: i for i, m in enumerate(mice)}
        for k, (a, b) in enumerate(data["consistency_pairs"]):
            a, b = str(a), str(b)
            if a in name_to_idx and b in name_to_idx:
                mat[name_to_idx[a], name_to_idx[b]] = data["consistency_scores"][k]
        if np.isnan(mat[np.tril_indices(n, -1)]).all():
            mat[np.tril_indices(n, -1)] = mat.T[np.tril_indices(n, -1)]
        np.fill_diagonal(mat, 1.0)
        im2 = axD.imshow(mat, aspect="equal", cmap="viridis",
                         vmin=np.nanmin(mat), vmax=1.0, origin="upper")
        axD.set_xticks(range(n)); axD.set_xticklabels(mice, rotation=90, fontsize=7)
        axD.set_yticks(range(n)); axD.set_yticklabels(mice, fontsize=7)
        axD.set_title("D. Multi-session consistency")
        fig.colorbar(im2, ax=axD, fraction=0.046, pad=0.04, label="Consistency")
    else:
        axD.text(0.5, 0.5, "No consistency data", ha="center", va="center")
        axD.set_axis_off()

    fig.suptitle("CEBRA position-decoding summary  (held-out trials, all task mice)",
                 fontsize=15, fontweight="bold")
    out = out_dir / "fig_headline_composite.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ------------------------------------------------------------------ #
# Embedding loaders
# ------------------------------------------------------------------ #


def _load_embeddings_mat(results_dir: Path) -> dict:
    """Load per-mouse embeddings from cebra_results.mat.

    Returns a dict {mouse_idx: ndarray (n_samples, 3)}.
    Requires scipy or falls back to h5py.
    """
    mat_path = results_dir / "cebra_results.mat"
    if not mat_path.is_file():
        raise FileNotFoundError(f"Embeddings file not found: {mat_path}")

    try:
        from scipy.io import loadmat
        m = loadmat(mat_path, squeeze_me=True)
        out = {}
        for k, v in m.items():
            if k.startswith("embedding_mouse"):
                idx = int(k.replace("embedding_mouse", ""))
                out[idx] = np.asarray(v)
        return out
    except (ImportError, NotImplementedError):
        import h5py
        out = {}
        with h5py.File(mat_path, "r") as f:
            for k in f.keys():
                if k.startswith("embedding_mouse"):
                    idx = int(k.replace("embedding_mouse", ""))
                    arr = np.array(f[k])
                    if arr.ndim == 2 and arr.shape[0] == 3:
                        arr = arr.T
                    out[idx] = arr
        return out


def _load_labels_for_mouse(data_dir: Path, mouse_idx: int,
                           label_keys: list[str]) -> dict | None:
    """Reload per-timepoint labels for one mouse, applying the same NaN-drop
    mask CEBRA used at fit time, so they line up row-for-row with the embedding.
    Returns None if the file is missing.
    """
    import h5py

    path = data_dir / f"cebra_mouse{mouse_idx}_data.mat"
    if not path.is_file():
        return None

    with h5py.File(path, "r") as f:
        # neural_data: post-h5 shape (n_trials, n_bins, n_neurons)
        nd_h5 = np.array(f["neural_data"])
        # transpose to MATLAB-native (n_neurons, n_bins, n_trials), then
        # reorder to (n_trials, n_bins, n_neurons) for time-major flatten
        nd = np.transpose(nd_h5, tuple(reversed(range(nd_h5.ndim))))
        n_neurons, n_bins, n_trials = nd.shape
        nd_tbN = np.transpose(nd, (2, 1, 0))
        neural = nd_tbN.reshape(n_trials * n_bins, n_neurons)

        bin_id   = np.tile(np.arange(n_bins), n_trials)
        trial_id = np.repeat(np.arange(n_trials), n_bins)

        def _shape_TB(name):
            if name not in f:
                return None
            arr = np.array(f[name])
            arr = np.transpose(arr, tuple(reversed(range(arr.ndim))))
            if arr.shape == (n_trials, n_bins): return arr
            if arr.shape == (n_bins, n_trials): return arr.T
            return None

        lick_rate_TB   = _shape_TB("lick_rate")
        velocity_TB    = _shape_TB("velocity")
        lick_errors_T  = (np.array(f["lick_errors"]).ravel()
                          if "lick_errors" in f else None)
        if lick_errors_T is not None and lick_errors_T.size != n_trials:
            lick_errors_T = lick_errors_T[:n_trials]
        lp = (float(np.array(f["learning_point"]).ravel()[0])
              if "learning_point" in f else float("nan"))

    labels = {
        "position":     bin_id.astype(float),
        "trial_number": trial_id.astype(float),
        "bin_id":       bin_id,
        "trial_id":     trial_id,
    }
    if lick_rate_TB   is not None: labels["lick_rate"]   = lick_rate_TB.reshape(-1)
    if velocity_TB    is not None: labels["velocity"]    = velocity_TB.reshape(-1)
    if lick_errors_T  is not None: labels["lick_errors"] = np.repeat(lick_errors_T, n_bins)

    # Reproduce the NaN-drop mask CEBRA applied at fit time.
    label_cols = []
    for k in label_keys:
        if k in labels:
            label_cols.append(labels[k])
    if label_cols:
        L = np.column_stack(label_cols)
        mask = ~(np.isnan(neural).any(axis=1) | np.isnan(L).any(axis=1))
    else:
        mask = ~np.isnan(neural).any(axis=1)

    out = {}
    for k, v in labels.items():
        out[k] = v[mask]
    out["learning_point"] = lp
    out["n_trials"]       = n_trials
    out["n_bins"]         = n_bins
    return out


# ------------------------------------------------------------------ #
# Plot 8 — per-mouse 3D embedding scatter, coloured 4 ways
# ------------------------------------------------------------------ #


def _scatter_emb(ax, emb: np.ndarray, c, cmap: str, title: str,
                 cbar_label: str = "", cbar_fig=None):
    sc = ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2],
                    c=c, cmap=cmap, s=2, alpha=0.7, edgecolor="none")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    if cbar_fig is not None and cbar_label:
        cb = cbar_fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, shrink=0.7)
        cb.set_label(cbar_label, fontsize=8)
        cb.ax.tick_params(labelsize=7)


def plot_embeddings_per_mouse(data: dict, results_dir: Path) -> None:
    """One 2x2 panel per mouse: embedding coloured by position, lick_rate,
    lick_errors, trial_number. Saved as fig_emb_mouseN.png each."""
    embeddings = _load_embeddings_mat(results_dir)
    if not embeddings:
        print("No embeddings found in cebra_results.mat.")
        return

    data_dir = Path("./cebra_data").resolve()
    out_dir = results_dir / "embedding_panels"
    out_dir.mkdir(parents=True, exist_ok=True)

    mice_drawn = 0
    for mouse_idx in sorted(embeddings.keys()):
        emb = embeddings[mouse_idx]
        if emb.ndim != 2 or emb.shape[1] != 3:
            print(f"Mouse {mouse_idx}: unexpected embedding shape {emb.shape}; skipping.")
            continue

        labels = _load_labels_for_mouse(data_dir, mouse_idx, list(data["label_keys"]))
        if labels is None:
            print(f"Mouse {mouse_idx}: no cebra_data file; skipping.")
            continue
        if labels["position"].size != emb.shape[0]:
            print(f"Mouse {mouse_idx}: label/embedding size mismatch "
                  f"({labels['position'].size} vs {emb.shape[0]}); skipping.")
            continue

        fig = plt.figure(figsize=(11, 9), constrained_layout=True)
        axs = [fig.add_subplot(2, 2, k + 1, projection="3d") for k in range(4)]

        _scatter_emb(axs[0], emb, labels["position"], "viridis",
                     f"Mouse {mouse_idx} — coloured by position",
                     "spatial bin", fig)
        _scatter_emb(axs[1], emb, labels["trial_number"], "plasma",
                     "by trial number", "trial #", fig)
        if "lick_rate" in labels:
            _scatter_emb(axs[2], emb, labels["lick_rate"], "magma",
                         "by lick rate", "lick rate", fig)
        else:
            axs[2].set_axis_off()
            axs[2].text2D(0.5, 0.5, "no lick_rate", transform=axs[2].transAxes,
                          ha="center", va="center")
        if "lick_errors" in labels:
            _scatter_emb(axs[3], emb, labels["lick_errors"], "coolwarm",
                         "by z-scored lick error", "z-error", fig)
        else:
            axs[3].set_axis_off()
            axs[3].text2D(0.5, 0.5, "no lick_errors", transform=axs[3].transAxes,
                          ha="center", va="center")

        lp = labels.get("learning_point", float("nan"))
        n_t = labels.get("n_trials", "?")
        fig.suptitle(f"CEBRA embedding — Mouse {mouse_idx}   "
                     f"(N={emb.shape[1]}D, T≈{n_t} trials, LP={lp})",
                     fontsize=12, fontweight="bold")
        out = out_dir / f"fig_emb_mouse{mouse_idx}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        mice_drawn += 1

    print(f"Wrote {mice_drawn} per-mouse embedding panels to {out_dir}/")


# ------------------------------------------------------------------ #
# Plot 9 — multi-mouse grid coloured by position
# ------------------------------------------------------------------ #


def plot_embedding_grid_by_position(data: dict, results_dir: Path) -> None:
    """One figure with all mice as 3D scatters coloured by position."""
    embeddings = _load_embeddings_mat(results_dir)
    if not embeddings:
        return

    data_dir = Path("./cebra_data").resolve()
    mouse_ids = sorted(embeddings.keys())
    n_mice = len(mouse_ids)
    if n_mice == 0:
        return

    n_cols = min(4, n_mice)
    n_rows = int(np.ceil(n_mice / n_cols))

    fig = plt.figure(figsize=(3.5 * n_cols, 3.2 * n_rows), constrained_layout=True)
    drawn = 0
    last_sc = None
    for plot_idx, mouse_idx in enumerate(mouse_ids):
        emb = embeddings[mouse_idx]
        if emb.ndim != 2 or emb.shape[1] != 3:
            continue
        labels = _load_labels_for_mouse(data_dir, mouse_idx, list(data["label_keys"]))
        if labels is None or labels["position"].size != emb.shape[0]:
            continue
        ax = fig.add_subplot(n_rows, n_cols, plot_idx + 1, projection="3d")
        sc = ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2],
                        c=labels["position"], cmap="viridis",
                        s=1.5, alpha=0.7, edgecolor="none")
        last_sc = sc
        lp = labels.get("learning_point", float("nan"))
        ax.set_title(f"Mouse {mouse_idx}  (LP={lp})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        drawn += 1

    fig.suptitle("Per-mouse multi-session CEBRA embeddings, coloured by position",
                 fontsize=14, fontweight="bold")
    if last_sc is not None:
        cb = fig.colorbar(last_sc, ax=fig.axes, shrink=0.5,
                          fraction=0.02, pad=0.02)
        cb.set_label("Spatial bin")

    out = results_dir / "fig_embedding_grid_by_position.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}  ({drawn}/{n_mice} mice drawn)")


# ------------------------------------------------------------------ #
# Plot 10 — embedding density by learning epoch
# ------------------------------------------------------------------ #


def _epoch_index(trial_id: np.ndarray, lp: float, w: int = 10) -> np.ndarray:
    """Return an integer 0/1/2/3 array tagging Naive (1:w), Pre-LP, Expert,
    or 'other'. NaN LP → only Naive is tagged, rest is 'other'."""
    out = np.full(trial_id.shape, 3, dtype=int)  # 3 = other
    out[trial_id < w] = 0  # Naive
    if not np.isnan(lp):
        out[(trial_id >= lp - w) & (trial_id < lp)] = 1  # Pre-LP
        out[(trial_id >= lp) & (trial_id < lp + w)] = 2  # Expert
    return out


def plot_embedding_density_by_epoch(data: dict, results_dir: Path) -> None:
    """For each mouse, project the 3D embedding onto its first two PCs and
    show a 2D KDE per epoch (Naive / Pre-LP / Expert). Saves a single figure
    with up to 6 mice (the ones with the largest non-Other epoch counts)."""
    embeddings = _load_embeddings_mat(results_dir)
    if not embeddings:
        return

    data_dir = Path("./cebra_data").resolve()

    # Pick mice that have both Naive and Expert tagged
    candidates = []
    for mid, emb in embeddings.items():
        labels = _load_labels_for_mouse(data_dir, mid, list(data["label_keys"]))
        if labels is None or labels["position"].size != emb.shape[0]:
            continue
        ep = _epoch_index(labels["trial_id"], labels["learning_point"])
        if (ep == 0).sum() > 30 and (ep == 2).sum() > 30:
            candidates.append((mid, emb, labels, ep))

    if not candidates:
        print("No mice have both Naive and Expert epochs; skipping density plot.")
        return

    candidates.sort(key=lambda t: -t[1].shape[0])
    candidates = candidates[:6]

    n = len(candidates)
    fig, axs = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False,
                            constrained_layout=True)
    epoch_names = ["Naive", "Pre-LP", "Expert"]

    for r, (mid, emb, labels, ep) in enumerate(candidates):
        # 2D projection: first two PCs of the 3D embedding for visual clarity
        emb_centered = emb - emb.mean(axis=0)
        u, s, vt = np.linalg.svd(emb_centered, full_matrices=False)
        emb2d = emb_centered @ vt[:2].T   # (n_samples, 2)

        # Common axis limits across the row
        xlim = (np.percentile(emb2d[:, 0], 1), np.percentile(emb2d[:, 0], 99))
        ylim = (np.percentile(emb2d[:, 1], 1), np.percentile(emb2d[:, 1], 99))

        for c, ep_id in enumerate([0, 1, 2]):
            ax = axs[r, c]
            mask = ep == ep_id
            if mask.sum() == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes); ax.set_axis_off(); continue

            # 2D histogram for density (no scipy KDE needed)
            H, xe, ye = np.histogram2d(emb2d[mask, 0], emb2d[mask, 1],
                                       bins=40, range=[xlim, ylim])
            H = H.T
            ax.imshow(H, origin="lower", extent=(*xlim, *ylim),
                      aspect="auto", cmap="viridis")
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"Mouse {mid}\nLP={labels['learning_point']}",
                              fontsize=9)
            if r == 0:
                ax.set_title(f"{epoch_names[c]} (n={int(mask.sum())} samples)",
                             fontsize=10)

    fig.suptitle("CEBRA embedding density by learning epoch  "
                 "(2D projection onto top-2 PCs of the 3D embedding)",
                 fontsize=12, fontweight="bold")
    out = results_dir / "fig_embedding_density_by_epoch.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ------------------------------------------------------------------ #
# Plot 11 — per-spatial-bin decoding accuracy
# ------------------------------------------------------------------ #


def _trialwise_split(trial_id: np.ndarray, test_frac: float, seed: int):
    """Same trial-wise split rule used inside cebra_analysis.fit_single_session."""
    rng = np.random.default_rng(seed)
    unique_trials = np.unique(trial_id)
    n_test = max(1, int(round(test_frac * unique_trials.size)))
    test_trials = rng.choice(unique_trials, size=n_test, replace=False)
    test_mask = np.isin(trial_id, test_trials)
    return ~test_mask, test_mask


def _per_mouse_held_out_predictions(data: dict, results_dir: Path,
                                    base_seed: int = 42,
                                    test_frac: float = 0.25,
                                    ridge_alpha: float = 1.0):
    """For each mouse: split trial-wise, fit ridge on the multi-session
    'all units' embedding, return (true_bin, pred_bin) on held-out timepoints."""
    from sklearn.linear_model import Ridge

    embeddings = _load_embeddings_mat(results_dir)
    if not embeddings:
        return {}, 0

    data_dir = Path("./cebra_data").resolve()
    label_keys = list(data["label_keys"])

    out = {}
    n_bins_seen = 0
    for mouse_idx, emb in embeddings.items():
        if emb.ndim != 2 or emb.shape[1] != 3:
            continue
        labels = _load_labels_for_mouse(data_dir, mouse_idx, label_keys)
        if labels is None or labels["position"].size != emb.shape[0]:
            continue

        trial_id = labels["trial_id"]
        bin_id   = labels["bin_id"]
        n_bins_seen = max(n_bins_seen, int(bin_id.max()) + 1)

        train_mask, test_mask = _trialwise_split(
            trial_id, test_frac, base_seed + mouse_idx)
        if test_mask.sum() < 5 or train_mask.sum() < 10:
            continue

        # Standardise embedding using train fold only
        mu = emb[train_mask].mean(axis=0)
        sd = emb[train_mask].std(axis=0); sd[sd == 0] = 1
        Xtr = (emb[train_mask] - mu) / sd
        Xte = (emb[test_mask]  - mu) / sd

        ridge = Ridge(alpha=ridge_alpha)
        ridge.fit(Xtr, bin_id[train_mask].astype(float))
        y_pred = ridge.predict(Xte)
        y_true = bin_id[test_mask].astype(float)

        out[mouse_idx] = (y_true, y_pred)

    return out, n_bins_seen


def plot_per_bin_decoding(data: dict, results_dir: Path) -> None:
    """Two figures:
       (a) Per-mouse and mean +/- SEM decoder error vs true spatial bin,
           with chance baseline and visual/reward zones marked.
       (b) Pooled confusion-style 2D histogram of true vs predicted bin.
    """
    per_mouse, n_bins = _per_mouse_held_out_predictions(data, results_dir)
    if not per_mouse:
        print("No held-out predictions; skipping per-bin decoding plot.")
        return

    n_mice = len(per_mouse)
    err_mat = np.full((n_mice, n_bins), np.nan)
    rmse_mat = np.full((n_mice, n_bins), np.nan)
    pos_means = []
    for i, (y_true, y_pred) in enumerate(per_mouse.values()):
        pos_means.append(np.mean(y_true))
        for b in range(n_bins):
            mk = y_true == b
            if mk.any():
                d = y_pred[mk] - b
                err_mat[i, b]  = float(np.mean(np.abs(d)))
                rmse_mat[i, b] = float(np.sqrt(np.mean(d * d)))
    chance_pos = float(np.nanmean(pos_means))
    chance_err = np.abs(np.arange(n_bins) - chance_pos)

    mu  = np.nanmean(err_mat, axis=0)
    sem = np.nanstd(err_mat, axis=0, ddof=1) / np.sqrt(
        np.maximum(np.sum(~np.isnan(err_mat), axis=0), 1))

    # Convert bins -> cm for a secondary axis (1 bin = 4 a.u. * 1.25 cm/au = 5 cm)
    BIN_CM = 5.0
    visual_zone = (20, 25)   # bins
    reward_zone = (25, 33)   # bins

    fig, ax = plt.subplots(figsize=(11, 5.2))

    # Per-mouse light traces
    for i in range(n_mice):
        ax.plot(np.arange(n_bins), err_mat[i, :],
                color="0.6", alpha=0.35, linewidth=0.8)

    # Mean +/- SEM
    ax.fill_between(np.arange(n_bins), mu - sem, mu + sem,
                    color="C0", alpha=0.25)
    ax.plot(np.arange(n_bins), mu, color="C0", linewidth=2.5,
            label=f"CEBRA (mean ± SEM, n={n_mice})")

    # Chance baseline
    ax.plot(np.arange(n_bins), chance_err, "k--", linewidth=1.4,
            label="Chance (predict global mean)")

    # Landmarks
    ax.axvspan(*visual_zone, color="dodgerblue", alpha=0.12,
               label="Visual zone")
    ax.axvspan(*reward_zone, color="forestgreen", alpha=0.12,
               label="Reward zone")

    ax.set_xlabel("True spatial bin (along corridor)")
    ax.set_ylabel("Decoding error  |predicted − true|  (bins)")
    ax.set_title("Per-spatial-bin decoder error\n"
                 "CEBRA multi-session 'all units' embedding, held-out trials")
    ax.set_xlim(0, n_bins - 1)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Secondary x-axis: cm
    secax = ax.secondary_xaxis(
        "top", functions=(lambda x: x * BIN_CM, lambda x: x / BIN_CM))
    secax.set_xlabel("Position along corridor (cm)")

    fig.tight_layout()
    out = results_dir / "fig_decoding_error_per_bin.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")

    # ---- Confusion-like heatmap ----
    all_true = np.concatenate([y[0] for y in per_mouse.values()])
    all_pred = np.concatenate([y[1] for y in per_mouse.values()])
    all_pred_int = np.clip(np.round(all_pred).astype(int), 0, n_bins - 1)
    H, _, _ = np.histogram2d(all_true.astype(int), all_pred_int,
                             bins=[n_bins, n_bins],
                             range=[[0, n_bins], [0, n_bins]])
    row_sums = H.sum(axis=1, keepdims=True); row_sums[row_sums == 0] = 1
    H_norm = H / row_sums  # P(predicted | true)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(H_norm.T, origin="lower", cmap="magma", aspect="auto",
                   extent=(-0.5, n_bins - 0.5, -0.5, n_bins - 0.5))
    ax.plot([0, n_bins - 1], [0, n_bins - 1], "w--",
            linewidth=1, alpha=0.7, label="Perfect")
    # Landmarks on both axes
    for span, c in [(visual_zone, "dodgerblue"), (reward_zone, "forestgreen")]:
        ax.axvspan(*span, color=c, alpha=0.10)
        ax.axhspan(*span, color=c, alpha=0.10)
    ax.set_xlabel("True spatial bin")
    ax.set_ylabel("Predicted bin (rounded)")
    ax.set_title("Decoder confusion matrix\n"
                 "P(predicted | true), pooled across mice (held-out trials)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(pred | true)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out2 = results_dir / "fig_decoder_confusion.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"Wrote {out2}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #


def main(argv: list[str]) -> int:
    results_dir = _resolve_results_dir(argv)
    if not results_dir.is_dir():
        print(f"ERROR: results directory does not exist: {results_dir}")
        return 1

    data = _load(results_dir)

    # Quick console summary
    r2 = data["single_decoder_r2"]
    print("\n========== Decoder R² by mouse × area ==========")
    print(f"Areas: {data['area_subsets']}")
    print(f"Mean R² (across mice): " +
          ", ".join(f"{a}={np.nanmean(r2[:, j]):.3f}"
                    for j, a in enumerate(data["area_subsets"])))
    print(f"N mice with data per area: " +
          ", ".join(f"{a}={int(np.sum(~np.isnan(r2[:, j])))}/{r2.shape[0]}"
                    for j, a in enumerate(data["area_subsets"])))
    print()

    plot_decoder_r2_heatmap(data, results_dir)
    plot_decoder_summary(data, results_dir)
    plot_paired_per_mouse(data, results_dir)
    plot_consistency_matrix(data, results_dir)
    plot_consistency_vs_decoding(data, results_dir)
    plot_r2_distribution(data, results_dir)
    plot_headline_composite(data, results_dir)

    # Embedding plots — require scipy + h5py + the per-mouse cebra_data files.
    # Skipped silently if dependencies aren't installed.
    try:
        plot_embeddings_per_mouse(data, results_dir)
        plot_embedding_grid_by_position(data, results_dir)
        plot_embedding_density_by_epoch(data, results_dir)
        plot_per_bin_decoding(data, results_dir)
    except (ImportError, FileNotFoundError) as e:
        print(f"\nSkipping embedding plots: {e}")
        print("Run from the cebra env (scipy + h5py available) to enable them.")

    print(f"\nAll figures written to {results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
