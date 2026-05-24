"""Stage 2 figures — round 7: two epochs (naive vs expert), four configs.

Loads results/stage2_<tag>.pkl for each robustness config

    res_fsexcl   res_fsincl   sig_fsexcl   sig_fsincl

(residual/signal CCA x fast-spiking units excluded/included; all z-scored over
the whole engaged period, held-out CC) and writes, per config:

  * stage2_comm_strength_<tag>.png   held-out CC over significant subspace
                                     dims, naive vs expert box + points
  * stage2_subspace_dim_<tag>.png    # significant subspace dims per pair
  * stage2_lag_curves_<tag>.png      held-out CC vs spatial lag over
                                     significant dims, line + shaded SEM
  * stage2_ifi_winNN_<tag>.png       IFI over significant dims, one figure
                                     per lag-integration window (1..10)

Statistics (learners only):
  * '*' above a box  -- that epoch's values differ from 0 (one-sample
                        Wilcoxon, n = significant dims)
  * pair p           -- naive vs expert, paired Wilcoxon over per-pair means
                        of the significant dims (n = pairs)
  * unpaired p       -- naive vs expert, Mann-Whitney over all significant
                        dims pooled (n = dims)

Run:  python scripts/plot_stage2.py
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

from striatum_cca import config  # noqa: E402

EPOCHS = config.EPOCH_NAMES
EPOCH_LABEL = ["naive", "expert"]
ALPHA = 0.05

CONFIGS = ("res_fsexcl", "res_fsincl", "sig_fsexcl", "sig_fsincl")
CONFIG_LABEL = {
    "res_fsexcl": "residual CCA, FS-excluded",
    "res_fsincl": "residual CCA, FS-included",
    "sig_fsexcl": "signal CCA, FS-excluded",
    "sig_fsincl": "signal CCA, FS-included",
}


def _configs_for(variant):
    """Temporal variants use the signal CCA only (no residual subtraction)."""
    if variant.startswith("t"):
        return tuple(c for c in CONFIGS if c.startswith("sig_"))
    return CONFIGS


def _describe(tag):
    """Human-readable config label from a file tag (with optional variant)."""
    for cfg_name, label in CONFIG_LABEL.items():
        if tag == cfg_name:
            return label
        if tag.endswith("_" + cfg_name):
            return f"{label}; {tag[: -len(cfg_name) - 1]}"
    return tag


# --- data access -------------------------------------------------------------
def learner_pairs(results, area_x, area_y):
    return [r for r in results
            if (r.area_x, r.area_y) == (area_x, area_y) and r.role == "learner"]


def sig_dims(epoch_analysis):
    """Indices of canonical dimensions passing the held-out significance test."""
    return np.where(epoch_analysis.p_per_dim < ALPHA)[0]


def dim_values(r, epoch, kind, window=0):
    """Significant-dim values for one pair x epoch ('cc' or 'ifi')."""
    ea = r.epochs[epoch]
    js = sig_dims(ea)
    if kind == "cc":
        return ea.held_out_cc[js]
    return ea.ifi_windows[js, window - 1]


def pooled(rs, epoch, kind, window=0):
    """All significant-dim values pooled across pairs (n = dims)."""
    if not rs:
        return np.array([])
    return np.concatenate([dim_values(r, epoch, kind, window) for r in rs])


def per_pair(rs, epoch, kind, window=0):
    """Per-pair mean over significant dims; NaN where a pair has none."""
    out = []
    for r in rs:
        v = dim_values(r, epoch, kind, window)
        out.append(float(np.nanmean(v)) if v.size else np.nan)
    return np.array(out)


# --- statistics --------------------------------------------------------------
def _wilcoxon_vs0(v):
    """One-sample Wilcoxon vs 0; NaN if too few non-zero values."""
    v = v[np.isfinite(v)]
    if v.size < 6 or not np.any(v != 0):
        return np.nan
    try:
        return stats.wilcoxon(v).pvalue
    except ValueError:
        return np.nan


def _paired(a, b):
    """Paired Wilcoxon over pairs finite in both epochs (n = pairs)."""
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 6 or not np.any(a != b):
        return np.nan
    try:
        return stats.wilcoxon(a, b).pvalue
    except ValueError:
        return np.nan


def _unpaired(a, b):
    """Mann-Whitney over all significant dims pooled (n = dims)."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 3 or b.size < 3:
        return np.nan
    try:
        return stats.mannwhitneyu(a, b).pvalue
    except ValueError:
        return np.nan


def _fmt_p(p):
    return f"{p:.3f}" if np.isfinite(p) else "n/a"


# --- plotting primitives -----------------------------------------------------
def _grid():
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.5))
    return fig, axes.ravel()


def _save(fig, name, tag):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    path = config.FIGURES_DIR / f"{name}_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def _box_by_epoch(ax, per_epoch, colour):
    """Box plot + jittered semi-transparent points for each epoch."""
    for i, v in enumerate(per_epoch):
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        bp = ax.boxplot([v], positions=[i], widths=0.55, showfliers=False,
                        patch_artist=True, medianprops=dict(color="k"))
        bp["boxes"][0].set(facecolor=colour, alpha=0.35)
        jit = (np.random.default_rng(i).random(v.size) - 0.5) * 0.3
        ax.scatter(i + jit, v, s=12, color=colour, alpha=0.5, zorder=3)
    n = len(per_epoch)
    ax.set_xticks(range(n))
    ax.set_xticklabels(EPOCH_LABEL[:n])
    ax.set_xlim(-0.6, n - 0.4)


def _star_vs0(ax, per_epoch):
    """Star above an epoch's box when its values differ from 0 (Wilcoxon)."""
    for i, v in enumerate(per_epoch):
        p = _wilcoxon_vs0(v)
        if np.isfinite(p) and p < ALPHA:
            ax.text(i, 0.93, "*", transform=ax.get_xaxis_transform(),
                    fontsize=14, ha="center")


# --- figures -----------------------------------------------------------------
def plot_comm_strength(results, tag):
    """Held-out CC over significant subspace dims, naive vs expert."""
    fig, axes = _grid()
    for ax, (ax_x, ax_y) in zip(axes, config.PAIRS):
        rs = learner_pairs(results, ax_x, ax_y)
        per_epoch = [pooled(rs, e, "cc") for e in EPOCHS]
        _box_by_epoch(ax, per_epoch, "tab:blue")
        ax.axhline(0, color="k", lw=0.6)
        _star_vs0(ax, per_epoch)
        pp = _paired(per_pair(rs, "naive", "cc"), per_pair(rs, "expert", "cc"))
        up = _unpaired(per_epoch[0], per_epoch[1])
        n = sum(v.size for v in per_epoch)
        ax.set_title(f"{ax_x}-{ax_y}  {n}d  "
                     f"pair={_fmt_p(pp)} unp={_fmt_p(up)}", fontsize=8)
    for ax in axes[::5]:
        ax.set_ylabel("held-out CC (significant dims)")
    fig.suptitle(f"Stage 2 — communication strength, naive vs expert "
                 f"({_describe(tag)}; '*' CC!=0; learners)")
    _save(fig, "stage2_comm_strength", tag)


def plot_subspace_dim(results, tag):
    """Number of significant communication-subspace dimensions per pair."""
    fig, axes = _grid()
    for ax, (ax_x, ax_y) in zip(axes, config.PAIRS):
        rs = learner_pairs(results, ax_x, ax_y)
        per_epoch = [np.array([float(r.epochs[e].n_significant) for r in rs])
                     for e in EPOCHS]
        _box_by_epoch(ax, per_epoch, "tab:green")
        pp = _paired(per_epoch[0], per_epoch[1])
        up = _unpaired(per_epoch[0], per_epoch[1])
        ax.set_ylim(bottom=0)
        ax.set_title(f"{ax_x}-{ax_y}  n={len(rs)}  "
                     f"pair={_fmt_p(pp)} unp={_fmt_p(up)}", fontsize=8)
    for ax in axes[::5]:
        ax.set_ylabel("# significant subspace dims")
    fig.suptitle(f"Stage 2 — communication-subspace dimensionality, naive vs "
                 f"expert ({_describe(tag)}; learners)")
    _save(fig, "stage2_subspace_dim", tag)


def plot_lag_curves(results, tag):
    """Held-out CC vs spatial lag over significant subspace dimensions."""
    fig, axes = _grid()
    colours = {"naive": "tab:blue", "expert": "tab:red"}
    for ax, (ax_x, ax_y) in zip(axes, config.PAIRS):
        rs = learner_pairs(results, ax_x, ax_y)
        n_dims = 0
        for epoch in EPOCHS:
            curves = np.array([r.epochs[epoch].lag_cc_per_dim[:, j]
                               for r in rs for j in sig_dims(r.epochs[epoch])])
            if curves.size == 0:
                continue
            n_dims += curves.shape[0]
            lags = rs[0].epochs[epoch].lags
            mean = np.nanmean(curves, axis=0)
            sem = np.nanstd(curves, axis=0) / np.sqrt(curves.shape[0])
            ax.plot(lags, mean, "-", color=colours[epoch], lw=1.6, label=epoch)
            ax.fill_between(lags, mean - sem, mean + sem,
                            color=colours[epoch], alpha=0.2)
        ax.axvline(0, color="k", lw=0.5, ls=":")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{ax_x}-{ax_y}  ({n_dims} dims)", fontsize=10)
    for ax in axes[::5]:
        ax.set_ylabel("held-out CC (significant dims)")
    for ax in axes[5:]:
        ax.set_xlabel("spatial lag (bins)   +ve: X leads Y")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Stage 2 — lagged-refit CC over significant subspace "
                 f"dimensions ({_describe(tag)}; learner mean +/- SEM)")
    _save(fig, "stage2_lag_curves", tag)


def plot_ifi_window(results, tag, window):
    """IFI over significant subspace dims for one lag-integration window."""
    fig, axes = _grid()
    for ax, (ax_x, ax_y) in zip(axes, config.PAIRS):
        rs = learner_pairs(results, ax_x, ax_y)
        per_epoch = [pooled(rs, e, "ifi", window) for e in EPOCHS]
        _box_by_epoch(ax, per_epoch, "tab:purple")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylim(-1.05, 1.05)
        _star_vs0(ax, per_epoch)
        pp = _paired(per_pair(rs, "naive", "ifi", window),
                     per_pair(rs, "expert", "ifi", window))
        up = _unpaired(per_epoch[0], per_epoch[1])
        n = sum(v.size for v in per_epoch)
        ax.set_title(f"{ax_x}-{ax_y}  {n}d  "
                     f"pair={_fmt_p(pp)} unp={_fmt_p(up)}", fontsize=8)
    for ax in axes[::5]:
        ax.set_ylabel("IFI")
    fig.suptitle(f"Stage 2 — Information Flow Index, |lag| <= {window} bins "
                 f"({_describe(tag)}; '*' IFI!=0; learners)")
    _save(fig, f"stage2_ifi_win{window:02d}", tag)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="s2p5",
                   help="s2p5, s5cm, t10, t20, t40")
    args = p.parse_args()
    for cfg_name in _configs_for(args.variant):
        tag = (cfg_name if args.variant == "s2p5"
               else f"{args.variant}_{cfg_name}")
        path = config.RESULTS_DIR / f"stage2_{tag}.pkl"
        if not path.exists():
            print(f"skip '{tag}': {path.name} not found")
            continue
        with open(path, "rb") as fh:
            results = pickle.load(fh)["results"]
        plot_comm_strength(results, tag)
        plot_subspace_dim(results, tag)
        plot_lag_curves(results, tag)
        max_window = results[0].epochs["naive"].ifi_windows.shape[1]
        for window in range(1, max_window + 1):
            plot_ifi_window(results, tag, window)
        print(f"config '{tag}' figures done.")
    print("Stage 2 figures done.")


if __name__ == "__main__":
    main()
