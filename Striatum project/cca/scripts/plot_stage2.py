"""Stage 2 figures -- committed config, circshift null, three epochs.

Loads results/stage2_committed_circshift.pkl (config.DEFAULT, circshift
surrogate, naive / intermediate / expert) and writes, with a `_committed`
suffix:

  * stage2_comm_strength_committed.png   held-out CC over significant subspace
                                         dims, per epoch (box + points)
  * stage2_subspace_dim_committed.png    # significant subspace dims per pair
  * stage2_lag_curves_committed.png      held-out CC vs spatial lag over
                                         significant dims, line + shaded SEM
  * stage2_ifi_winNN_committed.png       IFI over significant dims, one figure
                                         per lag-integration window (1..10)

Every metric is reported per area-pair (one panel per pair; no pooling across
pairs). Statistics, learners only:
  * '*' above a box  -- that epoch's values differ from 0 (one-sample
                        Wilcoxon, n = significant dims)
  * pair p           -- naive vs expert, paired Wilcoxon over per-pair means
                        of the significant dims (n = pairs)
  * unp p            -- naive vs expert, Mann-Whitney over all significant
                        dims pooled within the pair (n = dims)

Run:  python scripts/plot_stage2.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

from striatum_cca import config  # noqa: E402

EPOCHS = config.EPOCH_NAMES                       # naive, intermediate, expert
EPOCH_LABEL = ["naive", "inter", "expert"]
EPOCH_COLOUR = config.EPOCH_COLOURS               # consistent with MATLAB
ALPHA = 0.05
TAG = "committed"
RESULTS_PKL = config.RESULTS_DIR / "stage2_committed_circshift.pkl"
SUBTITLE = "committed config, circshift null"


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
    """All significant-dim values pooled across this pair's animals (n = dims)."""
    if not rs:
        return np.array([])
    return np.concatenate([dim_values(r, epoch, kind, window) for r in rs])


def per_pair(rs, epoch, kind, window=0):
    """Per-animal mean over significant dims; NaN where an animal has none."""
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
    """Paired Wilcoxon over animals finite in both epochs (n = animals)."""
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 6 or not np.any(a != b):
        return np.nan
    try:
        return stats.wilcoxon(a, b).pvalue
    except ValueError:
        return np.nan


def _unpaired(a, b):
    """Mann-Whitney over all significant dims pooled within the pair."""
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


def _save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    path = config.FIGURES_DIR / f"{name}_{TAG}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def _box_by_epoch(ax, per_epoch):
    """Box plot + jittered semi-transparent points for each epoch."""
    for i, (epoch, v) in enumerate(zip(EPOCHS, per_epoch)):
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        colour = EPOCH_COLOUR[epoch]
        bp = ax.boxplot([v], positions=[i], widths=0.55, showfliers=False,
                        patch_artist=True, medianprops=dict(color="k"))
        bp["boxes"][0].set(facecolor=colour, alpha=0.35)
        jit = (np.random.default_rng(i).random(v.size) - 0.5) * 0.3
        ax.scatter(i + jit, v, s=12, color=colour, alpha=0.5, zorder=3)
    ax.set_xticks(range(len(EPOCHS)))
    ax.set_xticklabels(EPOCH_LABEL)
    ax.set_xlim(-0.6, len(EPOCHS) - 0.4)


def _star_vs0(ax, per_epoch):
    """Star above an epoch's box when its values differ from 0 (Wilcoxon)."""
    for i, v in enumerate(per_epoch):
        p = _wilcoxon_vs0(v)
        if np.isfinite(p) and p < ALPHA:
            ax.text(i, 0.93, "*", transform=ax.get_xaxis_transform(),
                    fontsize=14, ha="center")


def _change_p(rs, per_epoch, kind, window=0):
    """Naive-vs-expert paired and unpaired p-values."""
    pp = _paired(per_pair(rs, "naive", kind, window),
                 per_pair(rs, "expert", kind, window))
    up = _unpaired(per_epoch[0], per_epoch[2])
    return pp, up


# --- figures -----------------------------------------------------------------
def plot_comm_strength(results):
    """Held-out CC over significant subspace dims, per epoch."""
    fig, axes = _grid()
    for ax, (ax_x, ax_y) in zip(axes, config.PAIRS):
        rs = learner_pairs(results, ax_x, ax_y)
        per_epoch = [pooled(rs, e, "cc") for e in EPOCHS]
        _box_by_epoch(ax, per_epoch)
        ax.axhline(0, color="k", lw=0.6)
        _star_vs0(ax, per_epoch)
        pp, up = _change_p(rs, per_epoch, "cc")
        n = sum(v.size for v in per_epoch)
        ax.set_title(f"{ax_x}-{ax_y}  n={len(rs)} {n}d  "
                     f"n->e pair={_fmt_p(pp)} unp={_fmt_p(up)}", fontsize=8)
    for ax in axes[::5]:
        ax.set_ylabel("held-out CC (significant dims)")
    fig.suptitle(f"Stage 2 -- communication strength across learning "
                 f"({SUBTITLE}; '*' CC!=0; learners)")
    _save(fig, "stage2_comm_strength")


def plot_subspace_dim(results):
    """Number of significant communication-subspace dimensions per pair."""
    fig, axes = _grid()
    for ax, (ax_x, ax_y) in zip(axes, config.PAIRS):
        rs = learner_pairs(results, ax_x, ax_y)
        per_epoch = [np.array([float(r.epochs[e].n_significant) for r in rs])
                     for e in EPOCHS]
        _box_by_epoch(ax, per_epoch)
        pp = _paired(per_epoch[0], per_epoch[2])
        up = _unpaired(per_epoch[0], per_epoch[2])
        ax.set_ylim(bottom=0)
        ax.set_title(f"{ax_x}-{ax_y}  n={len(rs)}  "
                     f"n->e pair={_fmt_p(pp)} unp={_fmt_p(up)}", fontsize=8)
    for ax in axes[::5]:
        ax.set_ylabel("# significant subspace dims")
    fig.suptitle(f"Stage 2 -- communication-subspace dimensionality across "
                 f"learning ({SUBTITLE}; learners)")
    _save(fig, "stage2_subspace_dim")


def plot_lag_curves(results):
    """Held-out CC vs spatial lag over significant subspace dimensions."""
    fig, axes = _grid()
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
            ax.plot(lags, mean, "-", color=EPOCH_COLOUR[epoch], lw=1.6,
                    label=epoch)
            ax.fill_between(lags, mean - sem, mean + sem,
                            color=EPOCH_COLOUR[epoch], alpha=0.2)
        ax.axvline(0, color="k", lw=0.5, ls=":")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{ax_x}-{ax_y}  ({n_dims} dims)", fontsize=10)
    for ax in axes[::5]:
        ax.set_ylabel("held-out CC (significant dims)")
    for ax in axes[5:]:
        ax.set_xlabel("spatial lag (bins)   +ve: X leads Y")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Stage 2 -- lagged-refit CC over significant subspace "
                 f"dimensions ({SUBTITLE}; learner mean +/- SEM)")
    _save(fig, "stage2_lag_curves")


def plot_ifi_window(results, window):
    """IFI over significant subspace dims for one lag-integration window."""
    fig, axes = _grid()
    for ax, (ax_x, ax_y) in zip(axes, config.PAIRS):
        rs = learner_pairs(results, ax_x, ax_y)
        per_epoch = [pooled(rs, e, "ifi", window) for e in EPOCHS]
        _box_by_epoch(ax, per_epoch)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylim(-1.05, 1.05)
        _star_vs0(ax, per_epoch)
        pp, up = _change_p(rs, per_epoch, "ifi", window)
        n = sum(v.size for v in per_epoch)
        ax.set_title(f"{ax_x}-{ax_y}  n={len(rs)} {n}d  "
                     f"n->e pair={_fmt_p(pp)} unp={_fmt_p(up)}", fontsize=8)
    for ax in axes[::5]:
        ax.set_ylabel("IFI  (+ve: X leads Y)")
    fig.suptitle(f"Stage 2 -- Information Flow Index, |lag| <= {window} bins "
                 f"({SUBTITLE}; '*' IFI!=0; learners)")
    _save(fig, f"stage2_ifi_win{window:02d}")


def main():
    if not RESULTS_PKL.exists():
        sys.exit(f"missing {RESULTS_PKL.name} -- run "
                 f"run_committed.py --stage 2 --null-type circshift")
    with open(RESULTS_PKL, "rb") as fh:
        results = pickle.load(fh)["results"]
    plot_comm_strength(results)
    plot_subspace_dim(results)
    plot_lag_curves(results)
    max_window = results[0].epochs["naive"].ifi_windows.shape[1]
    for window in range(1, max_window + 1):
        plot_ifi_window(results, window)
    print("Stage 2 figures done.")


if __name__ == "__main__":
    main()
