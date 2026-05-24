"""Stage 3 figures -- committed config, three epochs.

Loads results/stage3_committed.pkl (config.DEFAULT subspace driver -- Stage 3
is null-independent, so one run serves every figure) and writes, with a
`_committed` suffix:

  * stage3_principal_angles_committed.png   subspace reorientation: the three
                                            epoch-to-epoch principal angles
                                            vs the within-epoch split-half
                                            noise floor
  * stage3_gini_committed.png               communication-weight sparsity
                                            (Gini) across the three epochs
  * stage3_membership_overlap_committed.png unit membership: shared across an
                                            area's pairs, and stable n->expert

The communication subspace is the dominant canonical dimension (d_sub = 1):
the within-epoch split-half angle is already near-orthogonal beyond it at 10
trials/epoch. Learners only; one panel per area-pair (no pooling across pairs).

Run:  python scripts/plot_stage3.py
"""

from __future__ import annotations

import pickle
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

from striatum_cca import config, membership  # noqa: E402

EPOCHS = config.EPOCH_NAMES
EPOCH_LABEL = ["naive", "inter", "expert"]
TRANSITIONS = ("naive->intermediate", "intermediate->expert", "naive->expert")
TRANSITION_LABEL = ["n->i", "i->e", "n->e"]
RESULTS_PKL = config.RESULTS_DIR / "stage3_committed.pkl"


def load():
    if not RESULTS_PKL.exists():
        sys.exit(f"missing {RESULTS_PKL.name} -- run "
                 f"run_committed.py --stage 3")
    with open(RESULTS_PKL, "rb") as fh:
        return pickle.load(fh)["results"]


def learners(results, area_x, area_y):
    return [r for r in results
            if (r.area_x, r.area_y) == (area_x, area_y) and r.role == "learner"]


def _grid():
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.5))
    return fig, axes.ravel()


def _save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    path = config.FIGURES_DIR / f"{name}_committed.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


# ---------------------------------------------------------------------------
def plot_principal_angles(results):
    """Epoch-to-epoch subspace rotation vs the within-epoch split-half floor."""
    fig, axes = _grid()
    x = np.arange(4)                       # floor, n->i, i->e, n->e
    labels = ["split-half\nfloor", *TRANSITION_LABEL]
    for ax, (ax_x, ax_y) in zip(axes, config.PAIRS):
        ls = learners(results, ax_x, ax_y)
        rows = []
        for r in ls:
            floor = np.nanmean(
                [r.epochs[e].split_half_angle_x.max() for e in EPOCHS])
            angs = [r.angles_x[t].max() for t in TRANSITIONS]
            rows.append([floor, *angs])
            ax.plot(x, rows[-1], "-", color="0.8", lw=0.7, zorder=1)
        if rows:
            mat = np.array(rows)
            mean = np.nanmean(mat, axis=0)
            ax.plot(x, mean, "-o", color="tab:purple", lw=2.2, zorder=3)
            if len(ls) >= 5:
                # paired test: naive->expert angle vs the split-half floor
                diff = mat[:, 3] - mat[:, 0]
                diff = diff[np.isfinite(diff)]
                if diff.size >= 5 and stats.ttest_1samp(diff, 0.0).pvalue < 0.05:
                    ax.text(3, mean[3] + 0.06, "*", fontsize=15, ha="center")
        ax.axhline(np.pi / 2, color="k", ls="--", lw=0.6)
        ax.set_ylim(0, np.pi / 2 + 0.1)
        ax.set_xlim(-0.4, 3.4)
        ax.set_title(f"{ax_x}-{ax_y}  (n={len(ls)})", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
    for ax in axes[::5]:
        ax.set_ylabel("principal angle (rad)")
    fig.suptitle("Stage 3 -- communication-subspace reorientation across "
                 "learning (X side; dashed = orthogonal; '*' n->expert > floor)")
    _save(fig, "stage3_principal_angles")


def plot_gini(results):
    """Sparsity of the dominant-dimension weight profile across epochs."""
    fig, axes = _grid()
    x = np.arange(len(EPOCHS))
    for ax, (ax_x, ax_y) in zip(axes, config.PAIRS):
        ls = learners(results, ax_x, ax_y)
        for r in ls:
            ax.plot(x, [r.epochs[e].gini_x for e in EPOCHS], "-", color="0.8",
                    lw=0.7, zorder=1)
        if ls:
            for side, colour in (("gini_x", "tab:blue"),
                                 ("gini_y", "tab:orange")):
                mean = [np.nanmean([getattr(r.epochs[e], side) for r in ls])
                        for e in EPOCHS]
                ax.plot(x, mean, "-o", color=colour, lw=2.2, zorder=3,
                        label=ax_x if side == "gini_x" else ax_y)
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.4, len(EPOCHS) - 0.6)
        ax.set_title(f"{ax_x}-{ax_y}  (n={len(ls)})", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(EPOCH_LABEL)
        if ls:
            ax.legend(frameon=False, fontsize=7)
    for ax in axes[::5]:
        ax.set_ylabel("Gini (weight sparsity)")
    fig.suptitle("Stage 3 -- communication-weight sparsity across learning "
                 "(higher = fewer units carry the coupling)")
    _save(fig, "stage3_gini")


def _area_mask(pair_subspace, area, epoch):
    """Member mask for `area`'s units in one PairSubspace at one epoch."""
    es = pair_subspace.epochs[epoch]
    if area == pair_subspace.area_x:
        return es.member_x
    if area == pair_subspace.area_y:
        return es.member_y
    return None


def plot_membership_overlap(results):
    """Are the same units members across pairs, and stable across epochs?"""
    # cross-pair: per animal & area, Jaccard of member sets between pairs.
    cross_pair = {a: [] for a in config.AREAS}
    animals = sorted({r.animal_id for r in results})
    for animal in animals:
        ar = [r for r in results if r.animal_id == animal]
        for area in config.AREAS:
            for epoch in EPOCHS:
                masks = [m for r in ar
                         if (m := _area_mask(r, area, epoch)) is not None]
                for m1, m2 in combinations(masks, 2):
                    cross_pair[area].append(membership.jaccard(m1, m2))

    # cross-epoch: per pair, Jaccard of naive vs expert member sets.
    cross_epoch = []
    for r in results:
        if r.role != "learner":
            continue
        for member in ("member_x", "member_y"):
            j = membership.jaccard(getattr(r.epochs["naive"], member),
                                   getattr(r.epochs["expert"], member))
            if np.isfinite(j):
                cross_epoch.append(j)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # panel 1 -- cross-pair overlap by area
    areas = [a for a in config.AREAS if cross_pair[a]]
    means = [np.nanmean(cross_pair[a]) for a in areas]
    sems = [np.nanstd(cross_pair[a]) / np.sqrt(len(cross_pair[a]))
            for a in areas]
    axes[0].bar(areas, means, yerr=sems, color="teal", capsize=3)
    axes[0].axhline(0.25, color="k", ls=":", lw=0.8)   # chance for top-quartile
    axes[0].set_ylabel("member-set Jaccard across pairs")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Are the same units used across an area's pairs?\n"
                      "(dotted = chance for top-quartile sets)", fontsize=9)
    # panel 2 -- cross-epoch stability
    axes[1].hist(cross_epoch, bins=np.linspace(0, 1, 21), color="teal")
    axes[1].axvline(np.nanmean(cross_epoch), color="k", lw=2)
    axes[1].axvline(0.25, color="k", ls=":", lw=0.8)
    axes[1].set_xlabel("member-set Jaccard, naive vs expert")
    axes[1].set_ylabel("count (pair x side)")
    axes[1].set_title(f"Membership stability across learning\n"
                      f"(mean = {np.nanmean(cross_epoch):.2f})", fontsize=9)
    fig.suptitle("Stage 3 -- communication-subspace membership "
                 "(committed config; learners)")
    _save(fig, "stage3_membership_overlap")


def main():
    results = load()
    plot_principal_angles(results)
    plot_gini(results)
    plot_membership_overlap(results)
    print("Stage 3 figures done.")


if __name__ == "__main__":
    main()
