"""Stage 3 figures: subspace reorientation, weight sparsity, unit membership.

Loads results/stage3.pkl and writes PNG figures to figures/.

Run:  python scripts/plot_stage3.py
"""

from __future__ import annotations

import argparse
import pickle
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

from striatum_cca import config, membership  # noqa: E402

EPOCHS = config.EPOCH_NAMES


def load(tag: str = "main"):
    with open(config.RESULTS_DIR / f"stage3_{tag}.pkl", "rb") as fh:
        return pickle.load(fh)["results"]


# Cohort for group statistics: "learners" (clean) or "all" (+ yoked
# non-learners). Set by main(); figures produced for both.
COHORT = "learners"

# Spatial-binning variant; set by main(). Adds a filename token for non-default.
VARIANT = "s2p5"


def in_cohort(role: str) -> bool:
    return COHORT == "all" or role == "learner"


def learners(results, area_x, area_y):
    """Pairs for the current cohort (kept name for brevity)."""
    return [r for r in results
            if (r.area_x, r.area_y) == (area_x, area_y) and in_cohort(r.role)]


def _grid():
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.5))
    return fig, axes.ravel()


def _save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    vtok = "" if VARIANT == "s2p5" else f"{VARIANT}_"
    path = config.FIGURES_DIR / f"{name[:-4]}_{vtok}{COHORT}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


# ---------------------------------------------------------------------------
def plot_principal_angles(results):
    """Naive->expert subspace rotation vs the within-epoch split-half floor."""
    fig, axes = _grid()
    x = np.arange(2)
    labels = ["split-half\nfloor", "naive→expert"]
    for ax, (ax_x, ax_y) in zip(axes, config.PAIRS):
        ls = learners(results, ax_x, ax_y)
        rows = []
        for r in ls:
            floor = np.nanmean([r.epochs[e].split_half_angle_x.max() for e in EPOCHS])
            ang = r.angles_x["naive->expert"].max()
            rows.append([floor, ang])
            ax.plot(x, [floor, ang], "-", color="0.8", lw=0.7, zorder=1)
        if rows:
            mat = np.array(rows)
            mean = np.nanmean(mat, axis=0)
            ax.plot(x, mean, "-o", color="tab:purple", lw=2.2, zorder=3)
            if len(ls) >= 5:
                # paired test: naive->expert angle vs the split-half floor
                diff = mat[:, 1] - mat[:, 0]
                diff = diff[np.isfinite(diff)]
                if diff.size >= 5 and stats.ttest_1samp(diff, 0.0).pvalue < 0.05:
                    ax.text(1, mean[1] + 0.06, "*", fontsize=15, ha="center")
        ax.axhline(np.pi / 2, color="k", ls="--", lw=0.6)
        ax.set_ylim(0, np.pi / 2 + 0.1)
        ax.set_xlim(-0.4, 1.4)
        ax.set_title(f"{ax_x}-{ax_y}  (n={len(ls)})", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
    for ax in axes[::5]:
        ax.set_ylabel("principal angle (rad)")
    fig.suptitle("Stage 3 — communication-subspace reorientation across learning "
                 "(X side; dashed = orthogonal; '*' n→expert > floor)")
    _save(fig, "stage3_principal_angles.png")


def plot_gini(results):
    """Sparsity of the dominant-dimension weight profile across epochs."""
    fig, axes = _grid()
    x = np.arange(2)
    for ax, (ax_x, ax_y) in zip(axes, config.PAIRS):
        ls = learners(results, ax_x, ax_y)
        for r in ls:
            ax.plot(x, [r.epochs[e].gini_x for e in EPOCHS], "-", color="0.8",
                    lw=0.7, zorder=1)
        if ls:
            for side, colour in (("gini_x", "tab:blue"), ("gini_y", "tab:orange")):
                mean = [np.nanmean([getattr(r.epochs[e], side) for r in ls])
                        for e in EPOCHS]
                ax.plot(x, mean, "-o", color=colour, lw=2.2, zorder=3,
                        label=ax_x if side == "gini_x" else ax_y)
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.4, 1.4)
        ax.set_title(f"{ax_x}-{ax_y}  (n={len(ls)})", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(["naive", "expert"])
        ax.legend(frameon=False, fontsize=7)
    for ax in axes[::5]:
        ax.set_ylabel("Gini (weight sparsity)")
    fig.suptitle("Stage 3 — communication-weight sparsity across learning "
                 "(higher = fewer units carry the coupling)")
    _save(fig, "stage3_gini.png")


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
        if not in_cohort(r.role):
            continue
        for member in ("member_x", "member_y"):
            j = membership.jaccard(getattr(r.epochs["naive"], member),
                                   getattr(r.epochs["expert"], member))
            if np.isfinite(j):
                cross_epoch.append(j)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # panel 1 — cross-pair overlap by area
    areas = [a for a in config.AREAS if cross_pair[a]]
    means = [np.nanmean(cross_pair[a]) for a in areas]
    sems = [np.nanstd(cross_pair[a]) / np.sqrt(len(cross_pair[a])) for a in areas]
    axes[0].bar(areas, means, yerr=sems, color="teal", capsize=3)
    axes[0].axhline(0.25, color="k", ls=":", lw=0.8)   # chance for top-quartile
    axes[0].set_ylabel("member-set Jaccard across pairs")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Are the same units used across an area's pairs?\n"
                      "(dotted = chance for top-quartile sets)", fontsize=9)
    # panel 2 — cross-epoch stability
    axes[1].hist(cross_epoch, bins=np.linspace(0, 1, 21), color="teal")
    axes[1].axvline(np.nanmean(cross_epoch), color="k", lw=2)
    axes[1].axvline(0.25, color="k", ls=":", lw=0.8)
    axes[1].set_xlabel("member-set Jaccard, naive vs expert")
    axes[1].set_ylabel("count (pair x side)")
    axes[1].set_title(f"Membership stability across learning\n"
                      f"(mean = {np.nanmean(cross_epoch):.2f})", fontsize=9)
    fig.suptitle("Stage 3 — communication-subspace membership (D9)")
    _save(fig, "stage3_membership_overlap.png")


def main():
    global COHORT, VARIANT
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="s2p5", help="s2p5, s5cm, t10, t20, t40")
    VARIANT = p.parse_args().variant
    COHORT = "learners"            # committed to the learner cohort (round 7)
    results = load("main" if VARIANT == "s2p5" else VARIANT)
    plot_principal_angles(results)
    plot_gini(results)
    plot_membership_overlap(results)
    print("Stage 3 figures done.")


if __name__ == "__main__":
    main()
