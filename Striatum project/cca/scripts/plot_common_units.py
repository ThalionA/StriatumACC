"""Common-unit spatial activity profiles -- committed config.

For each area pair, splits the area's units into communication-subspace
members (top-quartile contributors to the dominant canonical dimension) and
non-members, and plots their mean z-scored activity across corridor position,
one curve per epoch. Shows whether the units carrying the inter-areal coupling
have a distinct spatial tuning profile, and whether it shifts across learning.

z-scored activity is re-derived exactly as the pipeline does it -- whole-
engaged-period per-unit z-scoring of the area tensor, before residualisation
(pipeline._zscore_area) -- and trial-averaged within each epoch. Membership is
read from results/stage3_committed.pkl. Profiles are pooled over the learner
animals of a pair (n = units); one panel per pair, no pooling across pairs.

Writes figures/stage3_common_units_x_committed.png and ..._y_committed.png
(X side and Y side of every pair).

Run:  python scripts/plot_common_units.py
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

from striatum_cca import config, dataio, pipeline  # noqa: E402

CFG = config.DEFAULT
EPOCHS = config.EPOCH_NAMES
EPOCH_COLOUR = {"naive": "tab:blue", "intermediate": "tab:green",
                "expert": "tab:red"}
RESULTS_PKL = config.RESULTS_DIR / "stage3_committed.pkl"

_ZCACHE: dict[tuple[int, str], np.ndarray] = {}


def ztensor(animal, area):
    """Whole-engaged-period z-scored area tensor (n_usable, n_bins, n_units)."""
    key = (animal.animal_id, area)
    if key not in _ZCACHE:
        tensor, _ = dataio.area_tensor(animal, area, CFG)
        _ZCACHE[key] = pipeline._zscore_area(tensor)
    return _ZCACHE[key]


def learners(results, area_x, area_y):
    return [r for r in results
            if (r.area_x, r.area_y) == (area_x, area_y) and r.role == "learner"]


def _mean_sem(rows):
    """Mean and SEM across units (rows: (n_units, n_bins)); NaN if empty."""
    if rows.shape[0] == 0:
        return None, None
    mean = np.nanmean(rows, axis=0)
    sem = np.nanstd(rows, axis=0) / np.sqrt(rows.shape[0])
    return mean, sem


def collect(results, animals, side):
    """Per pair x epoch, stacked member / non-member unit profiles.

    ``side`` is "x" or "y". Returns {(ax, ay): {epoch: (member, nonmember)}},
    each a (n_units, n_bins) array of trial-averaged z-scored profiles.
    """
    by_pair = {}
    for ax, ay in config.PAIRS:
        rs = learners(results, ax, ay)
        cells = {e: {"member": [], "nonmember": []} for e in EPOCHS}
        for r in rs:
            animal = animals[r.animal_id]
            area = r.area_x if side == "x" else r.area_y
            zt = ztensor(animal, area)
            windows = dataio.epoch_windows(r.lp, len(zt), CFG)
            if windows is None:
                continue
            for e in EPOCHS:
                prof = np.nanmean(zt[windows[e]], axis=0)      # (n_bins, units)
                es = r.epochs[e]
                mask = es.member_x if side == "x" else es.member_y
                cells[e]["member"].append(prof[:, mask].T)
                cells[e]["nonmember"].append(prof[:, ~mask].T)
        by_pair[(ax, ay)] = {
            e: (np.vstack(cells[e]["member"]) if cells[e]["member"]
                else np.empty((0, 0)),
                np.vstack(cells[e]["nonmember"]) if cells[e]["nonmember"]
                else np.empty((0, 0)))
            for e in EPOCHS
        }
    return by_pair


def plot_side(by_pair, side):
    """One 2x5 pair grid: member (solid) vs non-member (dashed) profiles."""
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.5))
    axes = axes.ravel()
    for ax, (ax_x, ax_y) in zip(axes, config.PAIRS):
        cells = by_pair[(ax_x, ax_y)]
        area = ax_x if side == "x" else ax_y
        n_member = 0
        for e in EPOCHS:
            member, nonmember = cells[e]
            colour = EPOCH_COLOUR[e]
            if member.size:
                n_bins = member.shape[1]
                pos = (np.arange(n_bins) + 0.5) * config.bin_size_cm(n_bins)
                mean, sem = _mean_sem(member)
                ax.plot(pos, mean, "-", color=colour, lw=1.8, label=e)
                ax.fill_between(pos, mean - sem, mean + sem,
                                color=colour, alpha=0.2)
                n_member += member.shape[0]
            if nonmember.size:
                n_bins = nonmember.shape[1]
                pos = (np.arange(n_bins) + 0.5) * config.bin_size_cm(n_bins)
                mean, _ = _mean_sem(nonmember)
                ax.plot(pos, mean, "--", color=colour, lw=0.9, alpha=0.7)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{ax_x}-{ax_y}  [{area}]  ({n_member} member units)",
                     fontsize=9)
    for ax in axes[::5]:
        ax.set_ylabel("mean z-scored activity")
    for ax in axes[5:]:
        ax.set_xlabel("corridor position (cm)")
    axes[0].legend(frameon=False, fontsize=8, title="solid=member\ndash=non-mem")
    fig.suptitle(f"Stage 3 -- spatial activity of communication-subspace member "
                 f"vs non-member units, {side.upper()} side "
                 f"(committed config; learners; mean +/- SEM over units)")
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    path = config.FIGURES_DIR / f"stage3_common_units_{side}_committed.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def main():
    if not RESULTS_PKL.exists():
        sys.exit(f"missing {RESULTS_PKL.name} -- run "
                 f"run_committed.py --stage 3")
    with open(RESULTS_PKL, "rb") as fh:
        results = pickle.load(fh)["results"]
    animals = {a.animal_id: a for a in dataio.load_animals()}
    for side in ("x", "y"):
        by_pair = collect(results, animals, side)
        plot_side(by_pair, side)
    print("common-unit figures done.")


if __name__ == "__main__":
    main()
