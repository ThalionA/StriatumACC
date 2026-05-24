"""Common-unit spatial activity profiles -- committed config.

One figure per area pair. Each figure is a 2 x 3 grid: the top row is the X
area, the bottom row is the Y area; the three columns are the naive,
intermediate and expert epochs. Within every panel the area's units are split
into communication-subspace members (top-quartile contributors to the
dominant canonical dimension) and non-members, and their mean z-scored
activity across corridor position is drawn -- members in the epoch colour,
non-members in grey, both solid, mean +/- SEM over units. Shows whether the
units carrying the inter-areal coupling have a distinct spatial tuning, and
whether it shifts across learning.

z-scored activity is re-derived exactly as the pipeline does it -- whole-
engaged-period per-unit z-scoring of the area tensor, before residualisation
(pipeline._zscore_area) -- and trial-averaged within each epoch. Membership is
read from results/stage3_committed.pkl. Profiles are pooled over the learner
animals of a pair (n = units); one figure per pair, no pooling across pairs.

Writes figures/stage3_common_units_<AX>-<AY>_committed.png for every pair.

Run:  python scripts/plot_common_units.py
"""

from __future__ import annotations

import argparse
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
EPOCH_COLOUR = config.EPOCH_COLOURS               # consistent with MATLAB

# Set by main() from --variant ("plain" or "partial"). The member masks come
# from the chosen Stage-3 run; the z-scored activity itself is variant-
# independent (it is raw activity, not a CCA output).
RESULTS_PKL = config.RESULTS_DIR / "stage3_committed.pkl"
SUFFIX = "committed"
VARIANT_NOTE = ""

_ZCACHE: dict[tuple[int, str], np.ndarray] = {}


def _configure(variant):
    """Point the script at the plain or the partial-CCA Stage-3 membership."""
    global RESULTS_PKL, SUFFIX, VARIANT_NOTE
    if variant == "partial":
        RESULTS_PKL = config.RESULTS_DIR / "stage3_committed_partial.pkl"
        SUFFIX = "committed_partial"
        VARIANT_NOTE = "  [PARTIAL membership -- all other areas removed]"


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


def _draw_panel(ax, epoch, member, nonmember):
    """Member (epoch colour) and non-member (grey) mean +/- SEM profiles."""
    for arr, colour, lw, label in (
            (nonmember, "0.6", 1.3, "non-member"),
            (member, EPOCH_COLOUR[epoch], 1.9, "member")):
        if not arr.size:
            continue
        n_bins = arr.shape[1]
        pos = (np.arange(n_bins) + 0.5) * config.bin_size_cm(n_bins)
        mean, sem = _mean_sem(arr)
        ax.plot(pos, mean, "-", color=colour, lw=lw, label=label)
        ax.fill_between(pos, mean - sem, mean + sem, color=colour, alpha=0.2)
    ax.axhline(0, color="k", lw=0.5)
    ax.text(0.03, 0.96, f"member {member.shape[0]} / non {nonmember.shape[0]}",
            transform=ax.transAxes, fontsize=7, va="top", color="0.3")


def plot_pair(area_x, area_y, cells_x, cells_y):
    """One 2 x 3 figure: rows = X / Y area, columns = epochs."""
    fig, axes = plt.subplots(2, len(EPOCHS), figsize=(13, 6.2),
                             sharex=True, sharey="row")
    for row, (side_label, area, cells) in enumerate(
            (("X", area_x, cells_x), ("Y", area_y, cells_y))):
        for col, epoch in enumerate(EPOCHS):
            ax = axes[row, col]
            _draw_panel(ax, epoch, *cells[epoch])
            if row == 0:
                ax.set_title(epoch, fontsize=11)
        axes[row, 0].set_ylabel(f"{side_label}: {area}\n"
                                f"mean z-scored activity")
    for ax in axes[1, :]:
        ax.set_xlabel("corridor position (cm)")
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Common-unit spatial activity -- {area_x}-{area_y} "
                 f"(committed config; learners; mean +/- SEM over units)"
                 + VARIANT_NOTE)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    path = (config.FIGURES_DIR
            / f"stage3_common_units_{area_x}-{area_y}_{SUFFIX}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=("plain", "partial"), default="partial")
    _configure(p.parse_args().variant)
    if not RESULTS_PKL.exists():
        sys.exit(f"missing {RESULTS_PKL.name} -- run "
                 f"run_committed.py --stage 3")
    with open(RESULTS_PKL, "rb") as fh:
        results = pickle.load(fh)["results"]
    animals = {a.animal_id: a for a in dataio.load_animals()}
    by_pair_x = collect(results, animals, "x")
    by_pair_y = collect(results, animals, "y")
    for area_x, area_y in config.PAIRS:
        if not learners(results, area_x, area_y):
            print(f"skip {area_x}-{area_y}: no learner pairs")
            continue
        plot_pair(area_x, area_y,
                  by_pair_x[(area_x, area_y)], by_pair_y[(area_x, area_y)])
    print("common-unit figures done.")


if __name__ == "__main__":
    main()
