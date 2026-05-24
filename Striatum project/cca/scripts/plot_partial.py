"""Partial-CCA figure -- plain vs partial held-out CC1, all pairs.

Reads results/partial_committed.pkl (run_partial.py) and draws a 2 x 5 grid,
one panel per area pair. Within each panel, for every epoch, the plain
held-out CC1 and the partial held-out CC1 (every other recorded area regressed
out) are shown side by side: faint per-animal points with a plain->partial
connector, and a bold mean +/- SEM. A consistent drop means the pair's
coupling is largely mediated by the other areas.

Learners only; panels annotated with the animal count and the mean number of
areas conditioned on. Writes figures/partial_cca_committed.png.

Run:  python scripts/plot_partial.py
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

from striatum_cca import config  # noqa: E402

EPOCHS = config.EPOCH_NAMES
EPOCH_LABEL = ["naive", "inter", "expert"]
RESULTS_PKL = config.RESULTS_DIR / "partial_committed.pkl"
PLAIN_C, PARTIAL_C = "0.30", "tab:red"
DX = 0.16


def load():
    if not RESULTS_PKL.exists():
        sys.exit(f"missing {RESULTS_PKL.name} -- run run_partial.py")
    with open(RESULTS_PKL, "rb") as fh:
        return pickle.load(fh)["rows"]


def _mean_sem(ax, x, values, colour):
    """Bold mean +/- SEM marker at position x."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return
    sem = np.std(v) / np.sqrt(v.size) if v.size > 1 else 0.0
    ax.errorbar(x, np.mean(v), yerr=sem, fmt="o", color=colour,
                ms=7, lw=2, capsize=3, zorder=4)


def plot_partial(rows):
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.6), sharey=True)
    axes = axes.ravel()
    for ax, (area_x, area_y) in zip(axes, config.PAIRS):
        pair = f"{area_x}-{area_y}"
        learner = [r for r in rows
                   if r["pair"] == pair and r["role"] == "learner"]
        n_animals = len({r["animal"] for r in learner})
        for ei, epoch in enumerate(EPOCHS):
            cells = [r for r in learner if r["epoch"] == epoch]
            plain = np.array([r["plain_cc1"] for r in cells])
            part = np.array([r["partial_cc1"] for r in cells])
            for p, q in zip(plain, part):
                ax.plot([ei - DX, ei + DX], [p, q], "-", color="0.8",
                        lw=0.7, zorder=1)
            ax.scatter(np.full(plain.size, ei - DX), plain, s=14,
                       color=PLAIN_C, alpha=0.6, zorder=2)
            ax.scatter(np.full(part.size, ei + DX), part, s=14,
                       color=PARTIAL_C, alpha=0.6, zorder=2)
            _mean_sem(ax, ei - DX, plain, PLAIN_C)
            _mean_sem(ax, ei + DX, part, PARTIAL_C)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(range(len(EPOCHS)))
        ax.set_xticklabels(EPOCH_LABEL)
        ax.set_xlim(-0.6, len(EPOCHS) - 0.4)
        n_ctrl = (np.mean([r["n_control"] for r in learner])
                  if learner else np.nan)
        ctrl = f"ctrl~{n_ctrl:.1f}" if np.isfinite(n_ctrl) else "ctrl n/a"
        ax.set_title(f"{area_x}-{area_y}  n={n_animals}  {ctrl}", fontsize=9)
    for ax in axes[::5]:
        ax.set_ylabel("held-out CC1")
    # legend proxies
    axes[0].scatter([], [], color=PLAIN_C, label="plain")
    axes[0].scatter([], [], color=PARTIAL_C, label="partial (others removed)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Partial CCA -- does each pair's coupling survive regressing "
                 "out every other recorded area?  (committed config; learners; "
                 "mean +/- SEM)", fontsize=11)
    fig.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / "partial_cca_committed.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def _summary(rows):
    print(f"\n{'pair':>9} {'epoch':>13}  {'plain->partial':>16}  {'n':>3}")
    for area_x, area_y in config.PAIRS:
        pair = f"{area_x}-{area_y}"
        for epoch in EPOCHS:
            cells = [r for r in rows if r["pair"] == pair
                     and r["epoch"] == epoch and r["role"] == "learner"]
            if not cells:
                continue
            pl = np.nanmean([r["plain_cc1"] for r in cells])
            pa = np.nanmean([r["partial_cc1"] for r in cells])
            print(f"  {pair:>9} {epoch:>13}  {pl:>7.3f}->{pa:<7.3f}  "
                  f"{len(cells):>3}")


def main():
    rows = load()
    plot_partial(rows)
    _summary(rows)
    print("partial-CCA figure done.")


if __name__ == "__main__":
    main()
