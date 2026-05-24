"""Within-area communication-subspace similarity across an area's pairs.

For each area and epoch, builds a partner x partner heatmap of how similar that
area's dominant communication-weight vector is between the pairs it
participates in -- |cosine| similarity, computed within an animal and averaged
across animals (crosspair.similarity_matrix). E.g. "in naive, is DMS's
subspace-with-DLS aligned with its subspace-with-ACC?".

Writes, learners only:
  * subspace_similarity_committed.png          3 epochs x 5 areas heatmap grid
  * subspace_similarity_summary_committed.png  mean pairwise similarity per
                                               area across epochs

Run:  python scripts/plot_subspace_similarity.py
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

from striatum_cca import config, crosspair  # noqa: E402

EPOCHS = config.EPOCH_NAMES

# Set by main() from --variant ("plain" or "partial").
RESULTS_PKL = config.RESULTS_DIR / "stage3_committed.pkl"
SUFFIX = "committed"
VARIANT_NOTE = ""


def _configure(variant):
    """Point the script at the plain or the partial-CCA Stage-3 results."""
    global RESULTS_PKL, SUFFIX, VARIANT_NOTE
    if variant == "partial":
        RESULTS_PKL = config.RESULTS_DIR / "stage3_committed_partial.pkl"
        SUFFIX = "committed_partial"
        VARIANT_NOTE = "  [PARTIAL -- all other recorded areas removed]"


def load_learners():
    if not RESULTS_PKL.exists():
        sys.exit(f"missing {RESULTS_PKL.name} -- run run_committed.py --stage 3")
    with open(RESULTS_PKL, "rb") as fh:
        results = pickle.load(fh)["results"]
    return [r for r in results if r.role == "learner"]


def partners_of(area):
    return [a for a in config.AREAS if a != area]


def _save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / f"{name}_{SUFFIX}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def plot_heatmaps(results):
    """3 epochs (rows) x 5 areas (columns) grid of partner-similarity heatmaps."""
    areas = config.AREAS
    fig, axes = plt.subplots(len(EPOCHS), len(areas), figsize=(17, 10.5))
    im = None
    for r, epoch in enumerate(EPOCHS):
        for c, area in enumerate(areas):
            ax = axes[r, c]
            partners = partners_of(area)
            mat, cnt = crosspair.similarity_matrix(
                results, area, epoch, partners)
            disp = mat.copy()
            np.fill_diagonal(disp, np.nan)        # diagonal is trivially 1
            im = ax.imshow(disp, vmin=0.0, vmax=1.0, cmap="magma")
            ax.set_xticks(range(len(partners)))
            ax.set_yticks(range(len(partners)))
            ax.set_xticklabels(partners, fontsize=7)
            ax.set_yticklabels(partners, fontsize=7)
            for i in range(len(partners)):
                for j in range(len(partners)):
                    if i == j or not np.isfinite(mat[i, j]):
                        continue
                    ax.text(j, i, f"{mat[i, j]:.2f}\nn{int(cnt[i, j])}",
                            ha="center", va="center", fontsize=6,
                            color="w" if disp[i, j] < 0.5 else "k")
            mp = crosspair.mean_pairwise(mat)
            title = f"{area}  mean={mp:.2f}" if np.isfinite(mp) else f"{area}"
            ax.set_title(title, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{epoch}\n\npartner area", fontsize=9)
    fig.suptitle("Within-area communication-subspace similarity across an "
                 "area's pairs  (|cos| of the dominant canonical weight "
                 "vector; committed config; learners)" + VARIANT_NOTE,
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.93, 1])
    cax = fig.add_axes([0.945, 0.28, 0.011, 0.45])
    fig.colorbar(im, cax=cax, label="|cosine| similarity")
    _save(fig, "subspace_similarity")


def plot_summary(results):
    """Mean pairwise within-area similarity per area across epochs."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(EPOCHS))
    for area in config.AREAS:
        partners = partners_of(area)
        means = []
        for epoch in EPOCHS:
            mat, _ = crosspair.similarity_matrix(
                results, area, epoch, partners)
            means.append(crosspair.mean_pairwise(mat))
        ax.plot(x, means, "-o", lw=2, label=area)
    ax.set_xticks(x)
    ax.set_xticklabels(EPOCHS)
    ax.set_ylim(0, 1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean pairwise |cos| similarity (off-diagonal)")
    title = ("Mean within-area subspace similarity across partners\n"
             "(higher = an area uses a more consistent subspace "
             "for all its partners)")
    if VARIANT_NOTE:
        title += "\n" + VARIANT_NOTE.strip()
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, title="area")
    fig.tight_layout()
    _save(fig, "subspace_similarity_summary")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=("plain", "partial"), default="partial")
    _configure(p.parse_args().variant)
    results = load_learners()
    plot_heatmaps(results)
    plot_summary(results)
    print("subspace-similarity figures done.")


if __name__ == "__main__":
    main()
