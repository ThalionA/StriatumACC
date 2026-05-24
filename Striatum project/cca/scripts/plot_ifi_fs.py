"""IFI at window 10 -- fast-spiking units included vs excluded.

Compares the Information Flow Index at the widest lag-integration window
(|lag| <= 10 bins) with FS units excluded (the committed default) vs included,
per area pair x epoch, over the significant subspace dimensions. One figure
for the plain committed pipeline and one for the partial-CCA pipeline (every
other recorded area regressed out).

Reads four Stage-2 pkls:
  plain    FS-excl  stage2_committed_circshift.pkl
  plain    FS-incl  stage2_committed_circshift_fsincl.pkl
  partial  FS-excl  stage2_committed_circshift_partial.pkl
  partial  FS-incl  stage2_committed_circshift_fsincl_partial.pkl

Writes figures/ifi_fs_win10_plain.png and ifi_fs_win10_partial.png.

Run:  python scripts/plot_ifi_fs.py
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
from matplotlib.patches import Patch  # noqa: E402
from scipy import stats  # noqa: E402

from striatum_cca import config  # noqa: E402

EPOCHS = config.EPOCH_NAMES
EPOCH_LABEL = ["naive", "inter", "expert"]
ALPHA = 0.05
WINDOW = 10
FS_COLOUR = {"excl": "tab:blue", "incl": "tab:orange"}
DX = 0.2

VARIANTS = {
    "plain": ("stage2_committed_circshift.pkl",
              "stage2_committed_circshift_fsincl.pkl"),
    "partial": ("stage2_committed_circshift_partial.pkl",
                "stage2_committed_circshift_fsincl_partial.pkl"),
}


def load(name):
    path = config.RESULTS_DIR / name
    if not path.exists():
        sys.exit(f"missing {name} -- run run_committed.py with --include-fs")
    with open(path, "rb") as fh:
        return pickle.load(fh)["results"]


def learner_pairs(results, area_x, area_y):
    return [r for r in results
            if (r.area_x, r.area_y) == (area_x, area_y) and r.role == "learner"]


def ifi_pool(results, area_x, area_y, epoch):
    """IFI(window 10) over significant dims, pooled across a pair's learners."""
    out = []
    for r in learner_pairs(results, area_x, area_y):
        ea = r.epochs[epoch]
        js = np.where(ea.p_per_dim < ALPHA)[0]
        out.extend(ea.ifi_windows[js, WINDOW - 1].tolist())
    return np.array([v for v in out if np.isfinite(v)])


def _wilcoxon_vs0(v):
    """One-sample Wilcoxon vs 0; NaN if too few non-zero values."""
    if v.size < 6 or not np.any(v != 0):
        return np.nan
    try:
        return stats.wilcoxon(v).pvalue
    except ValueError:
        return np.nan


def plot_variant(res_excl, res_incl, variant):
    """One 2x5 pair grid: IFI(w10) FS-excluded vs FS-included per epoch."""
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.6), sharey=True)
    axes = axes.ravel()
    for ax, (area_x, area_y) in zip(axes, config.PAIRS):
        n_excl = len(learner_pairs(res_excl, area_x, area_y))
        n_incl = len(learner_pairs(res_incl, area_x, area_y))
        for ei, epoch in enumerate(EPOCHS):
            for results, key, dx in ((res_excl, "excl", -DX),
                                     (res_incl, "incl", DX)):
                v = ifi_pool(results, area_x, area_y, epoch)
                if v.size == 0:
                    continue
                colour = FS_COLOUR[key]
                bp = ax.boxplot([v], positions=[ei + dx], widths=0.34,
                                showfliers=False, patch_artist=True,
                                medianprops=dict(color="k"))
                bp["boxes"][0].set(facecolor=colour, alpha=0.35)
                jit = (np.random.default_rng(ei).random(v.size) - 0.5) * 0.22
                ax.scatter(ei + dx + jit, v, s=10, color=colour, alpha=0.5,
                           zorder=3)
                p = _wilcoxon_vs0(v)
                if np.isfinite(p) and p < ALPHA:
                    ax.text(ei + dx, 0.93, "*",
                            transform=ax.get_xaxis_transform(),
                            fontsize=13, ha="center")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xticks(range(len(EPOCHS)))
        ax.set_xticklabels(EPOCH_LABEL)
        ax.set_xlim(-0.6, len(EPOCHS) - 0.4)
        ax.set_title(f"{area_x}-{area_y}  excl n={n_excl} / incl n={n_incl}",
                     fontsize=8)
    for ax in axes[::5]:
        ax.set_ylabel(f"IFI, |lag| <= {WINDOW} bins  (+ve: X leads Y)")
    axes[0].legend(handles=[
        Patch(facecolor=FS_COLOUR["excl"], alpha=0.35, label="FS excluded"),
        Patch(facecolor=FS_COLOUR["incl"], alpha=0.35, label="FS included")],
        frameon=False, fontsize=8)
    fig.suptitle(f"Information Flow Index, |lag| <= {WINDOW} bins -- "
                 f"fast-spiking units excluded vs included "
                 f"({variant} pipeline; circshift null; '*' IFI!=0; learners)")
    fig.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / f"ifi_fs_win10_{variant}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def main():
    for variant, (excl_pkl, incl_pkl) in VARIANTS.items():
        plot_variant(load(excl_pkl), load(incl_pkl), variant)
    print("IFI FS-comparison figures done.")


if __name__ == "__main__":
    main()
