"""IFI deep-dive at the committed config (config.DEFAULT, FS-excluded).

For each area-pair and each lag-integration window (|lag| <= w bins), reports:
  * the naive IFI and the expert IFI, each tested against 0 (Wilcoxon over the
    significant subspace dimensions pooled across that pair's learner animals);
  * the naive-vs-expert difference and whether it is significant
    (Mann-Whitney, unpaired -- the significant dims differ between epochs).

Writes figures/committed_ifi.csv and figures/committed_ifi.png.

Run:  python scripts/committed_ifi.py
"""

from __future__ import annotations

import csv
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

ALPHA = 0.05
# config.DEFAULT, FS-excluded -- the committed config's Stage-2 result.
COMMITTED_TAG = "s2p5_res_fsX_z1_mu06_lp7_samp15"
PAIR_NAMES = [f"{ax}-{ay}" for ax, ay in config.PAIRS]


def _sig(ea):
    return np.where(ea.p_per_dim < ALPHA)[0]


def _wilcoxon_vs0(v):
    v = v[np.isfinite(v)]
    if v.size < 6 or not np.any(v != 0):
        return np.nan
    try:
        return float(stats.wilcoxon(v).pvalue)
    except ValueError:
        return np.nan


def _mannwhitney(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 3 or b.size < 3:
        return np.nan
    try:
        return float(stats.mannwhitneyu(a, b).pvalue)
    except ValueError:
        return np.nan


def _pool(learners, epoch, window):
    """Significant-dim IFI at one lag window, pooled over a pair's animals."""
    if not learners:
        return np.array([])
    return np.concatenate(
        [r.epochs[epoch].ifi_windows[_sig(r.epochs[epoch]), window - 1]
         for r in learners])


def main():
    path = config.RESULTS_DIR / f"stage2_{COMMITTED_TAG}.pkl"
    if not path.exists():
        print(f"{path} not found -- run the sweep first")
        return
    with open(path, "rb") as fh:
        results = pickle.load(fh)["results"]
    n_win = results[0].epochs["naive"].ifi_windows.shape[1]
    windows = np.arange(1, n_win + 1)

    rows = []
    fig, axes = plt.subplots(2, 5, figsize=(17, 7.2))
    for ax, (ax_x, ax_y), name in zip(axes.ravel(), config.PAIRS, PAIR_NAMES):
        learners = [r for r in results
                    if (r.area_x, r.area_y) == (ax_x, ax_y)
                    and r.role == "learner"]
        ifi_n, ifi_e, p_n, p_e, p_d = [], [], [], [], []
        for w in windows:
            nv = _pool(learners, "naive", w)
            ev = _pool(learners, "expert", w)
            n_mean = float(np.nanmean(nv)) if nv.size else np.nan
            e_mean = float(np.nanmean(ev)) if ev.size else np.nan
            pn, pe = _wilcoxon_vs0(nv), _wilcoxon_vs0(ev)
            pd = _mannwhitney(nv, ev)
            ifi_n.append(n_mean)
            ifi_e.append(e_mean)
            p_n.append(pn)
            p_e.append(pe)
            p_d.append(pd)
            rows.append([name, int(w), nv.size, n_mean, pn,
                         ev.size, e_mean, pe, e_mean - n_mean, pd])
        ax.axhline(0, color="k", lw=0.7)
        ax.plot(windows, ifi_n, "-", color="tab:blue", lw=1.2, label="naive")
        ax.plot(windows, ifi_e, "-", color="tab:red", lw=1.2, label="expert")
        for w, v, p in zip(windows, ifi_n, p_n):
            ax.scatter(w, v, s=32, edgecolor="tab:blue", zorder=3,
                       facecolor="tab:blue" if (np.isfinite(p) and p < ALPHA)
                       else "white")
        for w, v, p in zip(windows, ifi_e, p_e):
            ax.scatter(w, v, s=32, edgecolor="tab:red", zorder=3,
                       facecolor="tab:red" if (np.isfinite(p) and p < ALPHA)
                       else "white")
        for w, p in zip(windows, p_d):
            if np.isfinite(p) and p < ALPHA:
                ax.text(w, 0.88, "*", ha="center", fontsize=13)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"{name}  (n={len(learners)} learners)", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("IFI", fontsize=8)
    for ax in axes[1, :]:
        ax.set_xlabel("lag-integration window (bins)", fontsize=8)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Committed config -- IFI per lag-integration window: naive "
                 "vs expert  (filled marker = that epoch's IFI significantly "
                 "!= 0; '*' = naive vs expert differ; n = significant "
                 "subspace dimensions)", fontsize=10)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "committed_ifi.png", dpi=150)
    plt.close(fig)
    print("saved figures/committed_ifi.png")

    with open(config.FIGURES_DIR / "committed_ifi.csv", "w",
              newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["pair", "window", "n_naive", "ifi_naive", "p_naive_vs0",
                     "n_expert", "ifi_expert", "p_expert_vs0",
                     "ifi_diff_expert_minus_naive", "p_naive_vs_expert"])
        for r in rows:
            wr.writerow([r[0], r[1]] + [
                "" if isinstance(x, float) and not np.isfinite(x)
                else (round(x, 4) if isinstance(x, float) else x)
                for x in r[2:]])
    print("saved figures/committed_ifi.csv")

    print("\nV1-ACC -- IFI per window (naive / expert, '<' = p<0.05):")
    for r in rows:
        if r[0] != "V1-ACC":
            continue
        nf = "<" if np.isfinite(r[4]) and r[4] < ALPHA else " "
        ef = "<" if np.isfinite(r[7]) and r[7] < ALPHA else " "
        df = "<" if np.isfinite(r[9]) and r[9] < ALPHA else " "
        print(f"  w{r[1]:>2}: naive {r[3]:+.3f}{nf}  expert {r[6]:+.3f}{ef}  "
              f"diff {r[8]:+.3f}{df}")


if __name__ == "__main__":
    main()
