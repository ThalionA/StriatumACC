"""Directionality (Information Flow Index) summary table + figure.

For the committed Stage-2 result, per area-pair, this reports the directional
read-out honestly at the animal level (the repeated-measures sampling), as a
companion to the per-dimension view on the Stage-2 IFI figures:

  * per epoch -- the per-animal mean IFI(|lag| <= 10 bins) over each animal's
    significant dims, tested against 0 (one-sample t-test, n = animals;
    signed-rank cannot reach significance below n=6);
  * across epochs -- the epoch effect on per-animal IFI by repeated-measures
    ANOVA (n = animals) + Holm-corrected paired-t post-hoc + a per-animal
    linear trend over the epoch index;
  * the mean / median peak-lag (bin at which held-out CC1 peaks) over the
    significant dims pooled across the pair's animals and all three epochs,
    in bins and in cm (an overall directionality summary).

Positive IFI / positive peak-lag => X (the first-listed area) leads Y.

Writes figures/directionality_<variant>.csv and a per-pair spaghetti figure
(per-animal IFI across the three epochs) figures/directionality_<variant>.png.

Run:  python scripts/directionality_table.py [--variant plain|partial]
                                             [--bin-cm 5.0]
"""

from __future__ import annotations

import argparse
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

from striatum_cca import aggregate, config, epoch_stats  # noqa: E402

EPOCHS = config.EPOCH_NAMES
EPOCH_LABEL = ["naive", "inter", "expert"]
EPOCH_IDX = np.arange(3, dtype=float)
IFI_WINDOW = 5   # bins; 25 cm at 5 cm bins (halved from 10 with the 2026-08-10 bin switch)
VARIANT_PKL = {"plain": "stage2_committed_circshift.pkl",
               "partial": "stage2_committed_circshift_partial.pkl"}


def _peak_lags(learners, epoch):
    """Significant-dim peak lags (bins) pooled over a pair's learner animals."""
    out = []
    for r in learners:
        ea = r.epochs[epoch]
        js = aggregate.sig_dims(ea)
        out.append(np.asarray(ea.peak_lag_per_dim)[js])
    return np.concatenate(out) if out else np.array([])


def _per_animal_trend(mat):
    """Mean per-animal slope of IFI vs epoch index + one-sample-t p."""
    if mat.shape[0] < 2:
        return np.nan, np.nan
    slopes = np.array([np.polyfit(EPOCH_IDX, mat[i], 1)[0]
                       for i in range(mat.shape[0])])
    p = (float(stats.ttest_1samp(slopes, 0.0).pvalue)
         if np.ptp(slopes) > 0 else np.nan)
    return float(np.mean(slopes)), p


COLS = ["pair", "n_animals",
        "ifi_naive", "p_naive_vs0_ttest", "ifi_inter", "p_inter_vs0_ttest",
        "ifi_expert", "p_expert_vs0_ttest",
        "rm_anova_F", "rm_anova_p", "holm_ni_p", "holm_ie_p", "holm_ne_p",
        "trend_slope", "trend_p",
        "peak_lag_mean_bins", "peak_lag_median_bins",
        "peak_lag_mean_cm", "n_sig_dims_pooled"]


def _round(v):
    if isinstance(v, float):
        return "" if not np.isfinite(v) else round(v, 4)
    return v


def build_rows(results, bin_cm):
    rows = []
    for area_x, area_y in config.PAIRS:
        learners = aggregate.learner_pairs(results, area_x, area_y)
        mat = aggregate.per_animal_matrix(learners, "ifi", IFI_WINDOW)
        rm = epoch_stats.rm_anova_posthoc(mat)
        slope, slope_p = _per_animal_trend(mat)
        per_epoch_vs0 = [(float(np.mean(mat[:, i])) if mat.shape[0] else np.nan,
                          epoch_stats.ttest_vs0(mat[:, i]) if mat.shape[0]
                          else np.nan)
                         for i in range(3)]
        peaks = np.concatenate([_peak_lags(learners, e) for e in EPOCHS]) \
            if learners else np.array([])
        pk_mean = float(np.mean(peaks)) if peaks.size else np.nan
        pk_med = float(np.median(peaks)) if peaks.size else np.nan
        rows.append([
            f"{area_x}-{area_y}", rm["n"],
            per_epoch_vs0[0][0], per_epoch_vs0[0][1],
            per_epoch_vs0[1][0], per_epoch_vs0[1][1],
            per_epoch_vs0[2][0], per_epoch_vs0[2][1],
            rm["F"], rm["p"], *rm["posthoc"], slope, slope_p,
            pk_mean, pk_med, pk_mean * bin_cm, int(peaks.size)])
    return rows


def write_csv(rows, variant):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / f"directionality_{variant}.csv"
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(COLS)
        for r in rows:
            wr.writerow([_round(v) for v in r])
    print(f"saved {path}")
    return path


def plot_spaghetti(results, variant):
    """Per-animal IFI(w10) across the three epochs, one panel per pair."""
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.5))
    for ax, (area_x, area_y) in zip(axes.ravel(), config.PAIRS):
        learners = aggregate.learner_pairs(results, area_x, area_y)
        mat = aggregate.per_animal_matrix(learners, "ifi", IFI_WINDOW)
        ax.axhline(0, color="k", lw=0.6)
        for row in mat:
            ax.plot(EPOCH_IDX, row, "-o", color="0.6", ms=3, lw=0.8, alpha=0.7)
        if mat.shape[0]:
            ax.plot(EPOCH_IDX, mat.mean(axis=0), "-o", color="tab:red", lw=2,
                    ms=5, label="mean")
        rm = epoch_stats.rm_anova_posthoc(mat)
        ax.set_xticks(EPOCH_IDX)
        ax.set_xticklabels(EPOCH_LABEL)
        ax.set_xlim(-0.3, 2.3)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"{area_x}-{area_y}  n={rm['n']}a  "
                     f"RM p={'n/a' if not np.isfinite(rm['p']) else round(rm['p'], 3)}",
                     fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("IFI(|lag|<=10)  (+ve: X leads Y)", fontsize=8)
    fig.suptitle(f"Directionality -- per-animal IFI across learning "
                 f"({variant} CCA, committed config; learners; "
                 f"per-animal mean over significant dims)", fontsize=11)
    fig.tight_layout()
    path = config.FIGURES_DIR / f"directionality_{variant}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=("plain", "partial"), default="partial")
    p.add_argument("--bin-cm", type=float, default=5.0,
                   help="spatial bin width in cm (committed = 5.0 since 2026-08-10)")
    args = p.parse_args()
    path = config.RESULTS_DIR / VARIANT_PKL[args.variant]
    if not path.exists():
        sys.exit(f"missing {path.name} -- run run_committed.py --stage 2")
    with open(path, "rb") as fh:
        results = pickle.load(fh)["results"]
    rows = build_rows(results, args.bin_cm)
    write_csv(rows, args.variant)
    plot_spaghetti(results, args.variant)
    print("directionality table done.")


if __name__ == "__main__":
    main()
