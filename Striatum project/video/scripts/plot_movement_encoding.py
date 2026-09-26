"""Summary of movement encoding: fraction of units modulated (vs the null's
empirical false-positive rate) and median dR2, per area and animal, for the two
contrasts (all movement beyond position; video beyond VR speed + licks).

Usage: python scripts/plot_movement_encoding.py
Reads results/movement_encoding.npz; writes figures/movement_encoding.{svg,png}.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AREAS = ("DMS", "DLS", "ACC", "V1", "CA1")
CONTRASTS = {"movement": "All movement\nbeyond position",
             "video_beyond_vr": "Video ROI ME (4)\nbeyond VR speed + licks",
             "face_svd_beyond_vr": "Face motion SVD (10)\nbeyond VR speed + licks"}
MARKERS = {1105: "o", 1106: "s", 1201: "^", 1206: "D"}


def main():
    t = np.load(ROOT / "results" / "movement_encoding.npz", allow_pickle=True)
    fig, ax = plt.subplots(2, len(CONTRASTS), figsize=(15, 8.5), constrained_layout=True)
    for col, (name, title) in enumerate(CONTRASTS.items()):
        fpr = np.mean(t[f"{name}__null_fpr"])
        for s, mk in MARKERS.items():
            for i, area in enumerate(AREAS):
                m = (t["area"] == area) & (t["session"] == s)
                if m.sum() == 0:
                    continue
                x = i + (list(MARKERS).index(s) - 1.5) * 0.12
                ax[0, col].plot(x, 100 * t[f"{name}__modulated"][m].mean(), mk, color="k", mfc="none",
                                label=str(s) if i == 0 or area == "DMS" else None)
                ax[1, col].plot(x, np.median(t[f"{name}__delta"][m]), mk, color="k", mfc="none")
        ax[0, col].axhline(100 * fpr, color="r", ls="--", lw=1, label=f"null false-positive rate ({100 * fpr:.1f}%)")
        ax[0, col].set(title=f"{title}: units modulated", ylabel="% of units above own null 95th pct",
                       xticks=range(len(AREAS)), xticklabels=AREAS, ylim=(0, 125), yticks=range(0, 101, 20), xlabel="area")
        ax[1, col].axhline(0, color="0.5", lw=0.8)
        ax[1, col].set(title=f"{title}: effect size", ylabel="median cross-validated ΔR² per unit",
                       xticks=range(len(AREAS)), xticklabels=AREAS, xlabel="area")
        handles, labels = ax[0, col].get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax[0, col].legend(uniq.values(), uniq.keys(), fontsize=6.5, ncol=3, loc="upper center")
    fig.suptitle("Single-unit firing per (trial, 5 cm bin): unique contribution of movement\n"
                 "(5-fold CV by trial; slow-drift terms in both models; circular-shift null)")
    (ROOT / "figures").mkdir(exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(ROOT / "figures" / f"movement_encoding.{ext}", dpi=105)  # 15 in x 105 dpi = 1575 px
    print("wrote figures/movement_encoding.svg/.png")


if __name__ == "__main__":
    main()
