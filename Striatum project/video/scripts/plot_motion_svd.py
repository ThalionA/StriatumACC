"""What each face motion-SVD component is: its spatial mask over the face ROI,
and its per-bin correlation with VR speed and lick fraction.

Usage: python scripts/plot_motion_svd.py 1105 1106 1201 1206
Reads results/<s>_motion_svd.npz and results/<s>_binned.npz; writes
figures/motion_svd_masks.{svg,png}.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N_SHOW = 6


def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[ok], b[ok])[0, 1]


def main(sessions):
    fig, ax = plt.subplots(len(sessions), N_SHOW, figsize=(14, 2.6 * len(sessions)), constrained_layout=True)
    ax = np.atleast_2d(ax)
    for row, s in enumerate(sessions):
        d = np.load(ROOT / "results" / f"{s}_motion_svd.npz")
        b = np.load(ROOT / "results" / f"{s}_binned.npz")
        var = d["singular_values"] ** 2
        for k in range(N_SHOW):
            mask = d["components"][:, k].reshape(tuple(d["mask_shape"]))
            lim = np.abs(mask).max()
            a = ax[row, k]
            a.imshow(mask, cmap="RdBu_r", vmin=-lim, vmax=lim)
            r_speed = corr(b[f"svd_{k + 1}"].ravel(), b["vr_speed"].ravel())
            r_lick = corr(b[f"svd_{k + 1}"].ravel(), b["lick_frac"].ravel())
            a.set_title(f"{s} SVD {k + 1} ({100 * var[k] / var.sum():.0f}%)\n"
                        f"r speed {r_speed:+.2f}, r lick {r_lick:+.2f}", fontsize=8)
            a.set_xticks([])
            a.set_yticks([])
    fig.suptitle("Face motion SVD: spatial masks over the face ROI (x 160-440, y 110-310 px; 5 px blocks). "
                 "Title: % of top-50 variance; per-bin r with |VR speed| and lick fraction")
    for ext in ("svg", "png"):
        fig.savefig(ROOT / "figures" / f"motion_svd_masks.{ext}", dpi=110)  # 14 in x 110 dpi = 1540 px
    print("wrote figures/motion_svd_masks.svg/.png")


if __name__ == "__main__":
    main(sys.argv[1:])
