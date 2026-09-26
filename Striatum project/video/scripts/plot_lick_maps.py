"""Where licking moves pixels: speed-matched lick minus lick-free mean |frame
difference|, beside a raw frame and the running (lick-free, fast minus slow)
map, with the current hand-drawn ROIs overlaid.

Usage: python scripts/plot_lick_maps.py 1206 1201
Reads results/<s>_lick_maps.npz; writes figures/lick_maps.{svg,png}.
"""

import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from striatum_video.rois import ROIS
from striatum_video.sessions import SESSIONS, video_parts

ROI_COLOURS = {"wheel": "#2ca02c", "mouth": "#d62728", "whiskers": "#ff7f0e"}


def lick_map(d):
    sums, counts = d["sums"], d["counts"]
    means = sums / np.maximum(counts, 1)[:, None, None]
    lick, free = means[0::2], means[1::2]
    w = counts[0::2] / counts[0::2].sum()
    return np.tensordot(w, lick - free, axes=1)


def run_map(d):
    """Lick-free frames, fastest minus slowest NON-EMPTY speed stratum: where
    running moves pixels. (Strata are lick-frame speed quintiles; when most licks
    happen at speed 0 the lowest quintiles collapse and are empty.)"""
    sums, counts = d["sums"], d["counts"]
    means = sums / np.maximum(counts, 1)[:, None, None]
    free = np.flatnonzero(counts[1::2] > 0)
    return means[1::2][free[-1]] - means[1::2][free[0]]


def a_frame(session):
    p = video_parts(SESSIONS[session]["video"])[0]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", "600", "-i", str(p), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(600, 500)


def main(sessions):
    fig, ax = plt.subplots(len(sessions), 3, figsize=(12, 5.2 * len(sessions)), constrained_layout=True)
    ax = np.atleast_2d(ax)
    for row, s in enumerate(sessions):
        d = np.load(ROOT / "results" / f"{s}_lick_maps.npz")
        n_lick = int(d["counts"][0::2].sum())
        panels = [(a_frame(s), "gray", f"{s}: frame at 10 min", None),
                  (lick_map(d), "RdBu_r", f"{s}: lick − lick-free |Δframe|\n(speed-matched, {n_lick} lick frames)", "Δ grey levels"),
                  (run_map(d), "RdBu_r", f"{s}: running: fastest − slowest\nspeed quintile (lick-free)", "Δ grey levels")]
        for a, (img, cmap, title, clabel) in zip(ax[row], panels):
            if cmap == "gray":
                im = a.imshow(img, cmap=cmap)
            else:
                lim = np.nanpercentile(np.abs(img), 99.5)
                im = a.imshow(img, cmap=cmap, vmin=-lim, vmax=lim)
                fig.colorbar(im, ax=a, shrink=0.8, label=clabel)
            for name, r in ROIS.items():
                a.add_patch(Rectangle((r.x, r.y), r.w, r.h, fill=False, lw=1.2, ec=ROI_COLOURS[name], label=name))
            a.set(title=title, xlabel="x (px)", ylabel="y (px)")
        ax[row, 0].legend(fontsize=7, loc="lower left")
    (ROOT / "figures").mkdir(exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(ROOT / "figures" / f"lick_maps.{ext}", dpi=120)  # 12 in x 120 dpi = 1440 px
    print("wrote figures/lick_maps.svg/.png")


if __name__ == "__main__":
    main(sys.argv[1:])
