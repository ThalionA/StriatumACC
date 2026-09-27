"""Trial x position maps and epoch profiles of the binned video features.

Usage: python scripts/plot_binned.py 1105 1106 1201 1206
Reads results/<s>_binned.npz; writes figures/<s>_binned.{svg,png}.
Rows: mouth ME, whisker ME, wheel ME, |VR speed| (reference). Left: every raw
trial (usable trials only; others blank) x 5 cm bin. Right: epoch mean +- SEM
across the 10 trials of each epoch.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BIN_CM = 5.0
REWARD_ZONE_CM = (100 / 4 * BIN_CM, 135 / 4 * BIN_CM)  # 100-135 a.u. (ProcessStriatumTask.m)
ROWS = [("me_mouth", "mouth ME (grey lvl)"), ("me_whiskers", "whisker ME (grey lvl)"),
        ("me_wheel", "wheel ME (grey lvl)"), ("vr_speed", "|VR speed| (a.u./s)")]
EPOCH_COLOURS = {"Naive": "#7f7f7f", "Intermediate": "#1f77b4", "Expert": "#d62728"}


def plot_session(s):
    d = np.load(ROOT / "results" / f"{s}_binned.npz")
    usable = d["usable"]
    n_trials, n_bins = d["durations"].shape
    x_cm = (np.arange(n_bins) + 0.5) * BIN_CM
    expert = d["epoch_Expert"]
    lp_raw = int(expert[0]) if expert.size else None

    fig, ax = plt.subplots(len(ROWS), 2, figsize=(12, 12), constrained_layout=True,
                           gridspec_kw={"width_ratios": [1.3, 1]})
    fig.suptitle(f"Session {s}: top-camera features by trial and position "
                 f"({usable.size}/{n_trials} usable trials; LP = good trial {d['lp']:.0f}, DP = raw trial {d['dp']:.0f})")
    for row, (key, label) in enumerate(ROWS):
        m = np.full((n_trials, n_bins), np.nan)
        m[usable] = d[key][usable]
        lo, hi = np.nanpercentile(m, [1, 99])
        a = ax[row, 0]
        im = a.imshow(m, aspect="auto", origin="lower", vmin=lo, vmax=hi, cmap="viridis",
                      extent=(0, n_bins * BIN_CM, 0.5, n_trials + 0.5), interpolation="nearest")
        fig.colorbar(im, ax=a, label=label)
        for z in REWARD_ZONE_CM:
            a.axvline(z, color="w", lw=0.8, ls="--")
        if lp_raw is not None:
            a.axhline(lp_raw + 1, color="r", lw=1, label="learning point")
            a.legend(loc="upper right", fontsize=7)
        a.set(xlabel="position (cm)", ylabel="raw trial", title=f"{label.split(' (')[0]}: every trial")

        a = ax[row, 1]
        for name, colour in EPOCH_COLOURS.items():
            idx = d[f"epoch_{name}"]
            if idx.size == 0:
                continue
            vals = d[key][idx]
            mean = np.nanmean(vals, 0)
            sem = np.nanstd(vals, 0) / np.sqrt(np.isfinite(vals).sum(0).clip(min=1))
            a.plot(x_cm, mean, color=colour, label=f"{name} (n={idx.size})")
            a.fill_between(x_cm, mean - sem, mean + sem, color=colour, alpha=0.25, lw=0)
        a.axvspan(*REWARD_ZONE_CM, color="0.92", zorder=0, label="reward zone")
        a.set(xlabel="position (cm)", ylabel=label, title=f"{label.split(' (')[0]}: epoch mean ± SEM")
        a.legend(fontsize=7)
    (ROOT / "figures").mkdir(exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(ROOT / "figures" / f"{s}_binned.{ext}", dpi=130)  # 12 x 12 in at 130 dpi = 1560 px
    plt.close(fig)
    print(f"wrote figures/{s}_binned.svg/.png")


if __name__ == "__main__":
    for s in sys.argv[1:]:
        plot_session(s)
