"""Per-epoch validation of the redesigned RL fit.

Splits each mouse's session into Naive / Intermediate / Expert epochs (the
learning-point machinery of `find_learning_points.m` / `epoch_indices.m`) and
compares the observed vs model-predicted spatial lick and velocity profiles in
each epoch.  The redesign's success criterion: the model must reproduce the
epoch-to-epoch *change*, not merely the session average.

Run:  python -m scripts.plot_epoch_validation
"""
from __future__ import annotations

import os
import sys

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_model.config import (VISUAL_LANDMARK_AU, REWARD_START_AU,            # noqa
                             REWARD_END_AU, BIN_SIZE_AU, TaskConfig)
from rl_model.io_real import load_real_cohort                                 # noqa

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(HERE, "figures")
REALDIR = os.path.join(HERE, "results", "real_fits_v5")
MAT = os.path.join(HERE, "..", "processed_data", "preprocessed_data5cm.mat")

VZ = VISUAL_LANDMARK_AU / BIN_SIZE_AU
RZ0 = REWARD_START_AU / BIN_SIZE_AU
RZ1 = REWARD_END_AU / BIN_SIZE_AU
EPOCH_W = 10
EPOCHS = [("Naive", "#b03030"), ("Intermediate", "#d9a300"), ("Expert", "#1a7a3a")]


def learning_point(zerr):
    """First sustained sub-threshold trial — mirrors find_learning_points.m."""
    zerr = np.asarray(zerr).squeeze().astype(float)
    n = len(zerr)
    if n < EPOCH_W:
        return np.nan
    passes = zerr <= -2.0
    wc = np.array([passes[i:i + EPOCH_W].sum() for i in range(n)])
    idx = np.where(passes & (wc >= 7))[0]
    return int(idx[0]) if len(idx) else np.nan


def epoch_idx(lp, nt):
    """Naive / Intermediate / Expert trial-index arrays (entries may be None)."""
    naive = np.arange(0, EPOCH_W) if nt >= EPOCH_W else None
    inter = expert = None
    if not np.isnan(lp):
        if lp - EPOCH_W >= 0:
            inter = np.arange(lp - EPOCH_W, lp)
        if lp + EPOCH_W <= nt:
            expert = np.arange(lp, lp + EPOCH_W)
    return [naive, inter, expert]


def profile(arr, mask, idx):
    """Trial-averaged spatial profile over an epoch's valid bins."""
    a, m = arr[idx], mask[idx]
    w = m.sum(0)
    return (a * m).sum(0) / np.maximum(w, 1.0)


def landmarks(ax):
    ax.axvline(VZ, c="#2b6cc4", ls="--", lw=1)
    ax.axvspan(RZ0, RZ1, color="#1a7a3a", alpha=0.08)


def main():
    cfg = TaskConfig()
    cohort = load_real_cohort(MAT, cfg)
    ids = [m["mouse"] for m in cohort]
    fits = {i: np.load(os.path.join(REALDIR, f"{i}.npz"), allow_pickle=True)
            for i in ids}

    # learning points straight from the .mat
    lps = {}
    with h5py.File(MAT, "r") as f:
        pd = f["preprocessed_data"]
        for i in range(len(ids)):
            lps[ids[i]] = learning_point(f[pd["zscored_lick_errors"][i, 0]])

    bins = np.arange(cfg.n_bins)
    summary = {"lick": [], "vel": []}

    for channel, obs_key, mod_key, ylab, title in [
        ("lick", None, "lat_lick_rate", "lick count / bin", "spatial lick profile"),
        ("vel", None, "lat_v_mean", "velocity (cm/s)", "velocity profile"),
    ]:
        fig, axes = plt.subplots(4, 4, figsize=(16, 14))
        for ax, mouse, m in zip(axes.flat, ids, cohort):
            ft = fits[mouse]
            nt = int(ft["n_trials"])
            mask = m["mask"]
            if channel == "lick":
                obs = m["licks"]
            else:
                obs = np.exp(m["logv"])
            mod = np.asarray(ft[mod_key])
            idxs = epoch_idx(lps[mouse], nt)

            d_obs = d_mod = None
            for (lab, col), idx in zip(EPOCHS, idxs):
                if idx is None:
                    continue
                po = profile(obs, mask, idx)
                pm = profile(mod, mask, idx)
                ax.plot(bins, po, c=col, lw=1.9, label=f"{lab} obs")
                ax.plot(bins, pm, c=col, lw=1.6, ls="--", label=f"{lab} model")
                if lab == "Naive":
                    n_obs, n_mod = po, pm
                if lab == "Expert":
                    d_obs = po - n_obs
                    d_mod = pm - n_mod
            # how well does the model reproduce the Naive->Expert change?
            r = np.nan
            if d_obs is not None and np.std(d_obs) > 1e-9 and np.std(d_mod) > 1e-9:
                r = float(np.corrcoef(d_obs, d_mod)[0, 1])
            summary[channel].append(r)
            landmarks(ax)
            lp = lps[mouse]
            ax.set_title(f"{mouse}  (LP={'NA' if np.isnan(lp) else int(lp)})  "
                         f"Δ r={r:.2f}", fontsize=9,
                         color="#1a7a3a" if (not np.isnan(r) and r > 0.5) else "#b03030")
            ax.tick_params(labelsize=7)
        axes.flat[0].legend(fontsize=6, ncol=1)
        rs = np.array(summary[channel], float)
        ok = np.sum(rs > 0.5)
        fig.suptitle(f"Per-epoch {title} — observed (solid) vs model (dashed)\n"
                     f"Naive=red Intermediate=amber Expert=green | "
                     f"Naive→Expert change reproduced (Δr>0.5) for {ok}/{len(rs)} mice "
                     f"(mean Δr={np.nanmean(rs):+.2f})", fontsize=12)
        fig.supxlabel("spatial bin (5 cm/bin)")
        fig.supylabel(ylab)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = os.path.join(FIGDIR, f"fig_epoch_validation_{channel}.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"wrote {out}")

    for ch in ("lick", "vel"):
        rs = np.array(summary[ch], float)
        print(f"  {ch:4s}  Naive->Expert change recovered: "
              f"{np.sum(rs > 0.5)}/{len(rs)} mice (Δr>0.5), mean Δr = "
              f"{np.nanmean(rs):+.3f}")
    print("DONE")


if __name__ == "__main__":
    main()
