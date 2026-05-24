"""Figures and latent export for the real-data fit of the belief-state RL model.

Reads the per-mouse fits produced by `fit_real_data.py`, re-loads the observed
behaviour, and writes:

  figures/fig_real_fit_quality.png   held-out log-likelihood vs the null
  figures/fig_real_lick_profiles.png observed vs predicted spatial lick profile
  figures/fig_real_latents_value.png value latent, all 16 mice
  figures/fig_real_latents_rpe.png   RPE latent, all 16 mice
  figures/fig_real_example.png       one mouse in detail
  results/rl_latents.mat             per-mouse latents for the neural analysis
  results/rl_latents.npz             same, NumPy

Run:  python -m scripts.plot_real_data
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import savemat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_model.config import (PARAM_NAMES, RZ_MASK, VISUAL_LANDMARK_AU,        # noqa
                             REWARD_START_AU, REWARD_END_AU, BIN_SIZE_AU, TaskConfig)
from rl_model.io_real import load_real_cohort                                  # noqa

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(HERE, "figures")
RESDIR = os.path.join(HERE, "results")
REALDIR = os.path.join(RESDIR, "real_fits_v3")
MAT = os.path.join(HERE, "..", "processed_data", "preprocessed_data.mat")

VZ = VISUAL_LANDMARK_AU / BIN_SIZE_AU
RZ0 = REWARD_START_AU / BIN_SIZE_AU
RZ1 = REWARD_END_AU / BIN_SIZE_AU
LATENT_KEYS = ["value", "rpe", "precision", "lick_rate", "v_mean",
               "belief_mean", "sigma"]


def landmarks(ax, horiz=False):
    fn = ax.axhline if horiz else ax.axvline
    fn(VZ, c="#2b6cc4", ls="--", lw=1.2)
    fn(RZ0, c="#1a7a3a", lw=1.2)
    fn(RZ1, c="#1a7a3a", lw=1.2)


def main():
    cfg = TaskConfig()
    cohort = {m["mouse"]: m for m in load_real_cohort(MAT, cfg)}
    ids = sorted(cohort)
    fits = {i: np.load(os.path.join(REALDIR, f"{i}.npz"), allow_pickle=True)
            for i in ids}

    # ---------------------------------------------------------------- Fig 1
    # cross-validated held-out log-likelihood, lick & velocity channels
    lick_gain = np.array([float(fits[i]["model_lick_test"]) -
                          float(fits[i]["null_lick_test"]) for i in ids])
    vel_gain = np.array([float(fits[i]["model_vel_test"]) -
                         float(fits[i]["null_vel_test"]) for i in ids])
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    x = np.arange(len(ids))
    ax[0].bar(x - 0.2, lick_gain, 0.4, label="lick channel",
              color=["#1a7a3a" if g > 0 else "#b03030" for g in lick_gain])
    ax[0].bar(x + 0.2, vel_gain, 0.4, label="velocity channel", color="#9aa0a6")
    ax[0].axhline(0, c="k", lw=0.8)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(ids, rotation=90, fontsize=7)
    ax[0].set_ylabel("held-out LL gain over null (nats / bin)")
    ax[0].set_title(f"Cross-validated fit quality\nlick: {(lick_gain>0).sum()}/16 "
                    f"mice beat the saturated per-bin null (mean {lick_gain.mean():+.3f})")
    ax[0].legend(fontsize=8)
    # pooled observed vs predicted spatial lick profile
    obs_all, pred_all = [], []
    for i in ids:
        m, ft = cohort[i], fits[i]
        obs_all.append(_profile(m["licks"], m["mask"]))
        pred_all.append(_profile(np.asarray(ft["lat_lick_rate"]), m["mask"]))
    obs_all, pred_all = np.array(obs_all), np.array(pred_all)
    bins = np.arange(cfg.n_bins)
    ax[1].plot(bins, obs_all.mean(0), c="k", lw=2.4, label="observed")
    ax[1].plot(bins, pred_all.mean(0), c="#b03030", lw=2.4, ls="--", label="model")
    ax[1].fill_between(bins, obs_all.mean(0) - obs_all.std(0) / 4,
                       obs_all.mean(0) + obs_all.std(0) / 4, color="k", alpha=0.12)
    landmarks(ax[1])
    ax[1].set_xlabel("spatial bin")
    ax[1].set_ylabel("lick count / bin")
    ax[1].set_title("Cohort spatial lick profile — observed vs model")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_real_fit_quality.png"), dpi=140)
    plt.close(fig)
    print("wrote fig_real_fit_quality.png")

    # ---------------------------------------------------------------- Fig 2
    # observed vs predicted lick profile, every mouse
    fig, axes = plt.subplots(4, 4, figsize=(15, 12))
    for ax, i in zip(axes.flat, ids):
        m, ft = cohort[i], fits[i]
        ax.plot(bins, _profile(m["licks"], m["mask"]), c="k", lw=1.8)
        ax.plot(bins, _profile(np.asarray(ft["lat_lick_rate"]), m["mask"]),
                c="#b03030", lw=1.8, ls="--")
        landmarks(ax)
        g = float(ft["model_lick_test"]) - float(ft["null_lick_test"])
        ax.set_title(f"{i}  ({int(ft['n_trials'])} tr)  LL gain {g:+.2f}",
                     fontsize=9, color="#1a7a3a" if g > 0 else "#b03030")
        ax.tick_params(labelsize=7)
    fig.suptitle("Observed (black) vs model (red) spatial lick profile — "
                 "per mouse", fontsize=13)
    fig.supxlabel("spatial bin")
    fig.supylabel("lick count / bin")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(FIGDIR, "fig_real_lick_profiles.png"), dpi=140)
    plt.close(fig)
    print("wrote fig_real_lick_profiles.png")

    # ----------------------------------------------------- Fig 3 & 4 latents
    _latent_grid("value", "Value V(b)", "viridis", ids, cohort, fits)
    _latent_grid("rpe", "RPE (TD error)", "RdBu_r", ids, cohort, fits)

    # ---------------------------------------------------------------- Fig 5
    _example_figure("M02", cohort, fits, cfg)

    # --------------------------------------------------------- latent export
    export = {}
    for k in LATENT_KEYS:
        arr = np.empty((1, len(ids)), dtype=object)
        for j, i in enumerate(ids):
            arr[0, j] = np.asarray(fits[i][f"lat_{k}"])
        export[k] = arr
    export["mouse_id"] = np.array(ids, dtype=object).reshape(1, -1)
    export["n_trials"] = np.array([int(fits[i]["n_trials"]) for i in ids])
    export["params"] = np.array([np.asarray(fits[i]["params"]) for i in ids])
    export["param_names"] = np.array(PARAM_NAMES, dtype=object).reshape(1, -1)
    export["test_idx"] = np.empty((1, len(ids)), dtype=object)
    for j, i in enumerate(ids):
        export["test_idx"][0, j] = np.asarray(fits[i]["test_idx"]) + 1   # MATLAB 1-based
    savemat(os.path.join(RESDIR, "rl_latents.mat"), {"rl_latents": export})
    np.savez_compressed(os.path.join(RESDIR, "rl_latents.npz"),
                        **{k: np.asarray([np.asarray(fits[i][f"lat_{k}"])
                                          for i in ids], dtype=object)
                           for k in LATENT_KEYS},
                        mouse_id=np.array(ids))
    print("wrote results/rl_latents.mat and .npz")
    print("DONE")


def _profile(arr, mask):
    """Trial-averaged spatial profile over valid bins."""
    w = mask.sum(0)
    return (arr * mask).sum(0) / np.maximum(w, 1.0)


def _latent_grid(key, label, cmap, ids, cohort, fits):
    fig, axes = plt.subplots(4, 4, figsize=(15, 12))
    for ax, i in zip(axes.flat, ids):
        arr = np.asarray(fits[i][f"lat_{key}"])
        nt = arr.shape[0]
        if key == "rpe":
            vlim = np.nanpercentile(np.abs(arr), 99) + 1e-9
            kw = dict(cmap=cmap, vmin=-vlim, vmax=vlim)
        else:
            kw = dict(cmap=cmap)
        im = ax.imshow(arr, aspect="auto", interpolation="nearest",
                       extent=[0, arr.shape[1], nt, 0], **kw)
        landmarks(ax)
        ax.set_title(f"{i} ({nt} tr)", fontsize=9)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(f"Inferred latent — {label} — per (trial × bin), all 16 mice",
                 fontsize=13)
    fig.supxlabel("spatial bin")
    fig.supylabel("trial")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(FIGDIR, f"fig_real_latents_{key}.png"), dpi=140)
    plt.close(fig)
    print(f"wrote fig_real_latents_{key}.png")


def _example_figure(mid, cohort, fits, cfg):
    m, ft = cohort[mid], fits[mid]
    nt = int(ft["n_trials"])
    maps = [("observed lick raster", m["licks"], "Greys", None),
            ("Value V(b)", np.asarray(ft["lat_value"]), "viridis", None),
            ("RPE (TD error)", np.asarray(ft["lat_rpe"]), "RdBu_r", "sym"),
            ("Precision 1/sigma", np.asarray(ft["lat_precision"]), "magma", None),
            ("Lick rate (model)", np.asarray(ft["lat_lick_rate"]), "Greys", None)]
    fig = plt.figure(figsize=(17, 6.2))
    for j, (lab, arr, cmap, sym) in enumerate(maps):
        ax = fig.add_subplot(1, 6, j + 1)
        kw = dict(cmap=cmap, aspect="auto", interpolation="nearest",
                  extent=[0, arr.shape[1], nt, 0])
        if sym == "sym":
            v = np.nanpercentile(np.abs(arr), 99) + 1e-9
            kw.update(vmin=-v, vmax=v)
        im = ax.imshow(arr, **kw)
        landmarks(ax)
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("bin", fontsize=8)
        if j == 0:
            ax.set_ylabel("trial")
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    # learning curve obs vs model
    ax = fig.add_subplot(1, 6, 6)
    rzw = slice(22, 31)
    obs = (m["licks"][:, rzw] * m["mask"][:, rzw]).sum(1) / \
          np.maximum(m["mask"][:, rzw].sum(1), 1)
    pred = np.asarray(ft["lat_lick_rate"])[:, rzw].mean(1)
    k = max(nt // 25, 1)
    sm = lambda a: np.convolve(a, np.ones(k) / k, mode="same")
    ax.plot(sm(obs), np.arange(nt), c="k", lw=2, label="observed")
    ax.plot(sm(pred), np.arange(nt), c="#b03030", lw=2, ls="--", label="model")
    ax.invert_yaxis()
    ax.set_xlabel("RZ-window lick rate", fontsize=8)
    ax.set_title("learning curve", fontsize=10)
    ax.legend(fontsize=7)
    fig.suptitle(f"Example mouse {mid}: observed behaviour and inferred latents",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIGDIR, "fig_real_example.png"), dpi=140)
    plt.close(fig)
    print("wrote fig_real_example.png")


if __name__ == "__main__":
    main()
