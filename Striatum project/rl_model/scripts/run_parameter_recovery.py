"""Parameter-recovery experiment for the belief-state RL model.

The TDD gate (CLAUDE.md): before any real animal data is touched, generate a
synthetic cohort with *known* ground-truth parameters, fit each mouse by
maximum a posteriori, and verify that

  (1) the subjective parameters are recovered, and
  (2) the per-(trial x bin) latents that will serve as neural regressors are
      recovered

even where individual parameters are weakly identified.

The script is *incremental*: each invocation fits a batch of mice (so a run
stays inside a short shell timeout) and the final invocation finalises the
analysis (recovery stats + figures).  Just run it repeatedly:

    python -m scripts.run_parameter_recovery        # repeat until it prints DONE

Outputs:
  results/fits/mouse_XX.npz      — per-mouse fitted parameters
  results/recovery_results.npz   — all numbers
  figures/fig_param_recovery.png
  figures/fig_latent_recovery.png
  figures/fig_behaviour.png
  figures/fig_example_latents.png
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_model.config import (  # noqa: E402
    PARAM_NAMES, RZ_MASK, VISUAL_LANDMARK_AU, REWARD_START_AU,
    REWARD_END_AU, BIN_SIZE_AU, TaskConfig,
)
from rl_model.agent import session_latents, to_constrained  # noqa: E402
from rl_model.synthetic import make_cohort  # noqa: E402
from rl_model.fitting import fit_mouse  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(HERE, "figures")
RESDIR = os.path.join(HERE, "results")
FITSDIR = os.path.join(RESDIR, "fits_v3")
RESULTS_NPZ = os.path.join(RESDIR, "recovery_v3.npz")
DONE_MARK = os.path.join(RESDIR, "DONE_v3")
for d in (FIGDIR, RESDIR, FITSDIR):
    os.makedirs(d, exist_ok=True)

N_MICE = 12
N_TRIALS = 160
N_RESTARTS = 1          # restart-1 (from defaults) reliably wins; see UNDERSTANDING
BATCH = 4               # mice fit per invocation (keeps each run inside the shell timeout)
LATENT_KEYS = ["value", "rpe", "precision", "lick_rate", "v_mean"]
LATENT_LABELS = {
    "value": "Value V(b)", "rpe": "RPE (TD error)", "precision": "Precision 1/sigma",
    "lick_rate": "Lick rate (lambda)", "v_mean": "Expected velocity",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pearson(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def get_cohort(cfg):
    """Deterministic — regenerated identically on every invocation."""
    return make_cohort(n_mice=N_MICE, n_trials=N_TRIALS, seed=2026, jitter=0.35, cfg=cfg)


# --------------------------------------------------------------------------
def fit_batch(cohort, cfg):
    """Fit up to BATCH not-yet-fitted mice.  Returns True when all are done."""
    done = {int(f[6:8]) for f in os.listdir(FITSDIR) if f.startswith("mouse_")}
    todo = [m for m in range(N_MICE) if m not in done]
    if not todo:
        return True
    for m in todo[:BATCH]:
        tm = time.time()
        res = fit_mouse(cohort["licks"][m], cohort["logv"][m], cfg=cfg,
                        n_restarts=N_RESTARTS, seed=100 + m)
        np.savez(os.path.join(FITSDIR, f"mouse_{m:02d}.npz"),
                 u_fit=res["u_fit"], nll=res["nll"])
        log(f"  fit mouse {m + 1:2d}/{N_MICE}  nll={res['nll']:9.1f}  "
            f"({time.time() - tm:.1f}s)")
    n_done = len(done) + len(todo[:BATCH])
    log(f"batch complete: {n_done}/{N_MICE} mice fitted")
    return n_done >= N_MICE


# --------------------------------------------------------------------------
def finalize(cohort, cfg):
    u_true = cohort["u_true"]
    licks, logv = cohort["licks"], cohort["logv"]
    u_fit = np.stack([np.load(os.path.join(FITSDIR, f"mouse_{m:02d}.npz"))["u_fit"]
                      for m in range(N_MICE)])
    nlls = np.array([float(np.load(os.path.join(FITSDIR, f"mouse_{m:02d}.npz"))["nll"])
                     for m in range(N_MICE)])

    p_true = np.array([[float(v) for v in to_constrained(jnp.asarray(u)).values()]
                       for u in u_true])
    p_fit = np.array([[float(v) for v in to_constrained(jnp.asarray(u)).values()]
                      for u in u_fit])
    param_r = {n: pearson(p_true[:, i], p_fit[:, i]) for i, n in enumerate(PARAM_NAMES)}

    # latent recovery: true-parameter vs fitted-parameter latents on the same data
    latent_r = {k: [] for k in LATENT_KEYS}
    true_lat0, fit_lat0 = {}, {}
    for m in range(N_MICE):
        lt = session_latents(jnp.asarray(u_true[m]), jnp.asarray(licks[m]),
                             jnp.asarray(logv[m]), cfg=cfg)
        lf = session_latents(jnp.asarray(u_fit[m]), jnp.asarray(licks[m]),
                             jnp.asarray(logv[m]), cfg=cfg)
        for k in LATENT_KEYS:
            latent_r[k].append(pearson(np.asarray(lt[k]), np.asarray(lf[k])))
        if m == 0:
            true_lat0 = {k: np.asarray(lt[k]) for k in LATENT_KEYS}
            fit_lat0 = {k: np.asarray(lf[k]) for k in LATENT_KEYS}
    latent_r_mean = {k: float(np.nanmean(latent_r[k])) for k in LATENT_KEYS}

    np.savez_compressed(
        RESULTS_NPZ,
        u_true=u_true, u_fit=u_fit, p_true=p_true, p_fit=p_fit, nll=nlls,
        param_names=np.array(PARAM_NAMES), latent_keys=np.array(LATENT_KEYS),
        latent_r=np.array([latent_r[k] for k in LATENT_KEYS]),
        param_r=np.array([param_r[n] for n in PARAM_NAMES]),
    )

    _fig_param_recovery(p_true, p_fit, param_r)
    _fig_latent_recovery(true_lat0, fit_lat0, latent_r_mean)
    _fig_behaviour(licks, cfg)
    _fig_example_latents(true_lat0, cfg)

    log("=" * 58)
    log("PARAMETER RECOVERY (Pearson r, true vs fitted, natural space)")
    for n in PARAM_NAMES:
        r = param_r[n]
        tag = "OK  " if (not np.isnan(r) and r >= 0.6) else "WEAK"
        log(f"   {tag}  {n:14s} r = {r:.3f}")
    log("LATENT RECOVERY (cohort-mean Pearson r)")
    for k in LATENT_KEYS:
        log(f"         {k:14s} r = {latent_r_mean[k]:.4f}")
    with open(DONE_MARK, "w") as fh:
        fh.write("ok\n")
    log("DONE")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _fig_param_recovery(p_true, p_fit, param_r):
    fig, axes = plt.subplots(4, 4, figsize=(13, 12.5))
    for i, name in enumerate(PARAM_NAMES):
        ax = axes.flat[i]
        t, f = p_true[:, i], p_fit[:, i]
        ax.scatter(t, f, s=36, c="#3457a6", edgecolor="white", linewidth=0.6, zorder=3)
        lo, hi = min(t.min(), f.min()), max(t.max(), f.max())
        pad = 0.08 * (hi - lo + 1e-9)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", c="0.5", lw=1)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        r = param_r[name]
        good = (not np.isnan(r)) and r >= 0.6
        ax.set_title(f"{name}    r = {r:.2f}", fontsize=10,
                     color="#1a7a3a" if good else "#b03030")
        ax.set_xlabel("true", fontsize=8)
        ax.set_ylabel("fitted", fontsize=8)
        ax.tick_params(labelsize=7)
    for j in range(len(PARAM_NAMES), axes.size):
        axes.flat[j].axis("off")
    fig.suptitle(f"Parameter recovery — synthetic cohort "
                 f"(n = {N_MICE} mice, {N_TRIALS} trials each)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIGDIR, "fig_param_recovery.png"), dpi=140)
    plt.close(fig)
    log("wrote fig_param_recovery.png")


def _fig_latent_recovery(true_lat, fit_lat, latent_r_mean):
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6))
    rng = np.random.default_rng(0)
    for i, k in enumerate(LATENT_KEYS):
        ax = axes[i]
        t = true_lat[k].ravel()
        f = fit_lat[k].ravel()
        idx = rng.choice(t.size, size=min(4000, t.size), replace=False)
        ax.scatter(t[idx], f[idx], s=6, alpha=0.25, c="#3457a6", edgecolor="none")
        lo, hi = float(np.nanmin(t)), float(np.nanmax(t))
        ax.plot([lo, hi], [lo, hi], "--", c="0.5", lw=1)
        rr = latent_r_mean[k]
        ax.set_title(f"{LATENT_LABELS[k]}    r = {rr:.3f}", fontsize=10,
                     color="#1a7a3a" if rr >= 0.9 else "#b03030")
        ax.set_xlabel("true-parameter latent", fontsize=8)
        ax.set_ylabel("fitted-parameter latent", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("Latent-variable recovery — the regressors handed to the neural "
                 "analysis (mouse 1; cohort-mean r in title)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(os.path.join(FIGDIR, "fig_latent_recovery.png"), dpi=140)
    plt.close(fig)
    log("wrote fig_latent_recovery.png")


def _fig_behaviour(licks, cfg):
    rzw = np.zeros(cfg.n_bins, bool)
    rzw[22:31] = True
    blocks = np.arange(0, N_TRIALS, 20)
    curve = np.array([[licks[m, b:b + 20][:, rzw].mean() for b in blocks]
                      for m in range(N_MICE)])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    bx = blocks + 10
    ax[0].plot(bx, curve.T, c="0.75", lw=0.8)
    ax[0].plot(bx, curve.mean(0), c="#b03030", lw=2.6, marker="o", label="cohort mean")
    ax[0].set_xlabel("trial")
    ax[0].set_ylabel("RZ-window lick rate (bins 22-30)")
    ax[0].set_title("Within-session learning of the licking strategy")
    ax[0].legend(fontsize=8)
    ax[1].imshow(licks[0], aspect="auto", cmap="Greys", interpolation="nearest",
                 extent=[0, cfg.n_bins, N_TRIALS, 0])
    ax[1].axvline(VISUAL_LANDMARK_AU / BIN_SIZE_AU, c="#2b6cc4", ls="--", lw=1.5)
    ax[1].axvline(REWARD_START_AU / BIN_SIZE_AU, c="#1a7a3a", lw=1.5)
    ax[1].axvline(REWARD_END_AU / BIN_SIZE_AU, c="#1a7a3a", lw=1.5)
    ax[1].set_xlabel("spatial bin")
    ax[1].set_ylabel("trial")
    ax[1].set_title("Example lick raster (mouse 1)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_behaviour.png"), dpi=140)
    plt.close(fig)
    log("wrote fig_behaviour.png")


def _fig_example_latents(true_lat, cfg):
    maps = [("value", "Value V(b)", "viridis"),
            ("rpe", "RPE (TD error)", "RdBu_r"),
            ("precision", "Precision 1/sigma", "magma"),
            ("lick_rate", "Lick rate (lambda)", "Greys")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (k, lab, cm) in zip(axes.flat, maps):
        arr = true_lat[k]
        vlim = float(np.nanpercentile(np.abs(arr), 99))
        kw = dict(cmap=cm, aspect="auto", interpolation="nearest",
                  extent=[0, cfg.n_bins, N_TRIALS, 0])
        if k == "rpe":
            kw.update(vmin=-vlim, vmax=vlim)
        im = ax.imshow(arr, **kw)
        ax.axvline(VISUAL_LANDMARK_AU / BIN_SIZE_AU, c="#2b6cc4", ls="--", lw=1.4)
        ax.axvline(REWARD_START_AU / BIN_SIZE_AU, c="#1a7a3a", lw=1.4)
        ax.axvline(REWARD_END_AU / BIN_SIZE_AU, c="#1a7a3a", lw=1.4)
        ax.set_title(lab)
        ax.set_xlabel("spatial bin")
        ax.set_ylabel("trial")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Model latents over a session (mouse 1) — the per-(trial x bin) "
                 "regressors", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIGDIR, "fig_example_latents.png"), dpi=140)
    plt.close(fig)
    log("wrote fig_example_latents.png")


# --------------------------------------------------------------------------
def main():
    cfg = TaskConfig()
    cohort = get_cohort(cfg)
    if fit_batch(cohort, cfg):
        log("all mice fitted — finalising")
        finalize(cohort, cfg)
    else:
        log("run again to fit the next batch")


if __name__ == "__main__":
    main()
