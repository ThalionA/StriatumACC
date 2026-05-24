"""Fit the belief-state RL model to the 16 real task mice, with cross-validation.

For each mouse:
  * interleaved cross-validation — every 5th trial is held out as a test set;
  * the model is fit by MAP on the train trials only (test trials still pass
    through the agent so within-session learning is uninterrupted, but they do
    not enter the fitting objective);
  * held-out predictive log-likelihood is evaluated on the test trials,
    separately for the lick channel and the velocity channel, and compared with
    a null model (per-bin Poisson rate + per-bin log-normal velocity estimated
    from the train trials — a saturated >100-parameter reference that captures
    the spatial profile but no trial-by-trial learning);
  * all latents are exported for every trial.

Incremental: each invocation fits as many mice as fit in a short time budget
and the last invocation writes the cross-validation summary.  Run repeatedly:

    python -m scripts.fit_real_data        # repeat until it prints DONE
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import jax.numpy as jnp
from scipy.special import gammaln

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_model.config import PARAM_NAMES, TaskConfig          # noqa: E402
from rl_model.agent import session_latents                    # noqa: E402
from rl_model.io_real import load_real_cohort                  # noqa: E402
from rl_model.fitting import fit_mouse                         # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESDIR = os.path.join(HERE, "results")
REALDIR = os.path.join(RESDIR, "real_fits_v5")   # v5: graded reward + deterministic velocity actor
os.makedirs(REALDIR, exist_ok=True)
MAT = os.path.join(HERE, "..", "processed_data", "preprocessed_data5cm.mat")

TEST_EVERY = 5            # interleaved CV: every 5th trial held out
TIME_BUDGET = 25.0        # seconds of fitting per invocation (stay under shell timeout)
LATENT_KEYS = ["value", "rpe", "precision", "lick_rate", "v_mean",
               "belief_mean", "sigma"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bucket_len(nt):
    """Common padded length — bucketed so short sessions are not over-padded
    (all mice in a bucket share one JIT compilation)."""
    return 256 if nt <= 256 else 512


def pad(arr, T):
    out = np.zeros((T,) + arr.shape[1:], dtype=float)
    out[:arr.shape[0]] = arr
    return out


def null_components(licks, logv, mask, train_idx):
    """Saturated per-bin null estimated on train trials.

    Returns (ll_lick, ll_vel), each a (n_trials, n_bins) per-bin log-likelihood:
      lick  ~ Poisson(rate_b),  rate_b = mean train lick count in bin b
      logv  ~ Normal(mu_b, sd_b) from the train trials of bin b
    """
    n_bins = licks.shape[1]
    lam_b = np.full(n_bins, 1e-6)
    mu_b = np.zeros(n_bins)
    sd_b = np.ones(n_bins)
    for b in range(n_bins):
        sel = mask[train_idx, b] > 0
        if sel.sum() > 2:
            lam_b[b] = max(licks[train_idx, b][sel].mean(), 1e-6)
            mu_b[b] = logv[train_idx, b][sel].mean()
            sd_b[b] = max(logv[train_idx, b][sel].std(), 1e-2)
    ll_lick = (licks * np.log(lam_b)[None, :] - lam_b[None, :]
               - gammaln(licks + 1.0)) * mask
    ll_vel = ((-0.5 * np.log(2 * np.pi) - np.log(sd_b)[None, :]
               - 0.5 * ((logv - mu_b[None, :]) / sd_b[None, :]) ** 2) * mask)
    return ll_lick, ll_vel


def _per_bin(ll, idx, mask, idx_set):
    """Total log-likelihood over `idx_set` trials, divided by their valid bins."""
    return ll[idx_set].sum() / max(mask[idx_set].sum(), 1.0)


def fit_one(mouse, cfg, n_restarts=1, maxiter=None):
    """Fit one mouse; return a dict of everything to persist."""
    nt = mouse["n_trials"]
    T = bucket_len(nt)
    licks, logv, mask = mouse["licks"], mouse["logv"], mouse["mask"]

    test_idx = np.arange(TEST_EVERY - 1, nt, TEST_EVERY)
    train_idx = np.setdiff1d(np.arange(nt), test_idx)

    licks_p, logv_p, mask_p = pad(licks, T), pad(logv, T), pad(mask, T)
    fit_mask = mask_p.copy()
    fit_mask[test_idx, :] = 0.0                     # exclude test trials from the fit

    # Long sessions get fewer optimiser iterations so each fit stays inside the
    # shell time slice; short sessions use the full count.
    mi = maxiter if maxiter is not None else (250 if nt > 200 else 400)
    res = fit_mouse(licks_p, logv_p, mask=fit_mask, cfg=cfg, n_restarts=n_restarts,
                    seed=int(mouse["mouse"][1:]), maxiter=mi)
    u_fit = res["u_fit"]

    lat = session_latents(jnp.asarray(u_fit), licks_p, logv_p, mask=mask_p, cfg=cfg)
    ll_lick = np.asarray(lat["loglik_lick"])
    ll_vel = np.asarray(lat["loglik_vel"])
    null_lick, null_vel = null_components(licks_p, logv_p, mask_p, train_idx)

    rec = dict(
        mouse=mouse["mouse"], n_trials=nt, u_fit=u_fit,
        params=np.array([res["params"][n] for n in PARAM_NAMES]),
        nll=res["nll"], test_idx=test_idx, train_idx=train_idx,
        # cross-validated log-likelihood per valid bin, lick / velocity channels
        model_lick_test=_per_bin(ll_lick, None, mask_p, test_idx),
        model_lick_train=_per_bin(ll_lick, None, mask_p, train_idx),
        model_vel_test=_per_bin(ll_vel, None, mask_p, test_idx),
        model_vel_train=_per_bin(ll_vel, None, mask_p, train_idx),
        null_lick_test=_per_bin(null_lick, None, mask_p, test_idx),
        null_vel_test=_per_bin(null_vel, None, mask_p, test_idx),
    )
    for k in LATENT_KEYS:
        rec[f"lat_{k}"] = np.asarray(lat[k])[:nt]
    return rec


def main():
    cfg = TaskConfig()
    t0 = time.time()
    cohort = load_real_cohort(MAT, cfg)
    log(f"loaded {len(cohort)} mice")

    done = set()                                     # only count intact files
    for f in os.listdir(REALDIR):
        if f.endswith(".npz"):
            try:
                np.load(os.path.join(REALDIR, f), allow_pickle=True)["mouse"]
                done.add(f.split(".")[0])
            except Exception:
                log(f"  (re-fitting {f}: corrupt)")
    todo = [m for m in cohort if m["mouse"] not in done]
    if not todo:
        return finalise(cohort)

    todo.sort(key=lambda m: m["n_trials"])           # same-bucket mice consecutive
    start_done = len(done)
    for mouse in todo:
        if time.time() - t0 > TIME_BUDGET and len(done) > start_done:
            break
        tm = time.time()
        r = fit_one(mouse, cfg)
        np.savez(os.path.join(REALDIR, f"{mouse['mouse']}.npz"), **r)
        done.add(mouse["mouse"])
        log(f"  {mouse['mouse']}: {time.time()-tm:5.1f}s | lick LL/bin "
            f"model {r['model_lick_test']:+.3f} null {r['null_lick_test']:+.3f} | "
            f"vel model {r['model_vel_test']:+.3f} null {r['null_vel_test']:+.3f}")

    if len(done) >= len(cohort):
        finalise(cohort)
    else:
        log(f"{len(done)}/{len(cohort)} mice fitted — run again")


def finalise(cohort):
    rows = [np.load(os.path.join(REALDIR, f"{m['mouse']}.npz"), allow_pickle=True)
            for m in cohort]
    log("=" * 70)
    log("CROSS-VALIDATED FIT QUALITY — held-out log-likelihood per valid bin")
    log(f"{'mouse':6s} {'trials':>6s} | {'lick model':>10s} {'lick null':>9s} "
        f"{'gain':>7s} | {'vel model':>9s} {'vel null':>8s} {'gain':>7s}")
    lg, vg = [], []
    for r in rows:
        lgi = float(r["model_lick_test"]) - float(r["null_lick_test"])
        vgi = float(r["model_vel_test"]) - float(r["null_vel_test"])
        lg.append(lgi)
        vg.append(vgi)
        log(f"{str(r['mouse']):6s} {int(r['n_trials']):6d} | "
            f"{float(r['model_lick_test']):+10.3f} {float(r['null_lick_test']):+9.3f} "
            f"{lgi:+7.3f} | {float(r['model_vel_test']):+9.3f} "
            f"{float(r['null_vel_test']):+8.3f} {vgi:+7.3f}")
    lg, vg = np.array(lg), np.array(vg)
    log("-" * 70)
    log(f"LICK: mean held-out gain over null {lg.mean():+.3f} nats/bin "
        f"({(lg > 0).sum()}/{len(lg)} mice positive)")
    log(f"VEL : mean held-out gain over null {vg.mean():+.3f} nats/bin "
        f"({(vg > 0).sum()}/{len(vg)} mice positive)")
    with open(os.path.join(RESDIR, "DONE_real"), "w") as fh:
        fh.write("ok\n")
    log("DONE")


if __name__ == "__main__":
    main()
