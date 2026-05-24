"""Pytest suite for the belief-state RL model.

Smoke tests on the agent/likelihood plus a fast end-to-end parameter-recovery
check on a small synthetic cohort.  Run:  python -m pytest tests/ -q
"""
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_model.config import N_PARAMS, TaskConfig
from rl_model.agent import (
    default_unconstrained, session_latents, session_loglik, simulate_session,
    to_constrained, to_unconstrained,
)
from rl_model.synthetic import make_cohort, sample_params
from rl_model.fitting import fit_mouse

CFG = TaskConfig()


# --------------------------------------------------------------------------
# Parameter transforms
# --------------------------------------------------------------------------
def test_param_roundtrip():
    u = default_unconstrained()
    u2 = to_unconstrained(to_constrained(u))
    assert np.allclose(np.asarray(u), np.asarray(u2), atol=1e-6)
    assert u.shape == (N_PARAMS,)


# --------------------------------------------------------------------------
# Agent forward pass
# --------------------------------------------------------------------------
def test_simulate_shapes_and_finiteness():
    u = default_unconstrained()
    out = simulate_session(u, jax.random.PRNGKey(0), 40, CFG)
    for k in ["value", "rpe", "precision", "lick_rate", "v_mean", "lick", "logv"]:
        a = np.asarray(out[k])
        assert a.shape == (40, CFG.n_bins)
        assert np.all(np.isfinite(a)), f"{k} not finite"
    # licks are non-negative integer counts; velocities positive
    licks = np.asarray(out["lick"])
    assert np.all(licks >= 0) and np.allclose(licks, np.round(licks))
    assert np.all(np.exp(np.asarray(out["logv"])) > 0)


def test_belief_is_a_distribution():
    """Precision (1/sigma) is positive and the belief stays normalised."""
    u = default_unconstrained()
    out = simulate_session(u, jax.random.PRNGKey(1), 20, CFG)
    assert np.all(np.asarray(out["precision"]) > 0)
    assert np.all(np.asarray(out["sigma"]) > 0)


def test_loglik_finite_and_gradient_flows():
    u = default_unconstrained()
    out = simulate_session(u, jax.random.PRNGKey(2), 60, CFG)
    licks, logv = out["lick"], out["logv"]
    ll = session_loglik(u, licks, logv, cfg=CFG)
    assert np.isfinite(float(ll))
    g = jax.grad(lambda uu: session_loglik(uu, licks, logv, cfg=CFG))(u)
    assert np.all(np.isfinite(np.asarray(g)))
    assert np.linalg.norm(np.asarray(g)) > 0


def test_mask_zeroes_likelihood_contribution():
    """A bin masked out must not contribute to the log-likelihood."""
    u = default_unconstrained()
    out = simulate_session(u, jax.random.PRNGKey(4), 30, CFG)
    licks, logv = out["lick"], out["logv"]
    full = np.ones((30, CFG.n_bins))
    half = full.copy()
    half[:, 25:] = 0.0
    ll_full = float(session_loglik(u, licks, logv, mask=full, cfg=CFG))
    ll_half = float(session_loglik(u, licks, logv, mask=half, cfg=CFG))
    assert ll_half != ll_full
    # masked-everything -> exactly zero
    assert abs(float(session_loglik(u, licks, logv,
                                    mask=np.zeros((30, CFG.n_bins)), cfg=CFG))) < 1e-6


def test_true_params_beat_wrong_params():
    """The data-generating parameters score higher likelihood than wrong ones."""
    u_true = sample_params(seed=5, jitter=0.3)
    out = simulate_session(u_true, jax.random.PRNGKey(9), 160, CFG)
    licks, logv = out["lick"], out["logv"]
    ll_true = float(session_loglik(u_true, licks, logv, cfg=CFG))
    rng = np.random.default_rng(0)
    for _ in range(5):
        u_wrong = u_true + rng.normal(scale=1.0, size=N_PARAMS)
        assert ll_true > float(session_loglik(jnp.asarray(u_wrong), licks, logv, cfg=CFG))


# --------------------------------------------------------------------------
# Behavioural realism: the agent learns to localise licking to the RZ
# --------------------------------------------------------------------------
def test_agent_localises_licking_within_session():
    """With the prior-task value `w_init`, learning is *carving*: licking starts
    elevated everywhere and is suppressed away from the RZ over trials."""
    u = default_unconstrained()
    out = simulate_session(u, jax.random.PRNGKey(7), 160, CFG)
    licks = np.asarray(out["lick"])
    rzw = np.zeros(CFG.n_bins, bool)
    rzw[22:31] = True
    far = np.zeros(CFG.n_bins, bool)
    far[:16] = True
    far[40:] = True
    early_far = licks[:20][:, far].mean()
    late_far = licks[-40:][:, far].mean()
    late_rzw = licks[-40:][:, rzw].mean()
    assert late_far < early_far, f"far licking not suppressed: {early_far:.3f}->{late_far:.3f}"
    assert late_rzw > late_far, f"licking not localised to RZ: rzw {late_rzw:.3f} far {late_far:.3f}"


# --------------------------------------------------------------------------
# End-to-end recovery (small + fast)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0])
def test_parameter_recovery_small(seed):
    """Well-identified parameters recover; latents recover regardless."""
    n = 3
    cohort = make_cohort(n_mice=n, n_trials=160, seed=seed, jitter=0.3, cfg=CFG)
    u_true, licks, logv = cohort["u_true"], cohort["licks"], cohort["logv"]

    u_fit = np.zeros_like(u_true)
    for m in range(n):
        res = fit_mouse(licks[m], logv[m], cfg=CFG, n_restarts=1, seed=m)
        u_fit[m] = res["u_fit"]

    # value/policy/velocity parameters should track truth
    well = ["eta_w", "beta", "theta", "v_base", "v_slope", "log_sigma_v"]
    from rl_model.config import PARAM_NAMES
    well_idx = [PARAM_NAMES.index(k) for k in well]
    err = np.abs(u_true[:, well_idx] - u_fit[:, well_idx])
    assert np.median(err) < 0.5, f"well-identified params drifted: {np.median(err):.3f}"

    # Latent recovery: cohort-level (per-mouse correlation is degenerate when a
    # true latent is nearly flat).
    rr = {k: [] for k in ["value", "rpe", "precision", "lick_rate"]}
    for m in range(n):
        lt = session_latents(jnp.asarray(u_true[m]), jnp.asarray(licks[m]),
                             jnp.asarray(logv[m]), cfg=CFG)
        lf = session_latents(jnp.asarray(u_fit[m]), jnp.asarray(licks[m]),
                             jnp.asarray(logv[m]), cfg=CFG)
        for k in rr:
            t, f = np.asarray(lt[k]).ravel(), np.asarray(lf[k]).ravel()
            if np.std(t) > 1e-9 and np.std(f) > 1e-9:
                rr[k].append(np.corrcoef(t, f)[0, 1])
    for k in ["value", "rpe", "precision"]:
        assert np.mean(rr[k]) > 0.85, f"latent {k} recovered poorly: {rr[k]}"
