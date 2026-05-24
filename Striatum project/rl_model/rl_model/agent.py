"""Belief-state RL agent: a two-process (perception + value) actor-critic.

The agent traverses a spatial corridor.  It maintains a Kalman belief over its
position (the *perceptual* process, fast) and a TD-learned critic ``V(belief)``
read out by stochastic lick and velocity policies (the *value* process, slow).

Discounted TD is used: the discount localises the value function into a bump at
the reward zone, so a value-thresholded lick policy licks *in* the RZ rather
than along the whole run-up.

Licking is a **Poisson count** per bin (real lick data are counts, 0..~10);
velocity is **log-normal**.  A per-bin **validity mask** lets the likelihood
ignore bins with missing behavioural data (NaN licks, zero-occupancy bins).

All computation is in JAX so the per-mouse log-likelihood is differentiable
w.r.t. the subjective parameters.  The same core step drives both generative
simulation and likelihood evaluation.

Conventions
-----------
* Positions are in arbitrary units (a.u.); velocities in cm/s.
* A "session" is ``(n_trials, n_bins)``.  Latents come out shaped the same way,
  aligned 1:1 with the neural ``spatial_binned_fr`` tensor.
* Fitting is *teacher-forced*: TD learning is driven by the mouse's observed
  licks/velocities; the free parameters are the subjective constants.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

from .config import (
    BIN_CENTRES_AU,
    BIN_SIZE_AU,
    BIN_SIZE_CM,
    PARAMS,
    PARAM_NAMES,
    RZ_MASK,
    VISUAL_LANDMARK_AU,
    TaskConfig,
)

_TRANSFORMS = tuple(p.transform for p in PARAMS)
_LOG2PI = 1.8378770664093453


# --------------------------------------------------------------------------
# Parameter transforms (unconstrained optimisation space <-> natural space)
# --------------------------------------------------------------------------
def _constrain_one(u, transform):
    if transform == "exp":
        return jnp.exp(u)
    if transform == "sigmoid":
        return jax.nn.sigmoid(u)
    return u  # identity


def _unconstrain_one(c, transform):
    if transform == "exp":
        return jnp.log(c)
    if transform == "sigmoid":
        return jnp.log(c) - jnp.log1p(-c)
    return c  # identity


def to_constrained(u):
    """Unconstrained parameter vector -> dict of natural-space parameters."""
    return {
        name: _constrain_one(u[i], _TRANSFORMS[i])
        for i, name in enumerate(PARAM_NAMES)
    }


def to_unconstrained(values):
    """dict or sequence of natural-space parameters -> unconstrained vector."""
    if isinstance(values, dict):
        seq = [values[name] for name in PARAM_NAMES]
    else:
        seq = list(values)
    return jnp.array(
        [_unconstrain_one(jnp.asarray(seq[i], float), _TRANSFORMS[i])
         for i in range(len(PARAM_NAMES))]
    )


def default_unconstrained():
    """Unconstrained vector at each parameter's `typical` value."""
    return to_unconstrained({p.name: p.typical for p in PARAMS})


# --------------------------------------------------------------------------
# Belief representation
# --------------------------------------------------------------------------
def belief_vector(mu, sigma, centres):
    """Normalised Gaussian belief over spatial bins (a probability vector)."""
    z = -0.5 * (centres - mu) ** 2 / jnp.maximum(sigma, 1e-3)
    return jax.nn.softmax(z)


# --------------------------------------------------------------------------
# Core session run.  `generate` is static: when True the agent samples its own
# actions (simulation); when False it scores observed actions (likelihood).
# Geometry is taken from module constants so array shapes stay static.
# --------------------------------------------------------------------------
@functools.partial(jax.jit, static_argnames=("generate", "sigma0", "reward_magnitude"))
def _run_session(u, licks_obs, logv_obs, mask_obs, keys, generate,
                 sigma0, reward_magnitude):
    n_bins = licks_obs.shape[1]
    p = to_constrained(u)
    centres = jnp.asarray(BIN_CENTRES_AU)
    rz_mask = jnp.asarray(RZ_MASK, dtype=float)
    x_true = centres                       # true position at each bin
    sigma_v = jnp.exp(p["log_sigma_v"])
    is_last = jnp.arange(n_bins) == (n_bins - 1)

    def bin_step(carry, xs):
        mu_pr, sig_pr, w, rewarded = carry
        xt, rz, last, lick_o, logv_o, valid, key = xs

        # --- perceptual update at the current bin (veridical-mean obs) ---
        R_obs = p["R_slope"] * jnp.abs(xt - VISUAL_LANDMARK_AU) + p["R0"]
        K = sig_pr / (sig_pr + R_obs)
        mu_post = mu_pr + K * (xt - mu_pr)
        sig_post = (1.0 - K) * sig_pr
        b_t = belief_vector(mu_post, sig_post, centres)

        # --- value & policy (stochastic readout of the critic) ---
        V_t = jnp.dot(w, b_t)
        z_lick = p["beta"] * (V_t - p["theta"])
        lam = p["lambda_max"] * jax.nn.sigmoid(z_lick) + 1e-9   # Poisson rate
        log_v_mean = p["v_base"] + p["v_slope"] * V_t

        k_lick, k_vel = jax.random.split(key)
        lick_gen = jax.random.poisson(k_lick, lam).astype(float)
        logv_gen = log_v_mean + sigma_v * jax.random.normal(k_vel)
        lick = lick_gen if generate else lick_o
        logv = logv_gen if generate else logv_o

        # --- per-bin emission log-likelihood (masked) ---
        ll_lick = (lick * jnp.log(lam) - lam - gammaln(lick + 1.0)) * valid
        ll_vel = (-0.5 * _LOG2PI - jnp.log(sigma_v)
                  - 0.5 * ((logv - log_v_mean) / sigma_v) ** 2) * valid
        ll_bin = ll_lick + ll_vel

        v = jnp.exp(logv)
        dt = BIN_SIZE_CM / jnp.clip(v, 1e-3, 1e4)

        # --- reward (water on the first RZ bin with >=1 lick) ---
        got = (lick > 0.5).astype(float) * rz * (1.0 - rewarded)
        r = got * reward_magnitude
        rewarded_new = jnp.clip(rewarded + got, 0.0, 1.0)

        # --- predict to the next bin ---
        mu_next = mu_post + BIN_SIZE_AU
        sig_next = sig_post + p["Q"] * BIN_SIZE_AU
        b_next = belief_vector(mu_next, sig_next, centres)
        V_next = jnp.where(last, 0.0, jnp.dot(w, b_next))

        # --- discounted TD update of the critic ---
        delta = r + p["gamma"] * V_next - V_t
        w_new = w + p["eta_w"] * delta * b_t

        carry_new = (mu_next, sig_next, w_new, rewarded_new)
        out = dict(
            value=V_t, rpe=delta, precision=1.0 / jnp.maximum(sig_post, 1e-3),
            sigma=sig_post, belief_mean=mu_post, lick_rate=lam,
            v_mean=jnp.exp(log_v_mean), reward=r, dt=dt,
            lick=lick, logv=logv, loglik=ll_bin,
            loglik_lick=ll_lick, loglik_vel=ll_vel,
        )
        return carry_new, out

    def trial_step(carry, xs):
        w, sigma_carry = carry
        lick_row, logv_row, mask_row, key_row = xs
        sig_start = sigma_carry + p["iti_inflation"]
        c0 = (0.0, sig_start, w, 0.0)
        xs_bins = (x_true, rz_mask, is_last, lick_row, logv_row, mask_row, key_row)
        (_, sig_end, w_end, _), outs = jax.lax.scan(bin_step, c0, xs_bins)
        return (w_end, sig_end), outs

    # Critic starts at a flat elevated value `w_init`: the residue of the prior
    # random-reward task, in which mice learned reward is available and licking
    # is worthwhile.  Task-2 learning then *carves* this down everywhere except
    # the RZ — the random-to-goal-directed transition.
    init = (jnp.full(n_bins, p["w_init"]), jnp.asarray(sigma0, float))
    _, session = jax.lax.scan(trial_step, init, (licks_obs, logv_obs, mask_obs, keys))
    return session


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def session_latents(u, licks, logv, mask=None, cfg: TaskConfig | None = None):
    """Run the agent teacher-forced on observed behaviour; return all latents.

    `mask` is an optional (n_trials, n_bins) array of 1.0 (valid) / 0.0 (missing
    behavioural data); defaults to all-valid.
    """
    cfg = cfg or TaskConfig()
    if mask is None:
        mask = jnp.ones_like(jnp.asarray(licks, dtype=float))
    keys = jnp.zeros((licks.shape[0], cfg.n_bins, 2), dtype=jnp.uint32)
    return _run_session(u, licks, logv, mask, keys, False,
                        cfg.sigma0, cfg.reward_magnitude)


def session_loglik(u, licks, logv, mask=None, cfg: TaskConfig | None = None):
    """Total teacher-forced (masked) log-likelihood of an observed session."""
    out = session_latents(u, licks, logv, mask, cfg)
    return jnp.sum(out["loglik"])


def simulate_session(u, key, n_trials, cfg: TaskConfig | None = None):
    """Generative roll-out: the agent samples its own licks and velocities."""
    cfg = cfg or TaskConfig()
    keys = jax.random.split(key, n_trials * cfg.n_bins)
    keys = keys.reshape(n_trials, cfg.n_bins, 2)
    dummy = jnp.zeros((n_trials, cfg.n_bins))
    mask = jnp.ones((n_trials, cfg.n_bins))
    return _run_session(u, dummy, dummy, mask, keys, True,
                        cfg.sigma0, cfg.reward_magnitude)
