"""Belief-state RL agent: a two-timescale actor-critic.

The agent traverses a spatial corridor and runs two coupled processes:

* **Fast appraisal.** A Kalman belief over position (the *perceptual* process)
  and a reward-only TD critic ``V(belief)`` (the *value* process). Both settle
  within a few trials: the mouse learns *where it is* and *where reward is*
  quickly.
* **Slow control.** Two *actors* — one for licking, one for running speed —
  that learn over tens of trials to stop emitting actions where they do not pay
  off: to suppress licks outside the reward zone, and to run fast through the
  corridor.

Discounted TD localises the critic into a bump at the reward zone.  **Licking is
a time-limited Poisson process**: the policy sets a lick *rate* (licks/s) and the
count emitted in a bin is ``Poisson(rate * dt)``, where ``dt`` is the time spent
in the bin.  Running fast shrinks ``dt`` and so the lick count — to lick enough
to collect RZ reward the agent must slow down.  Velocity is **log-normal**.  A
per-bin **validity mask** lets the likelihood — and the learning updates —
ignore bins with missing behavioural data.

Costs and the actor RPE
-----------------------
A per-lick cost ``c_lick`` and a time cost ``rho * dt`` are charged to the
*actor* learning signal only; the critic is reward-only. So the two TD errors
differ::

    delta_critic = r                       + gamma*V' - V      (drives V)
    delta_actor  = (r - c_lick*lick - rho*dt) + gamma*V' - V   (drives actors)

Because the critic never models the costs, ``delta_actor`` carries a persistent
negative error wherever the mouse acts unprofitably; it is not predicted away,
so it fades only as the actors learn. ``delta_actor`` is exported as the ``rpe``
latent — the within-session emergence signal.

Actor updates (teacher-forced, redesign 2026-05-24)
---------------------------------------------------
* **Lick actor** — a reward-modulated (three-factor) rule:
  ``w_lick += eta_a * delta_actor * lick * b``. Where licks are unprofitable
  (``delta_actor < 0``) the lick drive is depressed; the rule self-limits once
  licking stops. Starts neutral (``w_lick = 0``); early breadth comes from the
  elevated flat critic ``w_init``.
* **Velocity actor** — a *deterministic* gradient rule:
  ``w_vel += eta_a * g_vel * b`` where ``g_vel = d(delta_actor)/d(logv)`` is the
  analytic advantage gradient.  Because the lick count is ``Poisson(rate * dt)``
  and the graded reward saturates with RZ licks, ``g_vel`` is negative in the RZ
  (faster → fewer licks → less water) and positive in the far corridor (faster →
  less ``rho * dt`` time cost).  A deterministic gradient is used because the
  stochastic policy gradient is too weak — velocity exploration noise is small,
  so the noise/outcome covariance it would rely on is negligible.

All computation is in JAX so the per-mouse log-likelihood is differentiable
w.r.t. the subjective parameters. The same core step drives both generative
simulation and likelihood evaluation.

Conventions
-----------
* Positions are in arbitrary units (a.u.); velocities in cm/s.
* A "session" is ``(n_trials, n_bins)``. Latents come out shaped the same way,
  aligned 1:1 with the neural ``spatial_binned_fr`` tensor.
* Fitting is *teacher-forced*: learning is driven by the mouse's observed
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
    C_LICK,
    K_REWARD,
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
#
# Trial-level carry : (w_crit, w_lick, w_vel, sigma_carry)
# Bin-level carry   : (mu, sigma, w_crit, w_lick, w_vel, rewarded)
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
        mu_pr, sig_pr, w_crit, w_lick, w_vel, cum_rz = carry
        xt, rz, last, lick_o, logv_o, valid, key = xs

        # --- perceptual update at the current bin (veridical-mean obs) ---
        R_obs = p["R_slope"] * jnp.abs(xt - VISUAL_LANDMARK_AU) + p["R0"]
        K = sig_pr / (sig_pr + R_obs)
        mu_post = mu_pr + K * (xt - mu_pr)
        sig_post = (1.0 - K) * sig_pr
        b_t = belief_vector(mu_post, sig_post, centres)

        # --- fast critic value and the two slow actor fields ---
        V_t = jnp.dot(w_crit, b_t)
        a_lick = jnp.dot(w_lick, b_t)        # slow lick-control field
        a_vel = jnp.dot(w_vel, b_t)          # slow velocity-control field

        # --- policies: critic drive + actor field ---
        z_lick = p["beta"] * V_t + a_lick - p["theta"]
        sig_lick = jax.nn.sigmoid(z_lick)
        lam_rate = p["lambda_max"] * sig_lick + 1e-9            # lick RATE (licks/s)
        log_v_mean = p["v_base"] + p["v_slope"] * V_t + a_vel

        # --- velocity is resolved first: it sets dt, and dt sets how many licks
        #     fit in the bin.  lick count ~ Poisson(rate * dt) — running fast
        #     leaves no time to lick, so RZ reward needs the agent to slow. ---
        k_lick, k_vel = jax.random.split(key)
        logv_gen = log_v_mean + sigma_v * jax.random.normal(k_vel)
        logv = logv_gen if generate else logv_o
        v = jnp.exp(logv)
        dt = BIN_SIZE_CM / jnp.clip(v, 1e-3, 1e4)
        mu_lick = lam_rate * dt                                 # expected lick COUNT

        lick_gen = jax.random.poisson(k_lick, mu_lick).astype(float)
        lick = lick_gen if generate else lick_o

        # --- per-bin emission log-likelihood (masked) ---
        ll_lick = (lick * jnp.log(mu_lick) - mu_lick - gammaln(lick + 1.0)) * valid
        ll_vel = (-0.5 * _LOG2PI - jnp.log(sigma_v)
                  - 0.5 * ((logv - log_v_mean) / sigma_v) ** 2) * valid
        ll_bin = ll_lick + ll_vel

        # --- graded, saturating reward: total water collected over a trial is
        #     reward_magnitude*(1 - exp(-cumulative_RZ_licks / K_REWARD)); the
        #     per-bin reward is that total's increment.  Running fast through the
        #     RZ emits fewer licks and so collects less of the drop. ---
        cum_rz_new = cum_rz + lick * rz
        r = reward_magnitude * (jnp.exp(-cum_rz / K_REWARD)
                                - jnp.exp(-cum_rz_new / K_REWARD))

        # --- predict to the next bin; path-integration noise grows with speed
        #     (the speed/accuracy trade-off that makes velocity consequential) ---
        mu_next = mu_post + BIN_SIZE_AU
        Q_eff = p["Q"] * (1.0 + p["kappa_v"] * v)
        sig_next = sig_post + Q_eff * BIN_SIZE_AU
        b_next = belief_vector(mu_next, sig_next, centres)
        V_next = jnp.where(last, 0.0, jnp.dot(w_crit, b_next))

        # --- two TD errors: reward-only for the critic, reward-minus-cost for
        #     the actors.  The critic never models the costs, so delta_actor
        #     stays a persistent negative RPE until the actors suppress it. ---
        delta_critic = r + p["gamma"] * V_next - V_t
        cost = C_LICK * lick + p["rho"] * dt
        delta_actor = (r - cost) + p["gamma"] * V_next - V_t

        # --- fast critic update (reward-only TD) ---
        w_crit_new = w_crit + p["eta_w"] * delta_critic * b_t * valid

        # --- slow lick actor: reward-modulated three-factor rule.  Where licks
        #     carry a negative actor RPE the lick drive is depressed; the rule
        #     self-limits once licking stops (lick -> 0). ---
        w_lick_new = w_lick + p["eta_a"] * delta_actor * lick * b_t * valid

        # --- slow velocity actor: deterministic gradient ascent on the actor
        #     advantage.  g_vel = d(delta_actor)/d(logv), analytic: faster means
        #     less time cost and fewer costly licks (g_vel > 0 in the corridor)
        #     but fewer RZ licks and so less graded reward (g_vel < 0 in the RZ).
        #     Deterministic because the stochastic policy gradient is too weak. ---
        g_vel = (p["rho"] * dt + C_LICK * mu_lick
                 - rz * (reward_magnitude / K_REWARD) * mu_lick
                   * jnp.exp(-cum_rz_new / K_REWARD))
        w_vel_new = w_vel + p["eta_a"] * g_vel * b_t * valid

        carry_new = (mu_next, sig_next, w_crit_new, w_lick_new, w_vel_new,
                     cum_rz_new)
        out = dict(
            value=V_t, rpe=delta_actor, rpe_critic=delta_critic,
            precision=1.0 / jnp.maximum(sig_post, 1e-3),
            sigma=sig_post, belief_mean=mu_post,
            lick_rate=mu_lick, v_mean=jnp.exp(log_v_mean),
            a_lick=a_lick, a_vel=a_vel,
            reward=r, dt=dt, lick=lick, logv=logv, loglik=ll_bin,
            loglik_lick=ll_lick, loglik_vel=ll_vel,
        )
        return carry_new, out

    def trial_step(carry, xs):
        w_crit, w_lick, w_vel, sigma_carry = carry
        lick_row, logv_row, mask_row, key_row = xs
        sig_start = sigma_carry + p["iti_inflation"]
        c0 = (0.0, sig_start, w_crit, w_lick, w_vel, 0.0)
        xs_bins = (x_true, rz_mask, is_last, lick_row, logv_row, mask_row, key_row)
        (_, sig_end, w_crit_e, w_lick_e, w_vel_e, _), outs = jax.lax.scan(
            bin_step, c0, xs_bins)
        return (w_crit_e, w_lick_e, w_vel_e, sig_end), outs

    # Critic starts at a flat elevated value `w_init` — the residue of the prior
    # random-reward task — which gives broad, undirected early licking.  The
    # actors start neutral (zero) and slowly carve control on top of it.
    init = (jnp.full(n_bins, p["w_init"]),
            jnp.zeros(n_bins), jnp.zeros(n_bins),
            jnp.asarray(sigma0, float))
    _, session = jax.lax.scan(trial_step, init,
                              (licks_obs, logv_obs, mask_obs, keys))
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
