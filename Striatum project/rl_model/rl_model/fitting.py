"""Per-mouse maximum-likelihood fitting of the belief-state RL agent.

The negative teacher-forced log-likelihood is minimised in the unconstrained
parameter space with L-BFGS-B, using exact JAX gradients and several random
restarts to guard against local optima.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

from .agent import default_unconstrained, session_loglik, to_constrained
from .config import N_PARAMS, PARAM_NAMES, TaskConfig

_BOUND = 12.0  # generous box in unconstrained space, keeps the optimiser sane


def _objective(licks, logv, mask, cfg, prior_sd):
    """Return a jitted (value, grad) function of the unconstrained vector.

    With `prior_sd` set, this is a MAP objective: the negative log-likelihood
    plus a broad isotropic Gaussian prior centred on the default parameters.
    The prior is weak — it leaves well-identified parameters untouched and
    merely keeps flat (unidentified) directions from drifting to the bounds.
    """
    licks_j = jnp.asarray(licks, dtype=float)
    logv_j = jnp.asarray(logv, dtype=float)
    mask_j = jnp.asarray(mask, dtype=float)
    u0 = default_unconstrained()

    def nll(u):
        ll = -session_loglik(u, licks_j, logv_j, mask_j, cfg)
        if prior_sd is not None:
            ll = ll + 0.5 * jnp.sum(((u - u0) / prior_sd) ** 2)
        return ll

    return jax.jit(jax.value_and_grad(nll))


def fit_mouse(licks, logv, mask=None, cfg: TaskConfig | None = None,
              n_restarts: int = 6, seed: int = 0, prior_sd: float | None = 4.0):
    """Fit one mouse's session.  `licks`/`logv`/`mask` are (n_trials, n_bins).

    `mask` (1 = valid bin, 0 = missing behavioural data) defaults to all-valid.
    Returns a dict with the fitted unconstrained vector, the natural-space
    parameter dict, the negative log-likelihood, and per-restart diagnostics.
    """
    cfg = cfg or TaskConfig()
    if mask is None:
        mask = np.ones_like(np.asarray(licks, dtype=float))
    vg = _objective(licks, logv, mask, cfg, prior_sd)

    def scipy_obj(u_np):
        val, grad = vg(jnp.asarray(u_np))
        return float(val), np.asarray(grad, dtype=np.float64)

    base = np.asarray(default_unconstrained(), dtype=np.float64)
    rng = np.random.default_rng(seed)
    bounds = [(-_BOUND, _BOUND)] * N_PARAMS

    best = None
    nlls = []
    for r in range(n_restarts):
        u0 = base if r == 0 else base + 0.5 * rng.standard_normal(N_PARAMS)
        u0 = np.clip(u0, -_BOUND, _BOUND)
        res = minimize(scipy_obj, u0, jac=True, method="L-BFGS-B",
                       bounds=bounds, options=dict(maxiter=400, ftol=1e-10))
        nlls.append(float(res.fun))
        if best is None or res.fun < best.fun:
            best = res

    u_fit = np.asarray(best.x, dtype=np.float64)
    params = {k: float(v) for k, v in to_constrained(jnp.asarray(u_fit)).items()}
    return dict(
        u_fit=u_fit,
        params=params,
        nll=float(best.fun),
        success=bool(best.success),
        restart_nlls=nlls,
    )


def recovery_table(u_true, u_fit):
    """Per-parameter recovery: correlation of true vs fitted across the cohort.

    `u_true`, `u_fit` are (n_mice, N_PARAMS) in the unconstrained space.
    Returns a dict: param name -> (pearson_r, rmse).
    """
    u_true = np.asarray(u_true)
    u_fit = np.asarray(u_fit)
    table = {}
    for i, name in enumerate(PARAM_NAMES):
        t, f = u_true[:, i], u_fit[:, i]
        if np.std(t) < 1e-9 or np.std(f) < 1e-9:
            r = np.nan
        else:
            r = float(np.corrcoef(t, f)[0, 1])
        rmse = float(np.sqrt(np.mean((t - f) ** 2)))
        table[name] = (r, rmse)
    return table
