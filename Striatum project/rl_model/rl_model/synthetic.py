"""Synthetic-cohort generator: ground-truth data for parameter recovery.

A synthetic "mouse" has a known unconstrained parameter vector (jittered around
the `typical` values).  We roll the agent out generatively to produce licks and
log-velocities, which the fitter must then map back to the true parameters.
No real animal data is touched here.
"""
from __future__ import annotations

import numpy as np
import jax

from .agent import default_unconstrained, simulate_session
from .config import N_PARAMS, TaskConfig


def sample_params(seed: int, jitter: float = 0.35):
    """Draw one synthetic mouse's unconstrained parameter vector.

    Jitter is applied in the *unconstrained* space, so it is multiplicative for
    positive parameters and additive for real-valued ones automatically.
    """
    rng = np.random.default_rng(seed)
    base = np.asarray(default_unconstrained())
    u = base + jitter * rng.standard_normal(N_PARAMS)
    return u.astype(np.float64)


def make_cohort(n_mice: int = 16, n_trials: int = 160, seed: int = 0,
                jitter: float = 0.35, cfg: TaskConfig | None = None):
    """Generate a cohort of synthetic mice with known parameters.

    Returns a dict of stacked arrays:
        u_true : (n_mice, N_PARAMS)   ground-truth unconstrained parameters
        licks  : (n_mice, n_trials, n_bins)
        logv   : (n_mice, n_trials, n_bins)
    """
    cfg = cfg or TaskConfig()
    u_true, licks, logv = [], [], []
    for m in range(n_mice):
        u = sample_params(seed * 1000 + m, jitter)
        key = jax.random.PRNGKey(seed * 7919 + m)
        out = simulate_session(u, key, n_trials, cfg)
        u_true.append(u)
        licks.append(np.asarray(out["lick"]))
        logv.append(np.asarray(out["logv"]))
    return dict(
        u_true=np.stack(u_true),
        licks=np.stack(licks),
        logv=np.stack(logv),
        n_trials=n_trials,
        n_bins=cfg.n_bins,
    )


def save_cohort(cohort: dict, path: str):
    np.savez_compressed(path, **cohort)


def load_cohort(path: str) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}
