"""Biological-realism primitives for the population observation model.

These are small, pure, independently testable functions that make the synthetic
populations harder for linear methods (and more like real recordings) without
changing where the *communication* lives -- coupling stays at the latent level
(see :mod:`popsim.coupling`); these only shape the readout and noise:

- :func:`lognormal_gains`        -- per-neuron multiplicative gain heterogeneity.
- :func:`global_fluctuation`     -- a shared population-wide gain/offset signal.
- :func:`slow_drift`             -- nonstationary per-neuron gain over the session.
- :func:`subpoisson_counts`      -- regular (refractory) spike counts with a
                                    target Fano factor < 1.

The first three operate on / produce ``(n_timesteps, n_neurons)`` arrays (apply
per trial for trial-structured data); ``subpoisson_counts`` takes a rate array of
any shape. All compose with both the continuous and trialised simulation paths.
"""

from __future__ import annotations

import numpy as np

from .latents import ar1_latents

__all__ = [
    "lognormal_gains",
    "global_fluctuation",
    "slow_drift",
    "subpoisson_counts",
]


def _as_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def lognormal_gains(
    n_neurons: int, cv: float, rng: np.random.Generator | int | None = None
) -> np.ndarray:
    """Per-neuron multiplicative gains with mean 1 and coefficient of variation ``cv``.

    Real cortical firing rates are approximately log-normally distributed across
    neurons. Returns an array of shape ``(n_neurons,)``; ``cv == 0`` gives all
    ones (no heterogeneity).
    """
    if n_neurons <= 0:
        raise ValueError("n_neurons must be positive")
    if cv < 0:
        raise ValueError("cv must be non-negative")
    if cv == 0:
        return np.ones(n_neurons)
    rng = _as_rng(rng)
    # Parameterise the log-normal so E[gain] == 1 and CV == cv exactly.
    sigma = np.sqrt(np.log1p(cv**2))
    mu = -0.5 * sigma**2
    return rng.lognormal(mean=mu, sigma=sigma, size=n_neurons)


def global_fluctuation(
    n_timesteps: int,
    n_neurons: int,
    strength: float,
    tau: float = 30.0,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Shared population-wide fluctuation added to every neuron.

    A single smooth AR(1) signal (timescale ``tau``) is broadcast across the
    population with per-neuron weights, modelling slow shared excitability not
    captured by the area's latents. Returns ``(n_timesteps, n_neurons)``.
    ``strength == 0`` returns zeros.
    """
    if strength == 0:
        return np.zeros((n_timesteps, n_neurons))
    if strength < 0:
        raise ValueError("strength must be non-negative")
    rng = _as_rng(rng)
    shared = ar1_latents(n_timesteps, 1, tau=tau, rng=rng)[:, 0]  # unit variance
    # Non-negative per-neuron weights (shared signal is a common drive).
    weights = np.abs(rng.standard_normal(n_neurons))
    return strength * shared[:, None] * weights[None, :]


def slow_drift(
    n_timesteps: int,
    n_neurons: int,
    drift_std: float,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Nonstationary per-neuron multiplicative gain drifting over the session.

    Each neuron's gain follows a smooth random walk around 1 with standard
    deviation ``drift_std`` at the end of the session. Returns a strictly
    positive ``(n_timesteps, n_neurons)`` gain field; ``drift_std == 0`` returns
    all ones.
    """
    if drift_std < 0:
        raise ValueError("drift_std must be non-negative")
    if drift_std == 0:
        return np.ones((n_timesteps, n_neurons))
    rng = _as_rng(rng)
    # Cumulative random walk normalised so the endpoint std equals drift_std.
    steps = rng.standard_normal((n_timesteps, n_neurons))
    walk = np.cumsum(steps, axis=0) / np.sqrt(n_timesteps)
    gain = 1.0 + drift_std * walk
    return np.clip(gain, 0.05, None)  # keep gains positive


def subpoisson_counts(
    rate: np.ndarray,
    regularity: float,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Draw sub-Poisson (Fano < 1) spike counts with a target Fano factor.

    Refractoriness makes spike trains more *regular* than Poisson: the count
    variance is below the mean (Fano factor < 1). A Poisson draw cannot capture
    this -- and no independent thinning of a Poisson can, since thinning a
    Poisson leaves it Poisson (Fano == 1). Instead we draw counts from a
    moment-matched binomial, whose Fano factor is ``1 - p``::

        n = ceil(lambda / p),  p_eff = lambda / n,  count ~ Binomial(n, p_eff)

    with ``p = regularity``. This preserves the mean exactly (``E[count] ==
    lambda``) while giving Fano ``= 1 - p_eff ~= 1 - regularity``.
    ``regularity == 0`` falls back to a plain Poisson draw (Fano == 1).

    Parameters
    ----------
    rate:
        Non-negative expected counts (``lambda``), any shape.
    regularity:
        Target Fano reduction in ``[0, 1)``; larger -> more regular (lower Fano).
    rng:
        Seed / Generator / None.
    """
    if not (0.0 <= regularity < 1.0):
        raise ValueError("regularity must be in [0, 1)")
    rng = _as_rng(rng)
    rate = np.asarray(rate, dtype=float)
    if np.any(rate < 0):
        raise ValueError("rate must be non-negative")
    if regularity == 0:
        return rng.poisson(rate).astype(np.int64)
    p = regularity
    n = np.ceil(rate / p).astype(np.int64)
    p_eff = np.where(n > 0, rate / np.maximum(n, 1), 0.0)
    return rng.binomial(n, p_eff).astype(np.int64)
