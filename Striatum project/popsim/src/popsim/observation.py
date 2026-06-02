"""Project low-dimensional latents onto full neural populations.

A population's activity is a linear readout of its area's latents plus private
per-neuron noise. Two observation models are supported, selectable per area:

- ``"gaussian"``: continuous firing-rate-like signal
  ``x = z @ W.T + baseline + private_noise``.
- ``"poisson"``: spike counts drawn from ``Poisson(softplus(x))`` where ``x`` is
  the Gaussian rate above; the softplus keeps rates non-negative.

Optional biological-realism knobs (all default off, so the base model is the
clean linear-Gaussian one) are applied via :mod:`popsim.realism`: per-neuron
gain heterogeneity, a shared global fluctuation, slow nonstationary drift, and
(Poisson only) sub-Poisson refractory regularity.

Loading matrices have unit-norm columns so each latent contributes comparable
power across the population, and the requested ``snr`` sets the ratio of signal
variance to private-noise variance in the Gaussian case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import realism

__all__ = ["random_loadings", "project_population", "RealismParams"]


def _as_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


@dataclass
class RealismParams:
    """Optional realism knobs for :func:`project_population` (all default off).

    Attributes
    ----------
    neuron_gain_cv:
        Coefficient of variation of per-neuron multiplicative gains (log-normal).
    global_noise:
        Strength of a shared population-wide fluctuation added to every neuron.
    drift_std:
        End-of-session standard deviation of a per-neuron slow gain drift.
    refractory:
        Spike-count regularity in ``[0, 1)`` for the Poisson model; the counts
        become sub-Poisson with Fano factor ``~ 1 - refractory`` (0 = Poisson).
    """

    neuron_gain_cv: float = 0.0
    global_noise: float = 0.0
    drift_std: float = 0.0
    refractory: float = 0.0

    @property
    def any_enabled(self) -> bool:
        return bool(
            self.neuron_gain_cv or self.global_noise or self.drift_std
            or self.refractory
        )


def random_loadings(
    n_neurons: int,
    n_latents: int,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Random loading matrix with unit-norm columns.

    Returns an array of shape ``(n_neurons, n_latents)``; column ``k`` maps
    latent ``k`` onto the population and has Euclidean norm 1.
    """
    if n_neurons <= 0 or n_latents <= 0:
        raise ValueError("n_neurons and n_latents must be positive")
    rng = _as_rng(rng)
    W = rng.standard_normal((n_neurons, n_latents))
    norms = np.linalg.norm(W, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    return W / norms


def project_population(
    latents: np.ndarray,
    loadings: np.ndarray,
    model: str = "gaussian",
    snr: float = 2.0,
    baseline: float = 0.0,
    realism_params: RealismParams | None = None,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Project latents to a population with the chosen observation model.

    Parameters
    ----------
    latents:
        Array of shape ``(n_timesteps, n_latents)``.
    loadings:
        Array of shape ``(n_neurons, n_latents)``.
    model:
        ``"gaussian"`` or ``"poisson"``.
    snr:
        Per-neuron signal-to-noise ratio: ratio of projected-signal variance to
        private-noise variance (Gaussian). Higher -> cleaner.
    baseline:
        Additive offset. For Poisson this lifts the log-rate so counts are
        non-trivial.
    realism_params:
        Optional :class:`RealismParams`; ``None`` -> clean linear-Gaussian model.
    rng:
        Seed / Generator / None.

    Returns
    -------
    np.ndarray
        Shape ``(n_timesteps, n_neurons)``. Floats for Gaussian, integer counts
        for Poisson.
    """
    if latents.ndim != 2 or loadings.ndim != 2:
        raise ValueError("latents and loadings must be 2-D")
    if latents.shape[1] != loadings.shape[1]:
        raise ValueError("latents and loadings must share n_latents")
    if snr <= 0:
        raise ValueError("snr must be positive")
    rng = _as_rng(rng)
    rp = realism_params or RealismParams()

    signal = latents @ loadings.T  # (n_timesteps, n_neurons)
    n_t, n_neurons = signal.shape

    # Per-neuron gain heterogeneity and slow drift multiply the signal; the
    # shared fluctuation is an additive common drive. All are no-ops when off.
    if rp.neuron_gain_cv:
        signal = signal * realism.lognormal_gains(n_neurons, rp.neuron_gain_cv, rng)
    if rp.drift_std:
        signal = signal * realism.slow_drift(n_t, n_neurons, rp.drift_std, rng)
    if rp.global_noise:
        signal = signal + realism.global_fluctuation(
            n_t, n_neurons, rp.global_noise, rng=rng
        )

    # Private noise scaled per neuron so var(signal) / var(noise) == snr.
    sig_var = signal.var(axis=0, keepdims=True)
    sig_var = np.where(sig_var == 0, 1.0, sig_var)
    noise_std = np.sqrt(sig_var / snr)
    rate = signal + baseline + noise_std * rng.standard_normal(signal.shape)

    if model == "gaussian":
        return rate
    if model == "poisson":
        lam = np.logaddexp(0.0, rate)  # softplus, stable form
        if rp.refractory:
            # Sub-Poisson (regular/refractory) counts with Fano ~ 1 - refractory.
            return realism.subpoisson_counts(lam, rp.refractory, rng)
        return rng.poisson(lam).astype(np.int64)
    raise ValueError(f"unknown observation model: {model!r}")
