"""Project low-dimensional latents onto full neural populations.

A population's activity is a linear readout of its area's latents plus private
per-neuron noise. Two observation models are supported, selectable per area:

- ``"gaussian"``: continuous firing-rate-like signal
  ``x = z @ W.T + baseline + private_noise``.
- ``"poisson"``: spike counts drawn from ``Poisson(softplus(x))`` where ``x`` is
  the Gaussian rate above; the softplus keeps rates non-negative.

Loading matrices have unit-norm columns so each latent contributes comparable
power across the population, and the requested ``snr`` sets the ratio of signal
variance to private-noise variance in the Gaussian case.
"""

from __future__ import annotations

import numpy as np

__all__ = ["random_loadings", "project_population"]


def _as_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


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

    signal = latents @ loadings.T  # (n_timesteps, n_neurons)

    # Private noise scaled per neuron so var(signal) / var(noise) == snr.
    sig_var = signal.var(axis=0, keepdims=True)
    sig_var = np.where(sig_var == 0, 1.0, sig_var)
    noise_std = np.sqrt(sig_var / snr)
    rate = signal + baseline + noise_std * rng.standard_normal(signal.shape)

    if model == "gaussian":
        return rate
    if model == "poisson":
        # softplus keeps rates positive; logaddexp is the stable form.
        lam = np.logaddexp(0.0, rate)
        return rng.poisson(lam).astype(np.int64)
    raise ValueError(f"unknown observation model: {model!r}")
