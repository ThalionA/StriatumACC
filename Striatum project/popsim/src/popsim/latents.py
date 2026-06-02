"""Intrinsic latent dynamics for simulated neural areas.

Each area is driven by a small number (3-5) of latent variables whose
trajectories over time are temporally structured, mimicking the autocorrelated
low-dimensional structure typically recovered from neural populations.

Three interchangeable generators are provided, selectable by name via
:func:`generate_latents`:

- ``"ar1"``   -- order-1 autoregressive (Ornstein-Uhlenbeck-like) smooth noise.
- ``"lds"``   -- a rotational linear dynamical system (oscillatory + decay).
- ``"oscillatory"`` -- damped sinusoids at chosen frequencies plus noise.

All functions return arrays of shape ``(n_timesteps, n_latents)`` standardised
to approximately unit marginal variance per latent, so that downstream coupling
gains and population loadings control the signal scale.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ar1_latents",
    "lds_latents",
    "oscillatory_latents",
    "generate_latents",
]


def _as_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    """Coerce a seed / Generator / None into a numpy Generator."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _standardise(x: np.ndarray) -> np.ndarray:
    """Centre and scale each column to unit variance (safe for zero columns)."""
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return x / sd


def ar1_latents(
    n_timesteps: int,
    n_latents: int,
    tau: float = 20.0,
    dt: float = 1.0,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Generate AR(1) (OU-like) latent trajectories.

    The discrete update is ``z[t] = a * z[t-1] + sqrt(1 - a**2) * eps`` with
    ``a = exp(-dt / tau)``. The innovation scaling keeps the stationary marginal
    variance at 1.0 for every latent regardless of ``tau``.

    Parameters
    ----------
    n_timesteps, n_latents:
        Output shape.
    tau:
        Autocorrelation timescale in units of ``dt``; larger -> smoother.
    dt:
        Time-bin width; only matters relative to ``tau``.
    rng:
        Seed, ``numpy.random.Generator``, or ``None``.
    """
    if n_timesteps <= 0 or n_latents <= 0:
        raise ValueError("n_timesteps and n_latents must be positive")
    if tau <= 0:
        raise ValueError("tau must be positive")

    rng = _as_rng(rng)
    a = np.exp(-dt / tau)
    innovation_scale = np.sqrt(1.0 - a**2)

    z = np.empty((n_timesteps, n_latents))
    # Start from the stationary distribution (unit variance): no burn-in.
    z[0] = rng.standard_normal(n_latents)
    noise = rng.standard_normal((n_timesteps - 1, n_latents))
    for t in range(1, n_timesteps):
        z[t] = a * z[t - 1] + innovation_scale * noise[t - 1]
    return z


def lds_latents(
    n_timesteps: int,
    n_latents: int,
    freqs: np.ndarray | list[float] | None = None,
    decay: float = 0.02,
    dt: float = 1.0,
    noise_std: float = 0.1,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Generate latents from a rotational linear dynamical system.

    Latents are organised into 2-D rotation blocks ``z[t] = rho * R(theta) z[t-1]
    + noise`` where ``R(theta)`` is a planar rotation and ``rho = exp(-decay*dt)``
    sets the amplitude decay. A leftover odd dimension falls back to a scalar
    AR(1). This produces structured, oscillating trajectories with a clear
    autocorrelation, useful as a richer ground truth than smooth noise.

    Parameters
    ----------
    freqs:
        Rotation frequency (cycles per unit time) for each 2-D block. If
        ``None``, a geometric spread between 0.005 and 0.05 cycles/bin is used.
    decay:
        Per-step amplitude decay rate; larger -> faster-damped, noisier.
    noise_std:
        Standard deviation of the driving noise.
    """
    if n_timesteps <= 0 or n_latents <= 0:
        raise ValueError("n_timesteps and n_latents must be positive")
    rng = _as_rng(rng)

    n_blocks = n_latents // 2
    has_scalar = bool(n_latents % 2)

    if freqs is None:
        n_freqs = max(n_blocks + (1 if has_scalar else 0), 1)
        freqs = np.geomspace(0.005, 0.05, n_freqs)
    freqs = np.asarray(freqs, dtype=float)

    rho = np.exp(-decay * dt)
    z = np.zeros((n_timesteps, n_latents))
    z[0] = rng.standard_normal(n_latents)
    noise = noise_std * rng.standard_normal((n_timesteps, n_latents))

    # Pre-build the per-block rotation matrices.
    rotations = []
    for b in range(n_blocks):
        theta = 2 * np.pi * freqs[b] * dt
        c, s = np.cos(theta), np.sin(theta)
        rotations.append(rho * np.array([[c, -s], [s, c]]))
    a_scalar = rho

    for t in range(1, n_timesteps):
        for b in range(n_blocks):
            i = 2 * b
            z[t, i : i + 2] = rotations[b] @ z[t - 1, i : i + 2] + noise[t, i : i + 2]
        if has_scalar:
            z[t, -1] = a_scalar * z[t - 1, -1] + noise[t, -1]

    return _standardise(z)


def oscillatory_latents(
    n_timesteps: int,
    n_latents: int,
    freqs: np.ndarray | list[float] | None = None,
    dt: float = 1.0,
    noise_std: float = 0.1,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Generate sinusoidal latents at given frequencies plus additive noise.

    Each latent is ``sin(2*pi*f*t + phase) + noise``, then standardised to unit
    variance. Random phases decorrelate the latents at lag zero.
    """
    if n_timesteps <= 0 or n_latents <= 0:
        raise ValueError("n_timesteps and n_latents must be positive")
    rng = _as_rng(rng)

    if freqs is None:
        freqs = np.geomspace(0.01, 0.1, n_latents)
    freqs = np.asarray(freqs, dtype=float)
    if freqs.shape != (n_latents,):
        raise ValueError("freqs must have length n_latents")

    t = np.arange(n_timesteps) * dt
    phases = rng.uniform(0, 2 * np.pi, n_latents)
    signal = np.sin(2 * np.pi * freqs[None, :] * t[:, None] + phases[None, :])
    signal = signal + noise_std * rng.standard_normal((n_timesteps, n_latents))
    return _standardise(signal)


# Registry so callers (and SimConfig) can select dynamics by name.
_GENERATORS = {
    "ar1": ar1_latents,
    "lds": lds_latents,
    "oscillatory": oscillatory_latents,
}


def generate_latents(
    kind: str,
    n_timesteps: int,
    n_latents: int,
    dt: float = 1.0,
    rng: np.random.Generator | int | None = None,
    **kwargs,
) -> np.ndarray:
    """Dispatch to a named latent-dynamics generator.

    ``kind`` is one of ``"ar1"``, ``"lds"``, ``"oscillatory"``. Extra keyword
    arguments are forwarded to the chosen generator (e.g. ``tau`` for ``"ar1"``).
    """
    if kind not in _GENERATORS:
        raise ValueError(
            f"unknown dynamics {kind!r}; choose from {sorted(_GENERATORS)}"
        )
    return _GENERATORS[kind](n_timesteps, n_latents, dt=dt, rng=rng, **kwargs)
