"""popsim -- simulated multi-area neural population activity with ground truth.

A latent-driven generator for three (or more) neural areas. Inter-area
communication is specified at the *latent* level via directed, optionally lagged
and epoch-gated coupling edges, then projected to full populations (Gaussian
rates or Poisson spike counts).

Typical use::

    from popsim import scenarios, simulate
    result = simulate(scenarios.lagged())
    result.neural["A"]      # (n_timesteps, n_neurons)
    result.latents["B"]     # (n_timesteps, n_latents)
    result.metadata()       # JSON-serialisable ground truth
"""

from __future__ import annotations

from . import metrics, realism, scenarios
from .coupling import CouplingEdge, resolve_latents, topological_order
from .latents import (
    ar1_latents,
    generate_latents,
    lds_latents,
    oscillatory_latents,
)
from .observation import RealismParams, project_population, random_loadings
from .simulate import (
    AreaSpec,
    SharedNoise,
    SimConfig,
    SimResult,
    TrialResult,
    simulate,
    simulate_trials,
)

__all__ = [
    "AreaSpec",
    "SharedNoise",
    "SimConfig",
    "SimResult",
    "TrialResult",
    "simulate",
    "simulate_trials",
    "CouplingEdge",
    "resolve_latents",
    "topological_order",
    "ar1_latents",
    "lds_latents",
    "oscillatory_latents",
    "generate_latents",
    "project_population",
    "random_loadings",
    "RealismParams",
    "scenarios",
    "metrics",
    "realism",
]
