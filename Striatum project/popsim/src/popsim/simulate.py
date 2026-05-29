"""Top-level simulation: config -> ground-truth-carrying result.

A :class:`SimConfig` fully specifies a multi-area simulation (per-area latent
dimensions and population sizes, intrinsic dynamics, observation model, and the
inter-area coupling edges). :func:`simulate` runs it and returns a
:class:`SimResult` carrying everything needed to reproduce the run and to score
an analysis against the known ground truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .coupling import CouplingEdge, resolve_latents
from .latents import generate_latents
from .observation import project_population, random_loadings

__all__ = ["AreaSpec", "SimConfig", "SimResult", "simulate"]


@dataclass
class AreaSpec:
    """Specification for one area's latent and population structure."""

    name: str
    n_latents: int = 4
    n_neurons: int = 50
    dynamics: str = "ar1"  # "ar1", "lds", or "oscillatory"
    observation: str = "gaussian"  # "gaussian" or "poisson"
    tau: float = 20.0
    snr: float = 2.0
    baseline: float = 0.0
    dynamics_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (3 <= self.n_latents <= 5):
            # The design targets 3-5 latents per area.
            raise ValueError(
                f"area {self.name}: n_latents should be 3-5 (got {self.n_latents})"
            )
        if self.dynamics not in ("ar1", "lds", "oscillatory"):
            raise ValueError(f"unknown dynamics: {self.dynamics!r}")
        if self.observation not in ("gaussian", "poisson"):
            raise ValueError(f"unknown observation: {self.observation!r}")


@dataclass
class SimConfig:
    """Full specification of a multi-area simulation."""

    areas: list[AreaSpec]
    edges: list[CouplingEdge] = field(default_factory=list)
    n_timesteps: int = 3000
    dt: float = 1.0
    epoch_boundaries: list[int] = field(default_factory=list)
    seed: int = 0
    name: str = "sim"

    def area(self, name: str) -> AreaSpec:
        for a in self.areas:
            if a.name == name:
                return a
        raise KeyError(name)


@dataclass
class SimResult:
    """Outputs and ground truth of a simulation.

    Attributes
    ----------
    latents:
        ``area -> array(n_timesteps, n_latents)`` resolved latents (post-coupling).
    intrinsic_latents:
        ``area -> array(n_timesteps, n_latents)`` latents *before* coupling.
    neural:
        ``area -> array(n_timesteps, n_neurons)`` observed population activity.
    loadings:
        ``area -> array(n_neurons, n_latents)`` projection matrices.
    config:
        The :class:`SimConfig` that produced this result.
    """

    latents: dict[str, np.ndarray]
    intrinsic_latents: dict[str, np.ndarray]
    neural: dict[str, np.ndarray]
    loadings: dict[str, np.ndarray]
    config: SimConfig

    @property
    def area_names(self) -> list[str]:
        return [a.name for a in self.config.areas]

    def metadata(self) -> dict[str, Any]:
        """JSON-serialisable description of config and ground-truth couplings."""
        edges = []
        for e in self.config.edges:
            edges.append(
                {
                    "source": e.source,
                    "target": e.target,
                    "gain": e.gain,
                    "lag": e.lag,
                    "matrix": None if e.matrix is None else e.matrix.tolist(),
                    "epochs": e.epochs,
                }
            )
        return {
            "name": self.config.name,
            "n_timesteps": self.config.n_timesteps,
            "dt": self.config.dt,
            "epoch_boundaries": list(self.config.epoch_boundaries),
            "seed": self.config.seed,
            "areas": [asdict(a) for a in self.config.areas],
            "edges": edges,
        }


def _intrinsic_for_area(
    spec: AreaSpec, n_timesteps: int, dt: float, rng: np.random.Generator
) -> np.ndarray:
    kwargs = dict(spec.dynamics_kwargs)
    if spec.dynamics == "ar1":
        kwargs.setdefault("tau", spec.tau)
    return generate_latents(
        spec.dynamics, n_timesteps, spec.n_latents, dt=dt, rng=rng, **kwargs
    )


def simulate(config: SimConfig) -> SimResult:
    """Run a simulation and return latents, populations, and ground truth.

    A single seeded :class:`numpy.random.Generator` is split into independent
    child streams per area, so results are deterministic in ``config.seed`` and
    adding an area does not perturb earlier areas' random streams.
    """
    parent = np.random.default_rng(config.seed)
    children = parent.spawn(len(config.areas))
    streams = {a.name: np.random.default_rng(c) for a, c in zip(config.areas, children)}

    intrinsic: dict[str, np.ndarray] = {}
    loadings: dict[str, np.ndarray] = {}
    for spec in config.areas:
        rng = streams[spec.name]
        intrinsic[spec.name] = _intrinsic_for_area(
            spec, config.n_timesteps, config.dt, rng
        )
        loadings[spec.name] = random_loadings(spec.n_neurons, spec.n_latents, rng)

    resolved = resolve_latents(intrinsic, config.edges)

    neural: dict[str, np.ndarray] = {}
    for spec in config.areas:
        rng = streams[spec.name]
        neural[spec.name] = project_population(
            resolved[spec.name],
            loadings[spec.name],
            model=spec.observation,
            snr=spec.snr,
            baseline=spec.baseline,
            rng=rng,
        )

    return SimResult(
        latents=resolved,
        intrinsic_latents=intrinsic,
        neural=neural,
        loadings=loadings,
        config=config,
    )
