"""Top-level simulation: config -> ground-truth-carrying result.

A :class:`SimConfig` fully specifies a multi-area simulation (per-area latent
dimensions and population sizes, intrinsic dynamics, observation model, realism
knobs, inter-area coupling edges, and observation-level shared-noise groups).
Two entry points run it:

- :func:`simulate`        -- one continuous session, ``area -> (T, n_neurons)``.
- :func:`simulate_trials` -- trial-structured, ``area -> (n_trials, n_bins,
  n_neurons)`` with *independent* trials (each an independent latent draw), as
  required by trial-permutation nulls and the ``striatum_cca`` pipeline.

Both return a result object carrying everything needed to reproduce the run and
to score an analysis against the known ground truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .coupling import CouplingEdge, resolve_latents
from .latents import ar1_latents, generate_latents
from .observation import RealismParams, project_population, random_loadings

__all__ = [
    "AreaSpec",
    "SharedNoise",
    "SimConfig",
    "SimResult",
    "TrialResult",
    "simulate",
    "simulate_trials",
]


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
    realism: RealismParams = field(default_factory=RealismParams)

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
        if isinstance(self.realism, dict):
            self.realism = RealismParams(**self.realism)


@dataclass
class SharedNoise:
    """A shared additive fluctuation injected into several areas' populations.

    This is correlation at the *observation* level only -- there is no
    latent-level communication -- so it models a confound (e.g. shared arousal
    or motion) that a naive cross-area correlation would mistake for
    communication. Each named area receives a scaled copy of one smooth AR(1)
    time course (with non-negative per-neuron weights). All listed areas must
    use the ``"gaussian"`` observation model.

    Attributes
    ----------
    areas:
        Area names that share the fluctuation.
    strength:
        Multiplicative scale of the shared signal (relative to unit variance).
    tau:
        AR(1) timescale (bins) of the shared signal.
    """

    areas: list[str]
    strength: float = 0.3
    tau: float = 15.0


@dataclass
class SimConfig:
    """Full specification of a multi-area simulation."""

    areas: list[AreaSpec]
    edges: list[CouplingEdge] = field(default_factory=list)
    n_timesteps: int = 3000
    dt: float = 1.0
    epoch_boundaries: list[int] = field(default_factory=list)
    shared_noise: list[SharedNoise] = field(default_factory=list)
    seed: int = 0
    name: str = "sim"

    def area(self, name: str) -> AreaSpec:
        for a in self.areas:
            if a.name == name:
                return a
        raise KeyError(name)

    @property
    def max_lag(self) -> int:
        return max((e.lag for e in self.edges), default=0)


def _base_metadata(config: SimConfig, extra: dict[str, Any]) -> dict[str, Any]:
    edges = [
        {
            "source": e.source,
            "target": e.target,
            "gain": e.gain,
            "lag": e.lag,
            "matrix": None if e.matrix is None else e.matrix.tolist(),
            "epochs": e.epochs,
        }
        for e in config.edges
    ]
    meta = {
        "name": config.name,
        "dt": config.dt,
        "epoch_boundaries": list(config.epoch_boundaries),
        "seed": config.seed,
        "areas": [asdict(a) for a in config.areas],
        "edges": edges,
        "shared_noise": [asdict(s) for s in config.shared_noise],
    }
    meta.update(extra)
    return meta


def _check_shared_noise(config: SimConfig) -> None:
    """Validate shared-noise groups reference known, gaussian areas."""
    for grp in config.shared_noise:
        for area in grp.areas:
            spec = config.area(area)  # raises KeyError if unknown
            if spec.observation != "gaussian":
                raise ValueError(
                    f"shared_noise requires gaussian areas; {area} is "
                    f"{spec.observation!r}"
                )


def _apply_shared_noise(
    neural2d: dict[str, np.ndarray], config: SimConfig
) -> dict[str, np.ndarray]:
    """Add cross-area shared fluctuations to flat ``(time, neurons)`` arrays.

    Uses an RNG stream independent of the per-area streams (so toggling shared
    noise does not perturb the rest of the simulation) but still deterministic
    in ``config.seed``.
    """
    if not config.shared_noise:
        return neural2d
    rng = np.random.default_rng([config.seed, 0x5EED])
    n_t = next(iter(neural2d.values())).shape[0]
    out = {k: v.copy() for k, v in neural2d.items()}
    for grp in config.shared_noise:
        shared = ar1_latents(n_t, 1, tau=grp.tau, dt=config.dt, rng=rng)[:, 0]
        for area in grp.areas:
            n_neurons = out[area].shape[1]
            weights = np.abs(rng.standard_normal(n_neurons))
            out[area] = out[area] + grp.strength * shared[:, None] * weights[None, :]
    return out


@dataclass
class SimResult:
    """Outputs and ground truth of a continuous-session simulation.

    Attributes
    ----------
    latents:
        ``area -> array(T, n_latents)`` resolved latents (post-coupling).
    intrinsic_latents:
        ``area -> array(T, n_latents)`` latents *before* coupling.
    neural:
        ``area -> array(T, n_neurons)`` observed population activity.
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
        return _base_metadata(
            self.config,
            {"structure": "continuous", "n_timesteps": self.config.n_timesteps},
        )


@dataclass
class TrialResult:
    """Outputs and ground truth of a trial-structured simulation.

    Arrays are 3-D ``(n_trials, n_bins, ...)``; trials are independent latent
    draws sharing the same loadings and coupling, so they are exchangeable for
    trial-permutation nulls.
    """

    latents: dict[str, np.ndarray]      # area -> (n_trials, n_bins, n_latents)
    neural: dict[str, np.ndarray]       # area -> (n_trials, n_bins, n_neurons)
    loadings: dict[str, np.ndarray]     # area -> (n_neurons, n_latents)
    config: SimConfig
    n_trials: int
    n_bins: int

    @property
    def area_names(self) -> list[str]:
        return [a.name for a in self.config.areas]

    def metadata(self) -> dict[str, Any]:
        return _base_metadata(
            self.config,
            {"structure": "trials", "n_trials": self.n_trials, "n_bins": self.n_bins},
        )


def _streams(config: SimConfig) -> dict[str, np.random.Generator]:
    """One independent RNG stream per area, spawned from the master seed."""
    children = np.random.default_rng(config.seed).spawn(len(config.areas))
    return {
        a.name: np.random.default_rng(c)
        for a, c in zip(config.areas, children, strict=True)
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
    """Run a continuous-session simulation.

    Deterministic in ``config.seed``: one independent RNG stream per area, so
    adding an area does not perturb earlier areas' streams.
    """
    _check_shared_noise(config)
    streams = _streams(config)

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
            realism_params=spec.realism,
            rng=rng,
        )

    neural = _apply_shared_noise(neural, config)
    return SimResult(resolved, intrinsic, neural, loadings, config)


def simulate_trials(
    config: SimConfig,
    n_trials: int,
    n_bins: int,
    burn_in: int | None = None,
) -> TrialResult:
    """Run a trial-structured simulation with independent, exchangeable trials.

    Each trial is an independent draw of every area's intrinsic latents, coupled
    and projected through the *same* loadings. A per-trial ``burn_in`` (default
    ``max(20, max_edge_lag + 10)``) is generated and discarded so that lagged
    couplings are fully ramped up by the first retained bin.

    Loadings and any realism nuisance (per-neuron gains, global fluctuation,
    drift) are drawn once over the whole concatenated session, so neuron
    identity is consistent across trials and drift is smooth across trial index;
    the *signal* (latents) is independent across trials.
    """
    if n_trials <= 0 or n_bins <= 0:
        raise ValueError("n_trials and n_bins must be positive")
    if burn_in is None:
        burn_in = max(20, config.max_lag + 10)
    _check_shared_noise(config)

    streams = _streams(config)
    gen_len = burn_in + n_bins

    # Fixed loadings (drawn once, before trial generation).
    loadings = {
        spec.name: random_loadings(spec.n_neurons, spec.n_latents, streams[spec.name])
        for spec in config.areas
    }

    # Independent intrinsic draws per trial, coupled per trial, burn-in dropped.
    per_trial: dict[str, list[np.ndarray]] = {a.name: [] for a in config.areas}
    for _ in range(n_trials):
        intrinsic = {
            spec.name: _intrinsic_for_area(spec, gen_len, config.dt, streams[spec.name])
            for spec in config.areas
        }
        resolved = resolve_latents(intrinsic, config.edges)
        for name, z in resolved.items():
            per_trial[name].append(z[burn_in:])

    latents = {name: np.stack(trials, axis=0) for name, trials in per_trial.items()}

    neural: dict[str, np.ndarray] = {}
    for spec in config.areas:
        rng = streams[spec.name]
        flat = latents[spec.name].reshape(n_trials * n_bins, spec.n_latents)
        x = project_population(
            flat,
            loadings[spec.name],
            model=spec.observation,
            snr=spec.snr,
            baseline=spec.baseline,
            realism_params=spec.realism,
            rng=rng,
        )
        neural[spec.name] = x.reshape(n_trials, n_bins, spec.n_neurons)

    if config.shared_noise:
        flat = {a: neural[a].reshape(n_trials * n_bins, -1) for a in neural}
        flat = _apply_shared_noise(flat, config)
        neural = {
            spec.name: flat[spec.name].reshape(n_trials, n_bins, spec.n_neurons)
            for spec in config.areas
        }

    return TrialResult(latents, neural, loadings, config, n_trials, n_bins)
