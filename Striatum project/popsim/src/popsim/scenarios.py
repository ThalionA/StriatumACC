"""Predefined coupling scenarios with known ground truth.

Each builder returns a :class:`~popsim.simulate.SimConfig` for three areas
(``A``, ``B``, ``C``) with heterogeneous latent counts (4, 5, 3) and population
sizes, differing only in their inter-area coupling structure:

================  ============================================================
Scenario          Ground-truth coupling
================  ============================================================
``no_coupling``   none -- three independent areas.
``zero_lag``      A -> B instantaneously (peak cross-correlation at lag 0).
``lagged``        A -> B at +10 bins and C -> B at +25 bins; the same A->B
                  edge appears at lag -10 when the pair is ordered (B, A),
                  demonstrating positive and negative lags.
``mediated``      A -> C -> B with no direct A -> B; controlling for C
                  (partial CCA / partial correlation) collapses A-B coupling.
``epoch_varying`` A->B then B->A then A->B again across three epochs, changing
                  strength (gain), orientation (which latent dims), and
                  direction.
================  ============================================================

The coupling routing matrices are intentionally simple ("single-link" maps that
connect one source latent to one target latent) so that lag and mediation
structure are exactly recoverable in latent space; see :mod:`popsim.metrics`.
"""

from __future__ import annotations

import numpy as np

from .coupling import CouplingEdge
from .simulate import AreaSpec, SimConfig

__all__ = [
    "single_link_matrix",
    "orthonormal_matrix",
    "default_areas",
    "no_coupling",
    "zero_lag",
    "lagged",
    "mediated",
    "epoch_varying",
    "SCENARIOS",
]


def single_link_matrix(
    d_target: int, d_source: int, t_idx: int, s_idx: int, weight: float = 1.0
) -> np.ndarray:
    """Matrix routing one source latent to one target latent.

    All entries are zero except ``M[t_idx, s_idx] = weight``, making the coupling
    exactly traceable: ``z_target[:, t_idx]`` receives a scaled, possibly lagged
    copy of ``z_source[:, s_idx]``.
    """
    M = np.zeros((d_target, d_source))
    M[t_idx, s_idx] = weight
    return M


def orthonormal_matrix(d_target: int, d_source: int, seed: int = 0) -> np.ndarray:
    """Random matrix with orthonormal columns, mapping source -> target dims.

    Used for "full-rank" communication where every source latent contributes.
    """
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((d_target, d_source)))
    return q[:, :d_source]


def default_areas(
    dynamics: str = "ar1", observation: str = "gaussian"
) -> list[AreaSpec]:
    """Three areas with 4 / 5 / 3 latents and distinct population sizes."""
    return [
        AreaSpec("A", n_latents=4, n_neurons=60, dynamics=dynamics,
                 observation=observation, tau=20.0, snr=3.0),
        AreaSpec("B", n_latents=5, n_neurons=80, dynamics=dynamics,
                 observation=observation, tau=20.0, snr=3.0),
        AreaSpec("C", n_latents=3, n_neurons=40, dynamics=dynamics,
                 observation=observation, tau=20.0, snr=3.0),
    ]


def no_coupling(n_timesteps: int = 3000, seed: int = 0, **area_kw) -> SimConfig:
    """Three independent areas; no inter-area communication."""
    return SimConfig(
        areas=default_areas(**area_kw),
        edges=[],
        n_timesteps=n_timesteps,
        seed=seed,
        name="no_coupling",
    )


def zero_lag(
    n_timesteps: int = 3000, seed: int = 1, gain: float = 1.0, **area_kw
) -> SimConfig:
    """A communicates to B instantaneously through a full-rank map."""
    M = orthonormal_matrix(d_target=5, d_source=4, seed=100)
    edges = [CouplingEdge("A", "B", gain=gain, lag=0, matrix=M)]
    return SimConfig(
        areas=default_areas(**area_kw),
        edges=edges,
        n_timesteps=n_timesteps,
        seed=seed,
        name="zero_lag",
    )


def lagged(
    n_timesteps: int = 3000,
    seed: int = 2,
    lag_ab: int = 10,
    lag_cb: int = 25,
    gain: float = 1.2,
    **area_kw,
) -> SimConfig:
    """A leads B by ``lag_ab`` bins; C leads B by ``lag_cb`` bins.

    Single-link routing keeps the ground truth exact: ``z_B[:, 0] <- z_A[:, 0]``
    (lag ``lag_ab``) and ``z_B[:, 1] <- z_C[:, 0]`` (lag ``lag_cb``). The A->B
    edge shows up as a +``lag_ab`` peak in xcorr(z_A, z_B) and a -``lag_ab`` peak
    in xcorr(z_B, z_A).
    """
    edges = [
        CouplingEdge("A", "B", gain=gain, lag=lag_ab,
                     matrix=single_link_matrix(5, 4, t_idx=0, s_idx=0)),
        CouplingEdge("C", "B", gain=gain, lag=lag_cb,
                     matrix=single_link_matrix(5, 3, t_idx=1, s_idx=0)),
    ]
    return SimConfig(
        areas=default_areas(**area_kw),
        edges=edges,
        n_timesteps=n_timesteps,
        seed=seed,
        name="lagged",
    )


def mediated(
    n_timesteps: int = 3000, seed: int = 3, gain: float = 2.0, **area_kw
) -> SimConfig:
    """A -> C -> B with no direct A->B (a mediation chain).

    Both links are zero-lag single-link maps (A dim0 -> C dim0 -> B dim0), so A
    and B are correlated only through C. Partial CCA / partial correlation
    conditioning on C drives the A-B association toward zero.
    """
    edges = [
        CouplingEdge("A", "C", gain=gain, lag=0,
                     matrix=single_link_matrix(3, 4, t_idx=0, s_idx=0)),
        CouplingEdge("C", "B", gain=gain, lag=0,
                     matrix=single_link_matrix(5, 3, t_idx=0, s_idx=0)),
    ]
    return SimConfig(
        areas=default_areas(**area_kw),
        edges=edges,
        n_timesteps=n_timesteps,
        seed=seed,
        name="mediated",
    )


def epoch_varying(
    n_timesteps: int = 3000, seed: int = 4, gain: float = 1.4, lag: int = 2, **area_kw
) -> SimConfig:
    """Communication that changes across three equal epochs.

    - Epoch 1 ``[0, T/3)``:    A -> B, dim0 -> dim0 (direction A->B, orientation 1).
    - Epoch 2 ``[T/3, 2T/3)``: B -> A, dim0 -> dim0 (direction reversed).
    - Epoch 3 ``[2T/3, T)``:   A -> B, dim1 -> dim2, weaker gain (orientation 2,
      reduced strength).

    A small fixed ``lag`` is used so the reversed-direction edge does not create
    an instantaneous loop in the zero-lag dependency graph.
    """
    third = n_timesteps // 3
    e1, e2, e3 = [(0, third)], [(third, 2 * third)], [(2 * third, n_timesteps)]
    edges = [
        CouplingEdge("A", "B", gain=gain, lag=lag,
                     matrix=single_link_matrix(5, 4, t_idx=0, s_idx=0), epochs=e1),
        CouplingEdge("B", "A", gain=gain, lag=lag,
                     matrix=single_link_matrix(4, 5, t_idx=0, s_idx=0), epochs=e2),
        CouplingEdge("A", "B", gain=0.5 * gain, lag=lag,
                     matrix=single_link_matrix(5, 4, t_idx=2, s_idx=1), epochs=e3),
    ]
    return SimConfig(
        areas=default_areas(**area_kw),
        edges=edges,
        n_timesteps=n_timesteps,
        epoch_boundaries=[third, 2 * third],
        seed=seed,
        name="epoch_varying",
    )


# Registry for scripts / tests: name -> builder.
SCENARIOS = {
    "no_coupling": no_coupling,
    "zero_lag": zero_lag,
    "lagged": lagged,
    "mediated": mediated,
    "epoch_varying": epoch_varying,
}
