"""Predefined coupling scenarios with known ground truth.

Each builder returns a :class:`~popsim.simulate.SimConfig` for three areas
(``A``, ``B``, ``C``) with heterogeneous latent counts (4, 5, 3) and population
sizes, differing only in their inter-area coupling structure.

Base scenarios::

    no_coupling     none -- three independent areas.
    zero_lag        A -> B instantaneously (peak cross-correlation at lag 0).
    lagged          A -> B at +10 bins and C -> B at +25 bins; the A->B edge
                    appears at lag -10 when the pair is ordered (B, A).
    mediated        A -> C -> B with no direct A -> B; partial CCA(A,B|C)
                    collapses the apparent A-B coupling.
    epoch_varying   A->B, then B->A, then A->B again across three epochs,
                    changing strength, orientation, and direction.

Extended scenarios (Phase 2)::

    bidirectional   A -> B and B -> A at different positive lags on separate
                    latent dims (reciprocal communication).
    common_input    C drives both A and B (a shared-input confound); A-B
                    coupling collapses under partial CCA(A,B|C).
    rotated_subspace A -> B through a rank-r rotated (non-axis-aligned) subspace;
                    CCA recovers exactly r strong canonical correlations.
    partial_mediation part of A->B is relayed by C and part is direct (on a
                    separate dim pair), so partial CCA is reduced but not zeroed.
    noise_correlation no latent coupling, but A and B share additive
                    observation-level noise (correlation without communication).

Routing matrices are intentionally simple ("single-link" maps connecting one
source latent to one target latent, or explicit low-rank maps) so the lag,
mediation, and subspace structure are exactly recoverable; see
:mod:`popsim.metrics`.
"""

from __future__ import annotations

import numpy as np

from .coupling import CouplingEdge
from .simulate import AreaSpec, SharedNoise, SimConfig

__all__ = [
    "single_link_matrix",
    "orthonormal_matrix",
    "low_rank_matrix",
    "default_areas",
    "no_coupling",
    "zero_lag",
    "lagged",
    "mediated",
    "epoch_varying",
    "bidirectional",
    "common_input",
    "rotated_subspace",
    "partial_mediation",
    "noise_correlation",
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


def low_rank_matrix(
    d_target: int, d_source: int, rank: int, seed: int = 0, scale: float = 1.0
) -> np.ndarray:
    """Rank-``rank`` mixing matrix with rotated (non-axis-aligned) structure.

    Built as ``scale * U @ V.T`` with orthonormal ``U`` (d_target x rank) and
    ``V`` (d_source x rank), so communication flows through a ``rank``-dimensional
    rotated subspace of the source latents rather than along single axes.
    """
    if rank < 1 or rank > min(d_target, d_source):
        raise ValueError("rank must be in 1..min(d_target, d_source)")
    rng = np.random.default_rng(seed)
    u, _ = np.linalg.qr(rng.standard_normal((d_target, rank)))
    v, _ = np.linalg.qr(rng.standard_normal((d_source, rank)))
    return scale * (u @ v.T)


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


def bidirectional(
    n_timesteps: int = 3000,
    seed: int = 5,
    lag_ab: int = 5,
    lag_ba: int = 12,
    gain: float = 1.2,
    **area_kw,
) -> SimConfig:
    """Reciprocal communication: A -> B and B -> A on separate dims and lags.

    ``A0 -> B0`` at lag ``lag_ab`` and ``B1 -> A1`` at lag ``lag_ba``. Using
    distinct dimension pairs keeps the two directions independently recoverable,
    and using positive lags on both means there is no instantaneous loop. In the
    (A, B) cross-correlogram the forward link peaks at +``lag_ab`` (dim0) and the
    backward link at -``lag_ba`` (dim1).
    """
    edges = [
        CouplingEdge("A", "B", gain=gain, lag=lag_ab,
                     matrix=single_link_matrix(5, 4, t_idx=0, s_idx=0)),
        CouplingEdge("B", "A", gain=gain, lag=lag_ba,
                     matrix=single_link_matrix(4, 5, t_idx=1, s_idx=1)),
    ]
    return SimConfig(
        areas=default_areas(**area_kw),
        edges=edges,
        n_timesteps=n_timesteps,
        seed=seed,
        name="bidirectional",
    )


def common_input(
    n_timesteps: int = 3000, seed: int = 6, gain: float = 2.0, **area_kw
) -> SimConfig:
    """Shared-input confound: C drives both A and B, with no direct A-B link.

    ``C0 -> A0`` and ``C0 -> B0`` (zero-lag single-link). A and B are correlated
    only because they share C's input, so partial CCA(A, B | C) -- and the
    scalar partial correlation on the shared dimension -- collapse to zero, just
    like ``mediated`` but with C *upstream* of both rather than in the middle.
    """
    edges = [
        CouplingEdge("C", "A", gain=gain, lag=0,
                     matrix=single_link_matrix(4, 3, t_idx=0, s_idx=0)),
        CouplingEdge("C", "B", gain=gain, lag=0,
                     matrix=single_link_matrix(5, 3, t_idx=0, s_idx=0)),
    ]
    return SimConfig(
        areas=default_areas(**area_kw),
        edges=edges,
        n_timesteps=n_timesteps,
        seed=seed,
        name="common_input",
    )


def rotated_subspace(
    n_timesteps: int = 3000, seed: int = 7, rank: int = 2, gain: float = 1.6,
    **area_kw,
) -> SimConfig:
    """A -> B through a rank-``rank`` rotated communication subspace.

    The mixing matrix is an explicit rank-``rank`` map with orthonormal factors
    (see :func:`low_rank_matrix`), so communication occupies exactly ``rank``
    dimensions of a *rotated* (non-axis-aligned) subspace. CCA(A, B) should
    recover ``rank`` strong canonical correlations and a clear drop at the next.
    """
    M = low_rank_matrix(d_target=5, d_source=4, rank=rank, seed=200, scale=gain)
    edges = [CouplingEdge("A", "B", gain=1.0, lag=0, matrix=M)]
    return SimConfig(
        areas=default_areas(**area_kw),
        edges=edges,
        n_timesteps=n_timesteps,
        seed=seed,
        name="rotated_subspace",
    )


def partial_mediation(
    n_timesteps: int = 3000,
    seed: int = 8,
    gain: float = 1.6,
    direct_gain: float = 1.6,
    **area_kw,
) -> SimConfig:
    """Graded mediation: one A->B channel is relayed by C, another is direct.

    Two parallel channels: ``A0 -> C0 -> B0`` is mediated, while ``A1 -> B1`` is
    a direct edge that never touches C. The two levels of analysis tell
    complementary stories:

    - *Per latent dimension*: conditioning on C collapses the mediated pair
      (A0, B0) but leaves the direct pair (A1, B1) intact -- the grading is
      visible dimension-by-dimension (see ``test_partial_mediation_is_graded``).
    - *Population CCA*: the top canonical correlation is carried by the direct
      channel, so partial CCA(A, B | C) barely drops -- the A-B coupling
      *survives* partialling C, in clear contrast to ``mediated`` (which
      collapses) and ``common_input`` (also collapses).

    Routing the direct path through a *separate* dimension pair makes it
    identifiable: because C is collinear with A0, a direct path on dim0 would be
    absorbed when C is partialled out.
    """
    edges = [
        CouplingEdge("A", "C", gain=gain, lag=0,
                     matrix=single_link_matrix(3, 4, t_idx=0, s_idx=0)),
        CouplingEdge("C", "B", gain=gain, lag=0,
                     matrix=single_link_matrix(5, 3, t_idx=0, s_idx=0)),
        CouplingEdge("A", "B", gain=direct_gain, lag=0,
                     matrix=single_link_matrix(5, 4, t_idx=1, s_idx=1)),
    ]
    return SimConfig(
        areas=default_areas(**area_kw),
        edges=edges,
        n_timesteps=n_timesteps,
        seed=seed,
        name="partial_mediation",
    )


def noise_correlation(
    n_timesteps: int = 3000, seed: int = 9, strength: float = 0.6, **area_kw
) -> SimConfig:
    """Correlation without communication: A and B share additive output noise.

    There are no coupling edges, so the *latents* of A and B are independent
    (latent CCA ~ 0). A :class:`SharedNoise` group injects one common smooth
    fluctuation into A's and B's populations, so the recorded activity is
    correlated at the observation level only -- the canonical "correlation is
    not communication" confound. Requires gaussian observation (the default).
    """
    return SimConfig(
        areas=default_areas(**area_kw),
        edges=[],
        shared_noise=[SharedNoise(areas=["A", "B"], strength=strength, tau=15.0)],
        n_timesteps=n_timesteps,
        seed=seed,
        name="noise_correlation",
    )


# Registry for scripts / tests: name -> builder.
SCENARIOS = {
    "no_coupling": no_coupling,
    "zero_lag": zero_lag,
    "lagged": lagged,
    "mediated": mediated,
    "epoch_varying": epoch_varying,
    "bidirectional": bidirectional,
    "common_input": common_input,
    "rotated_subspace": rotated_subspace,
    "partial_mediation": partial_mediation,
    "noise_correlation": noise_correlation,
}
