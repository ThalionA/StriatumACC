"""Inter-area communication as directed, possibly time-varying latent coupling.

Communication between areas is modelled at the *latent* level. A directed
:class:`CouplingEdge` from a source area to a target area injects a linear,
optionally lagged, optionally epoch-gated copy of the source latents into the
target latents::

    z_target(t) += gain * M @ z_source(t - lag)        (when the edge is active)

The full set of edges plus the per-area intrinsic latents is resolved by
:func:`resolve_latents`, which time-steps the system in topological order so
that mediation chains (A -> C -> B) and zero-lag couplings compose correctly.

Conventions
-----------
- ``lag`` is in time bins and must be >= 0 (causal: the source's *past* drives
  the target). To obtain a relationship that peaks at a *negative* lag in the
  cross-correlogram of an ordered pair ``(X, Y)``, add the reverse edge
  ``Y -> X``; see ``scenarios.lagged`` for a worked example.
- ``matrix`` (``M``) has shape ``(n_latents_target, n_latents_source)``.
  ``None`` means an identity map (requires equal latent counts).
- ``epochs`` restricts an edge to a list of ``(start, stop)`` half-open bin
  ranges; ``None`` means the edge is always active. This is how communication
  changes strength / orientation / direction across epochs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["CouplingEdge", "resolve_latents", "topological_order"]


@dataclass
class CouplingEdge:
    """A directed latent-to-latent coupling between two areas."""

    source: str
    target: str
    gain: float = 1.0
    lag: int = 0
    matrix: np.ndarray | None = None
    epochs: list[tuple[int, int]] | None = None

    def __post_init__(self) -> None:
        if self.lag < 0:
            raise ValueError(
                f"lag must be >= 0 (got {self.lag}); reverse the edge direction "
                "to model a negative-lag relationship"
            )
        if self.source == self.target and self.lag == 0:
            raise ValueError("zero-lag self-edges create an instantaneous loop")
        if self.matrix is not None:
            self.matrix = np.asarray(self.matrix, dtype=float)
            if self.matrix.ndim != 2:
                raise ValueError("matrix must be 2-D (d_target, d_source)")

    def active_mask(self, n_timesteps: int) -> np.ndarray:
        """Boolean array of shape ``(n_timesteps,)`` for when the edge is on."""
        if self.epochs is None:
            return np.ones(n_timesteps, dtype=bool)
        mask = np.zeros(n_timesteps, dtype=bool)
        for start, stop in self.epochs:
            mask[start:stop] = True
        return mask


def topological_order(areas: list[str], edges: list[CouplingEdge]) -> list[str]:
    """Order area names so every zero-lag edge points 'forward'.

    Only zero-lag edges constrain the ordering, because lagged edges reference
    already-computed past values and therefore never create an evaluation-order
    dependency within a single time step.

    Raises
    ------
    ValueError
        If the zero-lag dependency graph contains a cycle.
    """
    incoming: dict[str, set[str]] = {a: set() for a in areas}
    for e in edges:
        if e.lag == 0 and e.source != e.target:
            if e.source not in incoming or e.target not in incoming:
                raise ValueError(
                    f"edge references unknown area: {e.source}->{e.target}"
                )
            incoming[e.target].add(e.source)

    order: list[str] = []
    resolved: set[str] = set()
    remaining = list(areas)
    while remaining:
        ready = [a for a in remaining if incoming[a] <= resolved]
        if not ready:
            raise ValueError(
                "zero-lag coupling graph has a cycle; cannot order areas"
            )
        for a in ready:  # preserve user-supplied order among ready nodes
            order.append(a)
            resolved.add(a)
            remaining.remove(a)
    return order


def resolve_latents(
    intrinsic: dict[str, np.ndarray],
    edges: list[CouplingEdge],
) -> dict[str, np.ndarray]:
    """Compose intrinsic latents with inter-area coupling.

    Parameters
    ----------
    intrinsic:
        Mapping ``area -> array(n_timesteps, n_latents)`` of intrinsic latents.
        All areas must share ``n_timesteps``.
    edges:
        Directed coupling edges between areas.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping ``area -> resolved latents`` of the same shapes as ``intrinsic``.
    """
    areas = list(intrinsic.keys())
    if not areas:
        return {}
    n_timesteps = intrinsic[areas[0]].shape[0]
    for z in intrinsic.values():
        if z.shape[0] != n_timesteps:
            raise ValueError("all areas must share n_timesteps")
    for e in edges:
        if e.source not in intrinsic or e.target not in intrinsic:
            raise ValueError(f"edge references unknown area: {e.source}->{e.target}")

    order = topological_order(areas, edges)
    edges_by_target: dict[str, list[CouplingEdge]] = {a: [] for a in areas}
    for e in edges:
        edges_by_target[e.target].append(e)
    masks = {id(e): e.active_mask(n_timesteps) for e in edges}

    out: dict[str, np.ndarray] = {a: intrinsic[a].copy() for a in areas}

    # Time-step in topological order. Within a step, a source's resolved value
    # is available to its zero-lag downstream targets because sources are
    # visited first. Lagged edges read already-finalised past values.
    for t in range(n_timesteps):
        for area in order:
            for e in edges_by_target[area]:
                src_t = t - e.lag
                if src_t < 0 or not masks[id(e)][t]:
                    continue
                src_val = out[e.source][src_t]
                contribution = src_val if e.matrix is None else e.matrix @ src_val
                out[area][t] += e.gain * contribution
    return out
