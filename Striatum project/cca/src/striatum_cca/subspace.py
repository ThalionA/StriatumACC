"""Canonical subspaces in neuron space and principal-angle reorientation (D10).

A canonical subspace is compared across epochs in *neuron space* -- the frame
common to all three epochs -- by back-projecting the canonical coefficients
through each epoch's PCA loadings. Principal angles between the epoch subspaces
quantify reorientation; a within-epoch split-half angle gives the noise floor.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg

from . import core


def canonical_weights(components: np.ndarray, coef: np.ndarray, d: int) -> np.ndarray:
    """Neuron-space canonical weights: ``components @ coef[:, :d]``.

    Parameters
    ----------
    components : ndarray (n_units, k)   PCA loadings for one area/epoch
    coef : ndarray (k, d_total)         canonical coefficients (A or B)
    d : int                             number of canonical dims to keep
    """
    d = min(d, coef.shape[1])
    return components @ coef[:, :d]


def principal_angles(basis_a: np.ndarray, basis_b: np.ndarray) -> np.ndarray:
    """Principal angles (radians, ascending) between two subspaces.

    Each basis is ``(n_units, d)`` with columns spanning the subspace. Returns
    ``min(d_a, d_b)`` angles in ``[0, pi/2]``: 0 = aligned, pi/2 = orthogonal.
    """
    angles = linalg.subspace_angles(basis_a, basis_b)   # descending
    return np.sort(angles)


def _flatten(tensor: np.ndarray) -> np.ndarray:
    return tensor.reshape(-1, tensor.shape[-1])


def split_half_angles(
    scores_x: np.ndarray,
    scores_y: np.ndarray,
    components_x: np.ndarray,
    components_y: np.ndarray,
    d: int,
    n_splits: int = 10,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Within-epoch noise floor: principal angle between split-half subspaces.

    Repeatedly splits the trials in two, fits CCA on each half, and measures
    the principal angles between the two canonical subspaces. The mean over
    splits is the angle expected from sampling noise alone -- the baseline that
    genuine cross-epoch reorientation must exceed.

    Returns
    -------
    (angles_x, angles_y) : each ndarray (d,), mean over splits, ascending.
    """
    rng = np.random.default_rng(seed)
    n_tr = scores_x.shape[0]
    ax, ay = [], []
    for _ in range(n_splits):
        perm = rng.permutation(n_tr)
        h1, h2 = perm[: n_tr // 2], perm[n_tr // 2:]
        if h1.size < 2 or h2.size < 2:
            continue
        c1 = core.cca_fit(_flatten(scores_x[h1]), _flatten(scores_y[h1]))
        c2 = core.cca_fit(_flatten(scores_x[h2]), _flatten(scores_y[h2]))
        dd = min(d, c1.A.shape[1], c2.A.shape[1])
        if dd < 1:
            continue
        ax.append(principal_angles(
            canonical_weights(components_x, c1.A, dd),
            canonical_weights(components_x, c2.A, dd)))
        ay.append(principal_angles(
            canonical_weights(components_y, c1.B, dd),
            canonical_weights(components_y, c2.B, dd)))
    if not ax:
        return np.full(d, np.nan), np.full(d, np.nan)
    min_d = min(a.size for a in ax)
    ax = np.mean([a[:min_d] for a in ax], axis=0)
    ay = np.mean([a[:min_d] for a in ay], axis=0)
    return ax, ay
