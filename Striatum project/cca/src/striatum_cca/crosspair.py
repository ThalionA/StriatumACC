"""Within-area communication-subspace similarity across an area's pairs.

Each area participates in several area pairs; in every pair it has a
communication subspace -- the dominant canonical weight vector in that area's
neuron space. This module asks whether an area uses a *consistent* subspace to
talk to its different partners: it compares that area's weight vector between
every pair of partners, within an animal (the vectors live in that animal's
neuron space, so cross-animal comparison is meaningless), and averages the
|cosine| similarity across animals.

Pure compute over the Stage-3 ``PairSubspace`` results; consumed by
scripts/plot_subspace_similarity.py.
"""

from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute cosine similarity between two vectors.

    Absolute because a canonical weight vector's sign is arbitrary, so an
    anti-parallel pair still describes the *same* one-dimensional subspace.
    Returns NaN if either vector is all-zero.
    """
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return np.nan
    return abs(float(np.dot(a, b)) / (na * nb))


def _area_weight(pair_subspace, area: str, epoch: str) -> np.ndarray | None:
    """Dominant canonical weight vector for ``area`` in one PairSubspace."""
    es = pair_subspace.epochs[epoch]
    if area == pair_subspace.area_x:
        return es.weights_x[:, 0]
    if area == pair_subspace.area_y:
        return es.weights_y[:, 0]
    return None


def area_partner_vectors(
    results, area: str, epoch: str
) -> dict[int, dict[str, np.ndarray]]:
    """``{animal_id: {partner_area: weight_vec}}`` for one area at one epoch."""
    out: dict[int, dict[str, np.ndarray]] = {}
    for ps in results:
        vec = _area_weight(ps, area, epoch)
        if vec is None:
            continue
        partner = ps.area_y if area == ps.area_x else ps.area_x
        out.setdefault(ps.animal_id, {})[partner] = vec
    return out


def similarity_matrix(results, area: str, epoch: str, partners: list[str]):
    """Mean |cos| similarity matrix for one area, plus per-cell animal counts.

    Entry (i, j) is the |cosine| similarity of ``area``'s dominant weight
    vector between its pair with ``partners[i]`` and its pair with
    ``partners[j]``, computed within each animal and averaged over the animals
    that have both pairs. Returns ``(mean_matrix, count_matrix)``; cells with
    no animal are NaN.
    """
    n = len(partners)
    idx = {p: i for i, p in enumerate(partners)}
    acc = np.zeros((n, n))
    cnt = np.zeros((n, n))
    for vecs in area_partner_vectors(results, area, epoch).values():
        present = [p for p in partners if p in vecs]
        for pa in present:
            for pb in present:
                if vecs[pa].shape != vecs[pb].shape:
                    continue
                s = cosine_similarity(vecs[pa], vecs[pb])
                if np.isfinite(s):
                    acc[idx[pa], idx[pb]] += s
                    cnt[idx[pa], idx[pb]] += 1
    mat = np.full((n, n), np.nan)
    nz = cnt > 0
    mat[nz] = acc[nz] / cnt[nz]
    return mat, cnt


def mean_pairwise(matrix: np.ndarray) -> float:
    """Mean of the off-diagonal finite entries of a similarity matrix."""
    n = matrix.shape[0]
    off = matrix[~np.eye(n, dtype=bool)]
    off = off[np.isfinite(off)]
    return float(np.mean(off)) if off.size else np.nan
