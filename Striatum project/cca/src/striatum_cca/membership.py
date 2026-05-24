"""Communication-subspace membership: which units carry the coupling (D9).

Each canonical dimension is scored back onto individual neurons two ways:

* **structure coefficient** -- the correlation between a neuron's (in-subspace)
  residual activity and the canonical variate. Stable under collinearity.
* **canonical weight** -- the raw coefficient back-projected through the PCA
  loadings. Matches the subspace paper's weight/Gini machinery.

A neuron is a *member* of the communication subspace if its scalar contribution
(L2 norm of its scores across the retained canonical dims) is in the top
quartile. Sparsity of the weight profile is summarised by the Gini coefficient.
"""

from __future__ import annotations

import numpy as np


def _flatten(tensor: np.ndarray) -> np.ndarray:
    return tensor.reshape(-1, tensor.shape[-1])


def structure_coefficients(
    scores: np.ndarray, components: np.ndarray, coef: np.ndarray, d: int
) -> np.ndarray:
    """Correlation of each neuron's in-subspace activity with each variate.

    Parameters
    ----------
    scores : ndarray (n_trials, n_bins, k)   PC scores for one area/epoch
    components : ndarray (n_units, k)        PCA loadings
    coef : ndarray (k, d_total)              canonical coefficients (A or B)
    d : int                                  canonical dims to keep

    Returns
    -------
    ndarray (n_units, d) of correlations in [-1, 1].
    """
    z = _flatten(scores)                          # (n_samples, k)
    d = min(d, coef.shape[1])
    variates = z @ coef[:, :d]                    # (n_samples, d)
    recon = z @ components.T                      # (n_samples, n_units)

    # Drop missing (trial, bin) samples before correlating.
    valid = (np.all(np.isfinite(recon), axis=1)
             & np.all(np.isfinite(variates), axis=1))
    recon, variates = recon[valid], variates[valid]

    rc = recon - recon.mean(axis=0)
    vc = variates - variates.mean(axis=0)
    cov = (rc.T @ vc) / rc.shape[0]               # (n_units, d)
    denom = np.outer(rc.std(axis=0), vc.std(axis=0))
    return np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)


def canonical_weight_scores(components: np.ndarray, coef: np.ndarray, d: int) -> np.ndarray:
    """Raw neuron-space canonical weights: ``components @ coef[:, :d]``."""
    d = min(d, coef.shape[1])
    return components @ coef[:, :d]


def subspace_contribution(score_matrix: np.ndarray) -> np.ndarray:
    """Per-neuron scalar contribution = L2 norm across canonical dims."""
    return np.linalg.norm(np.atleast_2d(score_matrix), axis=1)


def member_mask(contribution: np.ndarray, quantile: float = 0.75) -> np.ndarray:
    """Boolean mask of top-quantile neurons by |contribution|."""
    contribution = np.abs(np.asarray(contribution, dtype=float))
    if contribution.size == 0:
        return np.zeros(0, dtype=bool)
    return contribution >= np.quantile(contribution, quantile)


def gini(values: np.ndarray) -> float:
    """Gini coefficient of ``|values|`` -- 0 = uniform, 1 = one neuron carries all.

    Uses the N/(N-1) small-sample correction so regions with different cell
    counts are comparable (as in the subspace paper).
    """
    v = np.sort(np.abs(np.asarray(values, dtype=float)))
    n = v.size
    total = v.sum()
    if n < 2 or total == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    g = np.sum((2 * idx - n - 1) * v) / (n * total)
    return float(g * n / (n - 1))


def jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Jaccard overlap of two boolean member masks."""
    union = np.sum(mask_a | mask_b)
    if union == 0:
        return np.nan
    return float(np.sum(mask_a & mask_b) / union)
