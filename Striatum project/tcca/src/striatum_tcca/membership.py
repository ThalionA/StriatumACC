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


def variate_structure_coefficients(neurons: np.ndarray, variates: np.ndarray) -> np.ndarray:
    """Per-neuron structure coefficients: correlation of each neuron with each
    canonical variate. Method-agnostic membership — it reads the coupling off the
    *variate* (however computed: linear or kernel), so it is the only membership
    definition that ports to (p)KCCA, which has no neuron-space weights.

    Parameters
    ----------
    neurons : ndarray (n_samples, n_units)   raw (residualised) activity
    variates : ndarray (n_samples, d)        canonical variates for that area

    Returns
    -------
    ndarray (n_units, d) of correlations in [-1, 1] (0 where a column is constant).
    """
    A = np.asarray(neurons, dtype=float)
    V = np.asarray(variates, dtype=float)
    valid = (np.all(np.isfinite(A), axis=1) & np.all(np.isfinite(V), axis=1))
    A, V = A[valid], V[valid]
    if A.shape[0] < 3:
        return np.zeros((A.shape[1], V.shape[1]))
    Ac = A - A.mean(0)
    Vc = V - V.mean(0)
    cov = (Ac.T @ Vc) / Ac.shape[0]                  # (n_units, d)
    denom = np.outer(Ac.std(0), Vc.std(0))
    return np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)


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


def _zscore_cols(M: np.ndarray) -> np.ndarray:
    """Column-wise z-score; zero-variance columns map to all-zeros (no coupling)."""
    mu = M.mean(axis=0)
    sd = M.std(axis=0)
    sd_safe = np.where(sd > 0, sd, 1.0)
    z = (M - mu) / sd_safe
    z[:, sd == 0] = 0.0
    return z


def pearson_coupling_scores(X: np.ndarray, Y: np.ndarray):
    """CCA-INDEPENDENT per-neuron cross-area coupling magnitude.

    For each neuron, the L2 norm of its Pearson-correlation row with the *other*
    area's whole population — how strongly it covaries with the partner area,
    computed straight from pairwise correlations with **no CCA and no PCA**.
    Pairing this with ``gini`` gives a subspace-machinery-free control for the
    canonical-weight Gini (the de-sparsification headline): it mirrors
    ``subspace_contribution`` (an L2 norm over the partner dimensions) but reads
    the coupling off raw correlations instead of canonical weights, so if the
    weight-Gini fall is real — not an artefact of how (p)CCA distributes weights —
    the Gini of these scores tracks it.

    Scope: this removes the CCA/PCA *weight machinery* as the explanation, but it
    is computed on the *same* residualised, run-bin-masked matrices fed to (p)CCA
    (Z is already partialled out by the caller), so it does **not** control for
    anything entering via that shared residualisation/masking step. Interpret the
    within-animal *slope* (the noise floor of the L2-norm Gini is set by the
    partner unit count, so absolute levels are not comparable across pairs).

    Parameters
    ----------
    X : ndarray (n_samples, n_units_x)   residualised activity, area X
    Y : ndarray (n_samples, n_units_y)   residualised activity, area Y
        (the same residualised matrices fed to the (p)CCA fit).

    Returns
    -------
    (coupling_x, coupling_y) : per-neuron coupling magnitude, shapes
        ``(n_units_x,)`` / ``(n_units_y,)``. Zeros if < 2 finite samples.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    nx, ny = X.shape[1], Y.shape[1]
    valid = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(Y), axis=1)
    Xv, Yv = X[valid], Y[valid]
    if Xv.shape[0] < 2:
        return np.zeros(nx), np.zeros(ny)
    R = (_zscore_cols(Xv).T @ _zscore_cols(Yv)) / Xv.shape[0]   # (nx, ny) corr
    return np.linalg.norm(R, axis=1), np.linalg.norm(R, axis=0)
