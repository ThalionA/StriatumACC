"""Lightweight metrics for validating simulated coupling.

These small, dependency-light helpers are used by the tests and the recovery
benchmark to confirm that a simulation has the coupling structure it was
configured with. They are deliberately *not* a full communication-subspace
analysis library (the sibling ``striatum_cca`` package is that):

- :func:`cross_correlation` / :func:`lag_of_peak_xcorr` characterise the lag at
  which two scalar signals are maximally correlated.
- :func:`cca` returns canonical correlations (and weights) between two
  multivariate signals; :func:`canonical_variates` projects onto the top pair.
- :func:`partial_cca` / :func:`partial_correlation` remove the linear influence
  of a conditioning signal before measuring association -- how a mediated
  (A -> C -> B) coupling is shown to collapse once C is controlled for.
- :func:`pca_reduce` reduces a population to its top principal components, the
  standard pre-step before cross-area CCA on neural data.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "cross_correlation",
    "lag_of_peak_xcorr",
    "cca",
    "canonical_variates",
    "partial_cca",
    "partial_correlation",
    "regress_out",
    "pca_reduce",
]


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (x - mu) / sd


def cross_correlation(
    x: np.ndarray, y: np.ndarray, max_lag: int
) -> tuple[np.ndarray, np.ndarray]:
    """Normalised cross-correlation of two 1-D signals over a lag range.

    The value at lag ``k`` is ``corr(x[t], y[t + k])``. A positive peak lag
    therefore means ``x`` *leads* ``y`` (``y`` is a delayed copy of ``x``).

    Returns ``(lags, corr)`` where ``lags = arange(-max_lag, max_lag + 1)``.
    """
    x = _zscore(np.asarray(x, dtype=float).ravel())
    y = _zscore(np.asarray(y, dtype=float).ravel())
    n = x.size
    if y.size != n:
        raise ValueError("x and y must have equal length")
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.empty(lags.size)
    for i, k in enumerate(lags):
        if k >= 0:
            a, b = x[: n - k], y[k:]
        else:
            a, b = x[-k:], y[: n + k]
        corr[i] = np.nan if a.size < 2 else np.corrcoef(a, b)[0, 1]
    return lags, corr


def lag_of_peak_xcorr(x: np.ndarray, y: np.ndarray, max_lag: int) -> int:
    """Lag (in bins) at which ``|cross_correlation|`` is maximal."""
    lags, corr = cross_correlation(x, y, max_lag)
    return int(lags[np.nanargmax(np.abs(corr))])


def _inv_sqrt(C: np.ndarray) -> np.ndarray:
    """Symmetric inverse square root of a positive-definite matrix."""
    vals, vecs = np.linalg.eigh(C)
    vals = np.clip(vals, 1e-12, None)
    return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T


def cca(
    X: np.ndarray, Y: np.ndarray, n_components: int | None = None, reg: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonical correlation analysis between multivariate ``X`` and ``Y``.

    Parameters
    ----------
    X, Y:
        Arrays of shape ``(n_samples, n_features)``.
    n_components:
        Number of canonical pairs to return (default ``min`` of feature counts).
    reg:
        Ridge added to covariance diagonals for numerical stability.

    Returns
    -------
    (corrs, A, B):
        ``corrs`` are canonical correlations (descending); ``A`` and ``B`` map
        ``X`` and ``Y`` (centred) onto the canonical variates.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    n = Xc.shape[0]
    dx, dy = Xc.shape[1], Yc.shape[1]
    k = min(dx, dy) if n_components is None else min(n_components, dx, dy)

    Cxx = (Xc.T @ Xc) / n + reg * np.eye(dx)
    Cyy = (Yc.T @ Yc) / n + reg * np.eye(dy)
    Cxy = (Xc.T @ Yc) / n

    inv_sqrt_xx = _inv_sqrt(Cxx)
    inv_sqrt_yy = _inv_sqrt(Cyy)
    T = inv_sqrt_xx @ Cxy @ inv_sqrt_yy
    U, s, Vt = np.linalg.svd(T, full_matrices=False)
    corrs = np.clip(s[:k], 0.0, 1.0)
    A = inv_sqrt_xx @ U[:, :k]
    B = inv_sqrt_yy @ Vt[:k].T
    return corrs, A, B


def canonical_variates(
    X: np.ndarray, Y: np.ndarray, component: int = 0
) -> tuple[float, np.ndarray, np.ndarray]:
    """Top (or ``component``-th) canonical correlation and the variate pair.

    Returns ``(corr, u, v)`` where ``u`` and ``v`` are the 1-D canonical variates
    of (centred) ``X`` and ``Y`` for the requested component.
    """
    corrs, A, B = cca(X, Y)
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    return float(corrs[component]), Xc @ A[:, component], Yc @ B[:, component]


def regress_out(target: np.ndarray, conditioning: np.ndarray) -> np.ndarray:
    """Return ``target`` with the linear contribution of ``conditioning`` removed.

    Ordinary least-squares residuals; an intercept is included implicitly by
    centring both signals.
    """
    target = np.asarray(target, dtype=float)
    Z = np.asarray(conditioning, dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    squeeze = target.ndim == 1
    if squeeze:
        target = target[:, None]
    Zc = Z - Z.mean(axis=0, keepdims=True)
    Tc = target - target.mean(axis=0, keepdims=True)
    coef, *_ = np.linalg.lstsq(Zc, Tc, rcond=None)
    resid = Tc - Zc @ coef
    return resid[:, 0] if squeeze else resid


def partial_cca(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    n_components: int | None = None,
    reg: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CCA between ``X`` and ``Y`` after regressing out conditioning ``Z``.

    If ``X`` and ``Y`` are associated only through ``Z`` (mediation), the
    canonical correlations collapse toward zero.
    """
    return cca(regress_out(X, Z), regress_out(Y, Z), n_components=n_components, reg=reg)


def partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Pearson correlation of ``x`` and ``y`` after regressing out ``z``."""
    xr = regress_out(np.asarray(x).ravel(), z)
    yr = regress_out(np.asarray(y).ravel(), z)
    return float(np.corrcoef(xr.ravel(), yr.ravel())[0, 1])


def pca_reduce(X: np.ndarray, k: int) -> np.ndarray:
    """Project ``X`` (n_samples, n_features) onto its top ``k`` principal axes.

    Returns ``(n_samples, k)`` PC scores (centred). This is the standard
    dimensionality reduction applied to each area's population before cross-area
    CCA, mirroring ``striatum_cca``'s per-area PCA step.
    """
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    k = min(k, Xc.shape[1])
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:k].T
