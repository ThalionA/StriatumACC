"""Regularised kernel CCA (KCCA) — the nonlinear analogue of ``core.cca``.

Maps each area's activity into an RBF feature space and finds the maximally
correlated nonlinear projections, mirroring :func:`core.cca_fit`/``cca_score`` so
it slots into the same windowed pipeline (held-out CC, circular-shift surrogate,
structure-coefficient membership/Gini). Unregularised kernel CCA is *degenerate*
— a rich enough feature space makes the in-sample correlation 1 for ANY two sets
(the saturation the methods note warns about, in its most acute form), so the
ridge ``reg`` is mandatory and is the knob that controls overfitting.

Formulation (Hardoon, Szedmak & Shawe-Taylor 2004): with centred training Gram
matrices :math:`\\tilde K_x,\\tilde K_y` and ridge :math:`\\kappa`, the dual
directions solve

    (\\tilde K_x + \\kappa I)^{-1} \\tilde K_y (\\tilde K_y + \\kappa I)^{-1}
        \\tilde K_x\\, \\alpha = \\rho^2 \\alpha ,

with :math:`\\beta = \\rho^{-1}(\\tilde K_y + \\kappa I)^{-1}\\tilde K_x\\alpha`.
A new point is projected by its (train-centred) cross-kernel against the training
set, so the held-out canonical correlation is an honest out-of-sample estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigs as _sparse_eigs
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import KernelCenterer

from .core import _finite_rows


@dataclass
class KCCAResult:
    """A fitted regularised KCCA model (dual / kernel representation)."""

    alpha: np.ndarray          # (n_train, d) dual coefficients for X
    beta: np.ndarray           # (n_train, d) dual coefficients for Y
    r: np.ndarray              # (d,) ridge-penalised canonical correlations (sqrt of
                               # the regularised eigenvalue), descending. NOTE: this
                               # UNDER-estimates the variate Pearson correlation
                               # (more so for higher dims / larger reg); for the
                               # actual correlation use kcca_score. Only used here to
                               # scale beta (the scale cancels in any correlation).
    x_train: np.ndarray        # (n_train, p) training inputs (for cross-kernels)
    y_train: np.ndarray        # (n_train, q)
    gamma_x: float             # RBF bandwidth for X
    gamma_y: float
    centerer_x: KernelCenterer  # fit on the train Gram (centres test kernels too)
    centerer_y: KernelCenterer


def median_gamma(X: np.ndarray, max_n: int = 1000, seed: int = 0) -> float:
    """RBF bandwidth via the median heuristic: ``gamma = 1 / (2 * median d^2)``.

    ``median d^2`` is the median squared pairwise distance (sub-sampled to
    ``max_n`` rows for speed). Returns 1.0 if degenerate.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n > max_n:
        idx = np.random.default_rng(seed).choice(n, max_n, replace=False)
        X = X[idx]
    sq = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    med = np.median(sq[np.triu_indices(X.shape[0], k=1)])
    return float(1.0 / (2.0 * med)) if med > 0 else 1.0


def kcca_fit(x: np.ndarray, y: np.ndarray, reg: float,
             gamma_x: float | None = None, gamma_y: float | None = None,
             n_components: int = 5) -> KCCAResult:
    """Fit regularised RBF-kernel CCA. ``reg`` (>0) is the ridge on the centred
    Gram matrices — the overfitting control. Bandwidths default to the median
    heuristic per area. Returns up to ``n_components`` canonical directions."""
    if reg <= 0:
        raise ValueError("reg must be > 0 (unregularised KCCA is degenerate)")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = _finite_rows(x, y)
    x, y = x[mask], y[mask]
    n = x.shape[0]
    if n < 3:
        raise ValueError("need at least 3 finite paired samples for KCCA")
    if gamma_x is None:
        gamma_x = median_gamma(x)
    if gamma_y is None:
        gamma_y = median_gamma(y)

    Kx_raw = rbf_kernel(x, gamma=gamma_x)                   # compute each kernel once
    Ky_raw = rbf_kernel(y, gamma=gamma_y)
    cx = KernelCenterer().fit(Kx_raw)
    cy = KernelCenterer().fit(Ky_raw)
    Kx = cx.transform(Kx_raw)
    Ky = cy.transform(Ky_raw)
    Kx = 0.5 * (Kx + Kx.T)                                   # symmetrise
    Ky = 0.5 * (Ky + Ky.T)
    Ax = Kx + reg * np.eye(n)
    Ay = Ky + reg * np.eye(n)
    cAx = linalg.cho_factor(Ax, check_finite=False)
    cAy = linalg.cho_factor(Ay, check_finite=False)
    d = int(min(n_components, n))

    # Top-d eigenpairs of M = Ax^-1 Ky Ay^-1 Kx (rho^2 = eigenvalue). Never form M:
    # Cholesky-factor Ax,Ay once, hand ARPACK a matrix-free operator so each
    # matvec is O(n^2) (the full dense eig / repeated solves are the bottleneck).
    def _mv(v):
        return linalg.cho_solve(cAx, Ky @ linalg.cho_solve(cAy, Kx @ v,
                                                            check_finite=False),
                                check_finite=False)
    evals = evecs = None
    if 0 < d < n - 2:
        try:
            ev, vec = _sparse_eigs(LinearOperator((n, n), matvec=_mv), k=d,
                                   which="LR", maxiter=n * 20)
            evals, evecs = np.real(ev), np.real(vec)
        except Exception:                                   # noqa: BLE001
            evals = None
    if evals is None:                                       # tiny-n / fallback
        M = linalg.cho_solve(cAx, Ky @ linalg.cho_solve(cAy, Kx))
        ev, vec = linalg.eig(M)
        order = np.argsort(np.real(ev))[::-1][:d]
        evals, evecs = np.real(ev)[order], np.real(vec[:, order])
    order = np.argsort(evals)[::-1]
    r = np.sqrt(np.clip(evals[order], 0.0, 1.0))
    alpha = evecs[:, order]                                  # (n, d)
    # beta = rho^-1 Ay^-1 Kx alpha (guard rho ~ 0).
    beta = linalg.cho_solve(cAy, Kx @ alpha, check_finite=False)
    beta = beta / np.where(r > 1e-6, r, 1.0)[None, :]
    return KCCAResult(alpha=alpha, beta=beta, r=r, x_train=x, y_train=y,
                      gamma_x=float(gamma_x), gamma_y=float(gamma_y),
                      centerer_x=cx, centerer_y=cy)


def _variates(data: np.ndarray, train: np.ndarray, gamma: float,
              centerer: KernelCenterer, coef: np.ndarray) -> np.ndarray:
    """Project ``data`` onto the dual directions: centred cross-kernel @ coef."""
    K = centerer.transform(rbf_kernel(np.asarray(data, float), train, gamma=gamma))
    return K @ coef


def kcca_variates(x: np.ndarray, y: np.ndarray, model: KCCAResult):
    """Project data onto the fitted KCCA directions -> ``(u, v)`` canonical
    variates, each ``(n_samples, d)`` (the nonlinear analogue of ``X @ A``).
    Used for structure-coefficient membership (correlate neurons with these)."""
    u = _variates(x, model.x_train, model.gamma_x, model.centerer_x, model.alpha)
    v = _variates(y, model.y_train, model.gamma_y, model.centerer_y, model.beta)
    return u, v


def kcca_lagged_heldout(x_tr, y_tr, x_te, y_te, reg, lags, gamma_x=None,
                        gamma_y=None):
    """Held-out dominant KCCA correlation vs integer bin lag.

    For each lag ``L`` in ``lags`` (positive = Y delayed relative to X, i.e. X
    leads), shift Y by ``L``, fit KCCA on the (shifted) training rows and score
    the (shifted) held-out rows. Returns the lag curve of the held-out dominant
    correlation — the directionality readout for :func:`lagged.information_flow_index`
    (held-out, not in-sample, because the in-sample KCCA correlation saturates).
    """
    x_tr = np.asarray(x_tr, float); y_tr = np.asarray(y_tr, float)
    x_te = np.asarray(x_te, float); y_te = np.asarray(y_te, float)
    if gamma_x is None:
        gamma_x = median_gamma(x_tr)
    if gamma_y is None:
        gamma_y = median_gamma(y_tr)
    nt, nv = x_tr.shape[0], x_te.shape[0]
    out = np.full(len(lags), np.nan)
    for i, L in enumerate(lags):
        # positive L = Y delayed vs X (X leads): pair x[:n-L] with y[L:].
        if L >= 0:
            xt, yt = (x_tr[: nt - L], y_tr[L:]) if L else (x_tr, y_tr)
            xv, yv = (x_te[: nv - L], y_te[L:]) if L else (x_te, y_te)
        else:
            xt, yt = x_tr[-L:], y_tr[: nt + L]
            xv, yv = x_te[-L:], y_te[: nv + L]
        if xt.shape[0] < 20 or xv.shape[0] < 5:
            continue
        try:
            m = kcca_fit(xt, yt, reg=reg, gamma_x=gamma_x, gamma_y=gamma_y,
                         n_components=1)
            r = kcca_score(xv, yv, m)
            out[i] = r[0] if r.size else np.nan
        except Exception:                            # noqa: BLE001
            continue
    return out


def kcca_score(x: np.ndarray, y: np.ndarray, model: KCCAResult) -> np.ndarray:
    """Project held-out data through a fitted KCCA and correlate the variates.

    Returns the (d,) per-dimension canonical correlation on the supplied data —
    an unbiased out-of-sample estimate (a near-zero value means the nonlinear
    direction does not generalise; signed like :func:`core.cca_score`).
    """
    u = _variates(x, model.x_train, model.gamma_x, model.centerer_x, model.alpha)
    v = _variates(y, model.y_train, model.gamma_y, model.centerer_y, model.beta)
    mask = _finite_rows(u, v)
    u, v = u[mask], v[mask]
    d = model.alpha.shape[1]
    out = np.full(d, np.nan)
    if u.shape[0] < 3:
        return out
    for i in range(d):
        if np.std(u[:, i]) > 0 and np.std(v[:, i]) > 0:
            out[i] = np.corrcoef(u[:, i], v[:, i])[0, 1]
    return out
