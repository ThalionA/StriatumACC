"""Core numerical primitives: residualisation, PCA, CCA, cross-validation.

Array conventions
-----------------
* A 3-D activity tensor is ``(n_trials, n_bins, n_units)``.
* The samples for CCA are ``(trial, bin)`` pairs, flattened row-major so that
  sample ``t * n_bins + b`` is trial ``t``, bin ``b``.

These functions are pure (no data loading, no I/O) so they can be unit-tested
against synthetic data with known ground truth.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import linalg


# ---------------------------------------------------------------------------
# Missing-bin imputation
# ---------------------------------------------------------------------------
def impute_missing_bins(tensor: np.ndarray) -> np.ndarray:
    """Fill missing (unvisited) spatial-bin entries with the per-(bin, unit) mean.

    Spatial binning leaves NaN where the animal did not occupy a bin on a given
    trial (typically <1-3% of samples). Filling with the per-(bin, unit) mean
    across the trials that *did* visit the bin makes such samples contribute
    zero residual fluctuation after :func:`residualise`. A bin never visited in
    any trial is filled with 0.

    Parameters
    ----------
    tensor : ndarray, shape (n_trials, n_bins, n_units)
    """
    if tensor.ndim != 3:
        raise ValueError(f"expected 3-D (trials, bins, units), got {tensor.shape}")
    if not np.isnan(tensor).any():
        return tensor
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN bin slices
        bin_unit_mean = np.nanmean(tensor, axis=0, keepdims=True)
    bin_unit_mean = np.nan_to_num(bin_unit_mean, nan=0.0)
    return np.where(np.isnan(tensor), bin_unit_mean, tensor)


# ---------------------------------------------------------------------------
# Residualisation (D2)
# ---------------------------------------------------------------------------
def residualise(tensor: np.ndarray) -> np.ndarray:
    """Subtract the per-(bin, unit) mean across trials.

    This removes each unit's trial-averaged spatial tuning, leaving the
    trial-to-trial residual fluctuations ("interaction structure", H&H).

    Parameters
    ----------
    tensor : ndarray, shape (n_trials, n_bins, n_units)

    Returns
    -------
    ndarray, same shape, with zero mean along the trial axis.
    """
    if tensor.ndim != 3:
        raise ValueError(f"expected 3-D (trials, bins, units), got {tensor.shape}")
    return tensor - tensor.mean(axis=0, keepdims=True)


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
@dataclass
class PCAState:
    """A fitted PCA basis for one area."""

    mean: np.ndarray                      # (n_units,) sample mean
    components: np.ndarray                # (n_units, k) columns are PC axes
    explained_variance_ratio: np.ndarray  # (k,)


def _flatten(tensor: np.ndarray) -> np.ndarray:
    """(n_trials, n_bins, f) -> (n_trials * n_bins, f), row-major."""
    n_tr, n_bin, n_f = tensor.shape
    return tensor.reshape(n_tr * n_bin, n_f)


def pca_fit(tensor: np.ndarray, k: int) -> PCAState:
    """Fit PCA on the flattened (trial, bin) samples of one area's tensor.

    Parameters
    ----------
    tensor : ndarray, shape (n_trials, n_bins, n_units)
    k : int
        Number of principal components to keep.
    """
    samples = _flatten(tensor)
    mean = samples.mean(axis=0)
    centred = samples - mean
    _, sv, vt = linalg.svd(centred, full_matrices=False)
    total = float(np.sum(sv ** 2))
    k = int(min(k, vt.shape[0]))
    components = vt[:k].T
    evr = (sv[:k] ** 2) / total if total > 0 else np.zeros(k)
    return PCAState(mean=mean, components=components, explained_variance_ratio=evr)


def pca_transform(tensor: np.ndarray, state: PCAState) -> np.ndarray:
    """Project an activity tensor onto a fitted PCA basis.

    Returns
    -------
    ndarray, shape (n_trials, n_bins, k) of PC scores.
    """
    n_tr, n_bin, _ = tensor.shape
    centred = _flatten(tensor) - state.mean
    scores = centred @ state.components
    return scores.reshape(n_tr, n_bin, state.components.shape[1])


# ---------------------------------------------------------------------------
# Canonical correlation analysis
# ---------------------------------------------------------------------------
@dataclass
class CCAResult:
    """A fitted CCA model."""

    A: np.ndarray       # (p, d) canonical coefficients for X
    B: np.ndarray       # (q, d) canonical coefficients for Y
    r: np.ndarray       # (d,) in-sample canonical correlations, descending
    x_mean: np.ndarray  # (p,)
    y_mean: np.ndarray  # (q,)


def cca_fit(x: np.ndarray, y: np.ndarray) -> CCAResult:
    """Canonical correlation analysis via the QR/SVD (MATLAB ``canoncorr``).

    Inputs are assumed full column rank — guaranteed in this pipeline because
    they are PCA-reduced scores (``pca_fit`` keeps ``k <= rank``).

    Parameters
    ----------
    x : ndarray, shape (n_samples, p)
    y : ndarray, shape (n_samples, q)

    Returns
    -------
    CCAResult with ``d = min(p, q)`` canonical correlations and coefficients
    that produce unit-variance canonical variates.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.shape[0]
    if y.shape[0] != n:
        raise ValueError("x and y must have the same number of samples")
    if n < 3:
        raise ValueError("need at least 3 samples for CCA")

    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    xc = x - x_mean
    yc = y - y_mean

    qx, rx = linalg.qr(xc, mode="economic")
    qy, ry = linalg.qr(yc, mode="economic")

    # Rank guard: a near-zero pivot on the R diagonal means a rank-deficient
    # input, which the simple (non-pivoted) solve below cannot handle.
    _check_full_rank(rx, "x")
    _check_full_rank(ry, "y")

    u, sv, vt = linalg.svd(qx.T @ qy)
    d = min(x.shape[1], y.shape[1])
    r = np.clip(sv[:d], 0.0, 1.0)

    scale = np.sqrt(n - 1)
    a = linalg.solve_triangular(rx, u[:, :d]) * scale
    b = linalg.solve_triangular(ry, vt[:d].T) * scale
    return CCAResult(A=a, B=b, r=r, x_mean=x_mean, y_mean=y_mean)


def _check_full_rank(r_matrix: np.ndarray, name: str) -> None:
    diag = np.abs(np.diag(r_matrix))
    if diag.size == 0 or diag.max() == 0:
        raise ValueError(f"{name} has zero variance")
    if diag.min() / diag.max() < 1e-10:
        raise ValueError(
            f"{name} is rank-deficient; CCA inputs must be full column rank"
        )


def cca_score(x: np.ndarray, y: np.ndarray, model: CCAResult) -> np.ndarray:
    """Project held-out data through a fitted CCA and correlate the variates.

    Returns
    -------
    ndarray, shape (d,) — the correlation of each canonical variate pair on the
    supplied data. On held-out data this is an unbiased estimate of the true
    canonical correlation (signed: a negative value means the canonical
    direction does not generalise).
    """
    u = (np.asarray(x, float) - model.x_mean) @ model.A
    v = (np.asarray(y, float) - model.y_mean) @ model.B
    d = model.A.shape[1]
    out = np.full(d, np.nan)
    for i in range(d):
        if np.std(u[:, i]) > 0 and np.std(v[:, i]) > 0:
            out[i] = np.corrcoef(u[:, i], v[:, i])[0, 1]
    return out


def numerical_rank(matrix: np.ndarray, rel_tol: float = 1e-7) -> int:
    """Numerical rank: count of singular values above ``rel_tol * largest``."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 3:
        matrix = matrix.reshape(-1, matrix.shape[-1])
    if matrix.size == 0:
        return 0
    sv = linalg.svd(matrix, compute_uv=False)
    if sv[0] == 0:
        return 0
    return int(np.sum(sv > rel_tol * sv[0]))


def choose_k(
    n_units_x: int,
    n_units_y: int,
    n_samples: int,
    cfg,
    max_rank: int | None = None,
) -> int:
    """Pick the PC count per area (D4): samples-driven, symmetric, capped.

    ``k = floor(n_samples / samples_per_pc)``, capped by the smaller area's
    unit count, by ``k_cap``, and -- when given -- by ``max_rank`` (the
    per-epoch numerical rank of the residual data, so that every epoch's CCA
    receives full column rank input). Floor of 1.
    """
    caps = [n_samples // cfg.samples_per_pc, n_units_x, n_units_y, cfg.k_cap]
    if max_rank is not None:
        caps.append(max_rank)
    return int(max(1, min(caps)))


# ---------------------------------------------------------------------------
# Cross-validation (D8): 5-fold over whole trials
# ---------------------------------------------------------------------------
@dataclass
class CVResult:
    """Cross-validated CCA for one (animal, pair, epoch)."""

    held_out_r: np.ndarray         # (d,) mean held-out CC over folds
    held_out_r_folds: np.ndarray   # (n_folds, d) per-fold held-out CC
    in_sample_r: np.ndarray        # (d,) full-data in-sample CC (biased high)
    full: CCAResult                # CCA fitted on all samples (A, B kept)
    k: int
    n_samples: int
    samples_per_pc: float


def trial_folds(n_trials: int, n_folds: int, seed: int) -> list[np.ndarray]:
    """Partition trial indices into ``n_folds`` interleaved, balanced folds."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_trials)
    return [perm[i::n_folds] for i in range(n_folds)]


def cca_cv(px: np.ndarray, py: np.ndarray, cfg) -> CVResult:
    """Fit CCA with 5-fold whole-trial cross-validation.

    Folds hold out whole trials (not random samples) so that within-trial
    spatial-bin autocorrelation cannot leak between train and test (D8).

    Parameters
    ----------
    px, py : ndarray, shape (n_trials, n_bins, k)
        PCA-reduced scores for the two areas.
    """
    n_tr, n_bin, kx = px.shape
    ky = py.shape[2]

    full = cca_fit(_flatten(px), _flatten(py))
    d = full.r.shape[0]

    fold_r: list[np.ndarray] = []
    for test_tr in trial_folds(n_tr, cfg.n_folds, cfg.cv_seed):
        train_tr = np.setdiff1d(np.arange(n_tr), test_tr)
        if train_tr.size < 2 or test_tr.size < 1:
            continue
        try:
            model = cca_fit(_flatten(px[train_tr]), _flatten(py[train_tr]))
        except ValueError:
            # A train fold occasionally loses rank (units silent in those
            # trials); skip it rather than poison the held-out average.
            continue
        fold_r.append(cca_score(_flatten(px[test_tr]), _flatten(py[test_tr]), model)[:d])

    folds = np.array(fold_r) if fold_r else np.full((0, d), np.nan)
    held_out = np.nanmean(folds, axis=0) if folds.size else np.full(d, np.nan)
    return CVResult(
        held_out_r=held_out,
        held_out_r_folds=folds,
        in_sample_r=full.r,
        full=full,
        k=min(kx, ky),
        n_samples=n_tr * n_bin,
        samples_per_pc=(n_tr * n_bin) / max(kx, ky),
    )
