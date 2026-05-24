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
# Missing-data handling
# ---------------------------------------------------------------------------
# Spatial binning leaves NaN where the animal did not occupy a bin on a given
# trial. At 5 cm bins this is <1-3% of samples; at 2.5 cm bins it reaches ~33%
# for fast animals. Such (trial, bin) samples are *dropped* at every fit (CCA,
# PCA, rank) rather than imputed -- imputing a third of the data would corrupt
# the canonical correlations. The NaN is per-(trial, bin) across all units, so
# the valid-sample mask is shared by both areas of a pair.
def _finite_rows(*matrices: np.ndarray) -> np.ndarray:
    """Boolean mask of rows finite (no NaN/inf) across all given 2-D matrices."""
    mask: np.ndarray | None = None
    for m in matrices:
        ok = np.all(np.isfinite(m), axis=1)
        mask = ok if mask is None else (mask & ok)
    return mask


def n_valid_samples(tensor: np.ndarray) -> int:
    """Count of (trial, bin) samples with no missing units."""
    flat = _flatten(tensor) if tensor.ndim == 3 else tensor
    return int(np.sum(_finite_rows(flat)))


# ---------------------------------------------------------------------------
# Residualisation (D2)
# ---------------------------------------------------------------------------
def residualise(tensor: np.ndarray) -> np.ndarray:
    """Subtract the per-(bin, unit) mean across trials.

    Removes each unit's trial-averaged spatial tuning, leaving the trial-to-
    trial residual fluctuations ("interaction structure", H&H). Missing (NaN)
    samples are preserved as NaN and dropped later, at fit time.

    Parameters
    ----------
    tensor : ndarray, shape (n_trials, n_bins, n_units)
    """
    if tensor.ndim != 3:
        raise ValueError(f"expected 3-D (trials, bins, units), got {tensor.shape}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN bin slices
        mean = np.nanmean(tensor, axis=0, keepdims=True)
    return tensor - mean


def zscore_units(tensor: np.ndarray) -> np.ndarray:
    """Divide each unit by its std over the (trial, bin) samples.

    Re-weights units to unit variance before PCA. The pipeline applies this to
    the raw activity over the whole engaged period, before epoch slicing and
    residualisation (round 7). CCA is scale-invariant, so it only affects the
    PCA step. Missing samples are ignored in the std and preserved as NaN.
    """
    if tensor.ndim != 3:
        raise ValueError(f"expected 3-D (trials, bins, units), got {tensor.shape}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        std = np.nanstd(tensor, axis=(0, 1), keepdims=True)
    return tensor / np.where(std > 0, std, 1.0)


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
    samples = samples[_finite_rows(samples)]          # drop missing (trial,bin)
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


def cca_fit(x: np.ndarray, y: np.ndarray, rank_tol: float = 1e-9) -> CCAResult:
    """Canonical correlation analysis, SVD-based and rank-robust.

    Each population is reduced to an orthonormal basis of its (numerical)
    column space; the canonical correlations are the singular values of the
    cross-product of those bases. This handles rank-deficient inputs gracefully
    (returning ``d = min(rank_x, rank_y)`` canonical correlations) -- which
    arises when bin-slicing for the lagged CCA leaves a PC-score column with
    no variance over the slice.

    Parameters
    ----------
    x : ndarray, shape (n_samples, p)
    y : ndarray, shape (n_samples, q)
    rank_tol : float
        A singular value is treated as non-zero if it exceeds
        ``rank_tol * largest_singular_value``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.shape[0] != x.shape[0]:
        raise ValueError("x and y must have the same number of samples")
    p, q = x.shape[1], y.shape[1]

    # Drop missing (trial, bin) samples -- jointly, so x and y stay aligned.
    mask = _finite_rows(x, y)
    x, y = x[mask], y[mask]
    n = x.shape[0]
    if n < 3:
        raise ValueError("need at least 3 finite paired samples for CCA")
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)

    ux, sx, vxt = linalg.svd(x - x_mean, full_matrices=False)
    uy, sy, vyt = linalg.svd(y - y_mean, full_matrices=False)
    rx = int(np.sum(sx > rank_tol * sx[0])) if sx.size and sx[0] > 0 else 0
    ry = int(np.sum(sy > rank_tol * sy[0])) if sy.size and sy[0] > 0 else 0
    if rx == 0 or ry == 0:
        return CCAResult(
            A=np.zeros((p, 1)), B=np.zeros((q, 1)),
            r=np.zeros(1), x_mean=x_mean, y_mean=y_mean,
        )

    ux, sx, vxt = ux[:, :rx], sx[:rx], vxt[:rx]
    uy, sy, vyt = uy[:, :ry], sy[:ry], vyt[:ry]

    uc, rho, vct = linalg.svd(ux.T @ uy)
    d = min(rx, ry)
    r = np.clip(rho[:d], 0.0, 1.0)

    scale = np.sqrt(n - 1)
    a = (vxt.T @ (uc[:, :d] / sx[:, None])) * scale
    b = (vyt.T @ (vct[:d].T / sy[:, None])) * scale
    return CCAResult(A=a, B=b, r=r, x_mean=x_mean, y_mean=y_mean)


def cca_in_sample(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """In-sample canonical correlations from 3-D (trial, bin, k) score tensors.

    The fast (single-fit, no CV) statistic used for the surrogate permutation
    test — real and shuffles carry the same overfitting bias, so the test is
    valid. Use :func:`cca_cv` for the unbiased held-out effect size.
    """
    return cca_fit(_flatten(px), _flatten(py)).r


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
    mask = _finite_rows(u, v)                      # drop missing (trial,bin)
    u, v = u[mask], v[mask]
    d = model.A.shape[1]
    out = np.full(d, np.nan)
    if u.shape[0] < 3:
        return out
    for i in range(d):
        if np.std(u[:, i]) > 0 and np.std(v[:, i]) > 0:
            out[i] = np.corrcoef(u[:, i], v[:, i])[0, 1]
    return out


def numerical_rank(matrix: np.ndarray, rel_tol: float = 1e-7) -> int:
    """Numerical rank: count of singular values above ``rel_tol * largest``."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 3:
        matrix = matrix.reshape(-1, matrix.shape[-1])
    matrix = matrix[_finite_rows(matrix)]              # drop missing samples
    if matrix.shape[0] < 2:
        return 0
    sv = linalg.svd(matrix, compute_uv=False)
    if sv[0] == 0:
        return 0
    return int(np.sum(sv > rel_tol * sv[0]))


def k_for_variance(data: np.ndarray, threshold: float) -> int:
    """Number of principal components needed to reach ``threshold`` cumulative
    explained-variance ratio.

    ``data`` is 2-D ``(samples, units)`` or 3-D ``(trials, bins, units)``;
    missing (NaN) rows are dropped. Floor of 1.
    """
    samples = data.reshape(-1, data.shape[-1]) if data.ndim == 3 else data
    samples = samples[_finite_rows(samples)]
    if samples.shape[0] < 2:
        return 1
    sv = linalg.svd(samples - samples.mean(axis=0), compute_uv=False)
    total = float(np.sum(sv ** 2))
    if total <= 0:
        return 1
    cumvar = np.cumsum(sv ** 2) / total
    return int(np.searchsorted(cumvar, threshold) + 1)


def choose_k(
    n_units_x: int,
    n_units_y: int,
    n_samples: int,
    cfg,
    max_rank: int | None = None,
    variance_k: int | None = None,
) -> int:
    """Pick the PC count per area (D4): symmetric, capped.

    ``cfg.k_mode`` selects the base count -- "samples" gives
    ``floor(n_samples / samples_per_pc)``, "fixed" gives ``cfg.k_fixed``,
    "variance" uses ``variance_k`` (PCs reaching ``cfg.k_variance`` cumulative
    variance, computed by the caller). The base is capped by the smaller
    area's unit count, by ``k_cap``, and -- when given -- by ``max_rank``
    (per-epoch numerical rank, so every epoch's CCA gets full column rank).
    Floor of 1.
    """
    if cfg.k_mode == "fixed":
        base = cfg.k_fixed
    elif cfg.k_mode == "variance":
        base = variance_k if variance_k is not None else cfg.k_cap
    else:
        base = n_samples // cfg.samples_per_pc
    caps = [base, n_units_x, n_units_y, cfg.k_cap]
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
    n_valid = int(np.sum(_finite_rows(_flatten(px), _flatten(py))))

    fold_r: list[np.ndarray] = []
    for test_tr in trial_folds(n_tr, cfg.n_folds, cfg.cv_seed):
        train_tr = np.setdiff1d(np.arange(n_tr), test_tr)
        if train_tr.size < 2 or test_tr.size < 1:
            continue
        try:
            model = cca_fit(_flatten(px[train_tr]), _flatten(py[train_tr]))
            r_test = cca_score(_flatten(px[test_tr]), _flatten(py[test_tr]), model)
        except ValueError:
            continue                       # too few finite samples in this fold
        # A fold may have fewer canonical dims than the full fit (rank loss);
        # align to the full-fit dimension d, padding short folds with NaN.
        aligned = np.full(d, np.nan)
        aligned[: min(d, r_test.size)] = r_test[: min(d, r_test.size)]
        fold_r.append(aligned)

    folds = np.array(fold_r) if fold_r else np.full((0, d), np.nan)
    held_out = np.nanmean(folds, axis=0) if folds.size else np.full(d, np.nan)
    return CVResult(
        held_out_r=held_out,
        held_out_r_folds=folds,
        in_sample_r=full.r,
        full=full,
        k=min(kx, ky),
        n_samples=n_valid,
        samples_per_pc=n_valid / max(kx, ky),
    )
