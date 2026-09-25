"""Decoding, reliability and cross-area CCA on binned LFP band power.

Three questions, each with the guard that makes the answer mean something:

* **Spatial decoding** -- can position be read out of band power? Folds are split
  by *trial*, never by row: adjacent position bins within a trial are one
  continuous stretch of signal, so a row-wise split trains and tests on the same
  seconds of data and reports a number that is mostly autocorrelation.
* **Reliability** -- is a channel's spatial profile the same from trial to trial?
  Split-half over *interleaved* trials, so a slow drift across the session is not
  scored as unreliability, and Spearman-Brown corrected back to the full trial set.
* **Cross-area CCA** -- how much do two areas' band-power profiles share beyond
  what any pairing of trials would? Held-out CC1 against a trial-permutation
  null. DMS, DLS and ACC sit on one shank, so a shared field produces large
  values; this arm cannot separate field from communication. The separation-
  matched distance control (``distance.py``) is the test for that. (A within-area
  split-half "ceiling" was dropped on 2026-09-25: random channel halves split
  same-row pairs, so it measured zero separation and sat at ~1 by construction.)
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


def design_matrix(cube: np.ndarray, trials: np.ndarray | None = None):
    """``(X, bin_index, trial_index)`` from a ``(n_channels, n_bins, n_trials)`` cube.

    Rows are (trial, bin) samples and columns are channels. Samples with any
    non-finite channel are dropped, so ``X`` is complete and the returned labels
    stay aligned to it.
    """
    cube = np.asarray(cube, dtype=float)
    n_ch, n_bins, n_tr = cube.shape
    sel = np.arange(n_tr) if trials is None else np.asarray(trials, int)
    sel = sel[(sel >= 0) & (sel < n_tr)]
    if sel.size == 0:
        return np.empty((0, n_ch)), np.empty(0, int), np.empty(0, int)

    block = cube[:, :, sel]                                   # (ch, bin, trial)
    X = block.transpose(2, 1, 0).reshape(-1, n_ch)            # trial-major, then bin
    bin_index = np.tile(np.arange(n_bins), sel.size)
    trial_index = np.repeat(sel, n_bins)
    keep = np.all(np.isfinite(X), axis=1)
    return X[keep], bin_index[keep], trial_index[keep]


def _interleaved_halves(n: int):
    """Alternating trial indices, so a session-long drift falls in both halves."""
    idx = np.arange(n)
    return idx[0::2], idx[1::2]


def _corr_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r between corresponding rows of two ``(n, m)`` arrays, nan-safe."""
    a = a - np.nanmean(a, axis=1, keepdims=True)
    b = b - np.nanmean(b, axis=1, keepdims=True)
    num = np.nansum(a * b, axis=1)
    den = np.sqrt(np.nansum(a ** 2, axis=1) * np.nansum(b ** 2, axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    r[den == 0] = np.nan
    return r


def split_half_reliability(cube: np.ndarray, *, spearman_brown: bool = True) -> np.ndarray:
    """Per-channel reliability of the spatial profile across trials.

    Averages the ``(n_bins,)`` profile over each interleaved half of the trials,
    correlates the two halves per channel, and (by default) applies the
    Spearman-Brown correction ``2r/(1+r)`` so the number refers to the full trial
    set rather than to half of it.
    """
    cube = np.asarray(cube, dtype=float)
    n_tr = cube.shape[2]
    if n_tr < 2:
        return np.full(cube.shape[0], np.nan)
    a_idx, b_idx = _interleaved_halves(n_tr)
    with np.errstate(invalid="ignore"):
        a = np.nanmean(cube[:, :, a_idx], axis=2)
        b = np.nanmean(cube[:, :, b_idx], axis=2)
    r = _corr_rows(a, b)
    if not spearman_brown:
        return r
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(r > -1, 2 * r / (1 + r), np.nan)


def heldout_cca_grouped(A: np.ndarray, B: np.ndarray, groups: np.ndarray, *,
                        k: int = 5, seed: int = 0, test_size: float = 0.5) -> float:
    """Top canonical correlation between ``A`` and ``B``, evaluated out of sample.

    PCA to ``k`` dimensions and the CCA fit both happen on training trials only,
    and the correlation is measured on held-out *trials* (``GroupShuffleSplit``),
    not held-out rows. In-sample canonical correlations are biased upward, and a
    row-wise split leaves the bias almost intact because rows from one trial are
    highly autocorrelated.
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    groups = np.asarray(groups)
    ok = np.all(np.isfinite(A), 1) & np.all(np.isfinite(B), 1)
    A, B, groups = A[ok], B[ok], groups[ok]
    if A.shape[0] < 20 or np.unique(groups).size < 4:
        return np.nan
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(splitter.split(A, groups=groups))
    kA = int(min(k, A.shape[1], len(tr) - 1))
    kB = int(min(k, B.shape[1], len(tr) - 1))
    if kA < 1 or kB < 1 or len(te) < 5:
        return np.nan
    pA = PCA(n_components=kA).fit(A[tr])
    pB = PCA(n_components=kB).fit(B[tr])
    cca = CCA(n_components=1, max_iter=1000)
    try:
        cca.fit(pA.transform(A[tr]), pB.transform(B[tr]))
        u, v = cca.transform(pA.transform(A[te]), pB.transform(B[te]))
    except Exception:
        return np.nan
    r = np.corrcoef(u[:, 0], v[:, 0])[0, 1]
    return float(abs(r)) if np.isfinite(r) else np.nan


def paired_design(cube_a: np.ndarray, cube_b: np.ndarray, trials: np.ndarray | None = None):
    """``(Xa, Xb, trial_index)`` with rows complete in BOTH areas.

    Built from one stacked cube, so a (trial, bin) row is kept or dropped for
    both areas together and row ``i`` of ``Xa`` and ``Xb`` is always the same
    moment. Building the two design matrices separately and truncating to the
    shorter one misaligns every row after the first bin missing in one area.
    """
    n_a = cube_a.shape[0]
    X, _, groups = design_matrix(np.concatenate([cube_a, cube_b], axis=0), trials)
    return X[:, :n_a], X[:, n_a:], groups


def trial_shuffle_cca_null(cube_a: np.ndarray, cube_b: np.ndarray,
                           trials: np.ndarray | None = None, *,
                           n_shuffles: int = 25, k: int = 5, seed: int = 0) -> np.ndarray:
    """Null distribution: pair ``A``'s trials with a permutation of ``B``'s.

    Permuting whole trials -- rather than shuffling rows i.i.d. -- keeps each
    block's own temporal and spatial autocorrelation intact, so the null asks
    "is this pairing special?" instead of the far easier "is there any structure
    at all?". The permutation is applied to B's trial axis of the CUBE and the
    paired design matrix rebuilt, so bin ``j`` of A's trial is always paired with
    bin ``j`` of B's (shuffled) trial, whatever bins are missing.
    """
    rng = np.random.default_rng(seed)
    sel = np.arange(cube_a.shape[2]) if trials is None else np.asarray(trials, int)
    out = np.full(n_shuffles, np.nan)
    for s in range(n_shuffles):
        b = cube_b.copy()
        b[:, :, sel] = cube_b[:, :, rng.permutation(sel)]
        Xa, Xb, g = paired_design(cube_a, b, sel)
        out[s] = heldout_cca_grouped(Xa, Xb, g, k=k, seed=s)
    return out


def ridge_cv_decode(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    alpha: float = 1.0, n_splits: int = 5):
    """Group k-fold ridge regression (groups = trial ids -> no within-trial leakage).

    Standardises features on each train fold. Returns ``(r2, mae, y_pred)`` where
    r2/mae are cross-validated.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    keep = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y, groups = X[keep], y[keep], groups[keep]
    n_splits = int(min(n_splits, np.unique(groups).size))
    y_pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        model = Ridge(alpha=alpha).fit(sc.transform(X[tr]), y[tr])
        y_pred[te] = model.predict(sc.transform(X[te]))
    ok = np.isfinite(y_pred)
    ss_tot = np.sum((y[ok] - y[ok].mean()) ** 2)
    r2 = 1 - np.sum((y[ok] - y_pred[ok]) ** 2) / ss_tot if ss_tot > 0 else np.nan
    mae = float(np.mean(np.abs(y[ok] - y_pred[ok])))
    return float(r2), mae, y_pred


def circular_shift_targets(y: np.ndarray, groups: np.ndarray, rng) -> np.ndarray:
    """Null targets: circularly shift the position labels within each trial.

    Permuting *which trial* a sample belongs to does nothing here -- every trial
    carries the same 0..49 bin sequence, so a trial permutation leaves ``y``
    bit-identical and the "null" silently re-runs the real decoder. Rotating the
    labels inside each trial by an independent random offset is what actually
    breaks the position-to-power mapping while preserving the sequence structure,
    the fold geometry and the autocorrelation the real decoder benefits from.
    """
    y = np.asarray(y)
    out = y.copy()
    for g in np.unique(groups):
        m = np.flatnonzero(groups == g)
        if m.size < 2:
            continue
        out[m] = np.roll(y[m], int(rng.integers(1, m.size)))
    return out


def residualise_on(cube: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    """Remove the linear component of ``covariate`` from each channel of ``cube``.

    ``cube`` is ``(n_channels, n_bins, n_trials)`` and ``covariate`` is
    ``(n_bins, n_trials)`` -- typically log running speed.

    The slope is estimated from WITHIN-TRIAL variation only (each trial's power
    and speed profiles centred on their own trial means) and then applied to the
    whole covariate. Speed also rises across trials as the animal learns, so a
    slope pooled over all trials credits speed with part of any learning trend
    in power and removes it (pre-2026-09-25 behaviour). Bin-to-bin variation
    within a trial carries the speed-power relation without the trend.
    """
    cube = np.asarray(cube, dtype=float)
    cov = np.asarray(covariate, dtype=float)
    out = np.full_like(cube, np.nan)
    for c in range(cube.shape[0]):
        x = cube[c]
        ok = np.isfinite(x) & np.isfinite(cov)
        # Centre both on the SAME bins of each trial (those finite in both).
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            x_w = x - np.nanmean(np.where(ok, x, np.nan), axis=0, keepdims=True)
            cov_w = cov - np.nanmean(np.where(ok, cov, np.nan), axis=0, keepdims=True)
        denom = np.sum(cov_w[ok] ** 2)
        if ok.sum() < 10 or denom == 0:
            out[c] = x
            continue
        slope = np.sum(x_w[ok] * cov_w[ok]) / denom
        resid = x - slope * (cov - np.nanmean(cov[ok]))
        resid[~ok] = np.nan
        out[c] = resid
    return out


# --- The project's own moving-window reliability -----------------------------
# Ported from batch_triu_corr_mean.m and the stability section of
# IntegratedAll_v1.m:565-630, so the LFP trace is the same statistic on the same
# window as the single-unit one and the two can be put on the same axes.

MOVING_WINDOW_TRIALS = 5        # IntegratedAll_v1.m:565, ProcessStriatumTask.m:901


def batch_triu_corr_mean(cube: np.ndarray) -> np.ndarray:
    """Per-cell mean of the off-diagonal Pearson correlations between trials.

    Faithful port of ``batch_triu_corr_mean.m``: z-score each trial's profile
    across bins (a zero-SD profile gets SD 1 and NaNs become 0, exactly as the
    MATLAB does), form ``Z' Z / (bins - 1)``, and average the strict upper
    triangle. A cell whose pairs are all undefined returns ``nan`` rather than 0.

    ``cube`` is ``(n_cells, n_bins, n_trials_in_window)``.
    """
    cube = np.asarray(cube, dtype=float)
    n_cells, n_bins, w = cube.shape
    if w < 2 or n_bins < 2 or n_cells == 0:
        return np.full(n_cells, np.nan)

    mu = np.nanmean(cube, axis=1, keepdims=True)
    sd = np.nanstd(cube, axis=1, ddof=1, keepdims=True)
    flat = np.all(~np.isfinite(sd) | (sd == 0), axis=(1, 2))
    sd = np.where((sd == 0) | ~np.isfinite(sd), 1.0, sd)
    z = (cube - mu) / sd
    z = np.nan_to_num(z, nan=0.0)

    corr = np.matmul(z.transpose(0, 2, 1), z) / (n_bins - 1)   # (cells, w, w)
    iu = np.triu_indices(w, k=1)
    out = corr[:, iu[0], iu[1]].mean(axis=1)
    out[flat] = np.nan
    return out


def moving_window_indices(t: int, n_trials: int, window_size: int) -> np.ndarray:
    """Trial indices of the centred window at ``t``, clipped at both edges.

    ``max(1, t - half) : min(n, t + half)`` with ``half = floor(w / 2)``, in
    0-based form. Clipping rather than padding means the first and last two
    trials are averaged over 3 or 4 pairs instead of 5 -- noisier, not absent.
    """
    half = window_size // 2
    return np.arange(max(0, t - half), min(n_trials - 1, t + half) + 1)


def moving_window_reliability(cube: np.ndarray, *,
                              window_size: int = MOVING_WINDOW_TRIALS) -> np.ndarray:
    """Reliability at every trial from a centred, edge-clipped trial window.

    Returns ``(n_cells, n_trials)``. At each trial the statistic is the mean
    pairwise correlation between the spatial profiles of the trials in the
    window -- the same quantity the single-unit stability figures plot, on the
    same 5-trial window.
    """
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError(f"window_size must be odd and >= 3; got {window_size}")
    cube = np.asarray(cube, dtype=float)
    n_cells, _, n_trials = cube.shape
    out = np.full((n_cells, n_trials), np.nan)
    for t in range(n_trials):
        idx = moving_window_indices(t, n_trials, window_size)
        if idx.size < 2:
            continue
        out[:, t] = batch_triu_corr_mean(cube[:, :, idx])
    return out


def shuffle_trials(cube: np.ndarray, rng) -> np.ndarray:
    """Permute the trial axis -- the control IntegratedAll_v1.m:620 uses.

    Reliability is computed on a *window* of neighbouring trials, so permuting
    trial order destroys any trend or local similarity while leaving each single
    trial's profile untouched. Whatever the shuffled trace still shows is what
    the metric returns for unrelated trials of this data.
    """
    return cube[:, :, rng.permutation(cube.shape[2])]
