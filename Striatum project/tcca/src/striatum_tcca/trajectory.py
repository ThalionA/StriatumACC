"""Frame A: within-animal learning trajectory of the communication subspace.

The reference papers fit subspaces on a long continuous span and use the
*session* as the unit; Tom has one session per animal, so the power has to come
from *within* the animal. Here we slide a window over the animal's trials, fit a
(p)CCA on the continuous running bins inside each window (at the
sample-rich regime that `prototype_continuous_pcca.py` validated), read a
subspace metric per window, and regress it on learning progress (window-centre
trial fraction). Each animal yields a slope; the across-animal test is then a
sign-consistency test on those slopes -- a powered alternative to the
underpowered 3-epoch contrast (`STATE.md` §3, `OPPORTUNITIES.md` §4).

This module holds the pure, testable numerics. The driver
(`scripts/run_trajectory.py`) does the I/O and pCCA residualisation.
"""

from __future__ import annotations

import numpy as np

from . import core


def sliding_windows(unique_trials, window: int, step: int,
                    min_window: int | None = None) -> list[np.ndarray]:
    """Sliding windows over a sorted array of trial ids.

    Returns a list of trial-id arrays. A trailing partial window is kept only if
    it still holds at least ``min_window`` trials (default ``window``), so every
    returned window is comparable in size. Empty list if there are too few
    trials for even one window.
    """
    trials = np.asarray(sorted(unique_trials))
    n = trials.size
    if min_window is None:
        min_window = window
    if n < min_window:
        return []
    out = []
    start = 0
    while start < n:
        chunk = trials[start:start + window]
        if chunk.size < min_window:
            break
        out.append(chunk)
        if start + window >= n:
            break
        start += step
    return out


def _pca_project(train: np.ndarray, test: np.ndarray, k: int):
    mu = train.mean(0)
    _, _, vt = np.linalg.svd(train - mu, full_matrices=False)
    comp = vt[:k].T
    return (train - mu) @ comp, (test - mu) @ comp


def heldout_cc1(X: np.ndarray, Y: np.ndarray, group_ids: np.ndarray,
                k: int = 30, n_folds: int = 5, seed: int = 0
                ) -> tuple[float, float]:
    """Whole-group cross-validated dominant held-out canonical correlation.

    Folds hold out whole groups (trials) so within-trial autocorrelation cannot
    leak between train and test. Each fold PCA-reduces to ``k`` components per
    area then fits CCA on the train scores and scores the held-out group.
    Returns ``(cc1_test, cc1_train)`` averaged over folds; ``(nan, nan)`` if
    there are fewer groups than folds.
    """
    groups = np.asarray(group_ids)
    uniq = np.unique(groups)
    if uniq.size < n_folds:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(uniq), n_folds)
    cc_test, cc_train = [], []
    for test_g in folds:
        te = np.isin(groups, test_g)
        tr = ~te
        kk = int(min(k, X.shape[1], Y.shape[1], np.sum(tr) - 1))
        if kk < 2:
            continue
        sx_tr, sx_te = _pca_project(X[tr], X[te], kk)
        sy_tr, sy_te = _pca_project(Y[tr], Y[te], kk)
        model = core.cca_fit(sx_tr, sy_tr)
        r_te = core.cca_score(sx_te, sy_te, model)
        r_tr = core.cca_score(sx_tr, sy_tr, model)
        if r_te.size:
            cc_test.append(float(r_te[0]))
            cc_train.append(float(r_tr[0]))
    if not cc_test:
        return float("nan"), float("nan")
    return float(np.nanmean(cc_test)), float(np.nanmean(cc_train))


def linear_slope(x, y) -> tuple[float, float]:
    """Least-squares slope of ``y`` on ``x`` and the Pearson r.

    NaNs in either array are dropped pairwise. Returns ``(nan, nan)`` with fewer
    than 2 finite points or zero x-variance.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2 or np.std(x) == 0:
        return float("nan"), float("nan")
    slope = float(np.polyfit(x, y, 1)[0])
    r = float(np.corrcoef(x, y)[0, 1]) if np.std(y) > 0 else float("nan")
    return slope, r
