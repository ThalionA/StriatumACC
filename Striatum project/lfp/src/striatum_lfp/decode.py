"""Position decoding + cross-area CCA on LFP band power (PROVISIONAL, gated).

Position decoding doubles as an ALIGNMENT TEST: LFP band power can only decode
corridor position above chance if the export's time offset is roughly correct.
A positive result therefore validates the (unverified) alignment; a null is
consistent with misalignment OR absent position coding.

Cross-area LFP-LFP CCA is offset-robust (both areas share the file's single
offset) but volume-conduction limited (DMS/DLS/ACC sit on one probe and share a
large common signal). A within-area split-half CCA gives the VC ceiling: only
cross-area coupling *below* that ceiling but *above* the trial-shuffle null is
evidence of area-specific structure rather than shared field.

Everything is scale-relative (undocumented voltage units) and provisional.
"""

from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def bin_by_position(values: np.ndarray, sample_pos: np.ndarray, n_bins: int,
                    pos_max: float) -> np.ndarray:
    """Mean of ``values`` (samples x feat) within each of ``n_bins`` position bins.

    ``sample_pos`` is position (0..pos_max) per sample. Empty bins are NaN.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    edges = np.linspace(0, pos_max, n_bins + 1)
    idx = np.clip(np.digitize(sample_pos, edges) - 1, 0, n_bins - 1)
    out = np.full((n_bins, values.shape[1]), np.nan)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out[b] = np.nanmean(values[m], axis=0)
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


def residualise_by_group(values: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Subtract the per-group mean from each row (e.g. per position bin) ->
    trial-to-trial residual fluctuations (H&H 'communication' definition)."""
    values = np.asarray(values, float)
    out = values.copy()
    for g in np.unique(group):
        m = group == g
        out[m] -= np.nanmean(values[m], axis=0, keepdims=True)
    return out


def heldout_cca(A: np.ndarray, B: np.ndarray, k: int = 5, seed: int = 0) -> float:
    """Held-out top canonical correlation between A (n x p) and B (n x q).

    PCA to ``k`` dims each (fit on a train half), CCA fit on train, correlation of
    the projected *held-out* half. Split-half CV removes the in-sample upward bias.
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    ok = np.all(np.isfinite(A), 1) & np.all(np.isfinite(B), 1)
    A, B = A[ok], B[ok]
    n = len(A)
    if n < 8 or A.shape[1] < 1 or B.shape[1] < 1:
        return np.nan
    perm = np.random.default_rng(seed).permutation(n)
    tr, te = perm[: n // 2], perm[n // 2:]
    kA = int(min(k, A.shape[1], len(tr) - 1))
    kB = int(min(k, B.shape[1], len(tr) - 1))
    if kA < 1 or kB < 1:
        return np.nan
    pA = PCA(n_components=kA).fit(A[tr])
    pB = PCA(n_components=kB).fit(B[tr])
    cca = CCA(n_components=1, max_iter=1000)
    cca.fit(pA.transform(A[tr]), pB.transform(B[tr]))
    u, v = cca.transform(pA.transform(A[te]), pB.transform(B[te]))
    r = np.corrcoef(u[:, 0], v[:, 0])[0, 1]
    return float(abs(r)) if np.isfinite(r) else np.nan
