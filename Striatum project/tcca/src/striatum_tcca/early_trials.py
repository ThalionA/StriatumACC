"""Very-early-trial readout of the communication subspace (trials < ~10).

A single trial (~500–1150 running bins) is far too short to RE-FIT a 30-component
(p)CCA — that needs ~50 samples/component ≈ 1500 bins — but it is ample to PROJECT
onto a subspace fitted elsewhere and read a per-trial coupling. So for the metrics
that can be read off fixed canonical directions — the dominant canonical
correlation, the information-flow index, and a per-trial *participation* sparsity —
we fit the subspace once on the animal's LATER trials (the sample-rich reference)
and project each early trial through it, **leak-free** (the early trials are held
out of the fit). This gives a true per-trial readout (trial 1 vs 4 vs 7 vs 10).

Fit-only metrics — held-out CC over CV folds, #significant dims, the canonical
weight-Gini, rotation angles, and kernel CCA — cannot be read from a single trial
and are handled by the early-BLOCK path in the driver (≥5-trial blocks) instead.

Pure, testable numerics; the driver (``scripts/run_early_trials.py``) does the I/O
and the area-pair residualisation set-up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import core, lagged, membership, partial


@dataclass
class ReferenceFit:
    """Subspace fitted on the reference (later) trials, with every row projected
    so any held-out early trial can be scored against it."""
    Xr: np.ndarray            # (n_bins, n_units_x) residualised area-X activity, ALL bins
    Yr: np.ndarray            # (n_bins, n_units_y)
    Sx: np.ndarray            # (n_bins, kx) PCA scores, ALL bins (late-fit components)
    Sy: np.ndarray            # (n_bins, ky)
    model: core.CCAResult     # CCA fitted on the reference rows only


def _pca_fit(M: np.ndarray, k: int):
    mu = M.mean(0)
    _, _, vt = np.linalg.svd(M - mu, full_matrices=False)
    return mu, vt[:k].T


def reference_fit(X, Y, train_mask, Z=None, k: int = 30) -> ReferenceFit:
    """Residualise (partial out ``Z``), PCA, and CCA, all fit on ``train_mask``
    rows ONLY, then project EVERY row onto the resulting subspace so held-out
    (early) trials can be read leak-free."""
    train_mask = np.asarray(train_mask, dtype=bool)
    Xr = partial.partial_out_cv(X, Z, train_mask) if Z is not None else np.asarray(X, float)
    Yr = partial.partial_out_cv(Y, Z, train_mask) if Z is not None else np.asarray(Y, float)
    kx = int(min(k, Xr.shape[1]))
    ky = int(min(k, Yr.shape[1]))
    mux, compx = _pca_fit(Xr[train_mask], kx)
    muy, compy = _pca_fit(Yr[train_mask], ky)
    Sx = (Xr - mux) @ compx
    Sy = (Yr - muy) @ compy
    model = core.cca_fit(Sx[train_mask], Sy[train_mask])
    return ReferenceFit(Xr=Xr, Yr=Yr, Sx=Sx, Sy=Sy, model=model)


def _variates(fit: ReferenceFit, bins):
    """Dominant-and-beyond canonical variates of the reference fit on ``bins``."""
    u = (fit.Sx[bins] - fit.model.x_mean) @ fit.model.A
    v = (fit.Sy[bins] - fit.model.y_mean) @ fit.model.B
    return u, v


def trial_cc(fit: ReferenceFit, bins) -> float:
    """Dominant canonical correlation read on one trial's bins (held out of the
    fit). Signed: negative means the reference direction does not generalise."""
    r = core.cca_score(fit.Sx[bins], fit.Sy[bins], fit.model)
    return float(r[0]) if r.size else float("nan")


def trial_ifi(fit: ReferenceFit, bins, max_lag: int = 10):
    """``(ifi, optimal_lag)`` of the dominant canonical variates within one trial.
    Positive lag / IFI ⇒ X leads Y. Lags act within the single trial, so there is
    no trial-boundary leakage."""
    u, v = _variates(fit, bins)
    u, v = u[:, 0], v[:, 0]
    n = u.size
    lags = np.arange(-max_lag, max_lag + 1)
    cc = np.full(lags.size, np.nan)
    for i, L in enumerate(lags):
        if L >= 0:
            a, b = (u[:n - L], v[L:]) if L else (u, v)
        else:
            a, b = u[-L:], v[:n + L]
        if a.size >= 10 and np.std(a) > 0 and np.std(b) > 0:
            cc[i] = np.corrcoef(a, b)[0, 1]
    if not np.any(np.isfinite(cc)):
        return float("nan"), 0
    ifi = lagged.information_flow_index(lags, cc)
    opt = int(lags[int(np.nanargmax(cc))])
    return float(ifi), opt


def _gini_of_corr(M: np.ndarray, variate: np.ndarray) -> float:
    """Gini of each column's |Pearson correlation| with ``variate``."""
    M = np.asarray(M, float)
    variate = np.asarray(variate, float)
    if M.shape[0] < 3 or np.std(variate) == 0:
        return float("nan")
    Mc = M - M.mean(0)
    vc = variate - variate.mean()
    denom = np.sqrt((Mc ** 2).sum(0)) * np.sqrt((vc ** 2).sum())
    corr = np.where(denom > 0, (Mc.T @ vc) / np.where(denom > 0, denom, 1.0), 0.0)
    return membership.gini(np.abs(corr))


def trial_participation_gini(fit: ReferenceFit, bins):
    """Per-trial participation sparsity: the Gini of each neuron's |correlation|
    with the dominant canonical variate, within one trial. A per-trial analogue of
    the weight-Gini — low Gini ⇒ many neurons track the variate (de-sparsified)."""
    u, v = _variates(fit, bins)
    return (_gini_of_corr(fit.Xr[bins], u[:, 0]),
            _gini_of_corr(fit.Yr[bins], v[:, 0]))


def early_trial_metrics(X, Y, trial_ids, early_trials, train_trials, Z=None,
                        k: int = 30, max_lag: int = 10, min_bins: int = 10):
    """Per-early-trial projected readout against a reference subspace fit on
    ``train_trials`` (the later trials). ``X``/``Y`` are (n_bins, n_units) raw
    activity, ``trial_ids`` the per-bin trial id, ``Z`` the optional third-area
    confound. Returns one dict per id in ``early_trials`` (NaN metrics if the trial
    has fewer than ``min_bins`` bins)."""
    trial_ids = np.asarray(trial_ids)
    train_mask = np.isin(trial_ids, np.asarray(train_trials))
    fit = reference_fit(X, Y, train_mask, Z=Z, k=k)
    out = []
    for t in early_trials:
        bins = np.where(trial_ids == t)[0]
        if bins.size < min_bins:
            out.append(dict(trial=t, n_bins=int(bins.size), cc1=float("nan"),
                            ifi=float("nan"), optimal_lag=0,
                            gini_part_x=float("nan"), gini_part_y=float("nan")))
            continue
        ifi, opt = trial_ifi(fit, bins, max_lag)
        gx, gy = trial_participation_gini(fit, bins)
        out.append(dict(trial=t, n_bins=int(bins.size), cc1=trial_cc(fit, bins),
                        ifi=ifi, optimal_lag=opt, gini_part_x=gx, gini_part_y=gy))
    return out
