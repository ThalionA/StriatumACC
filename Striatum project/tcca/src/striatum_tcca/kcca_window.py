"""Full per-cell readout for NONLINEAR (regularised kernel) CCA — the kernel
analogue of :func:`subspace_window.window_subspace`.

Computes, for one window/epoch/condition cell, the same families of readout the
linear pipeline reports, so the two can be compared head to head:

* ``cc_kcca``    -- held-out KCCA canonical correlation per dim (leak-free CV)
* ``cc_linear``  -- held-out LINEAR CC on the SAME folds (the superiority baseline)
* ``n_sig``      -- significant KCCA dims (circular-shift surrogate)
* ``mi_sig``     -- Gaussian-MI surrogate over the significant dims (strength)
* ``ifi`` / ``optimal_lag`` -- direction of flow from a HELD-OUT lagged-KCCA curve
  (held-out, not in-sample: the in-sample kernel correlation saturates at our n)
* ``gini_x/y``   -- subspace sparsity from STRUCTURE COEFFICIENTS (corr of each
  neuron with the nonlinear canonical variate) — the only membership definition
  that ports to KCCA, which has no neuron-space weights.

Compute budget choices (KCCA is O(n^3)): a FIXED ridge ``reg`` (no per-cell ridge
selection -> a fair, optimism-free comparison to linear), bandwidths fixed once
per cell via the median heuristic, and modest surrogate / lag budgets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import core, kernel_cca, lagged, membership, partial
from .subspace_window import _pca_fit, _project

CC_CLIP = 0.999


@dataclass
class KCCAWindow:
    cc_kcca: np.ndarray        # (d,) held-out KCCA per-dim canonical correlation
    cc_linear: np.ndarray      # (d,) held-out LINEAR per-dim CC, same folds
    n_sig: int
    mi_sig: float
    ifi: float
    optimal_lag: int
    gini_x: float              # structure-coefficient Gini, area X
    gini_y: float
    n_units_x: int
    n_units_y: int


def _residualise(A, Z, train):
    return partial.partial_out_cv(A, Z, train) if Z is not None else A


def _heldout(X, Y, Z, groups, reg, gx, gy, kx, ky, dd, n_folds, seed):
    """Leak-free per-fold held-out CC for KCCA and linear CCA (same folds)."""
    uniq = np.unique(groups)
    if uniq.size < n_folds:
        return None, None
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(uniq), n_folds)
    kk, ll = [], []
    for test_g in folds:
        te = np.isin(groups, test_g)
        tr = ~te
        if np.sum(tr) < 5:
            continue
        Xr = _residualise(X, Z, tr); Yr = _residualise(Y, Z, tr)
        mux, cpx = _pca_fit(Xr[tr], kx); muy, cpy = _pca_fit(Yr[tr], ky)
        sx_tr, sx_te = _project(Xr[tr], mux, cpx), _project(Xr[te], mux, cpx)
        sy_tr, sy_te = _project(Yr[tr], muy, cpy), _project(Yr[te], muy, cpy)
        try:
            m = kernel_cca.kcca_fit(sx_tr, sy_tr, reg=reg, gamma_x=gx,
                                    gamma_y=gy, n_components=dd)
            kr = kernel_cca.kcca_score(sx_te, sy_te, m)[:dd]
        except Exception:                            # noqa: BLE001
            continue
        lr = core.cca_score(sx_te, sy_te, core.cca_fit(sx_tr, sy_tr))[:dd]
        # Count a fold toward BOTH means only when both estimators yield dd dims,
        # so the KCCA-vs-linear superiority Δ is averaged over the SAME folds.
        if kr.size == dd and lr.size == dd:
            kk.append(kr); ll.append(lr)
    kcca = np.nanmean(np.array(kk), axis=0) if kk else None
    linr = np.nanmean(np.array(ll), axis=0) if ll else None
    return kcca, linr


def _significance(Sx, Sy, groups, reg, gx, gy, dd, cc_kcca, n_shuffles, seed):
    """Circular-shift null on a single split: 95th-pct shuffled held-out dominant
    KCCA CC; a real per-dim held-out CC counts as significant if it clears it."""
    n = Sx.shape[0]
    if n_shuffles < 1 or cc_kcca is None:
        return np.zeros(dd, dtype=bool)
    rng = np.random.default_rng(seed + 3)
    cut = int(0.7 * n)
    null = np.empty(n_shuffles)
    for s in range(n_shuffles):
        Ys = np.roll(Sy, int(rng.integers(1, n)), axis=0)
        try:
            m = kernel_cca.kcca_fit(Sx[:cut], Ys[:cut], reg=reg, gamma_x=gx,
                                    gamma_y=gy, n_components=1)
            r = kernel_cca.kcca_score(Sx[cut:], Ys[cut:], m)
            null[s] = r[0] if r.size else 0.0
        except Exception:                            # noqa: BLE001
            null[s] = 0.0
    thr = float(np.nanquantile(null, 0.95))
    return np.asarray(cc_kcca) > thr


def kcca_window(X, Y, groups, Z=None, reg: float = 10.0, k: int = 30,
                max_lag: int = 6, n_shuffles: int = 20, n_folds: int = 5,
                d: int = 3, member_q: float = 0.75, seed: int = 0) -> KCCAWindow:
    """Full kernel-CCA readout for one cell. ``X``/``Y`` are raw ``(n, units)``
    activity; ``Z`` is the optional third-area control (partialled out leak-free,
    per fold, for the held-out CC; full-window for the in-sample readouts)."""
    Xr = partial.partial_out(X, Z) if Z is not None else X
    Yr = partial.partial_out(Y, Z) if Z is not None else Y
    kx = int(min(k, Xr.shape[1]))
    ky = int(min(k, Yr.shape[1]))
    mux, cpx = _pca_fit(Xr, kx); Sx = _project(Xr, mux, cpx)
    muy, cpy = _pca_fit(Yr, ky); Sy = _project(Yr, muy, cpy)
    dd = int(min(d, kx, ky))
    gx = kernel_cca.median_gamma(Sx)                 # fixed bandwidths per cell
    gy = kernel_cca.median_gamma(Sy)

    cc_kcca, cc_lin = _heldout(X, Y, Z, groups, reg, gx, gy, kx, ky, dd,
                               n_folds, seed)
    if cc_kcca is None:
        cc_kcca = np.full(dd, np.nan)
    if cc_lin is None:
        cc_lin = np.full(dd, np.nan)

    sig = _significance(Sx, Sy, groups, reg, gx, gy, dd, cc_kcca, n_shuffles, seed)
    n_sig = int(np.sum(sig))
    cc_pos = np.clip(cc_kcca, 0.0, CC_CLIP)
    if n_sig > 0:
        s = cc_pos[sig]; s = s[np.isfinite(s) & (s > 0)]
        mi_sig = float(-0.5 * np.sum(np.log1p(-s ** 2)))
    else:
        mi_sig = 0.0

    # Direction: held-out lagged-KCCA on one 70/30 (whole-group) split.
    lags = np.arange(-max_lag, max_lag + 1)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed + 5)
    test_g = rng.permutation(uniq)[: max(1, int(0.3 * uniq.size))]
    te = np.isin(groups, test_g); tr = ~te
    curve = kernel_cca.kcca_lagged_heldout(Sx[tr], Sy[tr], Sx[te], Sy[te],
                                           reg=reg, lags=lags, gamma_x=gx, gamma_y=gy)
    ifi = lagged.information_flow_index(lags, curve)
    optimal_lag = int(lags[int(np.nanargmax(curve))]) if np.any(np.isfinite(curve)) else 0

    # Membership / sparsity: structure coefficients from the full-window fit.
    gini_x = gini_y = float("nan")
    try:
        model = kernel_cca.kcca_fit(Sx, Sy, reg=reg, gamma_x=gx, gamma_y=gy,
                                    n_components=dd)
        u, v = kernel_cca.kcca_variates(Sx, Sy, model)
        scx = membership.variate_structure_coefficients(Xr, u)
        scy = membership.variate_structure_coefficients(Yr, v)
        gini_x = membership.gini(membership.subspace_contribution(scx))
        gini_y = membership.gini(membership.subspace_contribution(scy))
    except Exception:                                # noqa: BLE001
        pass

    return KCCAWindow(cc_kcca=cc_kcca, cc_linear=cc_lin, n_sig=n_sig,
                      mi_sig=mi_sig, ifi=ifi, optimal_lag=optimal_lag,
                      gini_x=gini_x, gini_y=gini_y,
                      n_units_x=X.shape[1], n_units_y=Y.shape[1])
