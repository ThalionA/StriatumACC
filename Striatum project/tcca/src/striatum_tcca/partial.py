"""Partial CCA (D3 add-on): communication after removing a third area.

For the DMS/DLS/ACC triplet (the only three areas recorded together in enough
animals), partial CCA asks whether each striatal-cingulate pair's communication
survives regressing out the third area's activity. A large drop from plain CC1
to partial CC1 means the pair's apparent coupling is largely explained by the
shared third area.
"""

from __future__ import annotations

import numpy as np

from . import core


def partial_out_cv(target: np.ndarray, confound: np.ndarray,
                   train_mask: np.ndarray) -> np.ndarray:
    """Residual of ``target`` on ``confound`` with coefficients fit on
    ``train_mask`` rows ONLY, then applied to every row.

    This is the leak-free form: when residualising before a cross-validated fit,
    the confound regression must not see the held-out fold, or the "held-out"
    score is optimistic. Coefficients use the finite training rows; the result is
    ``target - confound @ coef`` for all rows (NaN where inputs are NaN).
    """
    fit = (np.asarray(train_mask, dtype=bool)
           & np.all(np.isfinite(target), axis=1)
           & np.all(np.isfinite(confound), axis=1))
    coef, *_ = np.linalg.lstsq(confound[fit], target[fit], rcond=None)
    return target - confound @ coef


def partial_out(target: np.ndarray, confound: np.ndarray) -> np.ndarray:
    """Least-squares residual of ``target`` after regressing out ``confound``,
    with coefficients estimated on ALL finite rows (the in-sample / descriptive
    form). For a leak-free pre-CV residualisation use :func:`partial_out_cv`.
    """
    return partial_out_cv(target, confound,
                          np.ones(target.shape[0], dtype=bool))


def partial_out_tensor(tensor: np.ndarray, confound: np.ndarray) -> np.ndarray:
    """Regress ``confound`` out of a 3-D ``(n_trials, n_bins, n_features)``
    tensor over its flattened (trial, bin) samples; the shape is preserved.

    ``confound`` is ``(n_trials, n_bins, n_confound)`` with the same trial and
    bin axes. Used to partial other areas' activity out of an area's neuron
    tensor before the per-epoch PCA (see pipeline.prepare_pair_partial).
    """
    n_tr, n_bins, n_feat = tensor.shape
    flat = tensor.reshape(n_tr * n_bins, n_feat)
    fz = confound.reshape(n_tr * n_bins, -1)
    return partial_out(flat, fz).reshape(n_tr, n_bins, n_feat)


def partial_cca_cv(
    scores_x: np.ndarray, scores_y: np.ndarray, scores_z: np.ndarray, cfg
):
    """5-fold cross-validated CCA of X and Y after partialling out Z.

    Z is regressed out of both X and Y over the flattened (trial, bin) samples;
    the residuals are reshaped back to (n_trials, n_bins, k) and passed to the
    standard whole-trial cross-validated CCA.

    NOTE: the Z-regression here is fit on the FULL window (in-sample), so the
    held-out CC carries a small optimistic bias. This is the legacy
    landmark/spatial-arm path (run_partial.py). The continuous-regime path
    (subspace_window.window_subspace, Z=...) does the confound regression
    per-fold on training trials only and is leak-free; prefer it for new work.
    """
    n_tr, n_bins, _ = scores_x.shape
    fx = scores_x.reshape(n_tr * n_bins, -1)
    fy = scores_y.reshape(n_tr * n_bins, -1)
    fz = scores_z.reshape(n_tr * n_bins, -1)
    res_x = partial_out(fx, fz).reshape(n_tr, n_bins, -1)
    res_y = partial_out(fy, fz).reshape(n_tr, n_bins, -1)
    return core.cca_cv(res_x, res_y, cfg)
