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


def partial_out(target: np.ndarray, confound: np.ndarray) -> np.ndarray:
    """Least-squares residual of ``target`` after regressing out ``confound``.

    Both are ``(n_samples, n_features)``; the regression is over samples. The
    coefficients are estimated on the finite (non-missing) rows only; missing
    rows are returned as NaN and dropped later, at fit time.
    """
    valid = (np.all(np.isfinite(target), axis=1)
             & np.all(np.isfinite(confound), axis=1))
    coef, *_ = np.linalg.lstsq(confound[valid], target[valid], rcond=None)
    return target - confound @ coef


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

    Z is regressed out of X and Y INSIDE each fold (fitted on the training
    trials only; core.cca_cv zx=). Partialling over all samples first let the
    test trials shape the residualising projection and inflated held-out CC
    (tests/test_foldwise_partial.py).
    """
    return core.cca_cv(scores_x, scores_y, cfg, zx=scores_z)
