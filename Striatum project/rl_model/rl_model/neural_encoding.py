"""Per-neuron encoding analysis — do area neurons encode the RL latents?

Each neuron's **z-scored** spatially-binned firing ``F(trial x bin)`` is
regressed (ridge, trial-wise cross-validated) on the RL latents of interest —
``value``, ``RPE``, ``precision`` — with observed behaviour (lick count,
velocity) as nuisance covariates.  Two models are fit per neuron:

  'beh'         — behaviour controlled; the latents may also claim stationary
                  spatial structure.
  'beh_spatial' — additionally removes the per-bin mean firing, so a latent is
                  credited only for trial-by-trial modulation beyond stationary
                  spatial tuning.

Per latent the readout is its unique cross-validated contribution
``dR2 = R2(full) - R2(full minus that latent)``, with a trial-shuffle null for
per-neuron significance.

Pure NumPy; ridge is solved vectorised over all neurons of a mouse at once.
"""
from __future__ import annotations

import numpy as np
import h5py

LATENTS = ("value", "rpe", "precision")
AREAS = ("acc", "dms", "dls", "v1", "ca1", "dg")


def load_neural(mat_path):
    """Per-mouse z-scored firing tensor and per-cell area label.

    Returns a list of dicts: ``{fr: (n_total_trials, n_bins, n_cells),
    area: (n_cells,) object array, n_cells: int}``.
    """
    out = []
    with h5py.File(mat_path, "r") as f:
        pd = f["preprocessed_data"]
        n = pd["n_trials"].shape[0]
        for i in range(n):
            fr = np.asarray(f[pd["z_spatial_binned_fr_all"][i, 0]], dtype=float)
            n_cells = fr.shape[2]
            area = np.full(n_cells, "none", dtype=object)
            for a in AREAS:
                m = np.asarray(f[pd[f"is_{a}"][i, 0]]).ravel().astype(bool)
                area[m[:n_cells]] = a
            out.append(dict(fr=fr, area=area, n_cells=n_cells))
    return out


# --------------------------------------------------------------------------
# Numerics
# --------------------------------------------------------------------------
def _zscore(M):
    """Z-score each column over rows; constant columns map to zero."""
    sd = M.std(0)
    return (M - M.mean(0)) / np.where(sd > 1e-9, sd, 1.0)


def _demean_global(M):
    return M - M.mean(0)


def _demean_bin(M, bin_of_row, n_bins):
    """Subtract, per spatial bin, that bin's mean over rows (vectorised)."""
    M2 = np.asarray(M, float)
    flat2d = M2.reshape(len(M2), -1)
    bsum = np.zeros((n_bins, flat2d.shape[1]))
    np.add.at(bsum, bin_of_row, flat2d)
    bcnt = np.maximum(np.bincount(bin_of_row, minlength=n_bins), 1)
    out = flat2d - (bsum / bcnt[:, None])[bin_of_row]
    return out.reshape(M2.shape)


def _cv_r2(X, Y, fold_of_row, n_folds, lam):
    """Trial-wise cross-validated R2, vectorised over the columns of Y."""
    pred = np.empty_like(Y)
    eye = lam * np.eye(X.shape[1])
    for fdx in range(n_folds):
        te = fold_of_row == fdx
        tr = ~te
        Xtr = X[tr]
        beta = np.linalg.solve(Xtr.T @ Xtr + eye, Xtr.T @ Y[tr])
        pred[te] = X[te] @ beta
    sse = np.sum((Y - pred) ** 2, axis=0)
    sst = np.sum((Y - Y.mean(0)) ** 2, axis=0)
    return 1.0 - sse / np.maximum(sst, 1e-12)


# --------------------------------------------------------------------------
# Encoding analysis
# --------------------------------------------------------------------------
def _drift_basis(n_trials, n=5):
    """`n` smooth DCT-cosine functions over trials — a basis for slow drift."""
    t = (np.arange(n_trials) + 0.5) / n_trials
    return np.column_stack([np.cos(np.pi * k * t) for k in range(1, n + 1)])


def encode_mouse(fr, latents, licks, velocity, mask,
                 n_folds=5, n_shuffle=50, seed=0, lam=1.0, n_drift=5):
    """Encoding analysis for every neuron of one mouse.

    Nuisance covariates: observed lick count, observed velocity, and a smooth
    trial-drift basis — so a latent is never credited for generic slow session
    drift.  Two models per neuron ('beh', 'beh_spatial'); per latent the unique
    cross-validated dR2 is tested against a *trial-shuffle* null (trial-by-trial
    encoding) and a *bin-shuffle* null (spatial-profile encoding).

    Returns ``{model: {'r2_full': (C,), 'dR2': {lat:(C,)},
                       'pval': {lat:(C,)}, 'pval_bin': {lat:(C,)}}}``.
    """
    T, B, C = fr.shape
    vsel = (mask > 0).ravel()
    bin_of = np.tile(np.arange(B), T)[vsel]
    fold_tr = np.arange(T) % n_folds
    np.random.default_rng(seed).shuffle(fold_tr)
    fold_of = np.repeat(fold_tr, B)[vsel]

    Y0 = fr.reshape(T * B, C)[vsel]
    drift = np.repeat(_drift_basis(T, n_drift), B, axis=0)[vsel]
    nui0 = np.column_stack([licks.reshape(-1)[vsel],
                            velocity.reshape(-1)[vsel], drift])

    results = {}
    for model in ("beh", "beh_spatial"):
        prep = (_demean_global if model == "beh"
                else (lambda M: _demean_bin(M, bin_of, B)))
        Y = prep(Y0.copy())
        nui = _zscore(prep(nui0.copy()))
        Xint = _zscore(prep(np.column_stack(
            [latents[k].reshape(-1)[vsel] for k in LATENTS])))
        full = np.column_stack([Xint, nui])
        r2_full = _cv_r2(full, Y, fold_of, n_folds, lam)

        dR2, pval, pval_bin = {}, {}, {}
        for j, k in enumerate(LATENTS):
            keep = [c for c in range(len(LATENTS)) if c != j]
            reduced = np.column_stack([Xint[:, keep], nui])
            r2_red = _cv_r2(reduced, Y, fold_of, n_folds, lam)
            dR2[k] = r2_full - r2_red

            def _null(seed_n, permute):
                rng = np.random.default_rng(seed_n)
                acc = np.empty((n_shuffle, C))
                for s in range(n_shuffle):
                    col = _zscore(prep(permute(latents[k], rng)
                                       .reshape(-1)[vsel].reshape(-1, 1)))[:, 0]
                    Xsh = Xint.copy()
                    Xsh[:, j] = col
                    acc[s] = _cv_r2(np.column_stack([Xsh, nui]), Y,
                                    fold_of, n_folds, lam) - r2_red
                return (np.sum(acc >= dR2[k][None, :], axis=0) + 1.0) \
                    / (n_shuffle + 1.0)

            pval[k] = _null(seed + 101 * (j + 1),
                            lambda L, r: L[r.permutation(T)])
            pval_bin[k] = _null(seed + 211 * (j + 1),
                                lambda L, r: L[:, r.permutation(B)])
        results[model] = dict(r2_full=r2_full, dR2=dR2,
                              pval=pval, pval_bin=pval_bin)
    return results
