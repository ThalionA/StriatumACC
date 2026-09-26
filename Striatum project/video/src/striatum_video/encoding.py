"""How much of a unit's firing, per (trial, position bin), does movement explain
beyond position? Cross-validated by trial; all units of a session are fitted at
once because they share the design matrix.

Position model: one-hot position bin (carries each bin's mean rate, including
the average speed profile along the corridor). Full model: position + z-scored
movement covariates, which can only add trial-to-trial deviations.
"""

import numpy as np

RIDGE_ALPHA = 1.0


def _ridge_fit_predict(x_train, y_train, x_test, alpha):
    xtx = x_train.T @ x_train + alpha * np.eye(x_train.shape[1])
    return x_test @ np.linalg.solve(xtx, x_train.T @ y_train)


def _r2(y, pred):
    ss_res = ((y - pred) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1 - ss_res / ss_tot


def cv_delta_r2(fr, bins, movement, trials, n_folds=5, alpha=RIDGE_ALPHA, n_bins=None, nuisance=None):
    """Cross-validated R² of the position and position+movement models.

    fr: (n_rows,) or (n_rows, n_units); bins: position bin per row (0-based);
    movement: (n_rows, n_cov); trials: trial id per row (folds split by trial,
    interleaved so every fold spans the session). Rows must be NaN-free.
    nuisance: optional (n_rows, k) columns added to BOTH models (e.g.
    trial_drift_basis), so slow trends shared by firing and movement are not
    credited to movement.
    Returns (r2_pos, r2_full, {'pos', 'full'} held-out predictions).
    """
    fr = np.asarray(fr, float)
    squeeze = fr.ndim == 1
    y = fr[:, None] if squeeze else fr
    n_bins = int(bins.max()) + 1 if n_bins is None else n_bins
    onehot = np.eye(n_bins)[bins]
    uniq = np.unique(trials)
    fold_of_trial = {t: i % n_folds for i, t in enumerate(uniq)}
    fold = np.array([fold_of_trial[t] for t in trials])
    pred = {"pos": np.empty_like(y), "full": np.empty_like(y)}
    for k in range(n_folds):
        te, tr = fold == k, fold != k
        mu_y = y[tr].mean(0)
        mu_m, sd_m = movement[tr].mean(0), movement[tr].std(0)
        sd_m[sd_m == 0] = 1
        z = (movement - mu_m) / sd_m
        x_pos = onehot if nuisance is None else np.hstack([onehot, nuisance])
        x_full = np.hstack([x_pos, z])
        pred["pos"][te] = mu_y + _ridge_fit_predict(x_pos[tr], y[tr] - mu_y, x_pos[te], alpha)
        pred["full"][te] = mu_y + _ridge_fit_predict(x_full[tr], y[tr] - mu_y, x_full[te], alpha)
    r2_pos, r2_full = _r2(y, pred["pos"]), _r2(y, pred["full"])
    if squeeze:
        return float(r2_pos[0]), float(r2_full[0]), {k: v[:, 0] for k, v in pred.items()}
    return r2_pos, r2_full, pred


def trial_drift_basis(trials, n_basis=5):
    """Smooth functions of trial rank (Gaussian bumps evenly spaced over the
    session), one column each: absorbs slow drift over tens of trials."""
    uniq = np.unique(trials)
    rank = np.searchsorted(uniq, trials) / max(uniq.size - 1, 1)
    centres = np.linspace(0, 1, n_basis)
    width = 1.0 / max(n_basis - 1, 1)
    return np.exp(-0.5 * ((rank[:, None] - centres[None, :]) / width) ** 2)


def circular_shift_trials(movement, bins, trials, shift):
    """Null that keeps each covariate's slow structure: trial i takes the
    movement of trial (i + shift) mod n, bin by bin, over the ordered unique
    trials. Missing (trial, bin) partners keep their own row (conservative)."""
    uniq = np.unique(trials)
    partner = dict(zip(uniq, np.roll(uniq, -shift)))
    index = {(t, b): i for i, (t, b) in enumerate(zip(trials, bins))}
    out = movement.copy()
    for i, (t, b) in enumerate(zip(trials, bins)):
        j = index.get((partner[t], b))
        if j is not None:
            out[i] = movement[j]
    return out
