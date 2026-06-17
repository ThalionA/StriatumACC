"""Spatial-lag CCA and the Information Flow Index (D6).

At lag ``L``, area X's residual at spatial bin ``b`` is paired with area Y's
residual at bin ``b + L``; CCA is refit at every lag. With the animal running
through increasing bin index, a positive ``L`` means X's earlier-position
activity is matched to Y's later-position activity -- i.e. **X leads Y**.

The Information Flow Index summarises the lag curve into one bounded number:
    IFI = (mean CC1 over L>0  -  mean CC1 over L<0) / (their sum)
on held-out CC1 clipped at 0. IFI is in [-1, 1]: +1 = X leads, -1 = Y leads,
0 = symmetric.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import core


def lag_slice(
    x: np.ndarray, y: np.ndarray, lag: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pair ``x[:, b, :]`` with ``y[:, b + lag, :]``, trimming ``|lag|`` bins.

    Parameters
    ----------
    x, y : ndarray, shape (n_trials, n_bins, k)
    lag : int
        Spatial-bin offset. Positive => X leads Y.
    """
    n_bins = x.shape[1]
    if abs(lag) >= n_bins:
        raise ValueError(f"lag {lag} too large for {n_bins} bins")
    if lag >= 0:
        return x[:, : n_bins - lag, :], y[:, lag:, :]
    return x[:, -lag:, :], y[:, : n_bins + lag, :]


def information_flow_index(lags: np.ndarray, cc1: np.ndarray) -> float:
    """(X-leads - Y-leads) / (X-leads + Y-leads); held-out CC1 clipped at 0."""
    pos = np.clip(cc1[lags > 0], 0.0, None)
    neg = np.clip(cc1[lags < 0], 0.0, None)
    pos_mean = np.nanmean(pos) if np.any(np.isfinite(pos)) else 0.0
    neg_mean = np.nanmean(neg) if np.any(np.isfinite(neg)) else 0.0
    total = pos_mean + neg_mean
    if total <= 0:
        return 0.0
    return float((pos_mean - neg_mean) / total)


def ifi_by_window(lags: np.ndarray, cc: np.ndarray) -> np.ndarray:
    """IFI computed over progressively wider lag windows (D6 / point 4).

    Returns an array of length ``max(|lags|)``; entry ``w-1`` is the IFI using
    only lags with ``|lag| <= w``. Shows how the directionality readout depends
    on the integration window.
    """
    max_w = int(np.max(np.abs(lags))) if lags.size else 0
    out = np.full(max_w, np.nan)
    for w in range(1, max_w + 1):
        mask = np.abs(lags) <= w
        out[w - 1] = information_flow_index(lags[mask], cc[mask])
    return out


@dataclass
class LagResult:
    """Lagged-CCA directionality for one (animal, pair, epoch), all canonical
    dimensions."""

    lags: np.ndarray             # (n_lags,) integer bin lags
    cc_per_dim: np.ndarray       # (n_lags, n_dims) CC at each lag, NaN-padded
    ifi_per_dim: np.ndarray      # (n_dims,) IFI over the full lag range
    ifi_windows: np.ndarray      # (n_dims, max_window) IFI by lag window
    peak_lag_per_dim: np.ndarray  # (n_dims,) bin lag of the per-dim CC peak

    # Convenience accessors for the dominant canonical dimension.
    @property
    def cc1(self) -> np.ndarray:
        return self.cc_per_dim[:, 0]

    @property
    def ifi(self) -> float:
        return float(self.ifi_per_dim[0])

    @property
    def peak_lag(self) -> int:
        return int(self.peak_lag_per_dim[0])


def lag_curve(
    scores_x: np.ndarray,
    scores_y: np.ndarray,
    cfg,
    max_lag: int | None = None,
    held_out: bool = False,
) -> LagResult:
    """Refit CCA at every spatial lag and summarise direction, per dimension.

    Parameters
    ----------
    scores_x, scores_y : ndarray, shape (n_trials, n_bins, k)
        PCA-reduced residual scores for the two areas.
    held_out : bool
        If True use 5-fold cross-validated CC at each lag (the honest
        directionality curve); if False use the fast in-sample CC.
    """
    max_lag = cfg.max_lag_bins if max_lag is None else max_lag
    lags = np.arange(-max_lag, max_lag + 1)

    rows = []
    for lag in lags:
        xl, yl = lag_slice(scores_x, scores_y, int(lag))
        if held_out:
            rows.append(core.cca_cv(xl, yl, cfg).held_out_r)
        else:
            rows.append(core.cca_in_sample(xl, yl))

    n_dims = rows[len(rows) // 2].shape[0]          # dims at lag 0
    cc = np.full((lags.size, n_dims), np.nan)
    for i, r in enumerate(rows):
        m = min(n_dims, r.shape[0])
        cc[i, :m] = r[:m]

    ifi_per_dim = np.array(
        [information_flow_index(lags, cc[:, j]) for j in range(n_dims)]
    )
    ifi_windows = np.array([ifi_by_window(lags, cc[:, j]) for j in range(n_dims)])
    peak = np.array([
        int(lags[np.nanargmax(cc[:, j])]) if np.any(np.isfinite(cc[:, j])) else 0
        for j in range(n_dims)
    ])
    return LagResult(
        lags=lags,
        cc_per_dim=cc,
        ifi_per_dim=ifi_per_dim,
        ifi_windows=ifi_windows,
        peak_lag_per_dim=peak,
    )


def _segment_lagged_pairs(Sx, Sy, groups, lag):
    """Segment-aware lag of FLAT continuous scores: within each trial only, pair
    ``Sx`` at bin ``b`` with ``Sy`` at bin ``b+lag`` (positive lag => X leads Y).

    Crucially the pairing never crosses a trial boundary — the ``|lag|`` bins that
    would pair the end of one trial with the start of the next are dropped — so the
    lag curve is not contaminated by the running-bin concatenation (report §2.7
    caveat). Returns ``(Xp, Yp, group_ids)`` or ``None`` if no trial is long enough.
    """
    Xs, Ys, gs = [], [], []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        xt, yt = Sx[idx], Sy[idx]
        n = xt.shape[0]
        if n <= abs(lag) + 2:
            continue
        if lag >= 0:
            xp, yp = xt[: n - lag], yt[lag:]
        else:
            xp, yp = xt[-lag:], yt[: n + lag]
        Xs.append(xp); Ys.append(yp); gs.append(np.full(xp.shape[0], g))
    if not Xs:
        return None
    return np.vstack(Xs), np.vstack(Ys), np.concatenate(gs)


def heldout_lag_curve_flat(Sx, Sy, groups, max_lag, n_folds=5, seed=0):
    """Held-out dominant-dim canonical correlation vs integer bin lag, for the
    flat continuous-regime PCA scores — the honest directionality curve.

    At each lag the (segment-aware) lagged pairs are split into whole-trial folds;
    CCA is fit on the training trials and the dominant canonical correlation is read
    on the held-out trials (averaged over folds). This replaces the in-sample lag
    curve used by ``subspace_window`` (which biases every lag's CC upward).

    ``Sx``/``Sy`` are ``(n_samples, k)`` PCA scores; ``groups`` the per-bin trial id.
    Returns ``(lags, cc_test)`` — feed to :func:`ifi_by_window` for the window sweep.
    """
    lags = np.arange(-max_lag, max_lag + 1)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    folds = [f for f in np.array_split(rng.permutation(uniq), n_folds) if f.size]
    cc = np.full(lags.size, np.nan)
    for i, lag in enumerate(lags):
        pair = _segment_lagged_pairs(Sx, Sy, groups, int(lag))
        if pair is None:
            continue
        Xp, Yp, gp = pair
        rs = []
        for te in folds:
            tem = np.isin(gp, te)
            trm = ~tem
            if trm.sum() < max(5, Xp.shape[1] + 1) or tem.sum() < 3:
                continue
            model = core.cca_fit(Xp[trm], Yp[trm])
            r = core.cca_score(Xp[tem], Yp[tem], model)
            if r.size:
                rs.append(float(r[0]))
        if rs:
            cc[i] = float(np.nanmean(rs))
    return lags, cc
