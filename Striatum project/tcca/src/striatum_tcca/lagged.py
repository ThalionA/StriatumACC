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

from . import core, lagpairs
from .paired_stats import fdr_bh  # single BH implementation (re-exported; noqa: F401)


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


def ifi_sides(lags: np.ndarray, cc1: np.ndarray) -> tuple[float, float]:
    """The two halves of the IFI: mean clipped CC over positive lags (X leads) and over
    negative lags (Y leads). Lag 0 excluded; CCs clipped at 0; a side with no finite
    value contributes 0.0. Exposed so callers can flag the *degenerate* case
    (both sides 0 -> IFI 0.0 means "no coupling either way", not "balanced") without
    re-deriving the clipping rule."""
    lags = np.asarray(lags)
    cc1 = np.asarray(cc1, dtype=float)
    pos = np.clip(cc1[lags > 0], 0.0, None)
    neg = np.clip(cc1[lags < 0], 0.0, None)
    pos_mean = np.nanmean(pos) if np.any(np.isfinite(pos)) else 0.0
    neg_mean = np.nanmean(neg) if np.any(np.isfinite(neg)) else 0.0
    return float(pos_mean), float(neg_mean)


def information_flow_index(lags: np.ndarray, cc1: np.ndarray) -> float:
    """(X-leads - Y-leads) / (X-leads + Y-leads); held-out CC1 clipped at 0.
    Returns 0.0 when both sides clip to zero — see :func:`ifi_sides` to tell that
    apart from a genuinely balanced curve."""
    pos_mean, neg_mean = ifi_sides(lags, cc1)
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
    ix, iy = lagpairs.lag_pair_indices(groups, lag)
    if ix.size == 0:
        return None
    return Sx[ix], Sy[iy], np.asarray(groups)[ix]


def heldout_lag_curve_flat_perdim(Sx, Sy, groups, max_lag, n_dims=1, n_folds=5, seed=0):
    """Held-out PER-DIMENSION canonical correlation vs integer bin lag, for the flat
    continuous-regime PCA scores — the honest directionality curve for each of the
    leading ``n_dims`` canonical dimensions.

    At each lag the (segment-aware) lagged pairs are split into whole-trial folds;
    CCA is fit on the training trials and the per-dim canonical correlations are read
    on the held-out trials (averaged over folds). This is the leak-aware analogue of
    ``subspace_window._lag_curve_perdim`` (which is in-sample and biases every lag up).

    ``Sx``/``Sy`` are ``(n_samples, k)`` PCA scores; ``groups`` the per-bin trial id.
    Returns ``(lags, cc)`` with ``cc`` shape ``(n_lags, n_dims)`` (NaN where a lag
    has too few segment-aware pairs for any fold).
    """
    lags = np.arange(-max_lag, max_lag + 1)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    folds = [f for f in np.array_split(rng.permutation(uniq), n_folds) if f.size]
    cc = np.full((lags.size, n_dims), np.nan)
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
                rs.append(np.asarray(r, dtype=float))
        if rs:
            m = min(n_dims, min(r.size for r in rs))
            cc[i, :m] = np.nanmean([r[:m] for r in rs], axis=0)
    return lags, cc


def heldout_lag_curve_flat(Sx, Sy, groups, max_lag, n_folds=5, seed=0):
    """Held-out dominant-dim (CC1) canonical correlation vs lag — the d=0 slice of
    :func:`heldout_lag_curve_flat_perdim`. ``Sx``/``Sy`` are ``(n_samples, k)`` PCA
    scores; returns ``(lags, cc_test)`` — feed to :func:`ifi_by_window` for the sweep.
    """
    lags, cc = heldout_lag_curve_flat_perdim(
        Sx, Sy, groups, max_lag, n_dims=1, n_folds=n_folds, seed=seed)
    return lags, cc[:, 0]


@dataclass
class PerDimSignificance:
    """Per-dimension significance of a held-out canonical spectrum."""

    mask: np.ndarray        # (d,) bool — significant after correction
    p: np.ndarray           # (d,) empirical p-values
    threshold: np.ndarray   # (d,) null quantile at alpha, per dim
    null_mode: str          # "perdim" | "dominant"


def _heldout_perdim_at_zero(Sx, Sy, groups, n_dims, n_folds, seed):
    """Held-out per-dim CC at lag 0 only — the observed statistic's null counterpart."""
    _, cc = heldout_lag_curve_flat_perdim(Sx, Sy, groups, max_lag=0, n_dims=n_dims,
                                          n_folds=n_folds, seed=seed)
    return np.asarray(cc[0], dtype=float)


def perdim_significance(Sx, Sy, cc_heldout, groups=None, n_shuffles: int = 100,
                        alpha: float = 0.05, seed: int = 0, n_folds: int = 5,
                        null_mode: str = "perdim", correct: str = "fdr",
                        fdr_dims: int | None = None) -> PerDimSignificance:
    """Significance of each canonical dimension against a circular-shift null.

    Computed on **the scores that produced the lag curve**, thresholding **that curve's
    own lag-0 held-out CC**, so the flag and the curve describe the same dimension.
    `run_lag_curves` used to import this mask from a separate `window_subspace` fit and
    attach it by bare index (fixed 2026-08-06).

    ``null_mode``:
      * ``"perdim"`` (default) — dimension *j* is compared to the shuffled distribution
        of **dimension j**, computed HELD-OUT with the same fold structure as the
        observed value. Like-for-like.
      * ``"dominant"`` — every dimension is compared to the shuffled distribution of the
        **dominant** dimension, evaluated IN-SAMPLE. The original test of record. It is
        doubly conservative: the dominant dim is the hardest bar, and an in-sample
        shuffled r is inflated relative to the held-out observed r it gates. On this
        data it left a mean of 0.7 significant dims per cell.

    Testing many dimensions per cell needs a correction, which the dominant-dim null
    supplied implicitly by being a max-statistic; ``correct="fdr"`` (Benjamini-Hochberg)
    supplies it explicitly for the per-dim null. Pass ``correct=None`` for uncorrected.

    ⚠ **The empirical p-value floor limits what FDR can reject.** A permutation p cannot
    go below ``1/(n_shuffles+1)``, while BH needs the best of ``d`` tests to reach
    ``alpha/d``. With d = 30 that demands **more than 599 shuffles before any dimension
    can pass at all** — otherwise the corrected mask is empty for arithmetic reasons and
    not for scientific ones. ``fdr_dims`` therefore restricts the BH family to the
    leading dimensions (the tail sits at the held-out CC floor and only burns power):
    with ``fdr_dims=10`` and 200 shuffles the floor is 0.00498 against a rank-1
    threshold of 0.005, which is feasible.

    ⚠ Under ``"perdim"`` the mask is **not** monotone in ``cc_heldout`` — each dimension
    has its own threshold, and shuffled correlations fall with rank, so a smaller CC at
    high rank can legitimately pass where a larger one at low rank does not. Only the
    ``"dominant"`` mode (one scalar threshold) is monotone.
    """
    cc = np.asarray(cc_heldout, dtype=float)
    d = cc.size
    if n_shuffles < 1 or d == 0 or Sx.shape[0] == 0:
        return PerDimSignificance(np.zeros(d, dtype=bool), np.ones(d),
                                  np.full(d, np.nan), null_mode)
    rng = np.random.default_rng(seed + 7)
    n = Sx.shape[0]
    null = np.full((n_shuffles, d), np.nan)
    for s in range(n_shuffles):
        shift = int(rng.integers(1, max(2, n)))
        Ys = np.roll(Sy, shift, axis=0)
        try:
            if null_mode == "perdim":
                if groups is None:
                    raise ValueError("perdim null needs `groups` for held-out folds")
                rs = _heldout_perdim_at_zero(Sx, Ys, groups, d, n_folds, seed)
            else:
                model = core.cca_fit(Sx, Ys)
                rs = np.asarray(core.cca_score(Sx, Ys, model), dtype=float)
        except Exception:                                   # noqa: BLE001
            rs = np.array([])
        if rs.size:
            m = min(d, rs.size)
            null[s, :m] = rs[:m]
    if null_mode == "dominant":
        top = np.nan_to_num(null[:, 0], nan=0.0)
        null = np.repeat(top[:, None], d, axis=1)
    obs = np.nan_to_num(cc, nan=-np.inf)
    # +1 correction so an empirical p is never exactly zero
    ge = np.nansum(null >= obs[None, :], axis=0)
    valid = np.sum(np.isfinite(null), axis=0)
    p = (1.0 + ge) / (1.0 + np.maximum(valid, 1))
    p = np.where(np.isfinite(cc), p, 1.0)
    thr = np.array([np.nanquantile(null[:, j], 1 - alpha)
                    if np.any(np.isfinite(null[:, j])) else np.nan for j in range(d)])
    if correct == "fdr":
        fam = np.zeros(d, dtype=bool)
        fam[: (d if fdr_dims is None else min(d, fdr_dims))] = True
        mask = np.zeros(d, dtype=bool)
        mask[fam] = fdr_bh(p[fam], alpha)
    else:
        mask = p < alpha
    mask = mask & np.isfinite(cc)
    return PerDimSignificance(np.asarray(mask, dtype=bool), p, thr, null_mode)
