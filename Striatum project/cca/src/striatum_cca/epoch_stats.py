"""Parametric epoch-effect statistics: repeated-measures ANOVA and Holm.

scipy already covers the between-subjects one-way ANOVA (``f_oneway``) and
Tukey HSD (``tukey_hsd``); this module adds the two pieces it lacks -- a
one-way repeated-measures ANOVA (epoch as a within-subject factor, used for
the per-animal test) and the Holm-Bonferroni step-down correction for the
repeated-measures post-hoc.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def rm_anova(data: np.ndarray) -> tuple[float, float]:
    """One-way repeated-measures ANOVA for the condition (epoch) effect.

    ``data`` is ``(n_subjects, n_conditions)`` and must be complete -- pass
    complete cases only. Returns ``(F, p)`` for the condition effect:

      * ``(inf, 0.0)``  if there is a condition effect but zero residual
        variance (a perfect additive subject+condition model);
      * ``(nan, nan)``  if there are too few subjects, fewer than two
        conditions, or no variance at all.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("rm_anova expects a 2-D (subjects x conditions) array")
    n, k = data.shape
    if n < 2 or k < 2:
        return np.nan, np.nan
    grand = data.mean()
    subj = data.mean(axis=1)
    cond = data.mean(axis=0)
    ss_total = float(np.sum((data - grand) ** 2))
    ss_subj = k * float(np.sum((subj - grand) ** 2))
    ss_cond = n * float(np.sum((cond - grand) ** 2))
    ss_err = ss_total - ss_subj - ss_cond
    df_cond = k - 1
    df_err = (k - 1) * (n - 1)
    ms_err = ss_err / df_err
    if ms_err <= 0:
        # no residual variance: F is +inf if any condition effect remains.
        return (np.inf, 0.0) if ss_cond > 1e-12 else (np.nan, np.nan)
    f = (ss_cond / df_cond) / ms_err
    p = float(stats.f.sf(f, df_cond, df_err))
    return float(f), p


def holm(pvalues) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


# Epoch contrasts for the 3-epoch design, as (i, j) column pairs into a
# (subjects x 3) matrix: naive-inter, inter-expert, naive-expert.
_CONTRASTS = ((0, 1), (1, 2), (0, 2))


def rm_anova_posthoc(data: np.ndarray) -> dict:
    """One-way RM-ANOVA over the 3 epochs plus Holm-corrected paired-t post-hoc.

    ``data`` is ``(n_subjects, 3)`` and must be complete (drop incomplete cases
    first). Returns a dict with the omnibus ``F``/``p`` (from :func:`rm_anova`),
    the subject count ``n``, and ``posthoc`` -- the Holm-adjusted paired-t
    p-values for (naive-inter, inter-expert, naive-expert), in that order. Cells
    with no variance in a contrast are NaN.
    """
    data = np.asarray(data, dtype=float)
    n = data.shape[0]
    f, p = rm_anova(data)
    posthoc: tuple[float, float, float] = (np.nan, np.nan, np.nan)
    if data.ndim == 2 and data.shape[1] >= 3 and n >= 2:
        raw = np.array([
            stats.ttest_rel(data[:, a], data[:, b]).pvalue
            if np.any(data[:, a] != data[:, b]) else np.nan
            for a, b in _CONTRASTS])
        adj = np.full(3, np.nan)
        finite = np.isfinite(raw)
        if finite.any():
            adj[finite] = holm(raw[finite])
        posthoc = tuple(float(v) for v in adj)
    return {"n": int(n), "F": float(f), "p": float(p), "posthoc": posthoc}


def wilcoxon_vs0(values, min_n: int = 6) -> float:
    """Two-sided one-sample Wilcoxon signed-rank p vs 0 (non-parametric).

    For the per-dimension sampling (n = significant dims). NaN when fewer than
    ``min_n`` finite values or all values are zero (signed-rank is undefined
    below n=6 anyway -- its smallest two-sided p there is 0.0625).
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < min_n or not np.any(v != 0):
        return np.nan
    try:
        return float(stats.wilcoxon(v).pvalue)
    except ValueError:
        return np.nan


def ttest_vs0(values, min_n: int = 2) -> float:
    """Two-sided one-sample t-test p vs 0 (parametric).

    For the per-animal sampling (n = animals, typically 4-7), where the
    signed-rank test cannot reach significance; consistent with the parametric
    RM-ANOVA framing. NaN when fewer than ``min_n`` finite values or no variance.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < min_n or np.ptp(v) == 0:
        return np.nan
    return float(stats.ttest_1samp(v, 0.0).pvalue)


def linear_trend(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares slope of ``y`` on ``x`` and its two-sided p-value.

    Used for the epoch-index linear trend (x = 0/1/2). Returns ``(nan, nan)``
    when the fit is degenerate (fewer than 3 points or no spread in x).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or np.ptp(x) == 0:
        return np.nan, np.nan
    lr = stats.linregress(x, y)
    return float(lr.slope), float(lr.pvalue)
