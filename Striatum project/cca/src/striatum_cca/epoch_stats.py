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
