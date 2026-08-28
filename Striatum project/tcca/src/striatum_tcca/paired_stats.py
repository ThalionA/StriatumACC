"""Paired per-animal hypothesis-test helpers for learning-change analyses.

Two-sided Wilcoxon signed-rank on per-animal deltas, plus a Benjamini-Hochberg
FDR mask applied within a test family. Shared by the landmark- and spatial-arm
learning-change drivers so the statistics are defined in exactly one place.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def wilcoxon_signed(deltas) -> tuple[int, float, float, float]:
    """Two-sided Wilcoxon signed-rank on per-animal deltas.

    Returns ``(n, median, statistic, p)`` where ``n`` is the number of finite
    deltas (the sample size displayed by callers). The signed-rank test ignores
    zero deltas, so ``p`` is computed only when **at least 3 non-zero** deltas
    remain — otherwise ``p`` is NaN. Zeros are stripped *before* the scipy call
    so it never silently falls back to an invalid small-sample normal
    approximation (which it does when zeros are present at small n); the
    zero-free array uses the exact distribution.
    """
    arr = np.asarray(deltas, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    med = float(np.median(arr)) if n else float("nan")
    nz = arr[arr != 0.0]                                # signed-rank drops zeros
    if nz.size < 3:                                     # too few non-zero to test
        return (n, med, float("nan"), float("nan"))
    try:
        res = stats.wilcoxon(nz, alternative="two-sided", method="auto")
        return (n, med, float(res.statistic), float(res.pvalue))
    except ValueError:                                  # degenerate
        return (n, med, float("nan"), float("nan"))


def paired_t(deltas) -> tuple[int, float, float, float]:
    """Two-sided one-sample t-test of per-animal deltas vs 0 — i.e. the **paired
    t-test** (a paired two-condition comparison is the one-sample t on the per-animal
    differences). Drop-in for :func:`wilcoxon_signed`: returns ``(n, median, t, p)``
    with ``n`` the number of finite deltas.

    Preferred over signed-rank at the cohort n (~10 animals): the rank test's discrete
    p-floor wastes power and often can't reach p<0.05 at all, whereas the t-test is the
    field convention and more powerful (the normality-of-differences assumption is mild
    here). ``p`` is NaN with <2 finite deltas or zero variance.
    """
    arr = np.asarray(deltas, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    med = float(np.median(arr)) if n else float("nan")
    if n < 2 or np.ptp(arr) == 0.0:                     # need variance to test
        return (n, med, float("nan"), float("nan"))
    res = stats.ttest_1samp(arr, 0.0)
    return (n, med, float(res.statistic), float(res.pvalue))


def welch_t(a, b) -> float:
    """Two-sided Welch (unequal-variance) two-sample t-test p-value — the parametric
    analogue of Mann-Whitney for the unpaired dims-as-n group comparison. NaN if either
    group has <2 finite values."""
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float); b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def fdr_bh(pvals, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR mask. Operates on finite p-values only.

    Non-finite p-values are excluded from the family (they never pass and do not
    count toward ``n``); the correction is over the finite tests only.
    """
    p = np.asarray(pvals, dtype=float)
    finite = np.isfinite(p)
    if not np.any(finite):
        return np.zeros_like(p, dtype=bool)
    p_f = p[finite]
    n = p_f.size
    order = np.argsort(p_f)
    ranked = p_f[order]
    threshold = q * np.arange(1, n + 1) / n
    passed = ranked <= threshold
    cutoff = ranked[np.max(np.where(passed))] if np.any(passed) else -1
    out_f = p_f <= cutoff if cutoff >= 0 else np.zeros(n, dtype=bool)
    out = np.zeros_like(p, dtype=bool)
    out[finite] = out_f
    return out
