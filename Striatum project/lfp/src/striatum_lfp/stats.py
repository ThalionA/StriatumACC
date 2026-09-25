"""Across-animal tests: exact permutation tests, their floors, and BH-FDR.

The animal is the unit of inference everywhere in this project, so n is small
(3-16) and parametric tests are the wrong tool: Welch at 4 vs 3 animals returned
p = 0.0001 for a perfectly separated sample whose exact permutation p is 0.057,
and a one-sample t at n = 3 passed FDR. Exact permutation tests have an honest
floor, and :func:`can_reach` says when a comparison cannot reach 0.05 at all, so
a table can mark it rather than report an unreachable test as a null.

* :func:`sign_flip_test` -- one-sample / paired (e.g. Expert - Naive per animal).
* :func:`permutation_test_two_sample` -- two groups of animals (task vs control).

Both enumerate every relabelling up to ``EXACT_LIMIT`` and fall back to Monte
Carlo beyond it. Two-sided, on the mean.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np

EXACT_LIMIT = 2 ** 16
ALPHA = 0.05


def _finite(x) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    return x[np.isfinite(x)]


def sign_flip_floor(n: int) -> float:
    """Smallest two-sided p a sign-flip test on ``n`` animals can return."""
    return 2.0 / 2 ** n if n > 0 else np.nan


def two_sample_floor(n1: int, n2: int) -> float:
    """Smallest two-sided p a two-sample permutation test can return.

    With equal groups the observed labelling has a mirror image with the same
    |difference|, so the floor doubles.
    """
    if n1 < 1 or n2 < 1:
        return np.nan
    return (2.0 if n1 == n2 else 1.0) / comb(n1 + n2, n1)


def can_reach(floor: float, alpha: float = ALPHA) -> bool:
    """Whether a test with this floor can reject at ``alpha`` at all."""
    return bool(np.isfinite(floor) and floor < alpha)


def sign_flip_test(x, *, n_resamples: int = 100_000, seed: int = 0) -> float:
    """Two-sided p that the mean of ``x`` is zero, by flipping signs (NaNs dropped)."""
    x = _finite(x)
    n = x.size
    if n == 0:
        return np.nan
    observed = abs(x.mean())
    tol = 1e-12 * max(1.0, observed)
    if 2 ** n <= EXACT_LIMIT:
        signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
        means = np.abs(signs @ x) / n
        return float(np.mean(means >= observed - tol))
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n_resamples, n))
    means = np.abs(signs @ x) / n
    return float((np.sum(means >= observed - tol) + 1) / (n_resamples + 1))


def permutation_test_two_sample(a, b, *, n_resamples: int = 100_000, seed: int = 0) -> float:
    """Two-sided p that two groups of animals share a mean (NaNs dropped)."""
    a, b = _finite(a), _finite(b)
    n1, n2 = a.size, b.size
    if n1 == 0 or n2 == 0:
        return np.nan
    pooled = np.concatenate([a, b])
    total = pooled.sum()
    observed = abs(a.mean() - b.mean())
    tol = 1e-12 * max(1.0, observed)
    n = n1 + n2
    if comb(n, n1) <= EXACT_LIMIT:
        sums = np.array([pooled[list(idx)].sum() for idx in combinations(range(n), n1)])
    else:
        rng = np.random.default_rng(seed)
        sums = np.array([pooled[rng.permutation(n)[:n1]].sum() for _ in range(n_resamples)])
    diffs = np.abs(sums / n1 - (total - sums) / n2)
    hits = np.sum(diffs >= observed - tol)
    if comb(n, n1) <= EXACT_LIMIT:
        return float(hits / sums.size)
    return float((hits + 1) / (n_resamples + 1))


def fdr_bh(pvalues: np.ndarray, q: float = ALPHA):
    """Benjamini-Hochberg adjusted p-values and the reject mask at level ``q``.

    The project's standard correction (``fdr_correct.m``); a family is declared
    by the caller before the run. NaN entries are ignored and never rejected.
    """
    p = np.asarray(pvalues, dtype=float)
    ok = np.isfinite(p)
    adjusted = np.full(p.shape, np.nan)
    if not ok.any():
        return adjusted, np.zeros(p.shape, bool)
    vals = p[ok]
    order = np.argsort(vals)
    m = vals.size
    ranked = vals[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.clip(ranked, 0, 1)
    adjusted[ok] = adj
    return adjusted, np.nan_to_num(adjusted, nan=1.0) <= q
