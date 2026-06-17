"""Tests for the shared paired-stats helpers (Wilcoxon signed-rank + BH-FDR)."""

from __future__ import annotations

import warnings

import numpy as np

from striatum_tcca import paired_stats


def test_wilcoxon_basic_nonzero():
    # 6 concordant positive deltas -> exact two-sided p = 2/2**6 = 0.03125.
    n, med, _, p = paired_stats.wilcoxon_signed([0.2, 0.3, 0.25, 0.4, 0.35, 0.5])
    assert n == 6
    assert abs(med - 0.325) < 1e-9
    assert np.isfinite(p) and p < 0.05            # consistent positive shift


def test_wilcoxon_reports_finite_n_but_tests_only_nonzero():
    # 5 finite deltas, 3 of them zero -> only 2 non-zero remain -> not testable.
    n, med, _, p = paired_stats.wilcoxon_signed([0.0, 0.0, 0.0, 0.3, -0.1])
    assert n == 5                                  # sample size unchanged
    assert np.isnan(p)                             # <3 non-zero -> no p


def test_wilcoxon_zeros_do_not_trigger_normal_approx_warning():
    # The old code passed zeros to scipy at small n -> a normal-approx fallback
    # with a warning and an invalid p. Stripping zeros first must avoid that and
    # still return a finite exact p when >=3 non-zero deltas remain.
    deltas = [0.0, 0.0, 0.5, 0.4, 0.6, 0.3]        # 4 non-zero, 2 zero
    with warnings.catch_warnings():
        warnings.simplefilter("error")             # any warning -> test fails
        n, _, _, p = paired_stats.wilcoxon_signed(deltas)
    assert n == 6
    assert np.isfinite(p)


def test_wilcoxon_all_zero_is_nan():
    n, med, _, p = paired_stats.wilcoxon_signed([0.0, 0.0, 0.0, 0.0])
    assert n == 4 and med == 0.0 and np.isnan(p)


def test_wilcoxon_too_few_finite_is_nan():
    n, _, _, p = paired_stats.wilcoxon_signed([0.3, np.nan])
    assert n == 1 and np.isnan(p)


def test_fdr_bh_passes_only_below_threshold():
    p = np.array([0.001, 0.01, 0.2, 0.5, np.nan])
    mask = paired_stats.fdr_bh(p, q=0.05)
    assert mask[0] and mask[1]                     # the two small p's pass
    assert not mask[2] and not mask[3]
    assert not mask[4]                             # NaN never passes
