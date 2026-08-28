"""Tests for the shared paired-stats helpers: Wilcoxon signed-rank, the paired
and Welch t-tests, and the BH-FDR mask."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import stats

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


# ---------------------------------------------------------------------------
# paired_t / welch_t — the parametric siblings (added 2026-08-28)
# ---------------------------------------------------------------------------
def test_paired_t_matches_the_closed_form_on_a_hand_worked_case():
    # deltas 1..5: mean 3, sd(ddof=1) = sqrt(2.5), se = sqrt(2.5/5), t = 3/se
    n, med, t, p = paired_stats.paired_t([1.0, 2.0, 3.0, 4.0, 5.0])
    assert n == 5
    assert med == pytest.approx(3.0)
    assert t == pytest.approx(3.0 / np.sqrt(2.5 / 5), rel=1e-12)
    assert 0.0 < p < 0.05


def test_paired_t_ignores_non_finite_deltas():
    n, med, t, p = paired_stats.paired_t([1.0, np.nan, 2.0, np.inf, 3.0])
    assert n == 3
    assert med == pytest.approx(2.0)
    assert np.isfinite(t) and np.isfinite(p)


def test_paired_t_is_nan_without_enough_data_or_variance():
    for deltas in ([], [1.0], [2.0, 2.0, 2.0]):          # n<2, or zero variance
        _, _, t, p = paired_stats.paired_t(deltas)
        assert np.isnan(t) and np.isnan(p)
    assert paired_stats.paired_t([])[0] == 0


def test_paired_t_finds_no_effect_in_symmetric_noise():
    rng = np.random.default_rng(0)
    _, _, _, p = paired_stats.paired_t(rng.standard_normal(40))
    assert p > 0.05


def test_paired_t_beats_the_signed_rank_p_floor_at_cohort_n():
    """The reason for the port: at n ~ 10 the exact signed-rank test cannot reach a
    small p, while the t-test can. Same deltas, consistent positive effect."""
    deltas = np.array([0.11, 0.09, 0.14, 0.07, 0.12, 0.10, 0.13, 0.08, 0.15, 0.06])
    _, _, _, p_t = paired_stats.paired_t(deltas)
    _, _, _, p_w = paired_stats.wilcoxon_signed(deltas)
    # all ten deltas positive => the exact two-sided signed-rank p sits ON its floor
    assert p_w == pytest.approx(2.0 / 2 ** deltas.size)     # 0.001953125 at n = 10
    assert p_t < p_w / 100                                  # the t-test is unbounded below


def test_welch_t_is_symmetric_and_separates_shifted_groups():
    rng = np.random.default_rng(1)
    a = rng.standard_normal(30)
    b = rng.standard_normal(30) + 2.0
    p = paired_stats.welch_t(a, b)
    assert p < 1e-6
    assert p == pytest.approx(paired_stats.welch_t(b, a))


def test_welch_t_finds_no_difference_between_like_samples():
    rng = np.random.default_rng(2)
    assert paired_stats.welch_t(rng.standard_normal(50), rng.standard_normal(50)) > 0.05


def test_welch_t_is_nan_with_fewer_than_two_finite_values():
    assert np.isnan(paired_stats.welch_t([1.0], [1.0, 2.0, 3.0]))
    assert np.isnan(paired_stats.welch_t([np.nan, 1.0], [1.0, 2.0, 3.0]))
    assert np.isnan(paired_stats.welch_t([], []))


def test_welch_t_does_not_assume_equal_variance():
    """Unbalanced n with very unequal spread — the Welch p must differ from the
    pooled-variance (Student) p, otherwise the wrong test is being called."""
    a = np.array([0.0, 0.1, -0.1, 0.05, -0.05])                  # tight, n=5
    b = np.array([1.0, -8.0, 9.0, -7.0, 6.0, -5.0, 4.0, 12.0])   # wide, n=8
    welch = paired_stats.welch_t(a, b)
    student = float(stats.ttest_ind(a, b, equal_var=True).pvalue)
    assert not np.isclose(welch, student, rtol=1e-3)
