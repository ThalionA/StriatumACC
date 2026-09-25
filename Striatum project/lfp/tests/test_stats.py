"""Exact permutation tests with the animal as the unit, and their small-n floors.

Every across-animal test in lfp/ and infotheory/ goes through these. Known
answers: the exact p of a fully one-sided sample is the enumeration floor, a
symmetric null gives a uniform p, and the floor says when 0.05 is unreachable.
"""

from math import comb

import numpy as np
import pytest

from striatum_lfp import stats


def test_sign_flip_all_positive_hits_the_enumeration_floor():
    assert stats.sign_flip_test(np.arange(1.0, 7.0)) == pytest.approx(2 / 2**6)
    assert stats.sign_flip_test(np.arange(1.0, 5.0)) == pytest.approx(2 / 2**4)


def test_sign_flip_floor_says_when_significance_is_impossible():
    assert stats.sign_flip_floor(4) == pytest.approx(0.125)
    assert stats.sign_flip_floor(5) == pytest.approx(0.0625)
    assert stats.sign_flip_floor(6) == pytest.approx(0.03125)
    assert not stats.can_reach(stats.sign_flip_floor(5))
    assert stats.can_reach(stats.sign_flip_floor(6))


def test_sign_flip_is_calibrated_under_a_symmetric_null():
    rng = np.random.default_rng(0)
    ps = np.array([stats.sign_flip_test(rng.standard_normal(10)) for _ in range(400)])
    assert 0.03 < np.mean(ps < 0.05) < 0.08
    assert 0.4 < np.mean(ps) < 0.6


def test_two_sample_complete_separation_hits_the_floor():
    p = stats.permutation_test_two_sample(np.array([10.0, 11, 12]), np.array([1.0, 2, 3]))
    assert p == pytest.approx(2 / comb(6, 3))       # the labelling and its mirror
    assert stats.two_sample_floor(3, 3) == pytest.approx(2 / comb(6, 3))
    assert not stats.can_reach(stats.two_sample_floor(3, 3))


def test_two_sample_unequal_groups_floor():
    # 12 task vs 5 control: plenty of room below 0.05
    assert stats.two_sample_floor(12, 5) == pytest.approx(1 / comb(17, 5))
    a = np.arange(12.0) + 100
    b = np.arange(5.0)
    assert stats.permutation_test_two_sample(a, b) == pytest.approx(1 / comb(17, 5))


def test_two_sample_is_calibrated_under_the_null():
    rng = np.random.default_rng(1)
    ps = np.array([stats.permutation_test_two_sample(rng.standard_normal(8),
                                                     rng.standard_normal(6))
                   for _ in range(300)])
    assert 0.02 < np.mean(ps < 0.05) < 0.09


def test_nan_entries_are_dropped_not_propagated():
    assert np.isfinite(stats.sign_flip_test(np.array([1.0, 2, np.nan, 3, 4, 5, 6])))
    assert np.isnan(stats.sign_flip_test(np.array([np.nan, np.nan])))


def test_large_n_falls_back_to_monte_carlo_with_a_stable_answer():
    x = np.r_[np.ones(20), -0.1 * np.ones(10)]
    p1 = stats.sign_flip_test(x, n_resamples=20_000, seed=1)
    p2 = stats.sign_flip_test(x, n_resamples=20_000, seed=2)
    assert abs(p1 - p2) < 0.01


# --- BH-FDR ------------------------------------------------------------------

@pytest.mark.parametrize("p,expected,rejected", [
    # Values verified against statsmodels.stats.multitest.multipletests(method="fdr_bh").
    ([0.001, 0.008, 0.039, 0.041, 0.042],
     [0.005, 0.02, 0.042, 0.042, 0.042], [True] * 5),
    ([0.01, 0.2, 0.03, 0.9, 0.04],
     [0.05, 0.25, 0.0666667, 0.9, 0.0666667], [True, False, False, False, False]),
    ([0.001, 0.9], [0.002, 0.9], [True, False]),
])
def test_fdr_bh_matches_the_reference_implementation(p, expected, rejected):
    adj, rej = stats.fdr_bh(np.array(p), q=0.05)
    np.testing.assert_allclose(adj, expected, atol=1e-6)
    assert rej.tolist() == rejected


def test_fdr_bh_adjusted_p_are_monotone_in_the_raw_p():
    """The step-up correction must never let a larger raw p adjust to a smaller one."""
    rng = np.random.default_rng(0)
    p = np.sort(rng.uniform(size=50))
    adj, _ = stats.fdr_bh(p)
    assert np.all(np.diff(adj) >= -1e-12)


def test_fdr_bh_rejects_nothing_when_all_p_are_large():
    _, rej = stats.fdr_bh(np.array([0.4, 0.5, 0.9]))
    assert not rej.any()


def test_fdr_bh_ignores_nan_entries():
    adj, rej = stats.fdr_bh(np.array([0.001, np.nan, 0.9]))
    assert np.isnan(adj[1]) and not rej[1]
    assert rej[0]
