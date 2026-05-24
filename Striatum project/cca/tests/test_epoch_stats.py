"""Tests for epoch_stats -- repeated-measures ANOVA and Holm correction."""

from __future__ import annotations

import numpy as np
from scipy import stats

from striatum_cca import epoch_stats


# ---------------------------------------------------------------------------
# rm_anova
# ---------------------------------------------------------------------------
def test_rm_anova_two_conditions_matches_paired_t():
    # A one-way RM-ANOVA with two conditions is the paired t-test: F == t^2,
    # and the p-values agree exactly. This pins the SS decomposition.
    rng = np.random.default_rng(0)
    data = rng.standard_normal((15, 2))
    f, p = epoch_stats.rm_anova(data)
    t, p_t = stats.ttest_rel(data[:, 0], data[:, 1])
    assert abs(f - t ** 2) < 1e-9
    assert abs(p - p_t) < 1e-9


def test_rm_anova_detects_a_condition_effect():
    rng = np.random.default_rng(2)
    subject = rng.standard_normal((20, 1))
    effect = np.array([0.0, 1.0, 2.0])
    data = subject + effect + 0.3 * rng.standard_normal((20, 3))
    f, p = epoch_stats.rm_anova(data)
    assert f > 0 and p < 1e-6


def test_rm_anova_perfect_additive_model_has_zero_error():
    # subject offsets + an exact condition effect -> SS_error 0 -> F = inf.
    data = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
    f, p = epoch_stats.rm_anova(data)
    assert np.isinf(f) and p == 0.0


def test_rm_anova_too_few_subjects_is_nan():
    f, p = epoch_stats.rm_anova(np.ones((1, 3)))
    assert np.isnan(f) and np.isnan(p)


# ---------------------------------------------------------------------------
# holm
# ---------------------------------------------------------------------------
def test_holm_known_values():
    # m=3: sorted 0.01,0.03,0.04 -> 0.03, max(0.03,0.06)=0.06, max(0.06,0.04).
    adj = epoch_stats.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_holm_caps_at_one_and_is_monotone():
    adj = epoch_stats.holm([0.5, 0.9])
    assert np.all(adj <= 1.0)
    assert np.allclose(adj, [1.0, 1.0])


def test_holm_preserves_input_order():
    adj = epoch_stats.holm([0.04, 0.01, 0.03])
    assert adj[1] < adj[2] < adj[0] or np.isclose(adj[2], adj[0])
    assert np.isclose(adj[1], 0.03)            # smallest p -> 3 * 0.01
