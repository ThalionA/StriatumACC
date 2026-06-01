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


# ---------------------------------------------------------------------------
# rm_anova_posthoc — RM-ANOVA + Holm-corrected paired-t over the 3 epochs
# ---------------------------------------------------------------------------
def test_rm_anova_posthoc_matches_components():
    # The omnibus F/p must equal rm_anova; the post-hoc must equal the
    # Holm-adjusted paired-t over the (0,1),(1,2),(0,2) epoch contrasts.
    rng = np.random.default_rng(3)
    subject = rng.standard_normal((18, 1))
    data = subject + np.array([0.0, 0.5, 1.5]) + 0.4 * rng.standard_normal((18, 3))
    res = epoch_stats.rm_anova_posthoc(data)
    f, p = epoch_stats.rm_anova(data)
    assert abs(res["F"] - f) < 1e-9 and abs(res["p"] - p) < 1e-9
    raw = [stats.ttest_rel(data[:, a], data[:, b]).pvalue
           for a, b in ((0, 1), (1, 2), (0, 2))]
    assert np.allclose(res["posthoc"], epoch_stats.holm(raw))
    assert res["n"] == 18


def test_rm_anova_posthoc_detects_monotone_effect():
    rng = np.random.default_rng(4)
    subject = rng.standard_normal((25, 1))
    data = subject + np.array([0.0, 1.0, 2.0]) + 0.3 * rng.standard_normal((25, 3))
    res = epoch_stats.rm_anova_posthoc(data)
    assert res["p"] < 1e-6
    assert res["posthoc"][2] < 0.05            # naive vs expert separates


def test_rm_anova_posthoc_null_is_not_significant():
    rng = np.random.default_rng(5)
    data = rng.standard_normal((30, 3))        # no condition effect
    res = epoch_stats.rm_anova_posthoc(data)
    assert res["p"] > 0.05


def test_rm_anova_posthoc_too_few_subjects_is_nan():
    res = epoch_stats.rm_anova_posthoc(np.ones((1, 3)))
    assert np.isnan(res["F"]) and np.isnan(res["p"])
    assert res["n"] == 1
    assert all(np.isnan(v) for v in res["posthoc"])


# ---------------------------------------------------------------------------
# linear_trend — slope of value vs epoch index, with significance
# ---------------------------------------------------------------------------
def test_linear_trend_recovers_known_slope():
    x = np.array([0, 0, 1, 1, 2, 2], dtype=float)
    y = 2.0 * x + 1.0                          # exact slope 2
    slope, p = epoch_stats.linear_trend(x, y)
    assert abs(slope - 2.0) < 1e-9
    assert p < 1e-6


def test_linear_trend_flat_is_not_significant():
    rng = np.random.default_rng(6)
    x = np.repeat([0, 1, 2], 10).astype(float)
    y = rng.standard_normal(30)                # no trend
    slope, p = epoch_stats.linear_trend(x, y)
    assert p > 0.05


def test_linear_trend_degenerate_is_nan():
    slope, p = epoch_stats.linear_trend(np.array([1.0]), np.array([3.0]))
    assert np.isnan(slope) and np.isnan(p)
