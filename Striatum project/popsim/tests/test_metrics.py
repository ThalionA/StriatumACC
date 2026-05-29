"""Tests for validation metrics on synthetic ground truth."""

from __future__ import annotations

import numpy as np

from popsim.metrics import (
    cca,
    lag_of_peak_xcorr,
    partial_cca,
    partial_correlation,
)


def test_cross_correlation_detects_known_lag():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(5000)
    lag = 13
    y = np.zeros_like(x)
    y[lag:] = x[:-lag]  # y delayed by 13 -> x leads y -> positive peak.
    assert lag_of_peak_xcorr(x, y, max_lag=40) == lag


def test_cross_correlation_negative_lag_when_reversed():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(5000)
    lag = 9
    y = np.zeros_like(x)
    y[lag:] = x[:-lag]
    assert lag_of_peak_xcorr(y, x, max_lag=40) == -lag


def test_cca_recovers_strong_correlation():
    rng = np.random.default_rng(2)
    shared = rng.standard_normal((4000, 2))
    X = shared @ rng.standard_normal((2, 5)) + 0.1 * rng.standard_normal((4000, 5))
    Y = shared @ rng.standard_normal((2, 4)) + 0.1 * rng.standard_normal((4000, 4))
    corrs, _, _ = cca(X, Y)
    assert corrs[0] > 0.9


def test_cca_low_for_independent_signals():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((4000, 4))
    Y = rng.standard_normal((4000, 4))
    corrs, _, _ = cca(X, Y)
    assert corrs[0] < 0.15


def test_partial_correlation_removes_mediation():
    rng = np.random.default_rng(4)
    c = rng.standard_normal(6000)
    a = c + 0.3 * rng.standard_normal(6000)
    b = c + 0.3 * rng.standard_normal(6000)
    assert np.corrcoef(a, b)[0, 1] > 0.7
    assert abs(partial_correlation(a, b, c)) < 0.1


def test_partial_cca_collapses_mediated_coupling():
    rng = np.random.default_rng(5)
    c = rng.standard_normal((6000, 2))
    A = c @ rng.standard_normal((2, 4)) + 0.2 * rng.standard_normal((6000, 4))
    B = c @ rng.standard_normal((2, 5)) + 0.2 * rng.standard_normal((6000, 5))
    marginal, _, _ = cca(A, B)
    partial, _, _ = partial_cca(A, B, c)
    assert marginal[0] > 0.8
    assert partial[0] < 0.3
