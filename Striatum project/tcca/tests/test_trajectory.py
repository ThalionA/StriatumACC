"""Tests for tom_cca.trajectory (Frame A within-animal learning trajectory)."""

from __future__ import annotations

import numpy as np

from striatum_tcca import trajectory


# --------------------------------------------------------------------------
# sliding_windows
# --------------------------------------------------------------------------

def test_sliding_windows_basic():
    w = trajectory.sliding_windows(np.arange(1, 11), window=4, step=2)
    assert [list(x) for x in w] == [
        [1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8], [7, 8, 9, 10]]


def test_sliding_windows_includes_tail():
    # last partial window is kept if it still has >= min_window trials
    w = trajectory.sliding_windows(np.arange(1, 8), window=4, step=3,
                                   min_window=2)
    assert [list(x) for x in w] == [[1, 2, 3, 4], [4, 5, 6, 7]]


def test_sliding_windows_too_few():
    assert trajectory.sliding_windows(np.array([1, 2]), window=4, step=2) == []


# --------------------------------------------------------------------------
# heldout_cc1
# --------------------------------------------------------------------------

def _shared_xy(rng, n, n_units=6, shared_strength=0.9, n_groups=10):
    """X and Y sharing one strong latent dim; grouped for whole-group CV."""
    latent = rng.normal(0, 1, n)
    X = rng.normal(0, 1, (n, n_units))
    Y = rng.normal(0, 1, (n, n_units))
    X[:, 0] += shared_strength * latent
    Y[:, 0] += shared_strength * latent
    groups = rng.integers(0, n_groups, n)
    return X, Y, groups


def test_heldout_cc1_recovers_shared_dim():
    rng = np.random.default_rng(0)
    X, Y, g = _shared_xy(rng, 4000, shared_strength=1.5)
    cc_te, cc_tr = trajectory.heldout_cc1(X, Y, g, k=5, n_folds=5)
    assert cc_te > 0.6                       # strong shared dim recovered
    assert abs(cc_tr - cc_te) < 0.1          # large n -> no overfitting gap


def test_heldout_cc1_independent_is_low():
    rng = np.random.default_rng(1)
    n = 4000
    X = rng.normal(0, 1, (n, 6))
    Y = rng.normal(0, 1, (n, 6))
    g = rng.integers(0, 10, n)
    cc_te, _ = trajectory.heldout_cc1(X, Y, g, k=5, n_folds=5)
    assert cc_te < 0.2                       # no real shared structure


def test_heldout_cc1_too_few_groups():
    rng = np.random.default_rng(2)
    X, Y, _ = _shared_xy(rng, 200)
    g = np.zeros(200, dtype=int)             # one group -> cannot CV
    cc_te, cc_tr = trajectory.heldout_cc1(X, Y, g, k=5, n_folds=5)
    assert np.isnan(cc_te)


# --------------------------------------------------------------------------
# linear_slope
# --------------------------------------------------------------------------

def test_linear_slope_recovers_line():
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    y = 2.0 * x + 1.0
    slope, r = trajectory.linear_slope(x, y)
    assert abs(slope - 2.0) < 1e-9
    assert abs(r - 1.0) < 1e-9


def test_linear_slope_handles_nan_and_degenerate():
    x = np.array([0.0, 0.5, 1.0, np.nan])
    y = np.array([1.0, 2.0, 3.0, 5.0])
    slope, r = trajectory.linear_slope(x, y)
    assert abs(slope - 2.0) < 1e-9
    # fewer than 2 finite points -> NaN
    s2, _ = trajectory.linear_slope(np.array([1.0]), np.array([2.0]))
    assert np.isnan(s2)
