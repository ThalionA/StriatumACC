"""Tests for the full kernel-CCA per-cell readout (kcca_window)."""

from __future__ import annotations

import numpy as np

from striatum_tcca import kcca_window as kw


def _groups(n, n_groups, rng):
    return rng.integers(0, n_groups, n)


def test_kcca_window_detects_nonlinear_coupling():
    # Shared latent enters Y nonlinearly (s**2): linear CCA is weak, KCCA strong,
    # and the dim is significant against the circular-shift surrogate.
    rng = np.random.default_rng(0)
    n = 1400
    s = rng.standard_normal(n)
    X = rng.standard_normal((n, 8)) * 0.3; X[:, 0] += s
    Y = rng.standard_normal((n, 8)) * 0.3; Y[:, 0] += (s ** 2 - 1.0)
    g = _groups(n, 10, rng)
    r = kw.kcca_window(X, Y, g, reg=1.0, k=6, max_lag=3, n_shuffles=15,
                       n_folds=5, d=2)
    assert r.cc_kcca[0] > 0.4                        # KCCA finds it
    assert r.cc_kcca[0] > r.cc_linear[0] + 0.2       # and beats linear out-of-sample
    assert r.n_sig >= 1                              # clears its own surrogate
    assert 0.0 <= r.gini_x <= 1.0


def test_kcca_window_independent_is_not_significant():
    rng = np.random.default_rng(1)
    n = 1400
    X = rng.standard_normal((n, 8)); Y = rng.standard_normal((n, 8))
    g = _groups(n, 10, rng)
    r = kw.kcca_window(X, Y, g, reg=1.0, k=6, max_lag=3, n_shuffles=20, d=2)
    assert r.n_sig == 0                             # no false-positive coupling
    assert abs(r.cc_kcca[0]) < 0.25                 # held-out ~ 0


def test_kcca_window_membership_gini_reflects_concentration():
    # One X-unit carries the (nonlinear) coupling -> high structure-coeff Gini;
    # many units carrying it -> lower Gini.
    rng = np.random.default_rng(2)
    n = 1400
    s = rng.standard_normal(n)
    g = _groups(n, 10, rng)
    Xc = rng.standard_normal((n, 12)) * 0.3; Xc[:, 0] += s          # concentrated
    Xd = rng.standard_normal((n, 12)) * 0.3 + s[:, None]            # distributed
    Y = rng.standard_normal((n, 12)) * 0.3; Y[:, 0] += (s ** 2 - 1.0)
    rc = kw.kcca_window(Xc, Y, g, reg=1.0, k=8, max_lag=2, n_shuffles=5, d=2)
    rd = kw.kcca_window(Xd, Y, g, reg=1.0, k=8, max_lag=2, n_shuffles=5, d=2)
    assert rc.gini_x > rd.gini_x + 0.15


def test_kcca_window_z_control_collapses_confound_coupling():
    # X, Y coupled only through Z -> partialling Z out (leak-free, per fold)
    # collapses the held-out KCCA CC.
    rng = np.random.default_rng(3)
    n = 1400
    z = rng.standard_normal((n, 5))
    X = rng.standard_normal((n, 8)) * 0.3; X[:, :5] += z
    Y = rng.standard_normal((n, 8)) * 0.3; Y[:, :5] += z
    g = _groups(n, 10, rng)
    plain = kw.kcca_window(X, Y, g, reg=1.0, k=6, max_lag=2, n_shuffles=5, d=1)
    ctrl = kw.kcca_window(X, Y, g, Z=z, reg=1.0, k=6, max_lag=2, n_shuffles=5, d=1)
    assert plain.cc_kcca[0] > 0.4
    assert ctrl.cc_kcca[0] < plain.cc_kcca[0] - 0.2
