"""Tests for regularised kernel CCA (the nonlinear analogue of core.cca)."""

from __future__ import annotations

import numpy as np

from striatum_tcca import core, kernel_cca as kc


def _split(n):
    tr = np.zeros(n, dtype=bool)
    tr[: n // 2] = True
    return tr, ~tr


def test_kcca_recovers_a_linear_relationship():
    # Kernel CCA subsumes linear CCA: a linear shared latent is still found.
    rng = np.random.default_rng(0)
    n = 1200
    lat = rng.standard_normal(n)
    X = rng.standard_normal((n, 6)) * 0.3; X[:, 0] += lat
    Y = rng.standard_normal((n, 6)) * 0.3; Y[:, 0] += lat
    tr, te = _split(n)
    m = kc.kcca_fit(X[tr], Y[tr], reg=1.0, n_components=3)
    r_te = kc.kcca_score(X[te], Y[te], m)
    assert r_te[0] > 0.6


def test_kcca_recovers_nonlinear_coupling_that_linear_cca_misses():
    # Y carries s**2: linear corr(s, s**2) ~ 0, so linear CCA sees nothing, but
    # the RBF kernel recovers the dependence out-of-sample. This is the whole
    # point of going nonlinear.
    rng = np.random.default_rng(1)
    n = 1400
    s = rng.standard_normal(n)
    X = rng.standard_normal((n, 6)) * 0.3; X[:, 0] += s
    Y = rng.standard_normal((n, 6)) * 0.3; Y[:, 0] += (s ** 2 - 1.0)
    tr, te = _split(n)
    r_kcca = kc.kcca_score(X[te], Y[te], kc.kcca_fit(X[tr], Y[tr], reg=1.0))[0]
    lin = core.cca_fit(X[tr], Y[tr])
    r_lin = float(core.cca_score(X[te], Y[te], lin)[0])
    assert r_kcca > 0.5                         # KCCA finds the s->s**2 coupling
    assert abs(r_lin) < 0.2                     # linear CCA is blind to it
    assert r_kcca > abs(r_lin) + 0.3


def test_kcca_independent_held_out_is_near_zero():
    # No real coupling -> the held-out CC must be ~0 even though the in-sample
    # (regularised) CC is inflated: KCCA does not hallucinate out-of-sample.
    rng = np.random.default_rng(2)
    n = 1200
    X = rng.standard_normal((n, 6)); Y = rng.standard_normal((n, 6))
    tr, te = _split(n)
    m = kc.kcca_fit(X[tr], Y[tr], reg=1.0, n_components=3)
    r_te = kc.kcca_score(X[te], Y[te], m)
    assert abs(r_te[0]) < 0.2


def test_kcca_reg_controls_in_sample_saturation():
    # Smaller ridge -> richer feature space -> higher (more saturated) in-sample
    # CC. This is the overfitting knob the sample-size analysis hinges on.
    rng = np.random.default_rng(3)
    n = 1000
    X = rng.standard_normal((n, 6)); Y = rng.standard_normal((n, 6))
    r_small = kc.kcca_fit(X, Y, reg=0.01, n_components=1).r[0]
    r_large = kc.kcca_fit(X, Y, reg=10.0, n_components=1).r[0]
    assert r_small > r_large + 0.2              # ridge demonstrably reduces saturation


def test_kcca_score_on_training_data_recovers_in_sample():
    # Sanity of the projection path: scoring the TRAINING data reproduces the
    # in-sample canonical correlation the fit reports.
    rng = np.random.default_rng(4)
    n = 800
    lat = rng.standard_normal(n)
    X = rng.standard_normal((n, 5)) * 0.3; X[:, 0] += lat
    Y = rng.standard_normal((n, 5)) * 0.3; Y[:, 0] += lat
    m = kc.kcca_fit(X, Y, reg=1.0, n_components=2)
    r_self = kc.kcca_score(X, Y, m)
    assert abs(r_self[0] - m.r[0]) < 0.05


def test_kcca_variates_correlate_at_the_in_sample_rate():
    # The two projected variates should correlate ~ the reported in-sample r.
    rng = np.random.default_rng(7)
    n = 1000
    lat = rng.standard_normal(n)
    X = rng.standard_normal((n, 5)) * 0.3; X[:, 0] += lat
    Y = rng.standard_normal((n, 5)) * 0.3; Y[:, 0] += lat
    m = kc.kcca_fit(X, Y, reg=1.0, n_components=1)
    u, v = kc.kcca_variates(X, Y, m)
    assert u.shape == (n, 1) and v.shape == (n, 1)
    assert abs(abs(np.corrcoef(u[:, 0], v[:, 0])[0, 1]) - m.r[0]) < 0.05


def test_kcca_lagged_recovers_a_known_lead_direction():
    # Y depends NONLINEARLY on X three bins earlier (X leads Y by +3): the
    # held-out lag curve should peak at a positive lag.
    rng = np.random.default_rng(8)
    n = 2400
    s = rng.standard_normal(n)
    X = rng.standard_normal((n, 4)) * 0.3; X[:, 0] += s
    Y = rng.standard_normal((n, 4)) * 0.3
    Y[3:, 0] += (s[:-3] ** 2 - 1.0)               # Y_t = f(X_{t-3}) -> X leads by 3
    cut = 1700
    lags = np.arange(-5, 6)
    curve = kc.kcca_lagged_heldout(X[:cut], Y[:cut], X[cut:], Y[cut:],
                                   reg=1.0, lags=lags)
    assert np.any(np.isfinite(curve))
    opt = lags[int(np.nanargmax(curve))]
    assert opt > 0                                 # X leads Y (positive lag)


def test_kcca_rejects_nonpositive_reg():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((100, 4)); Y = rng.standard_normal((100, 4))
    for bad in (0.0, -1.0):
        try:
            kc.kcca_fit(X, Y, reg=bad)
        except ValueError:
            continue
        raise AssertionError(f"reg={bad} should have raised")


def test_median_gamma_positive_and_scale_sensitive():
    rng = np.random.default_rng(6)
    X = rng.standard_normal((300, 5))
    g1 = kc.median_gamma(X)
    g2 = kc.median_gamma(X * 10.0)              # 10x spread -> ~100x smaller gamma
    assert g1 > 0 and g2 > 0
    assert g2 < g1
