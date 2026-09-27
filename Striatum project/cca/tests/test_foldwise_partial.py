"""Partialling must happen INSIDE the cross-validation folds.

Regressing a confound out of an epoch's samples before cca_cv fits the
regression on the test trials too; the residualising projection then mixes
training rows into test rows and inflates held-out CC. Measured on independent
X, Y (k=20 PCs, 16 random confounds, 10 trials x 50 bins): +0.013 +- 0.002.
cca_cv(..., zx=, zy=) fits the confound regression on training trials only.
"""

from __future__ import annotations

import numpy as np

from striatum_cca import config, core, lagged, partial, surrogate

CFG = config.DEFAULT


def _null_draw(rng, k=20, q=16):
    x = rng.standard_normal((10, 50, k))
    y = rng.standard_normal((10, 50, k))
    z = rng.standard_normal((10, 50, q))
    return x, y, z


def test_in_sample_partialling_inflates_held_out_cc_under_the_null():
    """Documents the bug: paired over the same draws, in-sample partialling
    raises held-out CC1 of independent areas."""
    rng = np.random.default_rng(0)
    diff = []
    for _ in range(100):
        x, y, z = _null_draw(rng)
        fx, fy, fz = (a.reshape(-1, a.shape[-1]) for a in (x, y, z))
        rx = partial.partial_out(fx, fz).reshape(x.shape)
        ry = partial.partial_out(fy, fz).reshape(y.shape)
        diff.append(core.cca_cv(rx, ry, CFG).held_out_r[0] - core.cca_cv(x, y, CFG).held_out_r[0])
    assert np.mean(diff) > 0.006


def test_foldwise_partialling_is_unbiased_under_the_null():
    rng = np.random.default_rng(0)
    diff = []
    for _ in range(100):
        x, y, z = _null_draw(rng)
        diff.append(core.cca_cv(x, y, CFG, zx=z).held_out_r[0] - core.cca_cv(x, y, CFG).held_out_r[0])
    assert abs(np.mean(diff)) < 0.005


def test_foldwise_partialling_removes_confound_mediated_coupling():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((40, 50, 2))
    x = z @ rng.standard_normal((2, 6)) + 0.3 * rng.standard_normal((40, 50, 6))
    y = z @ rng.standard_normal((2, 6)) + 0.3 * rng.standard_normal((40, 50, 6))
    assert core.cca_cv(x, y, CFG).held_out_r[0] > 0.8
    assert core.cca_cv(x, y, CFG, zx=z).held_out_r[0] < 0.15


def test_foldwise_partialling_keeps_direct_coupling():
    rng = np.random.default_rng(2)
    shared = rng.standard_normal((40, 50, 1))
    x = shared * rng.standard_normal(6) + rng.standard_normal((40, 50, 6))
    y = shared * rng.standard_normal(6) + rng.standard_normal((40, 50, 6))
    z = rng.standard_normal((40, 50, 4))
    plain = core.cca_cv(x, y, CFG).held_out_r[0]
    assert abs(core.cca_cv(x, y, CFG, zx=z).held_out_r[0] - plain) < 0.03


def test_foldwise_partialling_tolerates_missing_rows():
    rng = np.random.default_rng(3)
    x, y, z = _null_draw(rng, k=5, q=3)
    x[2, 40:] = np.nan  # a short trial (temporal arm NaN tail)
    y[2, 40:] = np.nan
    z[2, 40:] = np.nan
    r = core.cca_cv(x, y, CFG, zx=z).held_out_r
    assert np.all(np.isfinite(r))


def test_partial_cca_cv_is_foldwise():
    rng = np.random.default_rng(4)
    x, y, z = _null_draw(rng)
    np.testing.assert_allclose(partial.partial_cca_cv(x, y, z, CFG).held_out_r,
                               core.cca_cv(x, y, CFG, zx=z).held_out_r)


def test_lag_curve_partials_each_area_on_its_own_lagged_confound():
    # X and Y both follow z at the SAME bin; at lag L, X's bins pair with Y's
    # bins shifted by L, so the confound must be sliced like each area.
    rng = np.random.default_rng(5)
    z = rng.standard_normal((40, 50, 2))
    x = z @ rng.standard_normal((2, 6)) + 0.3 * rng.standard_normal((40, 50, 6))
    y = z @ rng.standard_normal((2, 6)) + 0.3 * rng.standard_normal((40, 50, 6))
    res = lagged.lag_curve(x, y, CFG, max_lag=2, held_out=True, confound=z)
    assert np.nanmax(res.cc_per_dim[:, 0]) < 0.15


def test_surrogate_shuffles_y_together_with_its_confound():
    # Coupling is entirely via z: after fold-wise partialling the real CC is
    # ~0 and so is the null. If y were shuffled WITHOUT its confound, z would
    # no longer explain y and the null would carry spurious structure.
    rng = np.random.default_rng(6)
    z = rng.standard_normal((40, 50, 2))
    x = z @ rng.standard_normal((2, 6)) + 0.3 * rng.standard_normal((40, 50, 6))
    y = z @ rng.standard_normal((2, 6)) + 0.3 * rng.standard_normal((40, 50, 6))
    real = core.cca_cv(x, y, CFG, zx=z).held_out_r
    null = surrogate.build_null(x, y, real, CFG, confound=z)
    assert np.nanmedian(null.null_held_out[:, 0]) < 0.15
