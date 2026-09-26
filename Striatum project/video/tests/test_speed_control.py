"""Speed nuisance model: recovers a known nonlinear dependence, and removes it
from held-out data without absorbing a real added effect."""

import numpy as np
import pytest

from striatum_video.speed_control import SpeedModel, partial_corr


def _data(n, seed, effect=0.0):
    rng = np.random.default_rng(seed)
    speed = rng.gamma(2.0, 10.0, n)
    me = 5 + 8 * (1 - np.exp(-speed / 8)) + rng.normal(0, 0.3, n) + effect  # saturating in speed
    return speed, me


def test_model_recovers_a_saturating_speed_dependence():
    speed, me = _data(20000, 0)
    model = SpeedModel.fit(speed, me, n_bins=20)
    grid = np.array([2.0, 10.0, 30.0])
    np.testing.assert_allclose(model.predict(grid), 5 + 8 * (1 - np.exp(-grid / 8)), atol=0.2)


def test_residuals_keep_an_effect_that_is_absent_from_the_fit_data():
    s_fit, me_fit = _data(20000, 1)
    s_new, me_new = _data(5000, 2, effect=1.5)  # e.g. a learning effect, speed-independent
    model = SpeedModel.fit(s_fit, me_fit)
    assert np.mean(me_new - model.predict(s_new)) == pytest.approx(1.5, abs=0.05)


def test_prediction_is_clamped_outside_the_fitted_speed_range():
    speed, me = _data(5000, 3)
    model = SpeedModel.fit(speed, me)
    lo, hi = model.predict(np.array([-100.0, 1e6]))
    assert lo == pytest.approx(model.predict(np.array([speed.min()]))[0], abs=0.5)
    assert np.isfinite(hi)


def test_nans_are_ignored_in_the_fit_and_propagated_in_prediction():
    speed, me = _data(5000, 4)
    me[::10] = np.nan
    model = SpeedModel.fit(speed, me)
    out = model.predict(np.array([np.nan, 10.0]))
    assert np.isnan(out[0]) and np.isfinite(out[1])


def test_partial_corr_removes_a_shared_speed_drive():
    rng = np.random.default_rng(5)
    speed = rng.gamma(2.0, 10.0, 20000)
    x = np.sqrt(speed) + rng.normal(0, 0.5, speed.size)
    y_independent = -np.log1p(speed) + rng.normal(0, 0.5, speed.size)
    assert abs(np.corrcoef(x, y_independent)[0, 1]) > 0.3        # raw: spurious via speed
    assert abs(partial_corr(x, y_independent, speed)) < 0.05     # partial: gone
    link = rng.normal(0, 0.5, speed.size)
    assert partial_corr(x + link, y_independent + link, speed) > 0.3  # a real shared term survives
