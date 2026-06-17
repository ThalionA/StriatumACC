"""Tests for the very-early-trial projected readout (tom_cca.early_trials).

Ground-truth synthetic data: the reference subspace is fit on the LATE trials and
each early trial is projected through it. We check the projection recovers (1) a
coupling that is present in held-out coupled trials and absent in uncoupled ones,
(2) sparse vs dense per-trial participation, (3) the sign of a known lead/lag, and
(4) that the fit is genuinely leak-free (held-out rows do not change the model).
"""

import numpy as np

from striatum_tcca import core, early_trials


def _two_areas(rng, n_bins, n_trials, nx, ny, coupled_from):
    """Build (X, Y, trial_ids): a shared 1-D latent drives X in every trial and Y
    only in trials with id >= ``coupled_from`` (earlier Y is independent noise)."""
    Wx = rng.standard_normal((1, nx))
    Wy = rng.standard_normal((1, ny))
    X = np.zeros((n_trials * n_bins, nx))
    Y = np.zeros((n_trials * n_bins, ny))
    tids = np.repeat(np.arange(n_trials), n_bins)
    for t in range(n_trials):
        sl = slice(t * n_bins, (t + 1) * n_bins)
        s = rng.standard_normal((n_bins, 1))
        X[sl] = s @ Wx + 0.3 * rng.standard_normal((n_bins, nx))
        Y[sl] = (s @ Wy + 0.3 * rng.standard_normal((n_bins, ny))
                 if t >= coupled_from else rng.standard_normal((n_bins, ny)))
    return X, Y, tids


def test_per_trial_cc_detects_coupling_held_out():
    rng = np.random.default_rng(0)
    n_bins, n_trials = 300, 20
    X, Y, tids = _two_areas(rng, n_bins, n_trials, 8, 8, coupled_from=5)
    train = np.arange(10, n_trials)                  # reference = later trials
    # trials 5..9 are coupled but HELD OUT of the fit; 0..4 are genuinely uncoupled
    coupled = early_trials.early_trial_metrics(X, Y, tids, [5, 6, 7, 8, 9], train, k=6)
    uncoupled = early_trials.early_trial_metrics(X, Y, tids, [0, 1, 2, 3, 4], train, k=6)
    cc_coupled = np.nanmean([r["cc1"] for r in coupled])
    cc_uncoupled = np.nanmean([r["cc1"] for r in uncoupled])
    assert cc_coupled > 0.6                          # coupling recovered out-of-sample
    assert cc_uncoupled < 0.3                        # no coupling where there is none
    assert cc_coupled - cc_uncoupled > 0.4


def test_gini_of_corr_sparse_gt_dense():
    rng = np.random.default_rng(1)
    n = 500
    variate = rng.standard_normal(n)
    sparse = 0.01 * rng.standard_normal((n, 10))     # only neuron 0 tracks the variate
    sparse[:, 0] = variate + 0.1 * rng.standard_normal(n)
    dense = variate[:, None] + 0.5 * rng.standard_normal((n, 10))  # all neurons track it
    g_sparse = early_trials._gini_of_corr(sparse, variate)
    g_dense = early_trials._gini_of_corr(dense, variate)
    assert g_sparse > g_dense
    assert g_sparse > 0.5
    assert g_dense < 0.3


def test_trial_ifi_detects_known_lead():
    rng = np.random.default_rng(2)
    n, L = 400, 3
    u = rng.standard_normal(n)
    v = np.zeros(n)
    v[L:] = u[:n - L]                                # Y carries X delayed by L ⇒ X leads
    v[:L] = 0.01 * rng.standard_normal(L)
    model = core.CCAResult(A=np.array([[1.0]]), B=np.array([[1.0]]),
                           r=np.array([1.0]), x_mean=np.array([0.0]),
                           y_mean=np.array([0.0]))
    fit = early_trials.ReferenceFit(Xr=np.zeros((n, 1)), Yr=np.zeros((n, 1)),
                                    Sx=u[:, None], Sy=v[:, None], model=model)
    ifi, opt = early_trials.trial_ifi(fit, np.arange(n), max_lag=6)
    assert opt == L                                  # peak at the true lag
    assert ifi > 0                                   # positive ⇒ X leads Y


def test_reference_fit_is_leak_free():
    rng = np.random.default_rng(3)
    n_bins, n_trials = 200, 10
    X = rng.standard_normal((n_trials * n_bins, 5))
    Y = rng.standard_normal((n_trials * n_bins, 5))
    tids = np.repeat(np.arange(n_trials), n_bins)
    train = np.isin(tids, np.arange(5, 10))
    fit1 = early_trials.reference_fit(X, Y, train, k=4)
    X2, Y2 = X.copy(), Y.copy()
    X2[tids < 5] += 99.0                             # corrupt the held-out (early) rows
    Y2[tids < 5] *= -3.0
    fit2 = early_trials.reference_fit(X2, Y2, train, k=4)
    assert np.allclose(fit1.model.A, fit2.model.A)   # model depends only on train rows
    assert np.allclose(fit1.model.r, fit2.model.r)


def test_participation_gini_in_range_and_short_trial_nan():
    rng = np.random.default_rng(4)
    X, Y, tids = _two_areas(rng, 300, 20, 8, 8, coupled_from=0)
    train = np.arange(10, 20)
    res = early_trials.early_trial_metrics(X, Y, tids, [5, 6, 7], train, k=6)
    for r in res:
        assert 0.0 <= r["gini_part_x"] <= 1.0
        assert 0.0 <= r["gini_part_y"] <= 1.0
        assert np.isfinite(r["ifi"])
    # a trial shorter than min_bins yields NaN metrics
    short = early_trials.early_trial_metrics(X, Y, tids, [999], train, k=6, min_bins=10)
    assert np.isnan(short[0]["cc1"])
    assert short[0]["n_bins"] == 0
