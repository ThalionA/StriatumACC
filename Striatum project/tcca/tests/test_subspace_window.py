"""Tests for tom_cca.subspace_window (full per-window subspace readout)."""

from __future__ import annotations

import numpy as np

from striatum_tcca import subspace_window as sw


def _grouped(n, n_groups, rng):
    return rng.integers(0, n_groups, n)


def test_recovers_shared_dims_and_significance():
    rng = np.random.default_rng(0)
    n = 6000
    lat = rng.normal(0, 1, (n, 2))               # two shared latents
    X = rng.normal(0, 1, (n, 8))
    Y = rng.normal(0, 1, (n, 8))
    X[:, :2] += 1.5 * lat
    Y[:, :2] += 1.5 * lat
    g = _grouped(n, 10, rng)
    r = sw.window_subspace(X, Y, g, k=6, max_lag=5, n_shuffles=30)
    assert r.cc[0] > 0.5                          # dominant CC recovered
    assert r.n_sig >= 2                           # two significant dims
    assert r.mi_sig > 0
    assert r.weights_x.shape[0] == 8 and r.member_x.dtype == bool


def test_independent_has_no_significant_dims():
    rng = np.random.default_rng(1)
    n = 6000
    X = rng.normal(0, 1, (n, 8))
    Y = rng.normal(0, 1, (n, 8))
    g = _grouped(n, 10, rng)
    r = sw.window_subspace(X, Y, g, k=6, max_lag=5, n_shuffles=40)
    assert r.n_sig <= 1                           # ~0 false positives
    assert r.mi_sig < 0.05 or r.n_sig == 0


def test_optimal_lag_recovers_known_shift():
    # Y is X's shared latent delayed by +3 bins -> optimal lag should be near +3
    rng = np.random.default_rng(2)
    n = 8000
    lat = rng.normal(0, 1, n)
    X = rng.normal(0, 0.3, (n, 5))
    Y = rng.normal(0, 0.3, (n, 5))
    X[:, 0] += lat
    Y[3:, 0] += lat[:-3]                          # Y lags X by 3 bins
    g = (np.arange(n) // 800)                     # contiguous trial blocks
    r = sw.window_subspace(X, Y, g, k=4, max_lag=8, n_shuffles=10)
    assert abs(r.optimal_lag - 3) <= 1
    assert np.isfinite(r.ifi)


def test_split_half_floor_small_for_stable_subspace():
    # A strong, stable shared subspace -> the two halves agree -> small angle.
    rng = np.random.default_rng(7)
    n = 9000
    lat = rng.normal(0, 1, (n, 3))            # 3 shared latents == the 3 compared dims
    X = rng.normal(0, 0.3, (n, 8)); X[:, :3] += 1.8 * lat
    Y = rng.normal(0, 0.3, (n, 8)); Y[:, :3] += 1.8 * lat
    g = np.arange(n) // 300
    r = sw.window_subspace(X, Y, g, k=6, max_lag=3, n_shuffles=10)
    assert np.isfinite(r.split_half_x)
    assert r.split_half_x < 45     # halves agree -> well below orthogonal


def test_split_half_floor_large_for_noise():
    rng = np.random.default_rng(8)
    n = 8000
    X = rng.normal(0, 1, (n, 8)); Y = rng.normal(0, 1, (n, 8))
    g = np.arange(n) // 400
    r = sw.window_subspace(X, Y, g, k=6, max_lag=3, n_shuffles=10)
    # no real shared structure -> dominant direction unstable -> large angle
    assert r.split_half_x > 45


def test_per_dim_ifi_and_lag():
    # dim-1 latent: X leads Y by +3 bins; dim-2 latent: Y leads X by 2 bins.
    rng = np.random.default_rng(9)
    n = 9000
    l1 = rng.normal(0, 1, n); l2 = rng.normal(0, 1, n)
    X = rng.normal(0, 0.3, (n, 6)); Y = rng.normal(0, 0.3, (n, 6))
    X[:, 0] += l1; Y[3:, 0] += l1[:-3]              # Y lags X by 3 (X leads, +)
    Y[:, 1] += l2; X[2:, 1] += l2[:-2]              # X lags Y by 2 (Y leads, -)
    g = np.arange(n) // 450
    r = sw.window_subspace(X, Y, g, k=5, max_lag=8, n_shuffles=10)
    assert r.ifi_per_dim.shape == r.cc.shape
    assert r.lag_per_dim[0] > 0                     # dim-1: X leads (robust readout)
    assert r.lag_per_dim[1] < 0                     # dim-2: Y leads
    # IFI ordering tracks the lags (dim-1 more X-leading than dim-2); absolute
    # magnitude is weak because the lag curve is in-sample (a known caveat).
    assert r.ifi_per_dim[0] > r.ifi_per_dim[1]


def test_gini_and_members_present():
    rng = np.random.default_rng(3)
    n = 4000
    lat = rng.normal(0, 1, n)
    X = rng.normal(0, 1, (n, 10)); X[:, 0] += 2 * lat   # one unit dominates
    Y = rng.normal(0, 1, (n, 10)); Y[:, 0] += 2 * lat
    g = _grouped(n, 10, rng)
    r = sw.window_subspace(X, Y, g, k=5, max_lag=3, n_shuffles=10)
    assert 0.0 <= r.gini_x <= 1.0
    assert r.member_x.sum() >= 1                  # at least one member


def test_z_control_removes_confound_mediated_coupling_leakfree():
    # X and Y are coupled ONLY through a third area Z. Without Z the held-out CC
    # is high; partialling Z out (per-fold, train-only inside window_subspace)
    # collapses it — the leak-free held-out CC must not retain Z-driven coupling.
    rng = np.random.default_rng(5)
    n = 8000
    z = rng.normal(0, 1, (n, 6))
    X = rng.normal(0, 0.3, (n, 8)); X[:, :6] += z
    Y = rng.normal(0, 0.3, (n, 8)); Y[:, :6] += z       # shared only via z
    g = _grouped(n, 10, rng)
    plain = sw.window_subspace(X, Y, g, k=6, max_lag=3, n_shuffles=10)
    controlled = sw.window_subspace(X, Y, g, Z=z, k=6, max_lag=3, n_shuffles=10)
    assert plain.cc[0] > 0.6                             # strong Z-mediated coupling
    assert controlled.cc[0] < 0.3                        # collapses once Z removed
    assert controlled.n_sig <= plain.n_sig


def test_pearson_control_gini_present_and_tracks_concentration():
    # CCA-independent control. A coupling carried by ONE unit gives a higher
    # Pearson-coupling Gini than the SAME coupling spread over MANY units — the
    # de-sparsification the control is meant to corroborate.
    rng = np.random.default_rng(4)
    n = 5000
    lat = rng.normal(0, 1, n)
    g = _grouped(n, 10, rng)
    Xc = rng.normal(0, 0.3, (n, 12)); Xc[:, 0] += lat        # concentrated
    Yc = rng.normal(0, 0.3, (n, 12)); Yc[:, 0] += lat
    Xd = rng.normal(0, 0.3, (n, 12)) + lat[:, None]          # distributed
    Yd = rng.normal(0, 0.3, (n, 12)); Yd[:, 0] += lat
    rc = sw.window_subspace(Xc, Yc, g, k=6, max_lag=3, n_shuffles=10)
    rd = sw.window_subspace(Xd, Yd, g, k=6, max_lag=3, n_shuffles=10)
    assert 0.0 <= rc.gini_pearson_x <= 1.0
    assert 0.0 <= rd.gini_pearson_x <= 1.0
    assert rc.gini_pearson_x > rd.gini_pearson_x            # 1-unit > many-unit


def test_split_half_floor_tracks_true_dimensionality():
    """The split-half rotation floor must reflect the RECOVERABLE dimensionality,
    not be a fixed ~orthogonal value (a bug check: a near-orthogonal floor between
    two halves of one window is suspicious). With three genuinely shared dims the
    top-3 floor is small; with one shared dim only the CC1 floor is small while the
    top-3 max-angle is near-orthogonal (dims 2-3 are noise). This is why §3.4 reads
    rotation on CC1, not the top-3 subspace."""
    rng = np.random.default_rng(3)
    n_tr, n_bin, nx, ny = 14, 300, 12, 12
    g = np.repeat(np.arange(n_tr), n_bin)
    N = n_tr * n_bin

    def fit(n_shared, noise):
        S = rng.standard_normal((N, n_shared))
        Wx = rng.standard_normal((n_shared, nx)); Wy = rng.standard_normal((n_shared, ny))
        X = S @ Wx + noise * rng.standard_normal((N, nx))
        Y = S @ Wy + noise * rng.standard_normal((N, ny))
        return sw.window_subspace(X, Y, g, k=10, n_shuffles=5)

    three = fit(3, 0.4)                 # three genuinely shared dimensions
    assert three.split_half_x < 30.0    # top-3 subspace recovered in both halves -> SMALL
    assert three.split_half_x_cc1 < 30.0

    one = fit(1, 0.5)                    # only one shared dimension
    assert one.split_half_x_cc1 < 30.0  # CC1 direction is stable
    assert one.split_half_x > 60.0      # but top-3 max-angle ~orthogonal (dims 2-3 = noise)
