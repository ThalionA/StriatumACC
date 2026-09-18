"""Tests for communication-subspace membership scoring."""

from __future__ import annotations

import numpy as np

from striatum_tcca import membership


# ---------------------------------------------------------------------------
# Gini
# ---------------------------------------------------------------------------
def test_gini_uniform_is_zero():
    assert membership.gini(np.ones(20)) == 0.0


def test_gini_one_hot_is_one():
    v = np.zeros(20)
    v[0] = 1.0
    assert abs(membership.gini(v) - 1.0) < 1e-9


def test_gini_ignores_sign():
    assert membership.gini(np.array([-3.0, 3.0, -3.0])) == membership.gini(
        np.array([3.0, 3.0, 3.0])
    )


# ---------------------------------------------------------------------------
# structure coefficients
# ---------------------------------------------------------------------------
def test_structure_coefficients_recover_planted_alignment():
    # components = identity -> neuron j's reconstruction is PC j.
    # coef picks PC 0 as the variate -> neuron 0 correlates 1, others ~0.
    rng = np.random.default_rng(0)
    k = 5
    scores = rng.standard_normal((10, 50, k))
    components = np.eye(k)
    coef = np.zeros((k, 2))
    coef[0, 0] = 1.0
    coef[1, 1] = 1.0
    sc = membership.structure_coefficients(scores, components, coef, d=2)
    assert sc.shape == (k, 2)
    assert abs(sc[0, 0] - 1.0) < 1e-6          # neuron 0 IS variate 0
    assert abs(sc[1, 0]) < 0.15                # neuron 1 ~ uncorrelated (finite-sample)
    assert np.all(np.abs(sc) <= 1.0 + 1e-9)


# ---------------------------------------------------------------------------
# contribution / member mask
# ---------------------------------------------------------------------------
def test_subspace_contribution_is_l2_norm():
    mat = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])
    assert np.allclose(membership.subspace_contribution(mat), [5.0, 0.0, 1.0])


def test_member_mask_selects_top_quartile():
    contribution = np.arange(100, dtype=float)
    mask = membership.member_mask(contribution, quantile=0.75)
    assert 24 <= mask.sum() <= 26          # ~top 25%
    assert mask[-1] and not mask[0]


# ---------------------------------------------------------------------------
# variate structure coefficients (method-agnostic membership, used by KCCA)
# ---------------------------------------------------------------------------
def test_variate_structure_coefficients_recover_the_tracking_neuron():
    rng = np.random.default_rng(0)
    n = 2000
    variate = rng.standard_normal((n, 1))
    neurons = rng.standard_normal((n, 5)) * 0.5
    neurons[:, 2] += variate[:, 0]                 # neuron 2 tracks the variate
    sc = membership.variate_structure_coefficients(neurons, variate)
    assert sc.shape == (5, 1)
    assert sc[2, 0] > 0.7                          # tracking neuron loads high
    assert abs(sc[0, 0]) < 0.3                     # others ~ uncorrelated
    assert np.all(np.abs(sc) <= 1.0 + 1e-9)


def test_variate_structure_coefficient_gini_tracks_concentration():
    # Concentrated membership (one neuron tracks the variate) is higher-Gini than
    # distributed membership (all neurons track it) — the property the KCCA Gini
    # de-sparsification readout relies on.
    rng = np.random.default_rng(1)
    n = 3000
    v = rng.standard_normal((n, 1))
    conc = rng.standard_normal((n, 10)) * 0.5; conc[:, 0] += v[:, 0]
    dist = rng.standard_normal((n, 10)) * 0.5 + v
    g_conc = membership.gini(membership.subspace_contribution(
        membership.variate_structure_coefficients(conc, v)))
    g_dist = membership.gini(membership.subspace_contribution(
        membership.variate_structure_coefficients(dist, v)))
    assert g_conc > g_dist + 0.2


# ---------------------------------------------------------------------------
# Pearson coupling scores (CCA-independent control for the weight Gini)
# ---------------------------------------------------------------------------
def test_pearson_coupling_shapes():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 6))
    Y = rng.standard_normal((2000, 9))
    cx, cy = membership.pearson_coupling_scores(X, Y)
    assert cx.shape == (6,) and cy.shape == (9,)
    assert np.all(cx >= 0) and np.all(cy >= 0)


def test_pearson_coupling_isolates_the_coupled_neuron():
    # Only X-neuron 0 carries Y's shared latent -> it has the largest coupling.
    rng = np.random.default_rng(1)
    n = 4000
    s = rng.standard_normal(n)
    X = rng.standard_normal((n, 10)) * 0.3
    Y = rng.standard_normal((n, 10)) * 0.3
    X[:, 0] += s
    Y[:, 0] += s
    cx, _ = membership.pearson_coupling_scores(X, Y)
    assert int(np.argmax(cx)) == 0
    assert cx[0] > 3 * np.median(cx)              # stands well clear of the rest


def test_pearson_gini_concentrated_exceeds_distributed():
    # The property the control relies on: a coupling carried by ONE neuron is
    # high-Gini; the SAME coupling spread over all neurons is low-Gini. So a
    # de-sparsifying (1->many) shift shows up as Gini falling.
    rng = np.random.default_rng(2)
    n = 5000
    s = rng.standard_normal(n)
    # concentrated: only neuron 0 of X couples to Y
    Xc = rng.standard_normal((n, 12)) * 0.3
    Xc[:, 0] += s
    # distributed: every neuron of X carries the same coupling
    Xd = rng.standard_normal((n, 12)) * 0.3
    Xd += s[:, None]
    Y = rng.standard_normal((n, 12)) * 0.3
    Y[:, 0] += s
    gini_c = membership.gini(membership.pearson_coupling_scores(Xc, Y)[0])
    gini_d = membership.gini(membership.pearson_coupling_scores(Xd, Y)[0])
    assert gini_c > gini_d + 0.2


def test_pearson_coupling_drops_nonfinite_rows():
    rng = np.random.default_rng(3)
    n = 3000
    s = rng.standard_normal(n)
    X = rng.standard_normal((n, 5)) * 0.3; X[:, 0] += s
    Y = rng.standard_normal((n, 5)) * 0.3; Y[:, 0] += s
    cx_clean, _ = membership.pearson_coupling_scores(X, Y)
    Xn = X.copy(); Xn[7, 2] = np.nan; Yn = Y.copy(); Yn[11, 0] = np.inf
    cx_nan, _ = membership.pearson_coupling_scores(Xn, Yn)
    assert np.all(np.isfinite(cx_nan))
    assert np.allclose(cx_clean, cx_nan, atol=2e-2)   # two dropped rows ~ no change


def test_pearson_coupling_too_few_samples_is_zero():
    cx, cy = membership.pearson_coupling_scores(np.ones((1, 4)), np.ones((1, 3)))
    assert cx.shape == (4,) and np.all(cx == 0)
    assert cy.shape == (3,) and np.all(cy == 0)


def test_pearson_gini_noise_floor_is_sample_size_invariant():
    # The slope analysis relies on the no-coupling Gini floor being independent of
    # window sample count N — else shrinking late-session windows could fake a
    # Gini trend. Verify the mean uncoupled Gini matches across very different N.
    def mean_gini(n, reps=40):
        gs = []
        for r in range(reps):
            rng = np.random.default_rng(1000 + r)
            X = rng.standard_normal((n, 12)); Y = rng.standard_normal((n, 12))
            gs.append(membership.gini(membership.pearson_coupling_scores(X, Y)[0]))
        return float(np.mean(gs))
    assert abs(mean_gini(300) - mean_gini(6000)) < 0.02   # N-invariant floor


# ---------------------------------------------------------------------------
# Jaccard
# ---------------------------------------------------------------------------
def test_jaccard_identical_and_disjoint():
    a = np.array([True, True, False, False])
    b = np.array([False, False, True, True])
    assert membership.jaccard(a, a) == 1.0
    assert membership.jaccard(a, b) == 0.0
def _cca_weights(X, Y):
    """Minimal stand-in for core.cca_fit's A, r (avoids a circular import)."""
    from scipy import linalg

    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    ux, sx, vxt = linalg.svd(Xc, full_matrices=False)
    uy, sy, vyt = linalg.svd(Yc, full_matrices=False)
    tol = 1e-10
    rx = int((sx > tol * sx[0]).sum())
    ry = int((sy > tol * sy[0]).sum())
    ux, sx, vxt = ux[:, :rx], sx[:rx], vxt[:rx]
    uy, sy, vyt = uy[:, :ry], sy[:ry], vyt[:ry]
    uc, rho, _ = linalg.svd(ux.T @ uy)
    d = min(rx, ry)
    A = (vxt.T @ (uc[:, :d] / sx[:, None])) * np.sqrt(X.shape[0] - 1)
    return A, np.clip(rho[:d], 0.0, 1.0)


def _two_partners(seed=0, n=2000, p=12, q=12):
    """Same X; one partner strongly coupled to it, one pure noise."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    Y_coupled = X @ rng.standard_normal((p, q)) + 0.05 * rng.standard_normal((n, q))
    Y_noise = rng.standard_normal((n, q))
    components = rng.standard_normal((40, p))       # (n_units, k)
    return X, Y_coupled, Y_noise, components


def test_area_intrinsic_contribution_is_partner_invariant():
    """Documents the identity: the unweighted norm cannot see the partner."""
    X, Yc, Yn, comp = _two_partners()
    Ac, _ = _cca_weights(X, Yc)
    An, _ = _cca_weights(X, Yn)
    c_coupled = membership.subspace_contribution(comp @ Ac)
    c_noise = membership.subspace_contribution(comp @ An)
    assert np.allclose(c_coupled, c_noise, atol=1e-8)
    assert abs(membership.gini(c_coupled) - membership.gini(c_noise)) < 1e-9


def test_connection_contribution_depends_on_partner():
    """The rho-weighted norm must distinguish a coupled from an uncoupled partner."""
    X, Yc, Yn, comp = _two_partners()
    Ac, rc = _cca_weights(X, Yc)
    An, rn = _cca_weights(X, Yn)
    c_coupled = membership.subspace_contribution_connection(comp @ Ac, rc)
    c_noise = membership.subspace_contribution_connection(comp @ An, rn)
    assert not np.allclose(c_coupled, c_noise, atol=1e-3)


def test_connection_contribution_reduces_to_area_intrinsic_when_rho_flat():
    """Equal canonical correlations => same Gini (Gini is scale-invariant)."""
    X, Yc, _, comp = _two_partners()
    A, r = _cca_weights(X, Yc)
    flat = np.full_like(r, 0.7)
    plain = membership.subspace_contribution(comp @ A)
    weighted = membership.subspace_contribution_connection(comp @ A, flat)
    assert abs(membership.gini(plain) - membership.gini(weighted)) < 1e-9


def test_connection_contribution_tracks_the_communicating_dimension():
    """A unit loading on the one correlated dim must outrank one loading on a dead dim."""
    scores = np.array([[1.0, 0.0], [0.0, 1.0]])     # unit 0 -> dim 0; unit 1 -> dim 1
    r = np.array([0.9, 0.0])                        # dim 0 communicates, dim 1 does not
    c = membership.subspace_contribution_connection(scores, r)
    assert c[0] > c[1]
    assert c[1] == 0.0


def test_connection_contribution_zero_rho_gives_zero_everywhere():
    """No communication at all => no participation (not an arbitrary ranking)."""
    rng = np.random.default_rng(3)
    scores = rng.standard_normal((15, 4))
    c = membership.subspace_contribution_connection(scores, np.zeros(4))
    assert np.all(c == 0.0)


def test_connection_contribution_truncates_rho_to_available_dims():
    scores = np.ones((6, 3))
    c = membership.subspace_contribution_connection(scores, np.array([1.0, 1.0, 1.0, 1.0]))
    assert c.shape == (6,)


# ---------------------------------------------------------------------------
# signed_dim_weights — the export of signed per-dimension canonical weights (Tom)
# ---------------------------------------------------------------------------
def _planted_pair(seed=0, n=3000, kx=10, ky=8):
    from tom_cca import core
    rng = np.random.default_rng(seed)
    lat = rng.standard_normal((n, 2))
