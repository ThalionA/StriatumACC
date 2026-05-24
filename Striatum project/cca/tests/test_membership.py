"""Tests for communication-subspace membership scoring."""

from __future__ import annotations

import numpy as np

from striatum_cca import membership


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
# Jaccard
# ---------------------------------------------------------------------------
def test_jaccard_identical_and_disjoint():
    a = np.array([True, True, False, False])
    b = np.array([False, False, True, True])
    assert membership.jaccard(a, a) == 1.0
    assert membership.jaccard(a, b) == 0.0
