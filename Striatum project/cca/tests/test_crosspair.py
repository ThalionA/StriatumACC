"""Tests for crosspair -- within-area subspace similarity across an area's pairs."""

from __future__ import annotations

import numpy as np

from striatum_cca import crosspair, stage3


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------
def test_cosine_similarity_orthogonal_is_zero():
    assert crosspair.cosine_similarity(np.array([1.0, 0.0]),
                                       np.array([0.0, 1.0])) == 0.0


def test_cosine_similarity_parallel_is_one():
    assert crosspair.cosine_similarity(np.array([1.0, 0.0]),
                                       np.array([3.0, 0.0])) == 1.0


def test_cosine_similarity_is_sign_invariant():
    # a canonical weight vector's sign is arbitrary -> anti-parallel == same
    assert crosspair.cosine_similarity(np.array([1.0, 0.0]),
                                       np.array([-2.0, 0.0])) == 1.0


def test_cosine_similarity_forty_five_degrees():
    s = crosspair.cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 1.0]))
    assert abs(s - 1.0 / np.sqrt(2)) < 1e-12


def test_cosine_similarity_zero_vector_is_nan():
    assert np.isnan(crosspair.cosine_similarity(np.zeros(3), np.array([1.0, 0, 0])))


# ---------------------------------------------------------------------------
# similarity_matrix  (synthetic PairSubspace objects, known weight vectors)
# ---------------------------------------------------------------------------
def _epoch_subspace(epoch, wx, wy):
    """Minimal EpochSubspace carrying just the weight vectors crosspair reads."""
    nx, ny = wx.shape[0], wy.shape[0]
    return stage3.EpochSubspace(
        epoch=epoch, d_sub=1, cc=np.ones(1), weights_x=wx, weights_y=wy,
        struct_x=np.zeros((nx, 1)), struct_y=np.zeros((ny, 1)),
        contribution_x=np.zeros(nx), contribution_y=np.zeros(ny),
        member_x=np.zeros(nx, bool), member_y=np.zeros(ny, bool),
        gini_x=0.0, gini_y=0.0,
        split_half_angle_x=np.zeros(1), split_half_angle_y=np.zeros(1))


def _pair(animal, area_x, area_y, wx, wy):
    """A PairSubspace with one epoch ('naive') and the given weight vectors."""
    nx, ny = wx.shape[0], wy.shape[0]
    return stage3.PairSubspace(
        animal_id=animal, area_x=area_x, area_y=area_y, role="learner",
        lp=42, k=1, d_sub=1,
        unit_index_x=np.arange(nx), unit_index_y=np.arange(ny),
        epochs={"naive": _epoch_subspace("naive", wx, wy)},
        angles_x={}, angles_y={})


PARTNERS = ["DLS", "ACC", "V1", "CA1"]


def test_similarity_matrix_identical_vectors_score_one():
    # Animal 1's DMS weight vector is the SAME in its DLS and ACC pairs.
    v = np.array([[1.0], [0.0], [0.0]])
    results = [_pair(1, "DMS", "DLS", v, np.zeros((4, 1))),
               _pair(1, "DMS", "ACC", v, np.zeros((4, 1)))]
    mat, cnt = crosspair.similarity_matrix(results, "DMS", "naive", PARTNERS)
    assert mat.shape == (4, 4)
    assert mat[0, 1] == 1.0 and mat[1, 0] == 1.0          # DLS vs ACC
    assert cnt[0, 1] == 1
    assert np.isnan(mat[0, 2])                            # V1 pair absent


def test_similarity_matrix_orthogonal_vectors_score_zero():
    results = [_pair(1, "DMS", "DLS", np.array([[1.0], [0.0]]), np.zeros((4, 1))),
               _pair(1, "DMS", "ACC", np.array([[0.0], [1.0]]), np.zeros((4, 1)))]
    mat, _ = crosspair.similarity_matrix(results, "DMS", "naive", PARTNERS)
    assert mat[0, 1] == 0.0


def test_similarity_matrix_reads_y_side_when_area_is_y():
    # DMS is the Y area in V1-DMS; crosspair must read weights_y there.
    v = np.array([[1.0], [1.0], [0.0]])
    results = [_pair(1, "DMS", "DLS", v, np.zeros((4, 1))),        # DMS is X
               _pair(1, "V1", "DMS", np.zeros((5, 1)), v)]         # DMS is Y
    mat, _ = crosspair.similarity_matrix(results, "DMS", "naive", PARTNERS)
    assert abs(mat[0, 2] - 1.0) < 1e-12                   # DLS vs V1, same vec


def test_similarity_matrix_averages_across_animals():
    # animal 1 -> similarity 1, animal 2 -> similarity 0, mean 0.5.
    a = np.array([[1.0], [0.0]])
    b = np.array([[0.0], [1.0]])
    results = [_pair(1, "DMS", "DLS", a, np.zeros((4, 1))),
               _pair(1, "DMS", "ACC", a, np.zeros((4, 1))),       # sim 1
               _pair(2, "DMS", "DLS", a, np.zeros((4, 1))),
               _pair(2, "DMS", "ACC", b, np.zeros((4, 1)))]       # sim 0
    mat, cnt = crosspair.similarity_matrix(results, "DMS", "naive", PARTNERS)
    assert mat[0, 1] == 0.5
    assert cnt[0, 1] == 2


# ---------------------------------------------------------------------------
# mean_pairwise
# ---------------------------------------------------------------------------
def test_mean_pairwise_excludes_the_diagonal():
    mat = np.array([[1.0, 0.4, 0.6],
                    [0.4, 1.0, 0.8],
                    [0.6, 0.8, 1.0]])
    assert abs(crosspair.mean_pairwise(mat) - 0.6) < 1e-12


def test_mean_pairwise_ignores_nan_cells():
    mat = np.array([[1.0, 0.5, np.nan],
                    [0.5, 1.0, np.nan],
                    [np.nan, np.nan, np.nan]])
    assert crosspair.mean_pairwise(mat) == 0.5
