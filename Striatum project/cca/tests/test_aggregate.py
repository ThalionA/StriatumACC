"""Tests for the shared Stage-2 aggregation helpers (aggregate.py).

These cover the per-pair / per-significant-dimension / per-animal reshaping
used by both the figure scripts and the epoch-ANOVA table, so the two paths
stay consistent. Lightweight stand-in objects mimic the EpochAnalysis /
PairAnalysis attributes the helpers actually touch.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from striatum_cca import aggregate, config


def _epoch(p_per_dim, held_out_cc, ifi_w10):
    """A stand-in EpochAnalysis; ifi_windows has 10 columns, col 9 == w10."""
    d = len(p_per_dim)
    ifi_windows = np.zeros((d, 10))
    ifi_windows[:, 9] = ifi_w10
    return SimpleNamespace(
        p_per_dim=np.asarray(p_per_dim, float),
        held_out_cc=np.asarray(held_out_cc, float),
        ifi_windows=ifi_windows,
    )


def _pair(animal_id, area_x, area_y, role, epochs):
    return SimpleNamespace(
        animal_id=animal_id, area_x=area_x, area_y=area_y, role=role,
        epochs=epochs,
    )


def _cohort():
    """Two DMS-ACC learners + one non-learner, three epochs each."""
    en = config.EPOCH_NAMES
    a1 = _pair(1, "DMS", "ACC", "learner", {
        en[0]: _epoch([0.01, 0.5, 0.02], [0.4, 0.1, 0.3], [0.6, 0.0, 0.2]),
        en[1]: _epoch([0.01, 0.5, 0.5], [0.35, 0.1, 0.1], [0.1, 0.0, 0.0]),
        en[2]: _epoch([0.04, 0.5, 0.5], [0.2, 0.1, 0.1], [-0.4, 0.0, 0.0]),
    })
    a2 = _pair(2, "DMS", "ACC", "learner", {
        en[0]: _epoch([0.02], [0.45], [0.5]),
        en[1]: _epoch([0.02], [0.3], [0.2]),
        en[2]: _epoch([0.5], [0.25], [-0.3]),     # no significant dim in expert
    })
    a3 = _pair(3, "DMS", "ACC", "nonlearner", {
        en[0]: _epoch([0.01], [0.9], [0.9]),
        en[1]: _epoch([0.01], [0.9], [0.9]),
        en[2]: _epoch([0.01], [0.9], [0.9]),
    })
    return [a1, a2, a3]


# ---------------------------------------------------------------------------
def test_learner_pairs_filters_role_and_pair():
    results = _cohort()
    rs = aggregate.learner_pairs(results, "DMS", "ACC")
    assert [r.animal_id for r in rs] == [1, 2]       # non-learner excluded
    assert aggregate.learner_pairs(results, "DMS", "DLS") == []


def test_sig_dims_thresholds_at_alpha():
    ep = _epoch([0.01, 0.5, 0.04], [0, 0, 0], [0, 0, 0])
    assert np.array_equal(aggregate.sig_dims(ep), [0, 2])


def test_dim_values_selects_significant_cc_and_ifi():
    results = _cohort()
    a1 = aggregate.learner_pairs(results, "DMS", "ACC")[0]
    en = config.EPOCH_NAMES
    # naive: dims 0 and 2 significant -> CC [0.4, 0.3], IFI(w10) [0.6, 0.2]
    assert np.allclose(aggregate.dim_values(a1, en[0], "cc"), [0.4, 0.3])
    assert np.allclose(aggregate.dim_values(a1, en[0], "ifi", window=10),
                       [0.6, 0.2])


def test_per_dim_groups_pools_over_animals():
    results = _cohort()
    learners = aggregate.learner_pairs(results, "DMS", "ACC")
    groups = aggregate.per_dim_groups(learners, "cc")
    # naive: a1 has 2 sig dims (0.4,0.3), a2 has 1 (0.45) -> 3 pooled values.
    assert sorted(np.round(groups[0], 2)) == [0.3, 0.4, 0.45]
    # expert: a1 has 1 sig dim (0.2), a2 has none -> 1 value.
    assert np.allclose(groups[2], [0.2])


def test_per_animal_matrix_means_over_dims_and_drops_incomplete():
    results = _cohort()
    learners = aggregate.learner_pairs(results, "DMS", "ACC")
    mat = aggregate.per_animal_matrix(learners, "cc")
    # a2 has no significant expert dim -> NaN row -> dropped; only a1 remains.
    assert mat.shape == (1, 3)
    assert np.allclose(mat[0], [np.mean([0.4, 0.3]), 0.35, 0.2])


def test_per_animal_matrix_ifi_window():
    results = _cohort()
    learners = aggregate.learner_pairs(results, "DMS", "ACC")
    mat = aggregate.per_animal_matrix(learners, "ifi", window=10)
    assert mat.shape == (1, 3)                       # only a1 complete
    assert np.allclose(mat[0], [np.mean([0.6, 0.2]), 0.1, -0.4])
