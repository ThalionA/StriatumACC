"""Smoke test for the Stage-3 subspace driver."""

from __future__ import annotations

import numpy as np

from striatum_cca import config, core, pipeline, stage3


def make_prepared(seed: int = 0) -> pipeline.PreparedPair:
    """A PreparedPair with a shared latent and a stable cross-epoch subspace."""
    rng = np.random.default_rng(seed)
    k, n_tr, n_bins, n_units = 6, 14, config.N_BINS, 10
    load_x = rng.standard_normal((2, k))
    load_y = rng.standard_normal((2, k))
    scores_x, scores_y, pca_x, pca_y = {}, {}, {}, {}
    for epoch in config.EPOCH_NAMES:
        latent = rng.standard_normal((n_tr, n_bins, 2))
        scores_x[epoch] = latent @ load_x + 0.2 * rng.standard_normal((n_tr, n_bins, k))
        scores_y[epoch] = latent @ load_y + 0.2 * rng.standard_normal((n_tr, n_bins, k))
        comp = rng.standard_normal((n_units, k))
        pca_x[epoch] = core.PCAState(np.zeros(n_units), comp, np.ones(k) / k)
        pca_y[epoch] = core.PCAState(np.zeros(n_units), comp.copy(), np.ones(k) / k)
    return pipeline.PreparedPair(
        animal_id=1, area_x="DMS", area_y="ACC", role="learner", lp=42, k=k,
        n_units_x=n_units, n_units_y=n_units,
        unit_index_x=np.arange(n_units), unit_index_y=np.arange(n_units),
        scores_x=scores_x, scores_y=scores_y, pca_x=pca_x, pca_y=pca_y,
    )


def test_analyse_subspace_structure():
    result = stage3.analyse_subspace(make_prepared(), config.DEFAULT)
    assert isinstance(result, stage3.PairSubspace)
    assert set(result.epochs) == set(config.EPOCH_NAMES)
    for epoch in config.EPOCH_NAMES:
        es = result.epochs[epoch]
        assert es.weights_x.shape == (10, result.d_sub)
        assert es.member_x.shape == (10,)
        assert 0.0 <= es.gini_x <= 1.0
        assert es.member_x.sum() >= 1                       # quartile non-empty
    assert set(result.angles_x) == {
        "naive->intermediate", "intermediate->expert", "naive->expert"}
    for angles in result.angles_x.values():
        assert np.all((angles >= 0) & (angles <= np.pi / 2 + 1e-9))


def test_analyse_subspace_principal_angles_below_orthogonal():
    # With a trial-stable shared latent the canonical subspace barely rotates,
    # so the cross-epoch principal angles sit well below pi/2.
    result = stage3.analyse_subspace(make_prepared(seed=3), config.DEFAULT)
    max_angle = max(a.max() for a in result.angles_x.values())
    assert max_angle < np.pi / 2
