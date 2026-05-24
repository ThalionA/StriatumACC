"""End-to-end test for the Stage-2 analyse_pair driver."""

from __future__ import annotations

import dataclasses

import numpy as np

from striatum_cca import analysis, config, core, pipeline

CFG = dataclasses.replace(config.DEFAULT, n_shuffles=40)


def make_prepared(shared: bool, seed: int = 0) -> pipeline.PreparedPair:
    """A PreparedPair with (optionally) a genuine shared latent across areas."""
    rng = np.random.default_rng(seed)
    k, n_tr, n_bins, n_units = 5, 12, config.N_BINS, 8
    scores_x, scores_y, pca_x, pca_y = {}, {}, {}, {}
    for epoch in config.EPOCH_NAMES:
        latent = rng.standard_normal((n_tr, n_bins, k))
        scores_x[epoch] = latent + 0.3 * rng.standard_normal((n_tr, n_bins, k))
        if shared:
            scores_y[epoch] = latent + 0.3 * rng.standard_normal((n_tr, n_bins, k))
        else:
            scores_y[epoch] = rng.standard_normal((n_tr, n_bins, k))
        dummy = core.PCAState(np.zeros(n_units), np.zeros((n_units, k)), np.ones(k) / k)
        pca_x[epoch] = dummy
        pca_y[epoch] = dummy
    return pipeline.PreparedPair(
        animal_id=1, area_x="DMS", area_y="ACC", role="learner", lp=42, k=k,
        n_units_x=n_units, n_units_y=n_units,
        unit_index_x=np.arange(n_units), unit_index_y=np.arange(n_units),
        scores_x=scores_x, scores_y=scores_y, pca_x=pca_x, pca_y=pca_y,
    )


def test_analyse_pair_returns_complete_structure():
    result = analysis.analyse_pair(make_prepared(shared=True), CFG)
    assert isinstance(result, analysis.PairAnalysis)
    assert set(result.epochs) == set(config.EPOCH_NAMES)
    for epoch in config.EPOCH_NAMES:
        ea = result.epochs[epoch]
        assert ea.null_held_out.shape == (40, ea.held_out_cc.shape[0])
        assert ea.lags.shape == ea.lag_cc1.shape
        assert np.isfinite(ea.ifi)
        assert ea.cca_full.A.shape[1] == ea.held_out_cc.shape[0]


def test_analyse_pair_detects_communication():
    result = analysis.analyse_pair(make_prepared(shared=True), CFG)
    for epoch in config.EPOCH_NAMES:
        ea = result.epochs[epoch]
        assert ea.held_out_cc[0] > 0.5
        assert ea.p_cc1 < 0.05
        assert ea.n_significant >= 1


def test_analyse_pair_null_data_not_significant():
    result = analysis.analyse_pair(make_prepared(shared=False), CFG)
    for epoch in config.EPOCH_NAMES:
        assert result.epochs[epoch].p_cc1 > 0.05
