"""Stage 2 analysis: lagged CCA + surrogate significance per (animal, pair).

:func:`analyse_pair` is pure compute over the small per-epoch PC-score tensors
in a :class:`~striatum_cca.pipeline.PreparedPair`, so it parallelises cleanly
across processes (D11).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config, core, lagged, pipeline, surrogate


@dataclass
class EpochAnalysis:
    """Full Stage-2 result for one (animal, pair, epoch). All CC values are
    held-out (5-fold cross-validated)."""

    epoch: str
    k: int
    n_samples: int
    # Lag-0 CCA.
    held_out_cc: np.ndarray        # (d,) held-out canonical correlations
    in_sample_cc: np.ndarray       # (d,) in-sample CC (diagnostic only)
    cca_full: core.CCAResult       # lag-0 full-data fit (A, B for Stage 3)
    # Lagged CCA / directionality (D6) — held-out, all canonical dimensions.
    lags: np.ndarray               # (2*max_lag+1,) integer bin lags
    lag_cc_per_dim: np.ndarray     # (n_lags, d) held-out CC per lag per dim
    ifi_per_dim: np.ndarray        # (d,) IFI per dimension
    ifi_windows: np.ndarray        # (d, max_window) IFI by lag window (point 4)
    peak_lag_per_dim: np.ndarray   # (d,)
    # Subspace significance (D7) — held-out-CC permutation test, per dimension.
    null_held_out: np.ndarray      # (n_shuffles, d) trial-permutation null
    p_per_dim: np.ndarray          # (d,) per-dimension held-out p-values
    n_significant: int             # significant communication-subspace dims

    # Convenience accessors for the dominant canonical dimension.
    @property
    def lag_cc1(self) -> np.ndarray:
        return self.lag_cc_per_dim[:, 0]

    @property
    def ifi(self) -> float:
        return float(self.ifi_per_dim[0])

    @property
    def peak_lag(self) -> int:
        return int(self.peak_lag_per_dim[0])

    @property
    def p_cc1(self) -> float:
        return float(self.p_per_dim[0])


@dataclass
class PairAnalysis:
    """Stage-2 result for one (animal, area-pair) across all epochs."""

    animal_id: int
    area_x: str
    area_y: str
    role: str
    lp: int
    k: int
    n_units_x: int
    n_units_y: int
    unit_index_x: np.ndarray
    unit_index_y: np.ndarray
    pca_x: dict[str, core.PCAState]      # epoch -> PCA basis (for Stage 3)
    pca_y: dict[str, core.PCAState]
    epochs: dict[str, EpochAnalysis]


def analyse_pair(prepared: pipeline.PreparedPair, cfg=config.DEFAULT) -> PairAnalysis:
    """Run the lagged CCA and surrogate nulls for one prepared area pair."""
    epochs: dict[str, EpochAnalysis] = {}
    for epoch in config.EPOCH_NAMES:
        scores_x = prepared.scores_x[epoch]
        scores_y = prepared.scores_y[epoch]

        # Lag-0 held-out CCA -- unbiased effect size + the full A, B.
        cv0 = core.cca_cv(scores_x, scores_y, cfg)
        # Held-out lagged curve, all canonical dimensions -- directionality.
        lag = lagged.lag_curve(scores_x, scores_y, cfg, held_out=True)
        # Significance: held-out-CC permutation test, per dimension (the
        # in-sample test over-called -- spectrum shift; see surrogate.py).
        null = surrogate.build_null(scores_x, scores_y, cv0.held_out_r, cfg)

        epochs[epoch] = EpochAnalysis(
            epoch=epoch,
            k=cv0.k,
            n_samples=cv0.n_samples,
            held_out_cc=cv0.held_out_r,
            in_sample_cc=cv0.in_sample_r,
            cca_full=cv0.full,
            lags=lag.lags,
            lag_cc_per_dim=lag.cc_per_dim,
            ifi_per_dim=lag.ifi_per_dim,
            ifi_windows=lag.ifi_windows,
            peak_lag_per_dim=lag.peak_lag_per_dim,
            null_held_out=null.null_held_out,
            p_per_dim=null.p_per_dim,
            n_significant=null.n_significant,
        )

    return PairAnalysis(
        animal_id=prepared.animal_id,
        area_x=prepared.area_x,
        area_y=prepared.area_y,
        role=prepared.role,
        lp=prepared.lp,
        k=prepared.k,
        n_units_x=prepared.n_units_x,
        n_units_y=prepared.n_units_y,
        unit_index_x=prepared.unit_index_x,
        unit_index_y=prepared.unit_index_y,
        pca_x=prepared.pca_x,
        pca_y=prepared.pca_y,
        epochs=epochs,
    )
