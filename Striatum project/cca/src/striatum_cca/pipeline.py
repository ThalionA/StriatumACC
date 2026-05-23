"""Per-(animal, pair) orchestration of the residual CCA.

Composes the primitives in :mod:`core` and :mod:`dataio` into the Stage-1
pipeline: per-epoch residualisation, a per-epoch PCA per area, and 5-fold
cross-validated CCA for each of the three learning epochs.

PCA is fitted *per epoch* (not on a shared 30-trial basis). A shared basis
would let a component carry near-zero variance inside one epoch -- e.g. units
silent in the naive epoch but active later -- which makes that epoch's CCA
ill-conditioned. Cross-epoch subspace comparison (D10) is therefore done in
neuron space, the frame common to all epochs, by back-projecting the canonical
coefficients through each epoch's PCA loadings. See UNDERSTANDING.md edit log.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config, core, dataio


@dataclass
class PairFit:
    """The fitted residual CCA for one (animal, area-pair) across all epochs."""

    animal_id: int
    area_x: str
    area_y: str
    role: str                              # "learner" | "nonlearner"
    lp: int                                # learning point used (real or yoked)
    k: int                                 # PCs per area (D4), fixed over epochs
    n_units_x: int
    n_units_y: int
    unit_index_x: np.ndarray               # kept X units' indices into unit axis
    unit_index_y: np.ndarray
    pca_x_by_epoch: dict[str, core.PCAState]   # per-epoch PCA basis for X
    pca_y_by_epoch: dict[str, core.PCAState]
    epochs: dict[str, core.CVResult]       # epoch name -> cross-validated CCA


@dataclass
class SkippedPair:
    """A (animal, pair) that could not be fit, with the reason."""

    animal_id: int
    area_x: str
    area_y: str
    reason: str


def fit_pair(
    animal: dataio.Animal,
    area_x: str,
    area_y: str,
    entry: dataio.CohortEntry,
    cfg=config.DEFAULT,
) -> PairFit | SkippedPair:
    """Fit the residual CCA for one area pair in one animal.

    Steps (D2, D4, D8): build the two area tensors (FS units excluded,
    disengagement-truncated) -> per-epoch residualisation -> choose k, capped
    by the smallest per-epoch numerical rank -> per-epoch PCA per area ->
    5-fold cross-validated CCA per epoch.
    """
    tensor_x, idx_x = dataio.area_tensor(animal, area_x, cfg)
    tensor_y, idx_y = dataio.area_tensor(animal, area_y, cfg)

    n_units_x = tensor_x.shape[2]
    n_units_y = tensor_y.shape[2]
    if n_units_x < cfg.min_units or n_units_y < cfg.min_units:
        return SkippedPair(
            animal.animal_id, area_x, area_y,
            f"too few units ({area_x}={n_units_x}, {area_y}={n_units_y}; "
            f"min {cfg.min_units})",
        )

    n_use = tensor_x.shape[0]
    windows = dataio.epoch_windows(entry.lp, n_use, cfg)
    if windows is None:
        return SkippedPair(
            animal.animal_id, area_x, area_y,
            f"no valid epochs (lp={entry.lp}, usable trials={n_use})",
        )

    # Per-epoch residualisation (D2): subtract each epoch's per-(bin, unit)
    # trial mean. The non-subtracted variant is a Stage-2 supplementary check.
    res_x = {e: _residual(tensor_x[idx], cfg) for e, idx in windows.items()}
    res_y = {e: _residual(tensor_y[idx], cfg) for e, idx in windows.items()}

    # k from the per-epoch sample budget (D4), capped at the smallest
    # per-epoch numerical rank so every epoch's CCA gets full-rank input.
    n_samples_epoch = cfg.trials_per_epoch * config.N_BINS
    min_rank = min(
        min(core.numerical_rank(res_x[e]) for e in config.EPOCH_NAMES),
        min(core.numerical_rank(res_y[e]) for e in config.EPOCH_NAMES),
    )
    k = core.choose_k(n_units_x, n_units_y, n_samples_epoch, cfg, max_rank=min_rank)

    pca_x_by_epoch: dict[str, core.PCAState] = {}
    pca_y_by_epoch: dict[str, core.PCAState] = {}
    epochs: dict[str, core.CVResult] = {}
    for epoch in config.EPOCH_NAMES:
        pca_x = core.pca_fit(res_x[epoch], k)
        pca_y = core.pca_fit(res_y[epoch], k)
        scores_x = core.pca_transform(res_x[epoch], pca_x)
        scores_y = core.pca_transform(res_y[epoch], pca_y)
        pca_x_by_epoch[epoch] = pca_x
        pca_y_by_epoch[epoch] = pca_y
        epochs[epoch] = core.cca_cv(scores_x, scores_y, cfg)

    return PairFit(
        animal_id=animal.animal_id,
        area_x=area_x,
        area_y=area_y,
        role=entry.role,
        lp=entry.lp,
        k=k,
        n_units_x=n_units_x,
        n_units_y=n_units_y,
        unit_index_x=idx_x,
        unit_index_y=idx_y,
        pca_x_by_epoch=pca_x_by_epoch,
        pca_y_by_epoch=pca_y_by_epoch,
        epochs=epochs,
    )


def _residual(tensor: np.ndarray, cfg) -> np.ndarray:
    """Impute missing bins, then residualise (or, for the signal variant,
    remove only the per-unit grand mean)."""
    filled = core.impute_missing_bins(tensor)
    if cfg.subtract_trial_mean:
        return core.residualise(filled)
    return filled - filled.mean(axis=(0, 1), keepdims=True)
