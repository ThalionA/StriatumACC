"""Per-(animal, pair) orchestration of the residual CCA.

Two-step structure so the data-heavy preparation can be separated from the
parallelisable compute:

* :func:`prepare_pair` -- load, FS-select, residualise, per-epoch PCA. Produces
  small per-epoch PC-score tensors. Runs in the main process.
* :func:`fit_pair` (Stage 1) and :func:`striatum_cca.analysis.analyse_pair`
  (Stage 2) -- pure compute on those small score tensors.

PCA is fitted *per epoch* (not on a shared 30-trial basis): a shared basis
would let a component carry near-zero variance inside one epoch, making that
epoch's CCA ill-conditioned. Cross-epoch subspace comparison (D10) is therefore
done in neuron space. See UNDERSTANDING.md edit log v3.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from . import config, core, dataio, partial


@dataclass
class PreparedPair:
    """Per-epoch PCA-reduced residual scores for one (animal, area-pair).

    Small enough (~0.5 MB) to ship cheaply to worker processes.
    """

    animal_id: int
    area_x: str
    area_y: str
    role: str                              # "learner" | "nonlearner"
    lp: int
    k: int                                 # PCs per area (D4), fixed over epochs
    n_units_x: int
    n_units_y: int
    unit_index_x: np.ndarray
    unit_index_y: np.ndarray
    scores_x: dict[str, np.ndarray]        # epoch -> (n_trials, n_bins, k)
    scores_y: dict[str, np.ndarray]
    pca_x: dict[str, core.PCAState]        # epoch -> PCA basis for X
    pca_y: dict[str, core.PCAState]


@dataclass
class SkippedPair:
    """A (animal, pair) that could not be prepared, with the reason."""

    animal_id: int
    area_x: str
    area_y: str
    reason: str


@dataclass
class PairFit:
    """Stage-1 cross-validated CCA for one (animal, area-pair) over all epochs."""

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
    pca_x_by_epoch: dict[str, core.PCAState]
    pca_y_by_epoch: dict[str, core.PCAState]
    epochs: dict[str, core.CVResult]


def config_label(cfg) -> str:
    """One-line human description of an analysis config."""
    binning = (f"temporal {cfg.temporal_bin_ms} ms"
               if cfg.bin_mode == "temporal" else "spatial")
    cca = "residual" if cfg.subtract_trial_mean else "signal"
    fs = "FS-excl" if cfg.exclude_fast_spiking else "FS-incl"
    z = "z-on" if cfg.zscore_units else "z-off"
    return f"{binning}, {cca}, {fs}, {z}, k={cfg.k_mode}"


def prepare_pair(
    animal: dataio.Animal,
    area_x: str,
    area_y: str,
    entry: dataio.CohortEntry,
    cfg=config.DEFAULT,
) -> PreparedPair | SkippedPair:
    """Build per-epoch PCA-reduced residual scores for one area pair.

    Steps (D2, D4): area tensors (FS units excluded, disengagement-truncated)
    -> whole-engaged-period z-scoring -> per-epoch residualisation -> k from
    the sample budget, capped at the smallest per-epoch numerical rank ->
    per-epoch PCA.
    """
    tensor_x, idx_x = dataio.area_tensor(animal, area_x, cfg)
    tensor_y, idx_y = dataio.area_tensor(animal, area_y, cfg)
    if cfg.zscore_units:
        # Z-score each unit over the entire engaged period -- the first thing
        # done to the activity, before epoch slicing and residualisation.
        tensor_x = _zscore_area(tensor_x)
        tensor_y = _zscore_area(tensor_y)

    n_units_x = len(idx_x)
    n_units_y = len(idx_y)
    if n_units_x < cfg.min_units or n_units_y < cfg.min_units:
        return SkippedPair(
            animal.animal_id, area_x, area_y,
            f"too few units ({area_x}={n_units_x}, {area_y}={n_units_y}; "
            f"min {cfg.min_units})",
        )

    n_use = len(tensor_x)
    windows = dataio.epoch_windows(entry.lp, n_use, cfg)
    if windows is None:
        return SkippedPair(
            animal.animal_id, area_x, area_y,
            f"no valid epochs (lp={entry.lp}, usable trials={n_use})",
        )

    res_x = {e: _residual(_slice_epoch(tensor_x, idx), cfg)
             for e, idx in windows.items()}
    res_y = {e: _residual(_slice_epoch(tensor_y, idx), cfg)
             for e, idx in windows.items()}

    # k from the smallest *valid* (non-missing) per-epoch sample count.
    n_valid = min(core.n_valid_samples(res_x[e]) for e in config.EPOCH_NAMES)
    min_rank = min(
        min(core.numerical_rank(res_x[e]) for e in config.EPOCH_NAMES),
        min(core.numerical_rank(res_y[e]) for e in config.EPOCH_NAMES),
    )
    k = core.choose_k(n_units_x, n_units_y, n_valid, cfg, max_rank=min_rank,
                      variance_k=_variance_k([res_x, res_y], cfg))

    scores_x: dict[str, np.ndarray] = {}
    scores_y: dict[str, np.ndarray] = {}
    pca_x: dict[str, core.PCAState] = {}
    pca_y: dict[str, core.PCAState] = {}
    for epoch in config.EPOCH_NAMES:
        px = core.pca_fit(res_x[epoch], k)
        py = core.pca_fit(res_y[epoch], k)
        pca_x[epoch] = px
        pca_y[epoch] = py
        scores_x[epoch] = core.pca_transform(res_x[epoch], px)
        scores_y[epoch] = core.pca_transform(res_y[epoch], py)

    return PreparedPair(
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
        scores_x=scores_x,
        scores_y=scores_y,
        pca_x=pca_x,
        pca_y=pca_y,
    )


def _residual_tensors(
    animal: dataio.Animal, area: str, entry: dataio.CohortEntry, cfg
) -> tuple[dict[str, np.ndarray], np.ndarray] | None:
    """Per-epoch residualised (z-scored) neuron tensors for one area.

    Returns ``(res_by_epoch, unit_index)`` -- the same residual tensors
    :func:`prepare_pair` builds internally -- or None if the area has too few
    units or no valid epochs.
    """
    tensor, idx = dataio.area_tensor(animal, area, cfg)
    if len(idx) < cfg.min_units:
        return None
    if cfg.zscore_units:
        tensor = _zscore_area(tensor)
    windows = dataio.epoch_windows(entry.lp, len(tensor), cfg)
    if windows is None:
        return None
    res = {e: _residual(_slice_epoch(tensor, e_idx), cfg)
           for e, e_idx in windows.items()}
    return res, idx


def prepare_pair_partial(
    animal: dataio.Animal,
    area_x: str,
    area_y: str,
    entry: dataio.CohortEntry,
    cfg=config.DEFAULT,
) -> PreparedPair | SkippedPair:
    """Like :func:`prepare_pair`, but with every other recorded area removed.

    Each other area's PC scores are regressed out of X's and Y's residualised
    neuron tensors (neuron-level partialling, before the per-epoch PCA) so the
    PCA basis -- and hence the Stage-3 back-projection -- stays in X's / Y's own
    neuron space. Returns a SkippedPair if X or Y is unusable, or if the animal
    has no other area to condition on.
    """
    rx = _residual_tensors(animal, area_x, entry, cfg)
    ry = _residual_tensors(animal, area_y, entry, cfg)
    if rx is None or ry is None:
        return SkippedPair(animal.animal_id, area_x, area_y,
                           "X or Y unusable for partial preparation")
    res_x, idx_x = rx
    res_y, idx_y = ry

    confounds = []
    for area_z in config.AREAS:
        if area_z in (area_x, area_y):
            continue
        sz = prepare_area(animal, area_z, entry, cfg)
        if sz is not None:
            confounds.append(sz)
    if not confounds:
        return SkippedPair(animal.animal_id, area_x, area_y,
                           "no other area to partial out")

    res_xp, res_yp = {}, {}
    for epoch in config.EPOCH_NAMES:
        z = np.concatenate([c[epoch] for c in confounds], axis=-1)
        res_xp[epoch] = partial.partial_out_tensor(res_x[epoch], z)
        res_yp[epoch] = partial.partial_out_tensor(res_y[epoch], z)

    n_units_x, n_units_y = len(idx_x), len(idx_y)
    n_valid = min(core.n_valid_samples(res_xp[e]) for e in config.EPOCH_NAMES)
    min_rank = min(
        min(core.numerical_rank(res_xp[e]) for e in config.EPOCH_NAMES),
        min(core.numerical_rank(res_yp[e]) for e in config.EPOCH_NAMES),
    )
    k = core.choose_k(n_units_x, n_units_y, n_valid, cfg, max_rank=min_rank,
                      variance_k=_variance_k([res_xp, res_yp], cfg))

    scores_x: dict[str, np.ndarray] = {}
    scores_y: dict[str, np.ndarray] = {}
    pca_x: dict[str, core.PCAState] = {}
    pca_y: dict[str, core.PCAState] = {}
    for epoch in config.EPOCH_NAMES:
        px = core.pca_fit(res_xp[epoch], k)
        py = core.pca_fit(res_yp[epoch], k)
        pca_x[epoch] = px
        pca_y[epoch] = py
        scores_x[epoch] = core.pca_transform(res_xp[epoch], px)
        scores_y[epoch] = core.pca_transform(res_yp[epoch], py)

    return PreparedPair(
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
        scores_x=scores_x,
        scores_y=scores_y,
        pca_x=pca_x,
        pca_y=pca_y,
    )


def fit_pair(
    animal: dataio.Animal,
    area_x: str,
    area_y: str,
    entry: dataio.CohortEntry,
    cfg=config.DEFAULT,
) -> PairFit | SkippedPair:
    """Stage-1 cross-validated CCA for one area pair (lag 0, no surrogates)."""
    prepared = prepare_pair(animal, area_x, area_y, entry, cfg)
    if isinstance(prepared, SkippedPair):
        return prepared
    epochs = {
        e: core.cca_cv(prepared.scores_x[e], prepared.scores_y[e], cfg)
        for e in config.EPOCH_NAMES
    }
    return PairFit(
        animal_id=prepared.animal_id,
        area_x=area_x,
        area_y=area_y,
        role=prepared.role,
        lp=prepared.lp,
        k=prepared.k,
        n_units_x=prepared.n_units_x,
        n_units_y=prepared.n_units_y,
        unit_index_x=prepared.unit_index_x,
        unit_index_y=prepared.unit_index_y,
        pca_x_by_epoch=prepared.pca_x,
        pca_y_by_epoch=prepared.pca_y,
        epochs=epochs,
    )


def prepare_area(
    animal: dataio.Animal, area: str, entry: dataio.CohortEntry, cfg=config.DEFAULT
) -> dict[str, np.ndarray] | None:
    """Per-epoch residual PC scores for a single area (used by partial CCA).

    Returns ``{epoch: (n_trials, n_bins, k)}`` or None if the area has too few
    units or no valid epochs. k is chosen from the area's own unit count.
    """
    tensor, idx = dataio.area_tensor(animal, area, cfg)
    if len(idx) < cfg.min_units:
        return None
    if cfg.zscore_units:
        tensor = _zscore_area(tensor)               # whole engaged period
    windows = dataio.epoch_windows(entry.lp, len(tensor), cfg)
    if windows is None:
        return None
    res = {e: _residual(_slice_epoch(tensor, e_idx), cfg)
           for e, e_idx in windows.items()}
    n_units = len(idx)
    n_valid = min(core.n_valid_samples(res[e]) for e in config.EPOCH_NAMES)
    min_rank = min(core.numerical_rank(res[e]) for e in config.EPOCH_NAMES)
    k = core.choose_k(n_units, n_units, n_valid, cfg, max_rank=min_rank,
                      variance_k=_variance_k([res], cfg))
    return {
        e: core.pca_transform(res[e], core.pca_fit(res[e], k))
        for e in config.EPOCH_NAMES
    }


def _residual(tensor: np.ndarray, cfg) -> np.ndarray:
    """Per-epoch centring: residualise (subtract the per-(bin, unit) trial
    mean) or, for the signal variant, remove only the per-unit grand mean.

    Units are z-scored upstream over the whole engaged period (see
    :func:`prepare_pair`), so no per-unit scaling happens here. Missing
    samples stay NaN and are dropped later, at fit time.
    """
    if cfg.subtract_trial_mean:
        return core.residualise(tensor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        grand = np.nanmean(tensor, axis=(0, 1), keepdims=True)
    return tensor - grand


def _zscore_area(area_data):
    """Whole-engaged-period per-unit z-scoring (round 7).

    ``area_data`` is a spatial 3-D tensor or a temporal ragged list of
    per-trial 2-D arrays; both are divided by each unit's std over every
    (trial, bin) sample of the engaged period.
    """
    if isinstance(area_data, np.ndarray):
        return core.zscore_units(area_data)
    flat = np.concatenate(area_data, axis=0)            # (total_bins, n_units)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        std = np.nanstd(flat, axis=0)
    std = np.where(std > 0, std, 1.0)
    return [trial / std for trial in area_data]


def _slice_epoch(area_data, idx: np.ndarray) -> np.ndarray:
    """The 3-D ``(n_epoch_trials, n_bins, n_units)`` tensor for one epoch.

    Spatial data (a 3-D tensor) is sliced directly; temporal data (a ragged
    per-trial list) is NaN-padded to the epoch's longest trial.
    """
    if isinstance(area_data, np.ndarray):
        return area_data[idx]
    return dataio.pad_trials([area_data[i] for i in idx])


def _variance_k(area_residuals, cfg) -> int | None:
    """PCs reaching ``cfg.k_variance`` cumulative variance, pooled over epochs
    and taken as the symmetric (min) value across the supplied areas. Returns
    None unless ``cfg.k_mode == "variance"`` (the other modes ignore it).
    """
    if cfg.k_mode != "variance":
        return None
    ks = []
    for res in area_residuals:
        pooled = np.concatenate(
            [r.reshape(-1, r.shape[-1]) for r in res.values()])
        ks.append(core.k_for_variance(pooled, cfg.k_variance))
    return min(ks)
