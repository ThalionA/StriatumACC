"""Shared per-cohort fitting for the continuous-regime drivers (epochs, engagement).

The report's temporal pipeline fits, per (animal, area-pair, window), the full
``subspace_window`` readout on the **running, in-corridor** time bins of that
window, partialling out every other simultaneously recorded area (Z). This module
holds the reusable pieces so ``run_epochs`` (naive/intermediate/expert) and
``run_engagement`` (engaged vs disengaged) share identical fitting logic.

All inputs are the z-scored, run-masked per-area activity; the windowing differs
between drivers (epoch trial-windows vs engagement periods).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config, dataio, membership, subspace, subspace_window


# ---------------------------------------------------------------------------
# per-area activity on the running, in-corridor timeline
# ---------------------------------------------------------------------------
def build_present(streams: dataio.TemporalStreams, run: np.ndarray, animal, cfg,
                  ref_mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Z-scored, run-masked activity for every area with >= min_units.

    ``run`` is the in-trial running mask over the full timeline. Each kept area's
    activity is z-scored over ``ref_mask`` (defaults to ``run`` -- the engaged
    running reference, report Section 2.2) then restricted to the running bins.
    Returns ``{area: (n_run_bins, n_units_kept)}``.
    """
    if ref_mask is None:
        ref_mask = run
    present: dict[str, np.ndarray] = {}
    for area in config.AREAS:
        idx = dataio.select_units(animal, area, cfg)
        if len(idx) < cfg.min_units:
            continue
        spikes = streams.spikes[:, idx]
        if cfg.zscore_units:
            spikes = dataio.zscore_over_reference(spikes, ref_mask)
        present[area] = spikes[run]
    return present


def fit_window(present: dict[str, np.ndarray], trial_ids_run: np.ndarray,
               ax: str, ay: str, trials_in_window: np.ndarray, cfg, *,
               k: int, max_lag: int, n_shuffles: int, n_folds: int,
               min_cell_bins: int) -> subspace_window.WindowSubspace | None:
    """Fit ``subspace_window`` for one (pair, window) cell, or None if too sparse.

    Z = every other present area concatenated (partial CCA control, report 2.4).
    """
    if ax not in present or ay not in present:
        return None
    mask = np.isin(trial_ids_run, trials_in_window)
    if int(mask.sum()) < min_cell_bins:
        return None
    Xw, Yw = present[ax][mask], present[ay][mask]
    others = [present[z][mask] for z in present if z not in (ax, ay)]
    Z = np.concatenate(others, axis=1) if others else None
    return subspace_window.window_subspace(
        Xw, Yw, trial_ids_run[mask], Z=Z, k=k, max_lag=max_lag,
        n_shuffles=n_shuffles, n_folds=n_folds, seed=cfg.cv_seed,
    )


# ---------------------------------------------------------------------------
# cross-window subspace comparison (rotation + membership turnover)
# ---------------------------------------------------------------------------
def _max_angle(wa: np.ndarray, wb: np.ndarray, d_use: int) -> float:
    """Max principal angle (deg) between the top-``d_use`` columns of two weight sets."""
    d = int(min(d_use, wa.shape[1], wb.shape[1]))
    if d < 1:
        return float("nan")
    qa, _ = np.linalg.qr(wa[:, :d])
    qb, _ = np.linalg.qr(wb[:, :d])
    return float(np.degrees(np.nanmax(subspace.principal_angles(qa, qb))))


@dataclass
class CrossWindow:
    rot_x_cc1: float          # CC1 (dominant-dim) cross-window rotation, X side
    rot_y_cc1: float
    floor_x_cc1: float        # within-window split-half CC1 floor (mean of the two windows)
    floor_y_cc1: float
    rot_x_top3: float         # top-3 subspace rotation
    rot_y_top3: float
    jaccard_x: float          # member-set turnover
    jaccard_y: float


def cross_window(wa: subspace_window.WindowSubspace,
                 wb: subspace_window.WindowSubspace) -> CrossWindow:
    """Rotation (CC1 + top-3) and membership Jaccard between two windows' fits.

    A rotation counts as reorientation only if it exceeds the within-window
    split-half floor (averaged across the two windows); the driver records both.
    """
    return CrossWindow(
        rot_x_cc1=_max_angle(wa.weights_x, wb.weights_x, 1),
        rot_y_cc1=_max_angle(wa.weights_y, wb.weights_y, 1),
        floor_x_cc1=float(np.nanmean([wa.split_half_x_cc1, wb.split_half_x_cc1])),
        floor_y_cc1=float(np.nanmean([wa.split_half_y_cc1, wb.split_half_y_cc1])),
        rot_x_top3=_max_angle(wa.weights_x, wb.weights_x, 3),
        rot_y_top3=_max_angle(wa.weights_y, wb.weights_y, 3),
        jaccard_x=membership.jaccard(wa.member_x, wb.member_x),
        jaccard_y=membership.jaccard(wa.member_y, wb.member_y),
    )
