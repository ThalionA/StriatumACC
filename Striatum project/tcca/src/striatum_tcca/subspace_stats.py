"""Per-cell aggregation across canonical subspace dimensions.

Adapts the Gonzalez-et-al-style readout (pCCA paper, used as auxiliary tools
in the v5 spec): instead of treating dim-1 CC as THE strength stat and dim-1
IFI as THE directionality stat, aggregate across all (or all significant)
canonical dimensions.

Concretely, given one fit cell with d dimensions:

* ``n_sig`` -- number of dims passing the per-dim held-out-CC null test
  at alpha (default 0.05).
* ``mi_all`` -- Gaussian-MI surrogate ``MI = -0.5 * sum log(1 - cc_i^2)``
  summed over all d dims. Units: nats.
* ``mi_sig`` -- the same, summed only over significant dims.
* ``ifi_mean_all`` / ``ifi_mean_sig`` -- mean IFI across all / sig dims.
* ``ifi_weighted`` -- CC-magnitude-weighted IFI -- high-CC dims dominate
  the directionality readout.

These helpers are read-only over existing analysis pkls -- no re-run needed.

References
----------
Gonzalez et al. (pCCA paper) -- the Gaussian-MI surrogate over the
communication subspace; v5 spec extension notes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CC_CLIP = 0.999     # avoid log(0) when CC saturates


@dataclass
class CellSubspaceStats:
    """Subspace-aggregated statistics for one (animal, pair, epoch, landmark)."""

    n_dims: int                 # number of canonical dims fitted
    n_sig: int                  # dims passing per-dim alpha test
    peak_cc_per_dim: np.ndarray # (d,) held-out peak CC per dim, clipped
    sig_mask: np.ndarray        # (d,) bool -- per-dim significance
    mi_all: float               # nats; -0.5 * sum log(1 - cc^2) over all dims
    mi_sig: float               # nats; same restricted to sig dims (0 if n_sig=0)
    ifi_per_dim: np.ndarray     # (d,) -- copy for convenience
    ifi_mean_all: float         # mean IFI across all finite dims
    ifi_mean_sig: float         # mean IFI across sig dims (NaN if n_sig=0)
    ifi_weighted: float         # sum(IFI * CC) / sum(CC); NaN if all CC=0


def _aggregate(strength_cc: np.ndarray, p_dim: np.ndarray, ifi: np.ndarray,
               alpha: float, cc_clip: float) -> CellSubspaceStats:
    """Shared subspace aggregation given per-dim strength CC, p-values, IFI.

    ``strength_cc`` is the per-dim canonical correlation used for the MI
    strength readout (peak-across-lag for landmark/temporal cells; lag-0 for
    spatial epochs). ``p_dim`` is the matching per-dim significance.
    """
    # Clip to [0, cc_clip] -- negative held-out CCs (the direction did not
    # generalise) are floored; near-1 CCs are clipped so log(1-x^2) stays finite.
    peak_cc = np.clip(np.asarray(strength_cc, dtype=float), 0.0, cc_clip)
    p_dim = np.asarray(p_dim, dtype=float)
    sig = np.isfinite(p_dim) & (p_dim < alpha)
    n_sig = int(np.sum(sig))

    finite_cc = np.isfinite(peak_cc) & (peak_cc > 0)
    mi_all = float(-0.5 * np.sum(np.log1p(-peak_cc[finite_cc] ** 2)))
    if n_sig > 0:
        cc_sig = peak_cc[sig]
        cc_sig = cc_sig[np.isfinite(cc_sig) & (cc_sig > 0)]
        mi_sig = float(-0.5 * np.sum(np.log1p(-cc_sig ** 2)))
    else:
        mi_sig = 0.0

    ifi = np.asarray(ifi, dtype=float)
    finite_ifi = np.isfinite(ifi)
    if np.any(finite_ifi):
        ifi_mean_all = float(np.mean(ifi[finite_ifi]))
    else:
        ifi_mean_all = float("nan")
    if n_sig > 0 and np.any(sig & finite_ifi):
        ifi_mean_sig = float(np.mean(ifi[sig & finite_ifi]))
    else:
        ifi_mean_sig = float("nan")
    weights = peak_cc.copy()
    weights[~finite_cc] = 0.0
    weights[~finite_ifi] = 0.0
    if weights.sum() > 0:
        ifi_weighted = float(
            np.sum(np.where(finite_ifi & finite_cc, ifi * weights, 0.0))
            / weights.sum()
        )
    else:
        ifi_weighted = float("nan")
    return CellSubspaceStats(
        n_dims=int(peak_cc.size),
        n_sig=n_sig,
        peak_cc_per_dim=peak_cc,
        sig_mask=sig,
        mi_all=mi_all,
        mi_sig=mi_sig,
        ifi_per_dim=ifi,
        ifi_mean_all=ifi_mean_all,
        ifi_mean_sig=ifi_mean_sig,
        ifi_weighted=ifi_weighted,
    )


def cell_subspace_stats(cell, alpha: float = 0.05,
                        cc_clip: float = CC_CLIP) -> CellSubspaceStats | None:
    """Aggregate per-cell stats across canonical dimensions.

    ``cell`` is a ``LandmarkCellAnalysis`` or ``TemporalEpochAnalysis`` (anything
    that exposes ``lag_cc_per_dim``, ``p_peak_cc_per_dim``, ``ifi_per_dim``).
    Strength uses the peak-across-lag held-out CC. Returns ``None`` if the cell
    was skipped.
    """
    if getattr(cell, "skip_reason", None) is not None:
        return None
    if getattr(cell, "lag_cc_per_dim", None) is None:
        return None
    peak_cc_raw = np.nanmax(cell.lag_cc_per_dim, axis=0)
    return _aggregate(peak_cc_raw, cell.p_peak_cc_per_dim, cell.ifi_per_dim,
                      alpha, cc_clip)


def epoch_subspace_stats(epoch, alpha: float = 0.05,
                         cc_clip: float = CC_CLIP) -> CellSubspaceStats | None:
    """Spatial-arm analogue of :func:`cell_subspace_stats`.

    The spatial ``EpochAnalysis`` defines significance on the **lag-0** held-out
    CC (``p_per_dim``, the D7 permutation test), so strength and significance are
    both taken at lag 0: MI is summed over ``held_out_cc`` for dims with
    ``p_per_dim < alpha``. Directionality (``ifi_weighted``) uses ``ifi_per_dim``
    (the lagged-curve IFI) weighted by the lag-0 CC. Same formulas as the
    landmark cell, but on the spatial arm's native lag-0 quantities rather than
    the peak-across-lag quantities. Returns ``None`` if the epoch has no CC.
    """
    if getattr(epoch, "held_out_cc", None) is None:
        return None
    return _aggregate(epoch.held_out_cc, epoch.p_per_dim, epoch.ifi_per_dim,
                      alpha, cc_clip)


def per_dim_long_form(fits, alpha: float = 0.05) -> list[dict]:
    """Flatten a list of analysis results into one row per (cell × sig dim).

    Each row carries the cell coords plus the per-dim CC, IFI, peak lag bin,
    and a ``sig`` flag. Useful for "use all subspaces as n" cohort plots and
    paired tests.
    """
    out: list[dict] = []
    for r in fits:
        if not hasattr(r, "cells"):
            continue
        for (epoch, lm), cell in r.cells.items():
            stats = cell_subspace_stats(cell, alpha=alpha)
            if stats is None:
                continue
            for j in range(stats.n_dims):
                out.append({
                    "animal_id": r.animal_id,
                    "pair_x": r.area_x,
                    "pair_y": r.area_y,
                    "epoch": epoch,
                    "landmark": int(lm),
                    "dim": int(j + 1),
                    "peak_cc": float(stats.peak_cc_per_dim[j]),
                    "ifi": float(stats.ifi_per_dim[j])
                    if np.isfinite(stats.ifi_per_dim[j]) else float("nan"),
                    "sig": bool(stats.sig_mask[j]),
                })
    return out
