"""Shared Stage-2 aggregation: pair selection, significant-dim values, and the
per-animal reshaping used by both the figure scripts and the epoch-ANOVA table.

Kept in one place so ``plot_stage2.py``, ``epoch_anova.py``,
``directionality_table.py`` and ``committed_ifi.py`` all pool the data the same
way -- the figure annotations and the CSV tables are then guaranteed to agree.

A "value" is either held-out CC (``metric="cc"``) or the Information Flow Index
at a given lag-integration window (``metric="ifi"``, ``window`` bins). Both are
read per significant canonical dimension; ``ALPHA`` is the per-dimension
held-out-CC significance threshold (surrogate test).
"""

from __future__ import annotations

import numpy as np

from . import config

ALPHA = 0.05


def learner_pairs(results, area_x: str, area_y: str) -> list:
    """The learner-cohort results for one ordered area pair."""
    return [r for r in results
            if (r.area_x, r.area_y) == (area_x, area_y) and r.role == "learner"]


def sig_dims(epoch_analysis, alpha: float = ALPHA) -> np.ndarray:
    """Indices of canonical dimensions passing the held-out significance test."""
    return np.where(epoch_analysis.p_per_dim < alpha)[0]


def dim_values(pair_result, epoch: str, metric: str, window: int = 10) -> np.ndarray:
    """Significant-dim values for one pair x epoch ('cc' or 'ifi')."""
    ea = pair_result.epochs[epoch]
    js = sig_dims(ea)
    if metric == "cc":
        return np.asarray(ea.held_out_cc)[js]
    if metric == "ifi":
        return np.asarray(ea.ifi_windows)[js, window - 1]
    raise ValueError(f"unknown metric {metric!r} (expected 'cc' or 'ifi')")


def per_dim_groups(
    learners, metric: str, window: int = 10, epochs=config.EPOCH_NAMES
) -> list[np.ndarray]:
    """One finite array of significant-dim values per epoch, pooled over animals
    (the n = dims grouping)."""
    groups = []
    for epoch in epochs:
        vals = (np.concatenate([dim_values(r, epoch, metric, window)
                                for r in learners])
                if learners else np.array([]))
        groups.append(vals[np.isfinite(vals)])
    return groups


def per_animal_matrix(
    learners, metric: str, window: int = 10, epochs=config.EPOCH_NAMES
) -> np.ndarray:
    """``(n_complete_animals, n_epochs)`` of each animal's mean over its
    significant dims; rows missing any epoch are dropped (the n = animals
    grouping for the repeated-measures test)."""
    rows = []
    for r in learners:
        row = []
        for epoch in epochs:
            v = dim_values(r, epoch, metric, window)
            row.append(float(np.nanmean(v)) if v.size else np.nan)
        rows.append(row)
    mat = np.array(rows, dtype=float) if rows else np.empty((0, len(epochs)))
    return mat[np.all(np.isfinite(mat), axis=1)] if mat.size else mat
