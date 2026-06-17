"""striatum_tcca -- temporal communication-subspace CCA (striatum port of tom_cca).

Data layer (`config`, `dataio`) is striatum-specific; the numeric modules are
ported near-verbatim from TomLearning `tom_cca` (they are array+cfg only).
"""

from __future__ import annotations

from . import (
    config,
    core,
    crosspair,
    dataio,
    early_trials,
    kcca_window,
    kernel_cca,
    lagged,
    membership,
    mixed_effects,
    paired_stats,
    partial,
    subspace,
    subspace_stats,
    subspace_window,
    surrogate,
    trajectory,
)

__all__ = [
    "config",
    "core",
    "crosspair",
    "dataio",
    "early_trials",
    "kcca_window",
    "kernel_cca",
    "lagged",
    "membership",
    "mixed_effects",
    "paired_stats",
    "partial",
    "subspace",
    "subspace_stats",
    "subspace_window",
    "surrogate",
    "trajectory",
]
