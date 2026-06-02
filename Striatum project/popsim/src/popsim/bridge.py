"""Bridge popsim simulations into the sibling ``striatum_cca`` analysis pipeline.

``simulate_trials`` already produces, per area, the exact tensor shape the
``striatum_cca`` compute layer consumes -- ``(n_trials, n_bins, n_units)`` -- so
this module is a thin adapter that:

1. locates and imports ``striatum_cca`` (its compute modules are numpy/scipy/h5py
   only; we use its real ``core`` / ``lagged`` / ``partial`` / ``config``), and
2. runs that pipeline's cross-validated CCA, lagged directionality curve, and
   partial CCA on a popsim :class:`~popsim.simulate.TrialResult`.

This lets the known-ground-truth simulator drive the *actual* analysis code, so
the pipeline can be validated end-to-end against couplings we dialed in.

``striatum_cca`` is an optional dependency: it is imported lazily, and the path
to its ``src`` is auto-discovered relative to this file (``../striatum_cca`` =
the ``cca`` subproject) or taken from the ``STRIATUM_CCA_SRC`` env var.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType

import numpy as np

from .simulate import TrialResult

__all__ = [
    "striatum_cca_available",
    "to_area_tensor",
    "pca_scores",
    "BridgeResult",
    "analyse_pair",
]


def _candidate_src_dirs() -> list[str]:
    """Possible locations of the ``striatum_cca`` ``src`` directory."""
    here = os.path.dirname(os.path.abspath(__file__))
    # popsim/src/popsim -> popsim -> "Striatum project" -> cca/src
    project_root = os.path.normpath(os.path.join(here, "..", "..", ".."))
    candidates = [
        os.environ.get("STRIATUM_CCA_SRC", ""),
        os.path.join(project_root, "cca", "src"),
    ]
    return [c for c in candidates if c and os.path.isdir(c)]


@lru_cache(maxsize=1)
def _import_striatum_cca() -> ModuleType:
    """Import and return the ``striatum_cca`` package, adjusting sys.path.

    Raises
    ------
    ImportError
        If the package cannot be located or imported.
    """
    try:
        import striatum_cca  # noqa: F401  (maybe already importable)

        return striatum_cca
    except ImportError:
        pass
    for src in _candidate_src_dirs():
        if src not in sys.path:
            sys.path.insert(0, src)
    try:
        import striatum_cca

        return striatum_cca
    except ImportError as exc:  # pragma: no cover - exercised only when missing
        raise ImportError(
            "striatum_cca is not importable. Searched: "
            f"{_candidate_src_dirs()}. Set STRIATUM_CCA_SRC to its src dir, and "
            "ensure h5py is installed."
        ) from exc


def striatum_cca_available() -> bool:
    """Whether the ``striatum_cca`` package can be imported in this environment."""
    try:
        _import_striatum_cca()
        return True
    except ImportError:
        return False


def to_area_tensor(result: TrialResult, area: str) -> np.ndarray:
    """Return one area's ``(n_trials, n_bins, n_units)`` tensor from a TrialResult.

    This is the native shape ``striatum_cca.core`` expects; the function exists
    to make the contract explicit and to validate the area name.
    """
    if area not in result.neural:
        raise KeyError(f"unknown area {area!r}; have {result.area_names}")
    return np.asarray(result.neural[area], dtype=float)


def pca_scores(tensor: np.ndarray, k: int):
    """PCA-reduce an area tensor to ``k`` PCs using ``striatum_cca.core``.

    Returns ``(scores, state)`` where ``scores`` is ``(n_trials, n_bins, k)`` and
    ``state`` is the fitted ``striatum_cca.core.PCAState`` basis.
    """
    sca = _import_striatum_cca()
    state = sca.core.pca_fit(tensor, k)
    return sca.core.pca_transform(tensor, state), state


@dataclass
class BridgeResult:
    """Outputs of running the striatum_cca pipeline on one popsim area pair."""

    area_x: str
    area_y: str
    k: int
    held_out_cc: np.ndarray     # (k,) cross-validated canonical correlations
    in_sample_cc: np.ndarray    # (k,) in-sample canonical correlations
    lags: np.ndarray            # (2*max_lag+1,) integer bin lags
    lag_cc1: np.ndarray         # (n_lags,) held-out CC of the top dim vs lag
    peak_lag: int               # lag (bins) maximising lag_cc1
    partial_cc: np.ndarray | None        # (k,) partial CCA(X, Y | Z), or None
    partial_area: str | None             # the conditioning area Z, or None


def analyse_pair(
    result: TrialResult,
    area_x: str,
    area_y: str,
    k: int = 5,
    partial_area: str | None = None,
    cfg=None,
    max_lag: int | None = None,
) -> BridgeResult:
    """Run striatum_cca's CCA + lagged curve (+ optional partial CCA) on a pair.

    Parameters
    ----------
    result:
        A :class:`~popsim.simulate.TrialResult` (use :func:`simulate_trials`).
    area_x, area_y:
        The two areas to relate.
    k:
        PCs retained per area before CCA.
    partial_area:
        If given, also compute partial CCA(X, Y | this area).
    cfg:
        A ``striatum_cca`` config; defaults to ``striatum_cca.config.DEFAULT``.
    max_lag:
        Lag half-width (bins) for the directionality curve; defaults to
        ``cfg.max_lag_bins``.
    """
    sca = _import_striatum_cca()
    cfg = cfg if cfg is not None else sca.config.DEFAULT

    tx = to_area_tensor(result, area_x)
    ty = to_area_tensor(result, area_y)
    sx, _ = pca_scores(tx, k)
    sy, _ = pca_scores(ty, k)

    cv = sca.core.cca_cv(sx, sy, cfg)
    lag = sca.lagged.lag_curve(sx, sy, cfg, max_lag=max_lag, held_out=True)
    peak_lag = int(lag.lags[np.nanargmax(lag.cc_per_dim[:, 0])])

    partial_cc = None
    if partial_area is not None:
        tz = to_area_tensor(result, partial_area)
        sz, _ = pca_scores(tz, k)
        partial_cc = sca.partial.partial_cca_cv(sx, sy, sz, cfg).held_out_r

    return BridgeResult(
        area_x=area_x,
        area_y=area_y,
        k=k,
        held_out_cc=cv.held_out_r,
        in_sample_cc=cv.in_sample_r,
        lags=lag.lags,
        lag_cc1=lag.cc_per_dim[:, 0],
        peak_lag=peak_lag,
        partial_cc=partial_cc,
        partial_area=partial_area,
    )
