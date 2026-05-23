"""Configuration for the striatum CCA pipeline.

Single source of truth for paths, area definitions, and the analysis knobs
resolved in ../UNDERSTANDING.md (decisions D1-D12).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- Paths -------------------------------------------------------------------
# This file lives at  <Striatum project>/cca/src/striatum_cca/config.py
_PROJECT = Path(__file__).resolve().parents[3]          # ".../Striatum project"
PREPROCESSED_DATA = _PROJECT / "processed_data" / "preprocessed_data.mat"
CCA_DIR = _PROJECT / "cca"
RESULTS_DIR = CCA_DIR / "results"
FIGURES_DIR = CCA_DIR / "figures"

# --- Areas and pairs ---------------------------------------------------------
# DG excluded (UNDERSTANDING.md Won't-Do).
AREAS: tuple[str, ...] = ("DMS", "DLS", "ACC", "V1", "CA1")
AREA_FIELD: dict[str, str] = {
    "DMS": "is_dms",
    "DLS": "is_dls",
    "ACC": "is_acc",
    "V1": "is_v1",
    "CA1": "is_ca1",
}
# All 10 unordered pairs. The first element is "X" — it fixes the sign
# convention for the Information Flow Index (D6: positive IFI => X leads Y).
PAIRS: tuple[tuple[str, str], ...] = (
    ("DMS", "DLS"),
    ("DMS", "ACC"),
    ("DLS", "ACC"),
    ("V1", "DMS"),
    ("V1", "DLS"),
    ("V1", "ACC"),
    ("CA1", "DMS"),
    ("CA1", "DLS"),
    ("CA1", "ACC"),
    ("CA1", "V1"),
)

# --- Spatial geometry --------------------------------------------------------
N_BINS = 50            # spatial bins per corridor traversal (D5: keep 50)
BIN_SIZE_CM = 5.0      # 4 a.u. * 1.25 cm/a.u.

# --- Cell type ---------------------------------------------------------------
# final_neurontypes column 5 (0-indexed 4): 1 MSN, 2 FS, 3 TAN, 4 UIN
# (striatum); 2 FS, 5 RS (V1/CA1). Type 2 == fast-spiking in every area.
NTYPE_COL = 4
FS_TYPE_CODE = 2

# --- Epoch names (ordered naive -> intermediate -> expert) -------------------
EPOCH_NAMES: tuple[str, str, str] = ("naive", "intermediate", "expert")


@dataclass(frozen=True)
class Config:
    """Tunable analysis parameters. Frozen so a run's settings are immutable."""

    # Learning point detection (project rule; see UNDERSTANDING.md).
    lp_z_threshold: float = -2.0
    lp_window: int = 10
    lp_min_consecutive: int = 7
    # Animals forced to non-learner regardless of a detected LP. Animal 8's
    # detected LP (16) is implausibly early — likely a detection artefact;
    # it is yoked instead (UNDERSTANDING.md edit log v2).
    manual_nonlearners: frozenset[int] = frozenset({8})

    # Epochs.
    trials_per_epoch: int = 10

    # Unit inclusion.
    min_units: int = 5
    exclude_fast_spiking: bool = True

    # Residualisation (D2): subtract the per-(bin, unit) trial mean.
    subtract_trial_mean: bool = True

    # PCA / k rule (D4): k = floor(n_samples_per_epoch / samples_per_pc).
    samples_per_pc: int = 25
    k_cap: int = 30

    # Cross-validation (D8): 5-fold over whole trials.
    n_folds: int = 5
    cv_seed: int = 0


DEFAULT = Config()
