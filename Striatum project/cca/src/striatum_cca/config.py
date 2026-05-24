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
# Primary preprocessing: 100-bin (2.5 cm). The pipeline is bin-count-agnostic;
# the 50-bin (5 cm) file can still be passed explicitly to load_animals().
PREPROCESSED_DATA = _PROJECT / "processed_data" / "preprocessed_data2p5cm.mat"
PREPROCESSED_DATA_5CM = _PROJECT / "processed_data" / "preprocessed_data5cm.mat"
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
# The real pipeline auto-detects the bin count from the data file, so it works
# with the 50-bin (5 cm) or the 100-bin (2.5 cm) preprocessing unchanged.
# N_BINS is only a default for synthetic test fixtures.
N_BINS = 50
CORRIDOR_CM = 250.0    # 200 a.u. * 1.25 cm/a.u.


def bin_size_cm(n_bins: int) -> float:
    """Spatial bin width in cm for a given bin count."""
    return CORRIDOR_CM / n_bins

# --- Cell type ---------------------------------------------------------------
# final_neurontypes column 5 (0-indexed 4): 1 MSN, 2 FS, 3 TAN, 4 UIN
# (striatum); 2 FS, 5 RS (V1/CA1). Type 2 == fast-spiking in every area.
NTYPE_COL = 4
FS_TYPE_CODE = 2

# --- Epoch names (ordered naive -> intermediate -> expert) -------------------
# Intermediate re-added in round 10 (dropped round 7): naive = trials 1-10,
# intermediate = the 10 trials ending at LP, expert = the 10 after LP.
EPOCH_NAMES: tuple[str, str, str] = ("naive", "intermediate", "expert")

# Epoch plotting colours, kept byte-identical to the MATLAB pipeline
# (cfg.plot.colors.epoch_{early,middle,expert} in Run_TCA_pipeline.m) so that
# figures are colour-consistent across the MATLAB and Python halves of the
# project. Single source of truth -- every Python plot script imports this.
EPOCH_COLOURS: dict[str, tuple[float, float, float]] = {
    "naive": (0.298, 0.447, 0.690),          # epoch_early  (blue)
    "intermediate": (0.867, 0.518, 0.322),   # epoch_middle (orange)
    "expert": (0.333, 0.776, 0.333),         # epoch_expert (green)
}


@dataclass(frozen=True)
class Config:
    """Tunable analysis parameters. Frozen so a run's settings are immutable."""

    # Learning point detection (project rule; see UNDERSTANDING.md).
    lp_z_threshold: float = -2.0
    lp_window: int = 10
    lp_min_consecutive: int = 7        # round-8: kept at 7 (LP-8 cost 3 learners)
    # Animals forced to non-learner regardless of a detected LP. Animal 8's
    # detected LP (16) is implausibly early — likely a detection artefact;
    # it is yoked instead (UNDERSTANDING.md edit log v2).
    manual_nonlearners: frozenset[int] = frozenset({8})

    # Epochs.
    trials_per_epoch: int = 10

    # Binning mode (round 8). "spatial" bins corridor position
    # (spatial_binned_fr_all); "temporal" re-bins corridorData.binned_spikes
    # (1 ms spike counts) into temporal_bin_ms time bins from corridor onset.
    # Temporal trials keep their full natural length (no window, no clipping);
    # within an epoch they are NaN-padded to the longest trial at fit time.
    # Temporal CCA is signal-only -- no residualisation (Theo, round 8).
    # temporal_max_trial_ms: corridor traversals longer than this are treated
    # as disengaged and excluded from the temporal analysis (round 8, opt 3).
    bin_mode: str = "spatial"
    temporal_bin_ms: int = 20
    temporal_max_trial_ms: int = 60_000

    # Unit inclusion. min_units committed at 6 in round 8 (was 4); the
    # FS-included vs FS-excluded comparison flips exclude_fast_spiking;
    # FS-excluded is the primary committed default.
    min_units: int = 6
    exclude_fast_spiking: bool = True

    # Committed analysis (round 8): residual CCA, FS-excluded, z-scored units,
    # held-out CC, 2.5 cm bins, min_units 6, LP-criterion 7, samples-per-PC 15
    # -- the configuration the parameter sweep converged on. The sweep knobs
    # remain so alternatives can still be run.
    # Residualisation (D2): subtract the per-(bin, unit) trial mean.
    subtract_trial_mean: bool = True
    # Z-score each unit by its std over the entire engaged period, applied to
    # the raw activity *before* residualisation and epoch slicing (round 7).
    # CCA is scale-invariant, so this only re-weights units for the PCA step.
    zscore_units: bool = True

    # PCA / k rule (D4). k_mode selects how many PCs to keep per area:
    #   "samples"  -- k = floor(n_samples / samples_per_pc)   [default]
    #   "fixed"    -- k = k_fixed
    #   "variance" -- k = #PCs reaching k_variance cumulative variance
    # All are capped by the smaller area's unit count, k_cap and the
    # per-epoch numerical rank. The sweep (round 8) varies all of these.
    k_mode: str = "samples"
    k_fixed: int = 10
    k_variance: float = 0.90
    samples_per_pc: int = 15           # round-8 committed config (was 25)
    k_cap: int = 30

    # Cross-validation (D8): 5-fold over whole trials.
    n_folds: int = 5
    cv_seed: int = 0

    # Lagged CCA / directionality (D6): refit CCA at each spatial-bin lag.
    # 10 bins at 2.5 cm = +/-25 cm, matching the +/-5 bins x 5 cm of the
    # 50-bin preprocessing.
    max_lag_bins: int = 10

    # Surrogate null (D7): held-out-CC per-dimension permutation test.
    # null_type "circshift" -- per-trial circular shift of the bin axis by
    # >= circshift_min_bins (tests within-trial co-tuning, preserving each
    # area's spatial autocorrelation; the surrogate used by Gonzalez et al.).
    # "trials" -- permute the trial correspondence (H&H; tests trial-to-trial
    # communication). Committed to "circshift" in round 10: it is the
    # defensible null and flags ~2.6x more significant subspace dimensions.
    n_shuffles: int = 250          # round-14 lock-in: raised 200 -> 250
    surrogate_seed: int = 0
    null_type: str = "circshift"
    circshift_min_bins: int = 15

    # Parallelism (D11): processes for the cohort run.
    n_jobs: int = 4


DEFAULT = Config()
