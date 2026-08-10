"""Configuration for the striatum *temporal* CCA pipeline (striatum_tcca).

Single source of truth for paths, area definitions and the analysis knobs.
This is the striatum port of TomLearning's ``tom_cca`` temporal communication-
subspace pipeline (the "Hippocampus-V1 Communication-Subspace Learning Report",
method contract ``CCA_HH_Adapted``).

The data-level differences from the Tom pipeline are:

* one cohort ``.mat`` (``preprocessed_data5cm.mat``) rather than one file per
  animal; per-traversal ``corridorData.binned_spikes`` (1 ms counts), already
  corridor-only (dark stripped at ``trial_world > 6``);
* **no velocity channel** -- speed is derived from ``trial_position`` /
  ``trial_times`` (see :func:`dataio.trial_velocity`);
* areas are DMS/DLS/ACC/V1/CA1/DG (the project's six -- ``project_cfg.m``);
* FS exclusion is ``final_neurontypes(:,5) == 2`` applied to **every** area
  (type 2 == fast-spiking in all areas here), unlike Tom's V1/RSC/CA1/CA3 subset;
* learning point / epochs / disengagement follow the project's MATLAB rules.

Defaults follow the report "to the letter": Gaussian smoothing sigma 2.5 ms,
10 ms primary bin width, residual + partial CCA, held-out whole-trial CV, IFI
integrated over the +/-50 ms headline window.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- Paths -------------------------------------------------------------------
# This file lives at  <Striatum project>/tcca/src/striatum_tcca/config.py
_PROJECT = Path(__file__).resolve().parents[3]          # ".../Striatum project"
# Task (cued, learned) corridor -- the primary cohort. 5 cm products since
# 2026-08-10 (2.5 cm emission dropped); tcca reads only the 1 ms corridorData
# payload, so the spatial bin size of the container is irrelevant here.
TASK_DATA = _PROJECT / "processed_data" / "preprocessed_data5cm.mat"
# Control (blank habituation) corridor -- a separate cohort of animals, run
# through the identical preprocessing path (so the saved struct layout matches).
CONTROL_DATA = _PROJECT / "processed_data" / "preprocessed_data_control5cm.mat"
DATA_BY_GROUP: dict[str, Path] = {"task": TASK_DATA, "control": CONTROL_DATA}

TCCA_DIR = _PROJECT / "tcca"
RESULTS_DIR = TCCA_DIR / "results"
FIGURES_DIR = TCCA_DIR / "figures"

# --- Areas and pairs ---------------------------------------------------------
# The project's six areas (project_cfg.m). DG/CA1/V1 yields are low (1-4 animals)
# and are exploratory; the DMS/DLS/ACC triangle carries the group inference.
AREAS: tuple[str, ...] = ("DMS", "DLS", "ACC", "V1", "CA1", "DG")
AREA_FIELD: dict[str, str] = {
    "DMS": "is_dms",
    "DLS": "is_dls",
    "ACC": "is_acc",
    "V1": "is_v1",
    "CA1": "is_ca1",
    "DG": "is_dg",
}
# The 15 unordered pairs from project_cfg.m (cfg.area_pairs). The first element
# is "X" -- it fixes the IFI sign convention (positive IFI => X leads Y).
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
    ("DG", "DMS"),
    ("DG", "DLS"),
    ("DG", "ACC"),
    ("DG", "V1"),
    ("DG", "CA1"),
)

# --- Geometry ----------------------------------------------------------------
AU_TO_CM = 1.25         # VR position scale (project_cfg cfg.au_to_cm)
CORRIDOR_CM = 250.0     # 200 a.u. * 1.25 cm/a.u.
N_BINS = 50             # spatial-fixture default only; the temporal path is time-binned

# --- Cell type ---------------------------------------------------------------
# final_neurontypes column 5 (0-indexed 4): 1 MSN, 2 FS, 3 TAN, 4 UIN (striatum);
# 2 FS, 5 RS (V1/CA1/DG). Type 2 == fast-spiking in every area.
NTYPE_COL = 4
FS_TYPE_CODE = 2

# --- Surrogate / significance ------------------------------------------------
# Shuffle count for the circular-shift surrogate (95th-percentile threshold).
# 100 matches the report (Ns=100); centralised so every driver stays consistent.
SURROGATE_SHUFFLES: int = 100

# --- Epoch names + colours (byte-identical to the MATLAB / cca pipeline) ------
EPOCH_NAMES: tuple[str, str, str] = ("naive", "intermediate", "expert")
EPOCH_COLOURS: dict[str, tuple[float, float, float]] = {
    "naive": (0.298, 0.447, 0.690),          # blue
    "intermediate": (0.867, 0.518, 0.322),   # orange
    "expert": (0.333, 0.776, 0.333),         # green
}
# Engagement-contrast colours (engaged vs disengaged), distinct from the epochs.
ENGAGEMENT_COLOURS: dict[str, tuple[float, float, float]] = {
    "engaged": (0.20, 0.60, 0.40),           # teal-green
    "disengaged": (0.60, 0.60, 0.60),        # grey
}
# Area colours (project_cfg cfg.area_colors) for figures.
AREA_COLOURS: dict[str, tuple[float, float, float]] = {
    "DMS": (0.000, 0.447, 0.741),
    "DLS": (0.466, 0.674, 0.188),
    "ACC": (0.850, 0.325, 0.098),
    "V1":  (0.494, 0.184, 0.556),
    "CA1": (0.800, 0.100, 0.200),
    "DG":  (0.200, 0.700, 0.700),
}


@dataclass(frozen=True)
class Config:
    """Tunable analysis parameters. Frozen so a run's settings are immutable.

    Defaults reproduce the report's committed temporal pipeline.
    """

    # --- Learning point detection (project rule; find_learning_points.m) -----
    lp_z_threshold: float = -2.0
    lp_window: int = 10
    lp_min_consecutive: int = 7
    # Animal 8's detected LP (~16) is implausibly early -- yoked as a non-learner
    # (matches the striatum_cca decision, UNDERSTANDING.md edit log v2).
    manual_nonlearners: frozenset[int] = frozenset({8})

    # --- Epochs --------------------------------------------------------------
    trials_per_epoch: int = 10

    # --- Spike preprocessing (report Section 2.2) ----------------------------
    # 1 ms spikes -> Gaussian smooth (sigma_ms) -> sum into temporal_bin_ms bins.
    # Report primary: 10 ms bins, sigma 2.5 ms. Use 25 ms for the trajectory and
    # 50 ms as a robustness check (set via the drivers).
    temporal_bin_ms: int = 10
    gaussian_sd_ms: float = 2.5
    # Corridor traversals longer than this are treated as disengaged / pathological
    # and excluded from the running-state analysis (a single 300 s "trial" must not
    # dominate its epoch). For the disengaged-period contrast a driver may relax it.
    temporal_max_trial_ms: int = 60_000

    # --- Behavioural masks ---------------------------------------------------
    velocity_thresh_cm_s: float = 2.0     # running-speed gate (cm/s), report 2.0

    # --- Unit inclusion ------------------------------------------------------
    min_units: int = 5                    # report/Tom value (striatum spatial used 6)
    exclude_fast_spiking: bool = True     # FS-excluded primary; flip for the FS control

    # --- "Communication" definition + normalisation --------------------------
    subtract_trial_mean: bool = True      # residual CCA (report D2); flip for signal
    zscore_units: bool = True             # per-unit z over engaged running reference

    # --- PCA / k rule (report Section 2.5) -----------------------------------
    #   "samples"  -- k = floor(n_samples / samples_per_pc)   [default]
    #   "fixed"    -- k = k_fixed
    #   "variance" -- k = #PCs reaching k_variance cumulative variance
    # Capped by the smaller area's unit count, k_cap, and per-epoch numerical rank.
    k_mode: str = "samples"
    k_fixed: int = 10
    k_variance: float = 0.90
    samples_per_pc: int = 15
    k_cap: int = 30

    # --- Cross-validation (report Section 2.7): 5-fold over whole trials ------
    n_folds: int = 5
    cv_seed: int = 0

    # --- Directionality (report Section 2.9) ---------------------------------
    # Headline IFI integrates over +/-50 ms = +/-5 bins at 10 ms. The IFI window
    # sweep driver scans wider (out to +/-250 ms = 25 bins) and integrates
    # progressively; epoch/trajectory analyses use the headline window.
    max_lag_bins: int = 5

    # --- Surrogate null (report Section 2.8): per-dim held-out-CC permutation --
    # "circshift" -- per-trial circular roll of the bin axis by >= circshift_min_bins
    # (Gonzalez & Buzsaki; preserves each area's autocorrelation). "trials" --
    # permute trial correspondence (Han & Helmchen).
    n_shuffles: int = 100
    surrogate_seed: int = 0
    null_type: str = "circshift"
    circshift_min_bins: int = 15          # >= 150 ms at 10 ms

    # --- Parallelism ---------------------------------------------------------
    n_jobs: int = 4


DEFAULT = Config()


# ---- Derived quantities -----------------------------------------------------
def lag_window_ms(cfg: Config) -> int:
    """Physical half-width (ms) of the headline IFI integration window."""
    return cfg.max_lag_bins * cfg.temporal_bin_ms
