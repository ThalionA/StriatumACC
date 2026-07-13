"""Configuration for the striatum LFP pipeline (striatum_lfp).

Single source of truth for paths, Neuropixels geometry, LFP bands and the
extraction / QC / binning knobs.

The new data are 384-channel Neuropixels voltage exports, one .mat per mouse
(``RawData/LFP/voltage_data_384ch*.mat``). Their lengths are compatible with a
1000 Hz / 1 ms grid, and equal the corresponding ``binned_spikes`` lengths.
That establishes structural compatibility only: producer code, source band,
gain, anti-alias filtering and sample-accurate VR alignment are not available.
Do not position-bin or run temporal CCA until timing provenance is recovered.

Areas, learning-point rule and epoch geometry follow ``project_cfg.m``. Band
power per channel is the LFP analogue of a unit's firing rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- Paths -------------------------------------------------------------------
# This file lives at  <Striatum project>/lfp/src/striatum_lfp/config.py
_PROJECT = Path(__file__).resolve().parents[3]          # ".../Striatum project"
RAWDATA = _PROJECT / "RawData"
LFP_DIR = RAWDATA / "LFP"

# LFP file <-> mouse map (RawData/LFP/lfp_mapping.txt, keyed by file size).
FILE_BY_MOUSE: dict[int, str] = {
    1212: "voltage_data_384ch.mat",     # 16.33 GB, 11.4 M samples (190 min at 1 kHz)
    614: "voltage_data_384ch 2.mat",    # 11.38 GB,  8.4 M samples (140 min at 1 kHz)
    727: "voltage_data_384ch 3.mat",    # 11.65 GB,  8.4 M samples
    731: "voltage_data_384ch 4.mat",    # 11.48 GB,  8.4 M samples
}
LFP_MICE: tuple[int, ...] = (1212, 614, 727, 731)
# Positional order used by OrganiseStriatumDataIncV1.m and therefore by the
# preprocessed cohort struct / tcca Animal.animal_id.
TASK_MOUSE_IDS: tuple[int, ...] = (
    523, 614, 624, 727, 730, 731, 822, 823,
    1105, 1106, 1201, 1206, 1212, 409, 418, 703,
)

# Depth -> area boundaries (micrometres from probe tip). Probe 1 = striatum/cortex
# (DMS/DLS/ACC, all 4 mice); probe 2 = visual/hippocampal (V1/CA1/DG, only 1212).
# NB: the LFP .mat is 384 channels == ONE probe. The three non-1212 mice have only
# probe 1, so their LFP is striatal. For 1212 the probe identity of the LFP file is
# unconfirmed -- default to probe 1 (striatum); revisit V1/CA1 at Stage 3.
DEPTH_CSV = RAWDATA / "Neuropixels_Depth_Data.csv"
V1_CSV = RAWDATA / "Neuropixels_V1_Depth_Data.csv"

# Per-mouse spike + behaviour bundle (VR_data, VR_times_synched, binned_spikes).
RAW_MAT: dict[int, Path] = {m: RAWDATA / f"{m}_raw.mat" for m in LFP_MICE}
# The cohort struct the spike tensor was built from (corridorData, learning point,
# zscored_lick_errors, ...). Reused wholesale for the LFP drop-in behaviour fields.
PREPROC_MAT = _PROJECT / "processed_data" / "preprocessed_data2p5cm.mat"

PKG_DIR = _PROJECT / "lfp"
RESULTS_DIR = PKG_DIR / "results"
FIGURES_DIR = PKG_DIR / "figures"

# --- Sampling / grid ---------------------------------------------------------
FS = 1000       # inferred exported-grid Hz; source metadata is not shipped
DT_MS = 1       # inferred from length compatibility; precise offset unverified

# --- LFP bands (Hz) ----------------------------------------------------------
# Candidate bands only: provenance is unresolved, and the ~75 Hz narrow peak
# directly confounds the present 30-80 Hz definition. No band is approved for
# behavioural inference until timing/source metadata are recovered.
BANDS: dict[str, tuple[float, float]] = {
    "theta": (4.0, 8.0),
    "beta": (15.0, 30.0),
    "low_gamma": (30.0, 80.0),
    "broadband": (1.0, 100.0),
}
CONFOUNDED_BANDS: tuple[str, ...] = ("low_gamma",)

# --- Neuropixels 1.0 geometry ------------------------------------------------
# 384 channels, 2 per 20 um row -> depth[c] = (c // 2) * 20 um, spanning 0..3820
# um from tip. Coarse boundaries (100s of um) make the 2-per-row approximation
# adequate; the true channel_map.npy is not shipped with these mice. Low channel
# index == deep (near tip == ventral / striatum); high index == superficial
# (cortex / ACC).
PITCH_UM = 20.0
CH_PER_ROW = 2
N_CHANNELS = 384

AU_TO_CM = 1.25         # VR position scale (project_cfg cfg.au_to_cm)
CORRIDOR_CM = 250.0     # 200 a.u. * 1.25 cm/a.u.

# Areas labellable from the two depth CSVs (project_cfg cfg.areas order).
AREAS: tuple[str, ...] = ("DMS", "DLS", "ACC", "V1", "CA1", "DG")
AREA_FIELD: dict[str, str] = {
    "DMS": "is_dms",
    "DLS": "is_dls",
    "ACC": "is_acc",
    "V1": "is_v1",
    "CA1": "is_ca1",
    "DG": "is_dg",
}


@dataclass(frozen=True)
class Config:
    """Tunable extraction / QC / binning parameters (frozen: a run is immutable)."""

    # --- Out-of-core reader (overlap-save) -----------------------------------
    # block_samples must be a multiple of the HDF5 chunk row-count (42) so each
    # read lands on whole chunks. 100_800 = 2400 * 42 (~101 s at 1 kHz).
    block_samples: int = 100_800
    pad_samples: int = 3_000        # >= 3 s: covers the 1-4 Hz filter transient

    # --- Band-power extraction ----------------------------------------------
    filt_order: int = 4             # Butterworth order (SOS, zero-phase filtfilt)
    envelope: str = "hilbert"       # "hilbert" | "square_smooth"
    smooth_ms: float = 0.0          # optional post-envelope moving average (0 = off)

    # --- Legacy channel diagnostics (not an approved quality gate) -----------
    qc_probe_seconds: int = 60      # window (s) used to compute per-channel stats
    qc_probe_start_s: int = 1_000   # window start (s), well inside behaviour
    qc_hf_lo: float = 300.0         # descriptive split used by retired Stage 0
    qc_hf_frac_max: float = 0.05    # legacy threshold; do not infer good/bad LFP
    qc_std_mad_k: float = 4.0       # legacy within-group amplitude threshold

    # --- Spatial / temporal binning -----------------------------------------
    n_spatial_bins: int = 100       # 2.5 cm bins over the 250 cm corridor
    temporal_bin_ms: int = 50       # tcca running-state stream bin width
    velocity_thresh_cm_s: float = 2.0

    # --- Area gating / PCA ---------------------------------------------------
    min_sites: int = 5              # skip an area with fewer QC-good channels
    pca_k: int = 5                  # top PCs for the per-area diagnostic


DEFAULT = Config()
