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

# The June files were size-keyed via RawData/LFP/lfp_mapping.txt. The 2026-08
# download names every file after its animal, so the map is now derived from the
# directory listing: see ``cohort.discover_lfp_files`` (and ``cohort.parse_lfp_filename``
# for the two naming variants, ``<mouse>_...`` and ``<mouse>_v1_...``). Nothing here
# hard-codes a filename any more, and no code should assume one file per mouse:
# a mouse with both probes contributes two.
# Positional order used by OrganiseStriatumDataIncV1.m and therefore by the
# preprocessed cohort struct / tcca Animal.animal_id.
TASK_MOUSE_IDS: tuple[int, ...] = (
    523, 614, 624, 727, 730, 731, 822, 823,
    1105, 1106, 1201, 1206, 1212, 409, 418, 703,
)

# Depth -> area boundaries (micrometres from probe tip). Probe 1 = striatum/cortex
# (DMS/DLS/ACC); probe 2 = visual/hippocampal (V1/CA1/DG). Each LFP .mat is 384
# channels == ONE probe, and the filename now states which: ``<mouse>_v1_...`` is
# probe 2. The 2026-08 export also ships ``depth_to_save`` (0-3820 um), so the
# geometry assumption below is checkable against the file rather than assumed.
DEPTH_CSV = RAWDATA / "Neuropixels_Depth_Data.csv"
V1_CSV = RAWDATA / "Neuropixels_V1_Depth_Data.csv"

# Per-mouse spike + behaviour bundle (VR_data, VR_times_synched, binned_spikes),
# one per probe. ``<mouse>_raw.mat`` is probe 1; ``<mouse>_V1_raw.mat`` is probe 2
# and exists only for the five dual-probe mice. Behaviour fields are identical in
# both, so either file answers a behavioural question; spikes are probe-specific.
RAW_MAT: dict[int, Path] = {m: RAWDATA / f"{m}_raw.mat" for m in TASK_MOUSE_IDS}
V1_RAW_MAT: dict[int, Path] = {m: RAWDATA / f"{m}_V1_raw.mat" for m in TASK_MOUSE_IDS}


def raw_mat(mouse_id: int, probe: str = "striatum") -> Path:
    """Spike/behaviour bundle for one mouse and probe ("striatum" | "visual")."""
    table = RAW_MAT if probe == "striatum" else V1_RAW_MAT
    return table[mouse_id]
# The cohort struct the spike tensor was built from (corridorData, learning point,
# zscored_lick_errors, ...). Reused wholesale for the LFP drop-in behaviour fields.
PREPROC_MAT = _PROJECT / "processed_data" / "preprocessed_data5cm.mat"

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
    n_spatial_bins: int = 50        # 5 cm bins over the 250 cm corridor (project_cfg
                                    # cfg.n_bins_full; the 2.5 cm grid was retired 2026-08-10)
    max_bin: int = 30               # spatial truncation (project_cfg cfg.max_bin)
    temporal_bin_ms: int = 50       # tcca running-state stream bin width
    velocity_thresh_cm_s: float = 2.0

    # --- Area gating / PCA ---------------------------------------------------
    min_sites: int = 5              # skip an area with fewer QC-good channels
    pca_k: int = 5                  # top PCs for the per-area diagnostic


DEFAULT = Config()


# --- Backwards compatibility -------------------------------------------------
# The July drivers (scripts/run_sanity_audit.py, run_signal_identity.py,
# plot_sanity_audit.py, the quarantined learning/decode drivers) index
# ``config.FILE_BY_MOUSE[mouse]`` and iterate ``config.LFP_MICE``, both written
# for the four size-keyed June files. Rather than rewrite those drivers, resolve
# both lazily from the directory listing so they address the current export.
# Both cover probe 1 only, which is what those drivers assume; anything
# two-probe-aware should call ``cohort.discover_lfp_files`` directly.
# Module-level ``__getattr__`` (PEP 562) keeps this off the import path, which
# is what avoids a config <-> cohort import cycle.


def lfp_path(mouse_id: int, probe: str = "striatum") -> Path:
    """Path to one mouse's LFP export for one probe, resolved by filename."""
    from .cohort import discover_lfp_files

    return discover_lfp_files(LFP_DIR)[(mouse_id, probe)]


def lfp_mice(probe: str = "striatum") -> tuple[int, ...]:
    """Task mice with an LFP export for ``probe``, in cohort (positional) order."""
    from .cohort import discover_lfp_files

    present = {m for (m, p) in discover_lfp_files(LFP_DIR) if p == probe}
    return tuple(m for m in TASK_MOUSE_IDS if m in present)


def __getattr__(name: str):
    if name == "LFP_MICE":
        return lfp_mice()
    if name == "FILE_BY_MOUSE":
        from .cohort import discover_lfp_files

        return {
            m: path.name
            for (m, p), path in discover_lfp_files(LFP_DIR).items()
            if p == "striatum"
        }
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
