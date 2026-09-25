"""Configuration for the striatum LFP pipeline (striatum_lfp).

Single source of truth for paths, Neuropixels geometry, LFP bands and the
extraction and binning knobs.

The data are 384-channel Neuropixels voltage exports, one .mat per probe
(``RawData/LFP/<mouse>[_v1]_voltage_data_384ch.mat``), on the same 1 kHz / 1 ms
grid as ``binned_spikes``. Gain, physical units and the anti-alias filter are
undocumented, so every outcome is within-session relative.

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
DEPTH_CSV = RAWDATA / "Neuropixels_Depth_Data.csv"      # task; see Cohort below
V1_CSV = RAWDATA / "Neuropixels_V1_Depth_Data.csv"

# --- Cohorts -----------------------------------------------------------------
# Task and Control 1 are the same experiment recorded in two groups, but they are
# NOT interchangeable in code: the control probe-2 spike bundle is lowercase
# (``513_v1_raw.mat`` vs the task's ``1105_V1_raw.mat``), the depth boundaries
# live in separate CSVs, and epoch windows for yoked controls are anchored to the
# TASK cohort's average learning point rather than to one of their own. Bundling
# those differences here keeps every driver cohort-agnostic.
# Control 2 is dark-only (no corridor) and ships no voltage export, so it has no
# entry.


@dataclass(frozen=True)
class Cohort:
    """Everything that differs between the task and control recordings."""

    name: str
    rawdata: Path
    lfp_dir: Path
    depth_csv: Path
    v1_csv: Path
    mouse_ids: tuple[int, ...]
    preproc_mat: Path
    v1_raw_suffix: str
    #: Controls are yoked, so they take the task cohort's average learning point.
    learning_point_source: str          # "per_animal" | "task_average"


TASK = Cohort(
    name="task",
    rawdata=RAWDATA,
    lfp_dir=LFP_DIR,
    depth_csv=RAWDATA / "Neuropixels_Depth_Data.csv",
    v1_csv=RAWDATA / "Neuropixels_V1_Depth_Data.csv",
    # OrganiseStriatumDataIncV1.m:9 -- positional order into preprocessed_data.
    mouse_ids=TASK_MOUSE_IDS,
    preproc_mat=_PROJECT / "processed_data" / "preprocessed_data5cm.mat",
    v1_raw_suffix="_V1_raw.mat",
    learning_point_source="per_animal",
)

_CONTROL_RAW = _PROJECT / "RawDataControl"
CONTROL = Cohort(
    name="control",
    rawdata=_CONTROL_RAW,
    lfp_dir=_CONTROL_RAW / "LFP",
    depth_csv=_CONTROL_RAW / "Neuropixels_Depth_Data_control.csv",
    v1_csv=_CONTROL_RAW / "Neuropixels_V1_Depth_Data_control.csv",
    # OrganiseStriatumDataControlIncV1.m:20. 408 has a raw bundle and an LFP
    # export but is deliberately absent, exactly as 507 is on the task side.
    mouse_ids=(407, 513, 515, 817, 1205),
    preproc_mat=_PROJECT / "processed_data" / "preprocessed_data_control5cm.mat",
    v1_raw_suffix="_v1_raw.mat",
    learning_point_source="task_average",
)

COHORTS: dict[str, Cohort] = {c.name: c for c in (TASK, CONTROL)}


def add_cohort_argument(parser) -> None:
    """``--cohort task|control`` (default task), the same on every driver."""
    parser.add_argument("--cohort", type=str, default="task", choices=sorted(COHORTS))


def get_cohort(name: str) -> Cohort:
    """Look up a cohort by name, failing loudly on a typo."""
    if name not in COHORTS:
        raise KeyError(f"unknown cohort {name!r}; expected one of {sorted(COHORTS)}")
    return COHORTS[name]


def raw_mat(mouse_id: int, probe: str = "striatum", cohort: Cohort = TASK) -> Path:
    """Spike/behaviour bundle for one mouse and probe ("striatum" | "visual").

    Behaviour fields are identical in both probes of a session, so either answers
    a behavioural question; spikes are probe-specific.
    """
    suffix = "_raw.mat" if probe == "striatum" else cohort.v1_raw_suffix
    return cohort.rawdata / f"{mouse_id}{suffix}"


PKG_DIR = _PROJECT / "lfp"
RESULTS_DIR = PKG_DIR / "results"
FIGURES_DIR = PKG_DIR / "figures"
# Written by IntegratedAll_v1.m: the single-unit moving reliability per animal,
# the reference the LFP moving metric is compared against. Its `animal` column
# is the position in the organiser's mouse list, not the mouse id.
STABILITY_BY_ANIMAL_CSV = _PROJECT / "figures" / "stability_by_animal.csv"

# --- Sampling / grid ---------------------------------------------------------
FS = 1000       # inferred exported-grid Hz; source metadata is not shipped

# --- Neuropixels 1.0 geometry ------------------------------------------------
# 384 channels, 2 per 20 um row -> depth[c] = (c // 2) * 20 um, spanning 0..3820
# um from tip. Coarse boundaries (100s of um) make the 2-per-row approximation
# adequate; the true channel_map.npy is not shipped with these mice. Low channel
# index == deep (near tip == ventral / striatum); high index == superficial
# (cortex / ACC).
PITCH_UM = 20.0
CH_PER_ROW = 2
N_CHANNELS = 384
# The Neuropixels 1.0 internal reference site (0-based). It carries the
# reference, not tissue: SD 13.7x the median channel in 822, r ~ -0.1 with its
# neighbours (0.87 between ordinary neighbours). Never assigned to an area.
REFERENCE_CHANNELS: tuple[int, ...] = (191,)

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
    """Tunable reading / binning parameters (frozen: a run is immutable)."""

    # --- Out-of-core reader (overlap-save) -----------------------------------
    # block_samples must be a multiple of the HDF5 chunk row-count (42) so each
    # read lands on whole chunks. 100_800 = 2400 * 42 (~101 s at 1 kHz).
    block_samples: int = 100_800
    pad_samples: int = 3_000        # >= 3 s: covers the 1-4 Hz filter transient

    # --- Spatial binning (checked against project_cfg.m by a parity test) ----
    n_spatial_bins: int = 50        # 5 cm bins over the 250 cm corridor (cfg.n_bins_full)
    max_bin: int = 30               # spatial truncation (cfg.max_bin)

    # --- Area gating ---------------------------------------------------------
    min_sites: int = 5              # skip an area with fewer channels than this


DEFAULT = Config()
