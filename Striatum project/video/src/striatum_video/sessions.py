"""Where each session's top-camera video and VR log live, and the frame clock.

The top camera is triggered once per VR frame: frame i of the concatenated
video parts is row i of the VR log (counts identical; wheel speed vs VR velocity
r = 0.98 at zero shift in 1206; see GOTCHAS.md). Frames therefore take their
timestamps from the VR `time` column, and from there reach Neuropixels time via
the pipeline's VR_times_synched. Filename start times and the header frame rate
are NOT a clock.

Session -> video matching (2026-09-25): high-confidence task sessions only, each
with a single VR file.
"""

import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np

RAW_ROOT = Path(os.environ.get(
    "STRIATUM_RAW_ROOT", "/Volumes/INCR-RochefortLab/Zihao/Neuropixels Raw/Raw"))
VIDEO_DIR = RAW_ROOT / "video output" / "Top (02G55471207)"
VR_DIR = RAW_ROOT / "vroutput" / "VR_ZC_01"
VR_CACHE = Path(__file__).resolve().parents[2] / "results" / "vr_cache"

SESSIONS = {
    "1105": {"video": "Video_20241105190735390", "vr": "Test01_2024115_1920.csv"},
    "1106": {"video": "Video_20241106183745768", "vr": "Test01_2024116_196.csv"},
    "1201": {"video": "Video_20241201181115289", "vr": "Test01_2024121_1822.csv"},
    "1206": {"video": "Video_20241206181317609", "vr": "Test01_2024126_1824.csv"},
}

VR_COLUMNS = ["time", "x", "dtframe", "velocity", "world", "valve", "trial", "lick", "sync"]


def parse_part_start(name):
    """Wall-clock time a video part was opened, used only to order the parts.
    Base file: Video_YYYYMMDDHHMMSSmmm.avi; continuations:
    <base>_YYYYMMDD_HHMMSS_0.avi."""
    m = re.search(r"_(\d{8})_(\d{6})_\d+\.avi$", name)
    if m:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    m = re.search(r"Video_(\d{17})\.avi$", name)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S%f")
    raise ValueError(f"unrecognised video filename {name}")


def video_parts(base, video_dir=VIDEO_DIR):
    """All parts of one recording in order: the base file, then its continuations."""
    parts = [Path(video_dir) / f"{base}.avi"]
    parts += sorted(Path(video_dir).glob(f"{base}_*_0.avi"), key=lambda p: parse_part_start(p.name))
    return parts


def load_vr(name, vr_dir=VR_DIR):
    """VR log as {column: array}. Columns per Synch_NP_VR.m; time is s from VR start.
    Reads the local copy in results/vr_cache/ when present (the SMB share drops)."""
    cached = VR_CACHE / name
    data = np.loadtxt(cached if cached.exists() else Path(vr_dir) / name, delimiter=";")
    return {c: data[:, i] for i, c in enumerate(VR_COLUMNS)}


def frame_times(vr, n_frames):
    """VR time (s) of every video frame. Refuses to guess if the frame count
    differs from the VR row count: the one-trigger-per-VR-frame link is then
    broken somewhere and must be found, not interpolated over."""
    if n_frames != vr["time"].size:
        raise ValueError(f"{n_frames} video frames vs {vr['time'].size} VR rows: frame lock not established")
    return vr["time"].copy()
