#!/usr/bin/env python3
"""Collect the figures and tables for the 2026-09-28 meeting into
presentations/figures_2026-09-28/ (A = video/movement, B = spike CCA partial
fix, C = LFP + infotheory re-run on the fixed pipeline).

Copies only; every figure is drawn by its own pipeline (named in README.md).
Each figure is copied as its .svg + .png pair; a missing file, or one older
than the analysis it belongs to, is an error, never a silent skip.

    /opt/anaconda3/bin/python presentations/collect_meeting_2026-09-28.py
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures_2026-09-28"
VIDEO = ROOT / "video" / "figures"
CCA = ROOT / "cca" / "figures"
CCA_PREFIX = ROOT / "cca" / "results" / "_archive" / "figures_prefoldwise_2026-09-26"
LFP = ROOT / "lfp" / "figures"
INFO = ROOT / "infotheory" / "figures"

# (source stem, destination stem, not-older-than) -- stems without extension.
# The cca plotters emit PNG only (they predate the svg + png rule); B entries
# are therefore copied as PNG.
FIGURES = [
    (VIDEO / "frame_lock", "A1_video_frame_lock", "2026-09-25"),
    (VIDEO / "lick_maps", "A2_video_lick_maps", "2026-09-26"),
    (VIDEO / "motion_svd_masks", "A3_video_motion_svd_masks", "2026-09-26"),
    (VIDEO / "1201_binned", "A4_video_binned_1201", "2026-09-26"),
    (VIDEO / "1206_binned", "A4_video_binned_1206", "2026-09-26"),
    (VIDEO / "movement_encoding", "A5_movement_encoding_5cm", "2026-09-26"),
    (VIDEO / "cca_movement", "A6_cca_movement_removed", "2026-09-26"),
    (CCA_PREFIX / "stage2_comm_strength_committed_partial", "B1_cca_partial_strength_BEFORE_fix", "2026-08-12"),
    (CCA / "stage2_comm_strength_committed_partial", "B2_cca_partial_strength_AFTER_fix", "2026-09-26"),
    (CCA / "stage2_subspace_dim_committed_partial", "B3_cca_partial_subspace_dim", "2026-09-26"),
    (CCA / "directionality_partial", "B4_cca_partial_directionality", "2026-09-26"),
    (CCA / "partial_cca_committed", "B5_cca_plain_vs_partial", "2026-09-26"),
    (LFP / "lfp_task_vs_control", "C1_lfp_task_vs_control", "2026-09-26"),
    (LFP / "lfp_dls_theta_task_vs_control", "C2_lfp_dls_theta_task_vs_control", "2026-09-26"),
    (LFP / "lfp_evolution_log_task", "C3_lfp_evolution_log_task", "2026-09-26"),
    (LFP / "lfp_decoding_task", "C4_lfp_decoding_task", "2026-09-26"),
    (LFP / "lfp_reliability_moving_session_task_vs_control", "C5_lfp_reliability_task_vs_control", "2026-09-26"),
    (LFP / "lfp_cca_task", "C6_lfp_cca_task", "2026-09-26"),
    (LFP / "lfp_cca_vs_distance_task", "C7_lfp_cca_vs_distance_task", "2026-09-26"),
    (LFP / "lfp_distance_control", "C8_lfp_distance_control", "2026-09-26"),
    (LFP / "lfp_psi_direction", "C9_lfp_psi_direction", "2026-09-26"),
    (INFO / "mi_overview", "C10_spike_mi_overview", "2026-09-26"),
    (INFO / "lfp_mi_overview", "C11_lfp_mi_overview", "2026-09-26"),
]

# (source file, destination name, not-older-than)
TABLES = [
    (ROOT / "cca" / "results" / "_archive" / "epoch_stats_partial_prefoldwise_2026-09-26.csv",
     "B_epoch_stats_partial_BEFORE_fix.csv", "2026-08-12"),
    (ROOT / "cca" / "figures" / "epoch_stats_partial.csv",
     "B_epoch_stats_partial_AFTER_fix.csv", "2026-09-26"),
    (ROOT / "lfp" / "results" / "lfp_coupling_epochs_task.csv", "C_lfp_pac_coupling_epochs_task.csv", "2026-09-26"),
    (ROOT / "lfp" / "results" / "lfp_group_contrast.csv", "C_lfp_task_vs_control_contrast.csv", "2026-09-26"),
]


def _check(path: Path, not_older_than: str) -> None:
    if not path.exists():
        sys.exit(f"missing: {path}")
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    if mtime < datetime.fromisoformat(not_older_than):
        sys.exit(f"stale: {path} ({mtime:%Y-%m-%d %H:%M}) predates {not_older_than}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    for src, dst, since in FIGURES:
        for ext in ((".png",) if dst.startswith("B") else (".svg", ".png")):
            _check(src.with_suffix(ext), since)
            shutil.copy2(src.with_suffix(ext), OUT / f"{dst}{ext}")
    for src, dst, since in TABLES:
        _check(src, since)
        shutil.copy2(src, OUT / "tables" / dst)
    print(f"{len(FIGURES)} figures and {len(TABLES)} tables -> {OUT}")


if __name__ == "__main__":
    main()
