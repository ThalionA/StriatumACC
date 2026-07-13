"""Structural grid-compatibility guard (not a timing-alignment proof).

LFP ``n_samples`` equals ``binned_spikes`` bin-count and all VR timestamps fit
inside that nominal 1 ms grid. These necessary checks cannot detect a fixed
offset or prove that sample ``t`` corresponds to behavioural millisecond ``t``.
"""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np

from . import config, reader


@dataclass
class Alignment:
    mouse_id: int
    lfp_n_samples: int
    spike_n_bins: int
    vr_max_ms: int
    ok: bool

    def describe(self) -> str:
        return (f"mouse {self.mouse_id}: LFP {self.lfp_n_samples} samp, "
                f"spike bins {self.spike_n_bins}, VR max {self.vr_max_ms} ms -> "
                f"{'GRID-COMPATIBLE (offset unverified)' if self.ok else 'INCOMPATIBLE'}")


def spike_bincount(mouse_id: int, raw_mat=None) -> int:
    """Number of 1 ms spike bins for a mouse (``binned_spikes`` time axis)."""
    path = raw_mat or config.RAW_MAT[mouse_id]
    with h5py.File(path, "r") as f:
        return int(f["binned_spikes"].shape[0])   # h5py view (n_bins, n_units)


def vr_times_s(mouse_id: int, raw_mat=None) -> np.ndarray:
    """VR frame times in seconds (``VR_times_synched``)."""
    path = raw_mat or config.RAW_MAT[mouse_id]
    with h5py.File(path, "r") as f:
        return np.asarray(f["VR_times_synched"][()], dtype=float).ravel()


def check_alignment(mouse_id: int, lfp_path=None, raw_mat=None) -> Alignment:
    """Check necessary length/range conditions; do not infer exact alignment."""
    lfp_path = lfp_path or (config.LFP_DIR / config.FILE_BY_MOUSE[mouse_id])
    n_samp = reader.n_samples(lfp_path)
    n_bins = spike_bincount(mouse_id, raw_mat)
    vr = vr_times_s(mouse_id, raw_mat)
    vr_max_ms = int(round(float(vr.max()) * 1000.0))
    ok = (n_samp == n_bins) and (vr_max_ms <= n_samp)
    return Alignment(mouse_id, n_samp, n_bins, vr_max_ms, ok)
