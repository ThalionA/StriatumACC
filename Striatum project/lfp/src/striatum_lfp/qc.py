"""Legacy single-window channel diagnostics -- not approved for analysis.

This module underpinned the retired Stage-0 probe. A single 60 s window and an
undocumented absolute scale produced false "good/dead" channel conclusions.
Retained for its tested numerical primitives only; do not use its output as a
biological channel-quality verdict. Use the full-session audit in ``audit.py``
and ``sanity.py`` while signal provenance remains unresolved.

Two criteria, both computed on a short probe window:

* **flat-spectrum** -- legacy label for a fraction of power above ``qc_hf_lo``
  (default 300 Hz) exceeding ``qc_hf_frac_max``. The source band and anti-alias
  operation are unknown, so this must not be interpreted as a good/bad channel
  criterion for the current exports.
* **high-std** -- amplitude is an outlier *within its own area* (std >
  median + k*MAD over the area's channels), catching saturated / railing channels
  whose spectrum might still roll off.
"""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np
from scipy.signal import welch

from .config import FS, DEFAULT, Config
from .reader import DATASET


def _mad(x: np.ndarray) -> float:
    """Median absolute deviation, scaled to be a std estimate for normal data."""
    x = np.asarray(x, float)
    return float(np.median(np.abs(x - np.median(x))) * 1.4826)


@dataclass
class ChannelStats:
    std: np.ndarray          # (n_ch,) std over the probe window
    hf_frac: np.ndarray      # (n_ch,) fraction of PSD power above qc_hf_lo
    freqs: np.ndarray        # (F,) Welch frequency axis
    psd: np.ndarray          # (F, n_ch) Welch PSD
    start_sample: int


@dataclass
class QCResult:
    good: np.ndarray         # (n_ch,) legacy threshold pass; not analysis approval
    reason: np.ndarray       # (n_ch,) object -- "ok" | "flat-spectrum" | "high-std" | "not-in-area"
    std: np.ndarray
    hf_frac: np.ndarray

    def n_good(self) -> int:
        return int(self.good.sum())


def channel_stats(path, cfg: Config = DEFAULT) -> ChannelStats:
    """Per-channel std + high-frequency power fraction from a probe window."""
    s0 = cfg.qc_probe_start_s * FS
    dur = cfg.qc_probe_seconds * FS
    with h5py.File(path, "r") as f:
        d = f[DATASET]
        n = int(d.shape[0])
        s0 = int(min(s0, max(0, n - dur)))
        dur = int(min(dur, n))
        block = np.asarray(d[s0 : s0 + dur, :], dtype=np.float64)
    block = block - block.mean(axis=0, keepdims=True)
    std = block.std(axis=0)
    nper = int(min(4096, dur))
    freqs, psd = welch(block, fs=FS, nperseg=nper, axis=0)  # psd (F, n_ch)
    total = np.trapz(psd, freqs, axis=0)
    hf = freqs >= cfg.qc_hf_lo
    hf_pow = np.trapz(psd[hf], freqs[hf], axis=0)
    hf_frac = hf_pow / np.maximum(total, 1e-30)
    return ChannelStats(std=std, hf_frac=hf_frac, freqs=freqs, psd=psd, start_sample=s0)


def flag_bad_channels(std: np.ndarray, hf_frac: np.ndarray,
                      area_masks: dict[str, np.ndarray], cfg: Config = DEFAULT) -> QCResult:
    """Apply retired Stage-0 thresholds for numerical regression tests only.

    ``good`` means "passes the legacy thresholds", not "approved for LFP
    analysis". Channels in no nominal area are labelled ``not-in-area``.
    """
    std = np.asarray(std, float)
    hf_frac = np.asarray(hf_frac, float)
    n = std.size
    good = np.zeros(n, dtype=bool)
    reason = np.full(n, "not-in-area", dtype=object)
    for _area, mask in area_masks.items():
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        thr = np.median(std[idx]) + cfg.qc_std_mad_k * _mad(std[idx])
        for c in idx:
            if hf_frac[c] > cfg.qc_hf_frac_max:
                reason[c] = "flat-spectrum"
            elif std[c] > thr:
                reason[c] = "high-std"
            else:
                good[c] = True
                reason[c] = "ok"
    return QCResult(good=good, reason=reason, std=std, hf_frac=hf_frac)
