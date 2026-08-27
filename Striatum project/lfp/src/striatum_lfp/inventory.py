"""Per-file inventory of an LFP voltage export: structure, integrity, signal character.

One :func:`inventory_file` call answers, for a single ``(mouse, probe)`` export:

* **structure** -- shape/dtype/chunking, and whether the shipped ``depth_to_save``
  matches the Neuropixels 1.0 geometry the area mapping assumes;
* **grid compatibility** -- sample count against that probe's ``binned_spikes``;
* **integrity** -- a full single pass over every stored value: exact zeros,
  non-finite values, per-second RMS, and where the terminal zero padding starts;
* **signal character** -- per-window Welch spectra giving the 1/f slope, the
  low-to-high power ratio, mains and the two 2026-07 contamination peaks, plus
  adjacent- and distant-channel correlation and common-mode residual;
* **fingerprint** -- a content hash over fixed sample slices, so two files that
  are secretly the same recording cannot both be believed.

The identity check against spiking lives in :mod:`cohort`; this module supplies
the envelope it consumes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt, welch

from . import config, geometry
from .reader import DATASET

# Diagnostic bands (Hz). LF/HF is the 1/f test that separated real LFP from the
# scrambled June export (1-10 vs 100-200 Hz: 0.6-1.5 broadband vs 97-1400 LFP).
LF_BAND = (1.0, 10.0)
HF_BAND = (100.0, 200.0)
SLOPE_BAND = (2.0, 40.0)
# Narrowband peaks to police: mains, and the ~75/151 Hz pair that invalidated
# the 30-80 Hz band in the July audit.
LINE_HZ = (50.0, 75.0, 151.0)
LINE_NEIGHBOUR_HZ = 5.0     # ratio is peak / median power in +-5 Hz shoulders
COUPLING_BAND = (30.0, 90.0)


@dataclass
class FileInventory:
    mouse_id: int
    probe: str
    path: str
    size_bytes: int
    n_samples: int
    n_channels: int
    dtype: str
    chunks: tuple[int, ...] | None
    compression: str | None
    depth_matches_geometry: bool
    depth_max_abs_error_um: float
    channels_are_1_to_n: bool
    spike_n_bins: int | None
    grid_compatible: bool | None
    vr_first_s: float | None
    vr_last_s: float | None
    exact_zero_fraction: float
    nonfinite_fraction: float
    padding_start_s: float | None
    dead_channel_count: int
    median_rms: float
    lf_hf_ratio: float
    loglog_slope_2_40hz: float
    line_ratios: dict[str, float]
    adjacent_r: float
    distant_r: float
    common_mean_residual: float
    common_median_residual: float
    fingerprint: str
    area_channel_counts: dict[str, int]
    notes: list[str] = field(default_factory=list)


# --- structure ---------------------------------------------------------------

def read_structure(path: Path) -> dict:
    """Shape/dtype/chunking plus the shipped depth and channel vectors."""
    with h5py.File(path, "r") as handle:
        dset = handle[DATASET]
        n_samples, n_channels = map(int, dset.shape)
        depth = (
            np.asarray(handle["depth_to_save"]).ravel().astype(float)
            if "depth_to_save" in handle else np.array([])
        )
        channels = (
            np.asarray(handle["channels_to_save"]).ravel().astype(float)
            if "channels_to_save" in handle else np.array([])
        )
        return {
            "n_samples": n_samples,
            "n_channels": n_channels,
            "dtype": str(dset.dtype),
            "chunks": dset.chunks,
            "compression": dset.compression,
            "depth": depth,
            "channels": channels,
        }


def check_depth_against_geometry(depth: np.ndarray, n_channels: int) -> tuple[bool, float]:
    """Compare the shipped depths with ``geometry.channel_depths``.

    The area mapping assumes 2 channels per 20 um row; the 2026-08 export ships
    the depths, so the assumption is now falsifiable rather than inherited.
    """
    if depth.size != n_channels:
        return False, float("nan")
    predicted = geometry.channel_depths(n_channels)
    err = float(np.max(np.abs(depth - predicted)))
    return err == 0.0, err


# --- integrity (one full pass) ----------------------------------------------

def scan_integrity(path: Path, *, fs: int = config.FS, block_seconds: int = 42) -> dict:
    """Single pass over every stored value: zeros, non-finites, per-second RMS.

    ``block_seconds`` defaults to 42 so each read lands on whole HDF5 chunks.
    """
    block = fs * block_seconds
    zero_count = 0
    nonfinite_count = 0
    channel_zero = None
    rms_per_s: list[np.ndarray] = []
    zero_frac_per_s: list[np.ndarray] = []
    channel_sq = None

    with h5py.File(path, "r") as handle:
        dset = handle[DATASET]
        n_samples, n_channels = map(int, dset.shape)
        channel_zero = np.zeros(n_channels, dtype=np.int64)
        channel_sq = np.zeros(n_channels, dtype=np.float64)
        complete = (n_samples // fs) * fs

        for start in range(0, n_samples, block):
            stop = min(start + block, n_samples)
            raw = np.asarray(dset[start:stop, :], dtype=np.float32)
            is_zero = raw == 0.0
            zero_count += int(is_zero.sum(dtype=np.int64))
            nonfinite_count += int((~np.isfinite(raw)).sum(dtype=np.int64))
            channel_zero += is_zero.sum(axis=0, dtype=np.int64)
            channel_sq += np.square(raw, dtype=np.float64).sum(axis=0)

            stop_c = min(stop, complete)
            if stop_c <= start:
                continue
            usable = raw[: stop_c - start]
            n_win = usable.shape[0] // fs
            win = usable[: n_win * fs].reshape(n_win, fs, n_channels)
            rms_per_s.append(np.median(np.sqrt(np.mean(np.square(win, dtype=np.float64), axis=1)), axis=1))
            zero_frac_per_s.append((win == 0.0).mean(axis=(1, 2)))

    rms = np.concatenate(rms_per_s)
    zero_frac = np.concatenate(zero_frac_per_s)
    total = n_samples * n_channels
    return {
        "n_samples": n_samples,
        "n_channels": n_channels,
        "exact_zero_fraction": zero_count / total,
        "nonfinite_fraction": nonfinite_count / total,
        "channel_zero_fraction": channel_zero / n_samples,
        "channel_rms": np.sqrt(channel_sq / n_samples),
        "rms_per_s": rms,
        "zero_fraction_per_s": zero_frac,
    }


def padding_onset_s(zero_fraction_per_s: np.ndarray, threshold: float = 0.99) -> float | None:
    """Start (s) of the terminal all-zero block, or ``None`` if the file has none.

    Only a run that reaches the end of the file counts: an isolated dropout in
    the middle is a different defect and must not be reported as padding.
    """
    zero = np.asarray(zero_fraction_per_s) >= threshold
    if zero.size == 0 or not zero[-1]:
        return None
    idx = zero.size
    while idx > 0 and zero[idx - 1]:
        idx -= 1
    return float(idx)


# --- signal character --------------------------------------------------------

def window_starts(n_samples: int, n_windows: int, window_samples: int,
                  *, first: int = 0, last: int | None = None) -> np.ndarray:
    """Evenly spaced window starts inside ``[first, last)``, never overlapping the end."""
    last = n_samples if last is None else last
    hi = max(first, last - window_samples)
    if n_windows <= 1:
        return np.array([first], dtype=int)
    return np.linspace(first, hi, n_windows).astype(int)


def band_power(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """Mean PSD inside ``band`` for each channel (psd is ``(n_freqs, n_ch)``)."""
    lo, hi = band
    sel = (freqs >= lo) & (freqs <= hi)
    return psd[sel].mean(axis=0)


def line_ratio(freqs: np.ndarray, psd: np.ndarray, target_hz: float,
               neighbour_hz: float = LINE_NEIGHBOUR_HZ) -> float:
    """Power at ``target_hz`` over the median of its +-``neighbour_hz`` shoulders.

    1.0 means no narrowband peak; the July export showed >2 at mains and a large
    excess at ~75/151 Hz. Averaged over channels first, so one bad channel
    cannot manufacture a session-wide peak.
    """
    mean_psd = psd.mean(axis=1)
    peak_sel = np.abs(freqs - target_hz) <= 0.5
    shoulder = (np.abs(freqs - target_hz) <= neighbour_hz) & ~peak_sel
    if not peak_sel.any() or not shoulder.any():
        return float("nan")
    return float(mean_psd[peak_sel].max() / np.median(mean_psd[shoulder]))


def loglog_slope(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> float:
    """Least-squares slope of log10(PSD) vs log10(f) -- the 1/f exponent.

    Real LFP sits near -1 to -3; a flat slope means broadband/white content.
    """
    lo, hi = band
    sel = (freqs >= lo) & (freqs <= hi) & (freqs > 0)
    mean_psd = psd[sel].mean(axis=1)
    good = mean_psd > 0
    if good.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log10(freqs[sel][good]), np.log10(mean_psd[good]), 1)[0])


def spatial_correlations(block: np.ndarray, distant_offset: int = 100) -> tuple[float, float]:
    """Median correlation between adjacent channels and between ``+offset`` channels.

    Real LFP is smooth in depth (high adjacent r) but decorrelates over a
    millimetre (low distant r). The scrambled June export failed the second test.
    """
    x = block - block.mean(axis=0, keepdims=True)
    sd = x.std(axis=0)
    sd[sd == 0] = np.nan
    z = x / sd
    n = z.shape[1]
    adjacent = np.nanmean(z[:, :-1] * z[:, 1:], axis=0)
    distant = np.nanmean(z[:, : n - distant_offset] * z[:, distant_offset:], axis=0)
    return float(np.nanmedian(adjacent)), float(np.nanmedian(distant))


def common_mode_residuals(block: np.ndarray) -> tuple[float, float]:
    """SD of the across-channel mean and median, in units of a typical channel SD.

    Near 0 means that reference was already subtracted. The 2026-08-11 audit read
    0.12-0.20 (mean) vs 0.05-0.07 (median) as common-*median* referencing.
    """
    channel_sd = np.median(block.std(axis=0))
    if channel_sd == 0:
        return float("nan"), float("nan")
    return (float(block.mean(axis=1).std() / channel_sd),
            float(np.median(block, axis=1).std() / channel_sd))


def spectral_profile(path: Path, starts: np.ndarray, window_samples: int,
                     *, fs: int = config.FS) -> dict:
    """Welch spectra and spatial diagnostics averaged over the given windows."""
    psds, adjacent, distant, mean_res, median_res = [], [], [], [], []
    with h5py.File(path, "r") as handle:
        dset = handle[DATASET]
        for start in starts:
            block = np.asarray(dset[start:start + window_samples, :], dtype=np.float64)
            freqs, psd = welch(block, fs=fs, nperseg=fs, axis=0)
            psds.append(psd)
            a, d = spatial_correlations(block)
            adjacent.append(a)
            distant.append(d)
            m, md = common_mode_residuals(block)
            mean_res.append(m)
            median_res.append(md)
    psd = np.mean(psds, axis=0)
    lf = band_power(freqs, psd, LF_BAND).mean()
    hf = band_power(freqs, psd, HF_BAND).mean()
    return {
        "freqs": freqs,
        "psd": psd,
        "lf_hf_ratio": float(lf / hf) if hf > 0 else float("nan"),
        "loglog_slope_2_40hz": loglog_slope(freqs, psd, SLOPE_BAND),
        "line_ratios": {f"{hz:g}Hz": line_ratio(freqs, psd, hz) for hz in LINE_HZ},
        "adjacent_r": float(np.nanmedian(adjacent)),
        "distant_r": float(np.nanmedian(distant)),
        "common_mean_residual": float(np.nanmedian(mean_res)),
        "common_median_residual": float(np.nanmedian(median_res)),
    }


# --- fingerprint -------------------------------------------------------------

def fingerprint(path: Path, *, n_slices: int = 8, slice_samples: int = 4200) -> str:
    """SHA1 over evenly spaced sample slices -- cheap content identity.

    Two exports of different sessions cannot collide; two names for one file
    will. This is what would have caught the 614/731 duplicate at download time.
    """
    digest = hashlib.sha1()
    with h5py.File(path, "r") as handle:
        dset = handle[DATASET]
        n = int(dset.shape[0])
        for start in np.linspace(0, max(0, n - slice_samples), n_slices).astype(int):
            digest.update(np.asarray(dset[start:start + slice_samples, :],
                                     dtype=np.float32).tobytes())
    return digest.hexdigest()


# --- envelope for the identity test -----------------------------------------

def coupling_envelope(path: Path, start: int, n_samples_win: int,
                      *, channel_step: int = 4, band: tuple[float, float] = COUPLING_BAND,
                      fs: int = config.FS, bin_ms: int = 100) -> np.ndarray:
    """High-frequency amplitude envelope, binned, for the file-identity test.

    Uses a squared-and-smoothed envelope rather than a Hilbert transform: it is
    equivalent once binned to ``bin_ms`` and avoids allocating a complex copy of
    a multi-gigabyte block. ``channel_step`` subsamples channels -- the statistic
    is a mean over channels, so a quarter of the probe is ample.
    """
    from .cohort import bin_mean

    sos = butter(4, list(band), btype="band", fs=fs, output="sos")
    with h5py.File(path, "r") as handle:
        block = np.asarray(handle[DATASET][start:start + n_samples_win, ::channel_step],
                           dtype=np.float64)
    filtered = sosfiltfilt(sos, block, axis=0)
    power = uniform_filter1d(np.square(filtered), int(fs * bin_ms / 1000),
                             axis=0, mode="nearest")
    return bin_mean(np.sqrt(power), int(fs * bin_ms / 1000))
