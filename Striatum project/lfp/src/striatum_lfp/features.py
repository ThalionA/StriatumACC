"""Band-power feature extraction: the LFP analogue of a unit's firing rate.

For each band the voltage is bandpassed with a zero-phase Butterworth filter
(``sosfiltfilt`` on a second-order-sections design -> no phase distortion) and
its amplitude envelope is taken from the analytic (Hilbert) signal. The envelope
    stays on the nominal 1 kHz export-index grid. Equal array lengths make that
    grid structurally compatible with 1 ms spike bins, but the exact offset is
    unresolved; position/time binning remains gated.

Zero-phase filtering avoids adding a further deterministic lag. It cannot fix
the undocumented voltage-to-behaviour offset.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, hilbert, sosfiltfilt

from .config import BANDS, DEFAULT, FS, Config


def design_band_sos(band: tuple[float, float], fs: int = FS, order: int | None = None,
                    cfg: Config = DEFAULT) -> np.ndarray:
    """Second-order-sections Butterworth bandpass for ``(lo, hi)`` Hz."""
    lo, hi = band
    order = cfg.filt_order if order is None else order
    return butter(order, [lo, hi], btype="band", fs=fs, output="sos")


def band_envelope(x: np.ndarray, sos: np.ndarray, method: str = "hilbert",
                  smooth_samples: int = 0, axis: int = 0) -> np.ndarray:
    """Zero-phase bandpass + amplitude envelope of ``x`` along ``axis``.

    ``method="hilbert"``: ``abs(analytic(filtered))`` -- the instantaneous
    amplitude. ``method="square_smooth"``: ``sqrt`` of a boxcar-smoothed squared
    filtered signal (``smooth_samples`` window). Result has the shape of ``x`` and
    is non-negative.
    """
    filtered = sosfiltfilt(sos, x, axis=axis)
    if method == "hilbert":
        return np.abs(hilbert(filtered, axis=axis))
    if method == "square_smooth":
        power = np.square(filtered)
        if smooth_samples and smooth_samples > 1:
            power = uniform_filter1d(power, int(smooth_samples), axis=axis, mode="nearest")
        return np.sqrt(power)
    raise ValueError(f"unknown envelope method {method!r}")


def extract_bands(x: np.ndarray, *, bands: dict[str, tuple[float, float]] = BANDS,
                  fs: int = FS, cfg: Config = DEFAULT) -> dict[str, np.ndarray]:
    """``{band_name: envelope}`` for an in-memory (padded) block along axis 0.

    Call on a read window; the caller keeps the pad-free ``core`` (see
    :mod:`reader`). Each envelope has the same shape as ``x``.
    """
    smooth = int(round(cfg.smooth_ms * fs / 1000.0))
    out: dict[str, np.ndarray] = {}
    for name, band in bands.items():
        sos = design_band_sos(band, fs=fs, cfg=cfg)
        out[name] = band_envelope(x, sos, method=cfg.envelope, smooth_samples=smooth)
    return out
