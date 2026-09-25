"""Zero-phase band filtering: the filter design, envelope and phase every arm shares.

Each band is a Butterworth bandpass in second-order sections, applied with
``sosfiltfilt`` so it adds no phase lag. The amplitude envelope and the phase
are the modulus and angle of the analytic (Hilbert) signal of that output.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from .config import FS

FILTER_ORDER = 4


def design_band_sos(band: tuple[float, float], fs: int = FS,
                    order: int = FILTER_ORDER) -> np.ndarray:
    """Second-order-sections Butterworth bandpass for ``(lo, hi)`` Hz."""
    lo, hi = band
    return butter(order, [lo, hi], btype="band", fs=fs, output="sos")


def band_envelope(x: np.ndarray, sos: np.ndarray, axis: int = 0) -> np.ndarray:
    """Instantaneous amplitude of ``x`` after a zero-phase bandpass (shape of ``x``)."""
    return np.abs(hilbert(sosfiltfilt(sos, x, axis=axis), axis=axis))


def band_phase(x: np.ndarray, sos: np.ndarray, axis: int = 0) -> np.ndarray:
    """Instantaneous phase in ``(-pi, pi]`` of ``x`` after a zero-phase bandpass.

    The companion of :func:`band_envelope`: that returns ``abs`` of the analytic
    signal, this returns its angle. Zero-phase filtering matters here more than
    for the envelope -- a filter that shifted phase would shift every coupling
    measure built on it.
    """
    return np.angle(hilbert(sosfiltfilt(sos, x, axis=axis), axis=axis))
