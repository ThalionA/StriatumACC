"""Scale-aware, threshold-free diagnostics for continuous LFP exports.

The voltage scale is not yet documented.  Consequently, continuity is measured
from exact stored zeros and finite values; amplitude is reported but never used
as an absolute signal-present threshold.  Spectral flatness and common-mode
fraction are dimensionless and therefore invariant to V/mV/uV scaling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import welch


@dataclass(frozen=True)
class WindowMetrics:
    exact_zero_fraction: float
    nonzero_sample_fraction: float
    finite_fraction: float
    median_channel_rms: float
    median_channel_mad: float
    spectral_flatness: float
    common_mode_fraction: float


@dataclass(frozen=True)
class FixedWindowSummary:
    exact_zero_fraction: np.ndarray
    median_channel_rms: np.ndarray
    median_channel_abs: np.ndarray
    max_abs: np.ndarray


@dataclass(frozen=True)
class TemporalIdentityMetrics:
    median_lag1_correlation: float
    median_difference_to_signal_sd: float
    power_fraction_1_100hz: float
    loglog_slope_2_40hz: float


def boolean_runs(mask: np.ndarray) -> np.ndarray:
    """Inclusive ``[start, stop]`` bounds of contiguous True runs."""
    values = np.asarray(mask, dtype=bool).ravel()
    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1) - 1
    return np.column_stack([starts, stops]).astype(int)


def distance_to_nearest(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Absolute distance from each query value to the nearest reference value."""
    q = np.asarray(query, dtype=float)
    r = np.sort(np.asarray(reference, dtype=float))
    if r.size == 0:
        return np.full(q.shape, np.nan)
    pos = np.searchsorted(r, q)
    left = r[np.clip(pos - 1, 0, r.size - 1)]
    right = r[np.clip(pos, 0, r.size - 1)]
    return np.minimum(np.abs(q - left), np.abs(q - right))


def signed_distance_to_nearest(
    query: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Signed query-minus-reference distance to the nearest reference value.

    Positive values mean that the query occurred after its nearest reference;
    negative values mean that it occurred before. Ties are assigned to the
    earlier reference, making the convention deterministic.
    """
    q = np.asarray(query, dtype=float)
    r = np.sort(np.asarray(reference, dtype=float))
    if r.size == 0:
        return np.full(q.shape, np.nan)
    pos = np.searchsorted(r, q)
    left = r[np.clip(pos - 1, 0, r.size - 1)]
    right = r[np.clip(pos, 0, r.size - 1)]
    nearest = np.where(np.abs(q - left) <= np.abs(q - right), left, right)
    return q - nearest


def common_median_reference(voltage: np.ndarray) -> np.ndarray:
    """Subtract the across-channel median independently at each sample."""
    x = np.asarray(voltage, dtype=float)
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("voltage must be time x channel")
    return x - np.median(x, axis=1, keepdims=True)


def robust_synchronous_peak_sample(voltage: np.ndarray) -> int:
    """Index of the strongest transient shared across channels.

    Each channel is centred and scaled by its temporal median/MAD before the
    median absolute score is taken across channels. Thus one extreme electrode
    cannot define the event time, and multiplicative channel gains do not
    affect the result.
    """
    x = np.asarray(voltage, dtype=float)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("voltage must be non-empty time x channel")
    centre = np.median(x, axis=0, keepdims=True)
    centred = x - centre
    scale = 1.4826 * np.median(np.abs(centred), axis=0)
    fallback = np.std(centred, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    score = np.median(np.abs(centred / scale), axis=1)
    return int(np.argmax(score))


def robust_zscore(values: np.ndarray) -> np.ndarray:
    """Median/MAD z-score, with a finite standard-deviation fallback."""
    x = np.asarray(values, dtype=float)
    centre = np.nanmedian(x)
    scale = 1.4826 * np.nanmedian(np.abs(x - centre))
    if not np.isfinite(scale) or scale == 0:
        scale = np.nanstd(x)
    if not np.isfinite(scale) or scale == 0:
        return np.zeros_like(x)
    return (x - centre) / scale


def summarise_fixed_windows(
    voltage: np.ndarray, *, window_samples: int
) -> FixedWindowSummary:
    """Vectorised, scale-preserving summaries for equal-duration windows."""
    x = np.asarray(voltage, dtype=np.float64)
    if x.ndim != 2 or window_samples <= 0:
        raise ValueError("voltage must be time x channel and window_samples positive")
    if x.shape[0] % window_samples:
        raise ValueError("time dimension must be a multiple of window_samples")
    windows = x.reshape(-1, window_samples, x.shape[1])
    centred = windows - windows.mean(axis=1, keepdims=True)
    channel_rms = np.sqrt(np.mean(centred**2, axis=1))
    return FixedWindowSummary(
        exact_zero_fraction=np.mean(windows == 0.0, axis=(1, 2)),
        median_channel_rms=np.median(channel_rms, axis=1),
        median_channel_abs=np.median(np.abs(windows), axis=(1, 2)),
        max_abs=np.max(np.abs(windows), axis=(1, 2)),
    )


def median_welch_psd(
    voltage_windows: np.ndarray,
    *,
    fs_hz: float,
    nperseg: int = 2_048,
) -> tuple[np.ndarray, np.ndarray]:
    """Median Welch PSD across windows and channels.

    ``voltage_windows`` is ``window x time x channel``. Taking the median at
    both repeated-measure levels prevents one transient or noisy channel from
    defining the spectrum.
    """
    x = np.asarray(voltage_windows, dtype=np.float64)
    if x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 4 or x.shape[2] < 1:
        raise ValueError("voltage_windows must be window x time x channel")
    centred = x - x.mean(axis=1, keepdims=True)
    frequencies, psd = welch(
        centred,
        fs=fs_hz,
        nperseg=min(nperseg, x.shape[1]),
        axis=1,
    )
    # scipy output is window x frequency x channel when time axis is 1;
    # collapse the two repeated-measure axes and retain frequency.
    return frequencies, np.median(psd, axis=(0, 2))


def temporal_identity_metrics(
    voltage_windows: np.ndarray, *, fs_hz: float
) -> TemporalIdentityMetrics:
    """Dimensionless checks for LF-like temporal smoothness and concentration."""
    x = np.asarray(voltage_windows, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError("voltage_windows must be window x time x channel")
    centred = x - x.mean(axis=1, keepdims=True)
    numerator = np.sum(centred[:, :-1] * centred[:, 1:], axis=1)
    denominator = np.sum(centred[:, :-1] ** 2, axis=1)
    lag1 = numerator / np.maximum(denominator, np.finfo(float).tiny)
    signal_sd = np.std(centred, axis=1)
    difference_sd = np.std(np.diff(centred, axis=1), axis=1)
    difference_ratio = difference_sd / np.maximum(signal_sd, np.finfo(float).tiny)

    frequencies, psd = median_welch_psd(x, fs_hz=fs_hz)
    total_band = (frequencies >= 1) & (frequencies <= fs_hz / 2 - 1)
    low_band = (frequencies >= 1) & (frequencies <= 100)
    total_power = np.trapz(psd[total_band], frequencies[total_band])
    low_power = np.trapz(psd[low_band], frequencies[low_band])
    slope_band = (frequencies >= 2) & (frequencies <= 40) & (psd > 0)
    slope = np.polyfit(
        np.log10(frequencies[slope_band]), np.log10(psd[slope_band]), 1
    )[0]
    return TemporalIdentityMetrics(
        median_lag1_correlation=float(np.median(lag1)),
        median_difference_to_signal_sd=float(np.median(difference_ratio)),
        power_fraction_1_100hz=float(low_power / total_power),
        loglog_slope_2_40hz=float(slope),
    )


def frame_values_at_samples(
    sample_indices: np.ndarray,
    frame_indices: np.ndarray,
    frame_values: np.ndarray,
) -> np.ndarray:
    """Piecewise-constant frame values using the latest frame at each sample.

    Samples before the first frame are returned as NaN.  This avoids the future-
    frame error produced by bare ``searchsorted(..., side='left')``.
    """
    samples = np.asarray(sample_indices)
    frames = np.asarray(frame_indices)
    values = np.asarray(frame_values)
    order = np.argsort(frames, kind="stable")
    frames = frames[order]
    values = values[order]
    pos = np.searchsorted(frames, samples, side="right") - 1
    out = np.full(samples.shape, np.nan, dtype=float)
    valid = pos >= 0
    out[valid] = values[pos[valid]]
    return out


def robust_window_metrics(
    voltage: np.ndarray,
    *,
    fs_hz: float,
    spectral_band_hz: tuple[float, float] = (1.0, 100.0),
) -> WindowMetrics:
    """Summarise one time x channel window without an amplitude threshold."""
    x = np.asarray(voltage, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 4 or x.shape[1] < 1:
        raise ValueError("voltage must be time x channel with at least 4 samples")

    finite = np.isfinite(x)
    finite_fraction = float(finite.mean())
    exact_zero_fraction = float(np.equal(x, 0.0).mean())
    nonzero_sample_fraction = float(np.not_equal(x, 0.0).mean())
    x = np.where(finite, x, np.nan)
    centred = x - np.nanmean(x, axis=0, keepdims=True)
    channel_rms = np.sqrt(np.nanmean(centred**2, axis=0))
    channel_mad = 1.4826 * np.nanmedian(
        np.abs(x - np.nanmedian(x, axis=0, keepdims=True)), axis=0
    )

    # Dimensionless synchrony proxy: variance of the channel mean relative to
    # mean channel variance.  1 means identical channels; near 0 means largely
    # independent or cancelling activity.
    channel_var = np.nanmean(centred**2, axis=0)
    denom = float(np.nanmean(channel_var))
    common_mode_fraction = (
        float(np.nanvar(np.nanmean(centred, axis=1))) / denom if denom > 0 else 0.0
    )

    # Welch PSD after pooling channel PSDs. Geometric/arithmetic mean is 1 for
    # white spectra and approaches 0 for strongly structured spectra.
    clean = np.nan_to_num(centred, nan=0.0)
    nperseg = min(2_048, clean.shape[0])
    freqs, psd = welch(clean, fs=fs_hz, nperseg=nperseg, axis=0)
    pooled_psd = np.mean(psd, axis=1)
    band = (freqs >= spectral_band_hz[0]) & (freqs <= spectral_band_hz[1])
    positive = pooled_psd[band]
    positive = positive[positive > 0]
    if positive.size:
        spectral_flatness = float(
            np.exp(np.mean(np.log(positive))) / np.mean(positive)
        )
    else:
        spectral_flatness = 0.0

    return WindowMetrics(
        exact_zero_fraction=exact_zero_fraction,
        nonzero_sample_fraction=nonzero_sample_fraction,
        finite_fraction=finite_fraction,
        median_channel_rms=float(np.nanmedian(channel_rms)),
        median_channel_mad=float(np.nanmedian(channel_mad)),
        spectral_flatness=spectral_flatness,
        common_mode_fraction=common_mode_fraction,
    )
