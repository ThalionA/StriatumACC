"""Learning-related LFP power helpers -- scale-aware and artefact-masked.

Band power is reported in the file's undocumented "stored units"; all
learning/behaviour contrasts must therefore use *relative* measures (z-score
across trials within a session, or dimensionless ratios), never absolute power,
and must never be compared across mice in absolute terms. The periodic
high-amplitude instrument artefact is excluded per traversal by a robust
outlier rule on the traversal peak amplitude.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import welch


@dataclass(frozen=True)
class Traversal:
    trial: int
    start_ms: int
    end_ms: int


def corridor_traversals(
    world: np.ndarray, trial: np.ndarray, frame_ms: np.ndarray,
    corridor_world: float = 6, min_frames: int = 20,
) -> list[Traversal]:
    """Per-trial corridor ``[start_ms, end_ms]`` windows where ``world == corridor_world``."""
    world = np.asarray(world); trial = np.asarray(trial); ms = np.asarray(frame_ms)
    corridor = world == corridor_world
    out: list[Traversal] = []
    for tr in np.unique(trial[corridor]):
        idx = np.where(corridor & (trial == tr))[0]
        if idx.size < min_frames:
            continue
        a, b = int(ms[idx].min()), int(ms[idx].max())
        if b > a:
            out.append(Traversal(int(tr), a, b))
    return out


def valid_traversals(
    traversals: list[Traversal], *, n_samples: int, min_samples: int
) -> list[Traversal]:
    """Clip traversal bounds to the voltage array and drop short segments.

    Returning the filtered traversal objects keeps every downstream metric on
    the same index set; indexing the original list after filtering can silently
    associate PSDs with the wrong trial.
    """
    if n_samples < 0 or min_samples < 1:
        raise ValueError("n_samples must be non-negative and min_samples positive")
    out = []
    for traversal in traversals:
        start = max(0, traversal.start_ms)
        stop = min(n_samples, traversal.end_ms)
        if stop - start >= min_samples:
            out.append(Traversal(traversal.trial, start, stop))
    return out


def segment_band_power(
    segment: np.ndarray, fs: float, bands: dict[str, tuple[float, float]],
    nperseg: int = 1024,
) -> dict[str, np.ndarray]:
    """Per-channel band power (Welch PSD integrated over each band).

    ``segment`` is ``(time, channel)``. Returns ``{band: (channel,) power}`` in
    stored-units^2. Scale is preserved (never thresholded here).
    """
    seg = np.asarray(segment, dtype=float)
    seg = seg - seg.mean(axis=0, keepdims=True)
    freqs, psd = welch(seg, fs=fs, nperseg=min(nperseg, seg.shape[0]), axis=0)  # (F, ch)
    out: dict[str, np.ndarray] = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        out[name] = np.trapz(psd[mask], freqs[mask], axis=0)
    return out


def robust_log_outliers(values: np.ndarray, z_threshold: float = 4.0) -> np.ndarray:
    """Boolean mask of upper outliers of ``log10(values)`` by median/MAD z-score.

    Used to flag the periodic high-amplitude artefact by traversal peak amplitude,
    scale-invariantly (a multiplicative gain shifts the log by a constant only).
    """
    v = np.asarray(values, dtype=float)
    logv = np.full(v.shape, np.nan)
    positive = v > 0
    logv[positive] = np.log10(v[positive])
    centre = np.nanmedian(logv)
    scale = 1.4826 * np.nanmedian(np.abs(logv - centre))
    if not np.isfinite(scale) or scale == 0:
        return np.zeros(v.shape, dtype=bool)
    z = (logv - centre) / scale
    return np.nan_to_num(z, nan=-np.inf) > z_threshold


def rolling_mean_sem(values: np.ndarray, window: int = 21) -> tuple[np.ndarray, np.ndarray]:
    """Centred rolling mean and standard error over an ordered sequence.

    NaNs are ignored within each window. ``values`` should already be ordered
    (e.g. by trial). Returns ``(mean, sem)`` of the same length.
    """
    x = np.asarray(values, dtype=float)
    n = x.size
    half = window // 2
    mean = np.full(n, np.nan)
    sem = np.full(n, np.nan)
    for i in range(n):
        seg = x[max(0, i - half): min(n, i + half + 1)]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            mean[i] = seg.mean()
            sem[i] = seg.std() / np.sqrt(seg.size) if seg.size > 1 else 0.0
    return mean, sem


def phase_indices(
    n_trials: int, learning_point: int | None, trials_per_epoch: int = 10,
) -> tuple[dict[str, np.ndarray], str]:
    """0-indexed naive / intermediate / expert indices from a 1-indexed LP.

    Matches ``epoch_indices.m`` defaults exactly: naive = trials ``1:w``;
    intermediate = ``lp-w:lp-1``; expert = ``lp:lp+w-1``. Returned arrays are
    converted to Python's 0-based convention. Otherwise falls back to thirds.
    """
    w = trials_per_epoch
    if learning_point is not None and learning_point - w >= 1 and learning_point + w - 1 <= n_trials:
        lp = learning_point
        return (
            {
                "naive": np.arange(0, w),
                "intermediate": np.arange(lp - w - 1, lp - 1),
                "expert": np.arange(lp - 1, lp + w - 1),
            },
            "lp",
        )
    third = max(1, n_trials // 3)
    return (
        {
            "naive": np.arange(0, third),
            "intermediate": np.arange(third, 2 * third),
            "expert": np.arange(2 * third, n_trials),
        },
        "thirds",
    )
