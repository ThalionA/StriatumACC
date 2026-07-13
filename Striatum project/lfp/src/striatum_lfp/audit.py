"""Out-of-core, full-session integrity audit for LFP voltage exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .reader import DATASET
from .sanity import summarise_fixed_windows


@dataclass(frozen=True)
class VoltageAudit:
    n_samples: int
    n_channels: int
    dtype: str
    chunks: tuple[int, ...] | None
    compression: str | None
    fill_value: float | None
    exact_zero_count: int
    finite_count: int
    total_value_count: int
    channel_exact_zero_fraction: np.ndarray
    window_start_sample: np.ndarray
    window_exact_zero_fraction: np.ndarray
    window_median_channel_rms: np.ndarray
    window_median_channel_abs: np.ndarray
    window_max_abs: np.ndarray


def audit_voltage_file(
    path: str | Path,
    *,
    fs_hz: int,
    window_seconds: int = 1,
    block_windows: int = 42,
) -> VoltageAudit:
    """Read every stored value once and return exact continuity summaries.

    Only complete windows are included in the time series. Global and channel
    counts include every sample, including a final partial window if present.
    """
    window_samples = int(fs_hz * window_seconds)
    if window_samples <= 0 or block_windows <= 0:
        raise ValueError("window length and block_windows must be positive")

    zero_count = 0
    finite_count = 0
    channel_zero_count: np.ndarray | None = None
    starts: list[np.ndarray] = []
    zero_windows: list[np.ndarray] = []
    rms_windows: list[np.ndarray] = []
    abs_windows: list[np.ndarray] = []
    max_windows: list[np.ndarray] = []

    with h5py.File(path, "r") as handle:
        dataset = handle[DATASET]
        n_samples, n_channels = map(int, dataset.shape)
        channel_zero_count = np.zeros(n_channels, dtype=np.int64)
        block_samples = window_samples * block_windows
        complete_n = (n_samples // window_samples) * window_samples

        for start in range(0, n_samples, block_samples):
            stop = min(start + block_samples, n_samples)
            raw = np.asarray(dataset[start:stop, :])
            is_zero = raw == 0.0
            zero_count += int(is_zero.sum(dtype=np.int64))
            finite_count += int(np.isfinite(raw).sum(dtype=np.int64))
            channel_zero_count += is_zero.sum(axis=0, dtype=np.int64)

            summary_stop = min(stop, complete_n)
            if summary_stop <= start:
                continue
            summary_raw = raw[: summary_stop - start]
            summary = summarise_fixed_windows(
                summary_raw, window_samples=window_samples
            )
            n_windows = summary.exact_zero_fraction.size
            starts.append(start + np.arange(n_windows) * window_samples)
            zero_windows.append(summary.exact_zero_fraction)
            rms_windows.append(summary.median_channel_rms)
            abs_windows.append(summary.median_channel_abs)
            max_windows.append(summary.max_abs)

        return VoltageAudit(
            n_samples=n_samples,
            n_channels=n_channels,
            dtype=str(dataset.dtype),
            chunks=dataset.chunks,
            compression=dataset.compression,
            fill_value=None if dataset.fillvalue is None else float(dataset.fillvalue),
            exact_zero_count=zero_count,
            finite_count=finite_count,
            total_value_count=n_samples * n_channels,
            channel_exact_zero_fraction=channel_zero_count / n_samples,
            window_start_sample=np.concatenate(starts),
            window_exact_zero_fraction=np.concatenate(zero_windows),
            window_median_channel_rms=np.concatenate(rms_windows),
            window_median_channel_abs=np.concatenate(abs_windows),
            window_max_abs=np.concatenate(max_windows),
        )
