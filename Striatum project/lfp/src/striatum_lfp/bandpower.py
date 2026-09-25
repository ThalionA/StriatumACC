"""LFP band power binned exactly like the unit pipeline bins firing rate.

The product is the drop-in analogue of ``spatial_binned_fr_all`` and
``temp_binned_dark_fr``: for each trial, mean band power per 5 cm corridor bin
and per 100 ms dark bin, per channel. Indexing matches the MATLAB arrays, so an
LFP array can be substituted wherever a unit array is used.

**One deliberate difference from the unit pipeline.** A unit's rate is spikes
divided by the bin's *occupancy* -- a denominator that carries the known
``(k-1)*dt`` defect and makes the estimate speed-dependent. Band power needs no
counting denominator at all: the value stored here is the plain mean of the
envelope power over the samples inside the bin. That removes the speed bias, but
it also means the corridor arm is a **different estimand** from the unit corridor
arm, which is why the two must never be captioned as "the same analysis".

Timing follows ``OrganiseStriatumDataIncV1.m`` (crop at
``ceil(VR_times_synched(1)*1000)``), ``cut_data_per_trial.m`` (trial edges from
``diff(vr_trial) ~= 0``), ``separate_dark_and_corridor_periods.m`` (corridor
opens at the frame after ``world > 6``) and ``spatial_binning.m`` (a bin spans
the first to the last VR frame inside it, and needs at least two frames).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import iirnotch, sosfiltfilt, tf2sos

from .filtering import design_band_sos

from . import config

# --- Bands -------------------------------------------------------------------
# Analysis bands. ``total`` is the 1/f denominator: reporting a band as a
# fraction of it separates a genuine band change from an aperiodic-slope change,
# which is the confound that sank the previous learning analysis.
ANALYSIS_BANDS: dict[str, tuple[float, float]] = {
    "theta": (4.0, 8.0),
    "beta": (15.0, 30.0),
    "low_gamma": (30.0, 80.0),
    "high_gamma": (80.0, 150.0),
    "total": (1.0, 150.0),
}
# Mains and its harmonics are notched on every file, unconditionally: the
# measured 50 Hz excess ranges from 1.4x to 1348x across this cohort, so notching
# only the bad sessions would make the estimator a function of the mouse.
NOTCH_HZ = (50.0, 100.0, 150.0)
NOTCH_Q = 30.0

CORRIDOR_END_AU = 200.0
BIN_SIZE_AU = 4.0               # project_cfg cfg.bin_size_au -> 5 cm
N_SPATIAL_BINS = 50
DARK_BIN_MS = 100
N_DARK_BINS = 50                # temp_bin_edges = 1:100:5001


def spatial_bin_edges() -> np.ndarray:
    """``0:4:200`` with the final edge widened to 204, as ProcessStriatumTask does.

    The widened last edge exists so a position of exactly 200 a.u. lands in bin
    50 rather than falling outside ``histcounts``.
    """
    edges = np.arange(0.0, CORRIDOR_END_AU + BIN_SIZE_AU, BIN_SIZE_AU)
    edges[-1] = CORRIDOR_END_AU + BIN_SIZE_AU
    return edges


# --- Trial geometry ----------------------------------------------------------

def trial_boundaries(vr_trial: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inclusive ``(start, end)`` VR-frame indices per trial.

    Mirrors ``cut_data_per_trial.m``: a boundary is *any* change in the trial
    channel, not an increment, and the final trial runs to the last frame.
    """
    v = np.asarray(vr_trial).ravel()
    change = np.flatnonzero(np.diff(v) != 0)
    ends = np.append(change, v.size - 1)
    starts = np.insert(change + 1, 0, 0)
    return starts, ends


def npx_index(times_ms: np.ndarray, n_npx: int) -> np.ndarray:
    """Nearest sample index for VR times on the ``0:n_npx-1`` millisecond grid.

    Reproduces ``interp1(npx_time, 1:n, t, 'nearest', 'extrap')`` minus MATLAB's
    1-based offset, including its clipping at both ends.
    """
    idx = np.rint(np.asarray(times_ms, dtype=float)).astype(np.int64)
    return np.clip(idx, 0, n_npx - 1)


def corridor_start(trial_world: np.ndarray, trial_times_zeroed: np.ndarray):
    """``(vr_index, corridor_onset_ms)`` for one trial, or ``None`` if there is none.

    ``separate_dark_and_corridor_periods.m`` finds the first frame with
    ``world > 6`` and then reads the time of the frame *after* it; a trial where
    either step fails is silently dropped by its ``try/catch`` and becomes a
    non-good trial. Both failure modes return ``None`` here rather than
    improvising a corridor onset.
    """
    world = np.asarray(trial_world).ravel()
    times = np.asarray(trial_times_zeroed, dtype=float).ravel()
    hits = np.flatnonzero(world > 6)
    if hits.size == 0:
        return None
    vr_idx = int(hits[0])
    if vr_idx + 1 >= times.size:
        return None
    return vr_idx, float(times[vr_idx + 1] - times[0])


def spatial_bin_segments(trial_position: np.ndarray, trial_times_zeroed: np.ndarray,
                         bin_edges: np.ndarray) -> list[tuple[int, int] | None]:
    """Per spatial bin, the corridor-relative ``(first_ms, last_ms)`` sample range.

    A bin is used only if at least two VR frames fall in it (``sum(idx_in_bin) > 1``
    in ``spatial_binning.m``), and it spans the first to the last frame in the bin
    even if the animal left and re-entered -- both behaviours are copied
    deliberately so the bin map is identical to the unit one.
    """
    position = np.asarray(trial_position, dtype=float).ravel()
    times = np.asarray(trial_times_zeroed, dtype=float).ravel()
    if position.size == 0 or times.size == 0:
        return [None] * (len(bin_edges) - 1)
    times = times - times[0]
    bin_idx = np.digitize(position, bin_edges) - 1
    bin_idx[(position < bin_edges[0]) | (position >= bin_edges[-1])] = -1

    out: list[tuple[int, int] | None] = []
    for b in range(len(bin_edges) - 1):
        in_bin = np.flatnonzero(bin_idx == b)
        if in_bin.size <= 1:
            out.append(None)
            continue
        bin_times = times[in_bin]
        out.append((int(round(bin_times[0])), int(round(bin_times[-1]))))
    return out


def dark_bin_segments(n_dark_samples: int) -> list[tuple[int, int] | None]:
    """50 consecutive 100 ms bins from the start of the dark period.

    ``temp_bin_edges = 1:100:5001`` in ProcessStriatumTask, i.e. the first 5 s.
    Bins beyond the dark period return ``None``; a partial final bin is kept,
    matching ``histcounts`` assigning whatever samples exist.
    """
    out: list[tuple[int, int] | None] = []
    for b in range(N_DARK_BINS):
        start = b * DARK_BIN_MS
        if start >= n_dark_samples:
            out.append(None)
            continue
        out.append((start, min(start + DARK_BIN_MS, n_dark_samples) - 1))
    return out


# --- Filtering ---------------------------------------------------------------

def _notch_sos(fs: int) -> np.ndarray:
    return np.vstack([tf2sos(*iirnotch(hz, NOTCH_Q, fs=fs)) for hz in NOTCH_HZ])


def apply_notches(x: np.ndarray, fs: int = config.FS) -> np.ndarray:
    """Zero-phase notch at 50/100/150 Hz along axis 0."""
    return sosfiltfilt(_notch_sos(fs), x, axis=0)


def band_power_series(x: np.ndarray, band: tuple[float, float], *,
                      fs: int = config.FS, order: int = 4) -> np.ndarray:
    """Instantaneous band power: the square of the zero-phase bandpassed signal.

    Squaring rather than taking a Hilbert envelope is deliberate -- the value is
    averaged over a bin immediately afterwards, which does the smoothing, and it
    avoids allocating a complex copy of a multi-gigabyte block. The mean of this
    series over a window equals that window's band power, so a unit-amplitude
    sinusoid inside the band reads 1/2.
    """
    sos = design_band_sos(band, fs=fs, order=order)
    return np.square(sosfiltfilt(sos, x, axis=0))



# --- Accumulation ------------------------------------------------------------

@dataclass
class SegmentAccumulator:
    """Streaming mean over arbitrary sample ranges ("cells") of a long recording.

    Blocks arrive in order; each segment ``(cell, start, stop)`` is clipped to the
    block and its sum and sample count added to that cell. A cell split across a
    block boundary therefore gets the same mean as if the recording had been read
    in one piece, which is what makes out-of-core extraction exact rather than
    approximate.
    """

    n_cells: int
    n_channels: int

    def __post_init__(self) -> None:
        self.sums = np.zeros((self.n_cells, self.n_channels), dtype=np.float64)
        self.counts = np.zeros(self.n_cells, dtype=np.int64)

    def add_block(self, data: np.ndarray, block_start: int,
                  segments: list[tuple[int, int, int]]) -> None:
        if data.shape[1] != self.n_channels:
            raise ValueError(
                f"block has {data.shape[1]} channels, accumulator expects {self.n_channels}"
            )
        block_stop = block_start + data.shape[0]
        for cell, start, stop in segments:
            lo = max(start, block_start)
            hi = min(stop, block_stop)
            if hi <= lo:
                continue
            self.sums[cell] += data[lo - block_start:hi - block_start].sum(axis=0)
            self.counts[cell] += hi - lo

    def result(self) -> np.ndarray:
        """``(n_cells, n_channels)`` means; ``nan`` for cells with no samples."""
        out = np.full((self.n_cells, self.n_channels), np.nan)
        seen = self.counts > 0
        out[seen] = self.sums[seen] / self.counts[seen, None]
        return out


def truncated_trials(trial_end_ms: np.ndarray, n_lfp_samples: int,
                     crop_start0: int) -> np.ndarray:
    """Boolean mask of trials whose VR end falls past the end of the EXPORT.

    ``trial_end_ms`` is crop-relative (ms after ``crop_start0``). A trial is cut
    short only if the voltage file stops before it ends, so the test is against
    the samples the file holds after the crop start -- not against the crop
    itself, whose ``floor(t_end) - ceil(t0)`` length always sits 0.1-1.7 ms short
    of the last VR frame and used to flag the last trial of every session.

    Needed because ``npx_index`` clips to the crop, so a trial that runs off the
    end of a short export (1212's August file, 407's) comes back looking like a
    trial that happens to finish exactly at the last sample.
    """
    available = int(n_lfp_samples) - int(crop_start0)
    return np.rint(np.asarray(trial_end_ms, dtype=float)) > (available - 1)
