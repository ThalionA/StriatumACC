"""Out-of-core reader for the 384-channel LFP ``.mat`` files.

The voltage array is ~8-11 M samples x 384 channels of float32 (~12-16 GB), too
large for RAM. h5py opens it lazily; :func:`iter_time_blocks` yields contiguous
time blocks with an **overlap-save** pad on each side so a downstream zero-phase
filter (:func:`features.band_envelope`) sees no seam between blocks: the padded
region is filtered but discarded, only the pad-free ``core`` is kept.

MATLAB stores the array column-major, so an h5py view of ``data_to_save`` has
shape ``(n_samples, 384)`` -- time on axis 0, channels on axis 1 -- chunked
``(42, 384)``: whole rows of all 384 channels. Reading a slice of rows is
therefore cheap and naturally returns every channel, so channel selection is
done in numpy after the read.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .config import DEFAULT, Config

DATASET = "data_to_save"


def n_samples(path: str | Path) -> int:
    """Number of time samples (== 1 ms bins) in the LFP file."""
    with h5py.File(path, "r") as f:
        return int(f[DATASET].shape[0])


def n_channels(path: str | Path) -> int:
    """Number of channels (expected 384)."""
    with h5py.File(path, "r") as f:
        return int(f[DATASET].shape[1])


def block_spans(
    n: int, block_samples: int, pad_samples: int
) -> list[tuple[int, int, int, int]]:
    """``(read0, t0, t1, read1)`` per block: core ``[t0:t1)``, read window ``[read0:read1)``.

    Cores tile ``[0, n)`` exactly once; the read window pads each core by up to
    ``pad_samples`` on each side, clipped to ``[0, n)``. Pure/indexing-only so the
    overlap-save geometry is unit-testable without any I/O.
    """
    spans: list[tuple[int, int, int, int]] = []
    t0 = 0
    while t0 < n:
        t1 = min(t0 + block_samples, n)
        read0 = max(0, t0 - pad_samples)
        read1 = min(n, t1 + pad_samples)
        spans.append((read0, t0, t1, read1))
        t0 = t1
    return spans


@dataclass
class Block:
    """One padded time block.

    ``data`` is the ``(read_len, n_ch)`` read window (float64) including pads;
    ``core`` slices it back to the pad-free interior; ``t0``/``t1`` are the
    absolute sample bounds of that interior.
    """

    t0: int
    t1: int
    read0: int
    data: np.ndarray

    @property
    def core(self) -> slice:
        return slice(self.t0 - self.read0, self.t1 - self.read0)

    @property
    def core_data(self) -> np.ndarray:
        return self.data[self.core]


def iter_time_blocks(
    path: str | Path,
    *,
    block_samples: int | None = None,
    pad_samples: int | None = None,
    channels: np.ndarray | slice | None = None,
    cfg: Config = DEFAULT,
) -> Iterator[Block]:
    """Yield padded :class:`Block`\\ s covering the whole recording.

    ``channels`` (bool mask, index array, or ``None`` for all) is applied in
    numpy after each chunk-aligned row read. Data is upcast to float64 for the
    downstream zero-phase filtering.
    """
    bs = block_samples if block_samples is not None else cfg.block_samples
    ps = pad_samples if pad_samples is not None else cfg.pad_samples
    with h5py.File(path, "r") as f:
        dset = f[DATASET]
        n = int(dset.shape[0])
        for read0, t0, t1, read1 in block_spans(n, bs, ps):
            raw = np.asarray(dset[read0:read1, :], dtype=np.float64)
            data = raw if channels is None else raw[:, channels]
            yield Block(t0=t0, t1=t1, read0=read0, data=data)
