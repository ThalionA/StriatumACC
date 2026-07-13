"""Reader tests: block geometry + the overlap-save filtering guarantee.

The key correctness property is that filtering block-by-block (filter the padded
read window, keep only the pad-free core) reproduces filtering the whole signal
at once -- otherwise every block seam would inject an edge artifact into the band
power. This is what justifies streaming a 12 GB file instead of loading it.
"""

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from striatum_lfp import config, reader


def test_block_spans_tile_and_pad():
    n, bs, ps = 1000, 300, 40
    spans = reader.block_spans(n, bs, ps)
    cores = [(t0, t1) for _, t0, t1, _ in spans]
    # cores tile [0, n) exactly once, contiguous
    assert cores[0][0] == 0 and cores[-1][1] == n
    for (a0, a1), (b0, b1) in zip(cores, cores[1:]):
        assert a1 == b0
    assert sum(t1 - t0 for t0, t1 in cores) == n
    # read window contains the core and pads by <= ps, clipped to [0, n]
    for read0, t0, t1, read1 in spans:
        assert read0 <= t0 <= t1 <= read1
        assert read0 == max(0, t0 - ps) and read1 == min(n, t1 + ps)


def test_block_spans_single_block_when_large():
    spans = reader.block_spans(500, 10_000, 100)
    assert spans == [(0, 0, 500, 500)]


def _blockwise_filter(x, sos, n, block_samples, pad_samples):
    out = np.empty_like(x, dtype=float)
    for read0, t0, t1, read1 in reader.block_spans(n, block_samples, pad_samples):
        seg = sosfiltfilt(sos, x[read0:read1], axis=0)
        out[t0:t1] = seg[t0 - read0 : t1 - read0]
    return out


def test_overlap_save_matches_whole_signal_representative_band():
    # A representative band (beta 15-30 Hz) has a short transient, so pad = 3000
    # makes block-wise filtering essentially identical to whole-signal filtering.
    rng = np.random.default_rng(0)
    fs, n = 1000, 60_000
    t = np.arange(n) / fs
    sig = np.sin(2 * np.pi * 6 * t) + 0.5 * np.sin(2 * np.pi * 20 * t) + 0.3 * rng.standard_normal(n)
    x = np.stack([sig, np.roll(sig, 111)], axis=1)  # (n, 2 channels)
    sos = butter(4, [15, 30], btype="band", fs=fs, output="sos")

    whole = sosfiltfilt(sos, x, axis=0)
    out = _blockwise_filter(x, sos, n, 10_000, 3_000)
    assert np.max(np.abs(out - whole)) < 1e-6 * np.max(np.abs(whole))


def test_overlap_save_broadband_within_tolerance():
    # Broadband 1-100 Hz is the worst case: its 1 Hz edge has the longest transient.
    # At pad = 3000 the residual is ~5e-4 relative -- a smooth decayed tail near each
    # core start, negligible once enveloped and bin-averaged for band power.
    rng = np.random.default_rng(0)
    fs, n = 1000, 60_000
    t = np.arange(n) / fs
    sig = np.sin(2 * np.pi * 6 * t) + 0.5 * np.sin(2 * np.pi * 20 * t) + 0.3 * rng.standard_normal(n)
    x = np.stack([sig, np.roll(sig, 111)], axis=1)
    sos = butter(4, [1, 100], btype="band", fs=fs, output="sos")

    whole = sosfiltfilt(sos, x, axis=0)
    out = _blockwise_filter(x, sos, n, 10_000, 3_000)
    assert np.max(np.abs(out - whole)) < 2e-3 * np.max(np.abs(whole))


def test_overlap_save_small_pad_is_visibly_worse():
    # sanity that the pad matters: a tiny pad injects a much larger seam error
    # than the configured pad, so the overlap-save guarantee is doing real work.
    rng = np.random.default_rng(1)
    fs, n = 1000, 30_000
    t = np.arange(n) / fs
    x = (np.sin(2 * np.pi * 2 * t) + 0.3 * rng.standard_normal(n))[:, None]
    sos = butter(4, [1, 100], btype="band", fs=fs, output="sos")
    whole = sosfiltfilt(sos, x, axis=0)
    err_small = np.max(np.abs(_blockwise_filter(x, sos, n, 5_000, 50) - whole))
    err_big = np.max(np.abs(_blockwise_filter(x, sos, n, 5_000, 3_000) - whole))
    assert err_small > 20 * err_big


_LFP_614 = config.LFP_DIR / config.FILE_BY_MOUSE[614]


@pytest.mark.skipif(not _LFP_614.exists(), reason="LFP file not present")
def test_reader_real_file_small_block():
    assert reader.n_channels(_LFP_614) == 384
    assert reader.n_samples(_LFP_614) == 8_400_000
    blk = next(reader.iter_time_blocks(_LFP_614, block_samples=4200, pad_samples=420))
    assert blk.t0 == 0 and blk.read0 == 0
    assert blk.core_data.shape == (4200, 384)
    assert blk.data.dtype == np.float64
    assert np.isfinite(blk.core_data).all()
