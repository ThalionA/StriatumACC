"""Regression tests for the retired Stage-0 channel-threshold primitives.

These tests verify numerical behaviour only. Source-band provenance is absent,
so passing/failing these thresholds is not a current LFP-quality verdict.
"""

import numpy as np
import pytest

from striatum_lfp import config, geometry, qc


def test_flag_rejects_flat_spectrum_and_high_std():
    rng = np.random.default_rng(0)
    n = 384
    std = rng.normal(0.08, 0.004, n)
    hf_frac = rng.uniform(0.0, 0.02, n)  # all good by default
    # one area = channels 100..119
    area = np.zeros(n, bool)
    area[100:120] = True
    std[105] = 0.30            # amplitude outlier -> high-std
    hf_frac[110] = 0.40        # flat spectrum -> flat-spectrum

    res = qc.flag_bad_channels(std, hf_frac, {"DMS": area})
    assert res.reason[110] == "flat-spectrum" and not res.good[110]
    assert res.reason[105] == "high-std" and not res.good[105]
    assert res.good[100] and res.good[119]           # normal in-area channels kept
    assert not res.good[0] and res.reason[0] == "not-in-area"
    # 20 in area, 2 rejected
    assert res.n_good() == 18


def test_flat_spectrum_checked_before_high_std():
    # a channel that is both flat AND high-std is labelled flat-spectrum (primary)
    n = 384
    std = np.full(n, 0.08)
    hf_frac = np.full(n, 0.005)
    area = np.zeros(n, bool)
    area[10:30] = True
    std[15] = 0.5
    hf_frac[15] = 0.6
    res = qc.flag_bad_channels(std, hf_frac, {"DMS": area})
    assert res.reason[15] == "flat-spectrum"


def test_marginal_channel_survives():
    # just under both thresholds -> kept
    n = 384
    std = np.full(n, 0.08)
    hf_frac = np.full(n, 0.005)
    area = np.zeros(n, bool)
    area[0:20] = True
    hf_frac[5] = config.DEFAULT.qc_hf_frac_max - 0.005  # just under 0.05
    res = qc.flag_bad_channels(std, hf_frac, {"DMS": area})
    assert res.good[5]


_LFP_614 = config.LFP_DIR / config.FILE_BY_MOUSE[614]


@pytest.mark.skipif(not _LFP_614.exists(), reason="LFP file not present")
def test_channel_stats_real_614_spans_legacy_hf_threshold():
    stats = qc.channel_stats(_LFP_614)
    assert stats.std.shape == (384,)
    assert np.all(np.isfinite(stats.hf_frac))
    assert np.all((stats.hf_frac >= 0) & (stats.hf_frac <= 1.0 + 1e-9))
    # The real export spans both sides of the retired threshold. This confirms
    # the diagnostic has dynamic range, not that either side is physiological.
    assert (stats.hf_frac > config.DEFAULT.qc_hf_frac_max).any()
    assert (stats.hf_frac < 0.01).sum() >= 20

    masks = geometry.channel_area_masks(
        geometry.channel_depths(), geometry.load_area_boundaries(614)
    )
    res = qc.flag_bad_channels(stats.std, stats.hf_frac, masks)
    assert 0 < res.n_good() <= sum(int(m.sum()) for m in masks.values())
