"""Per-area signals for coupling and PSI: which channels, which pairs, which filter.

Before 2026-09-25 these signals were never mains-notched (50 Hz was 55-96 % of
low-gamma power in 1105), the bipolar pairs were two sites in the same row, and
the probe's reference site sat inside eleven areas.
"""

import numpy as np

from striatum_lfp import area_signals, config, geometry

FS = config.FS


def _cube_masks(**areas):
    """A minimal band-power-cache stand-in: depths plus is_<area> masks."""
    depths = geometry.channel_depths()
    z = {"channel_depth_um": depths}
    for area in config.AREAS:
        z[f"is_{area.lower()}"] = np.zeros(depths.size, bool)
    for area, (lo, hi) in areas.items():
        z[f"is_{area.lower()}"][lo:hi] = True
    return _NpzLike(z)


class _NpzLike(dict):
    @property
    def files(self):
        return list(self)


def test_area_channels_drop_the_reference_even_from_an_old_cache():
    chans = area_signals.area_channels(_cube_masks(ACC=(180, 200)))
    assert 191 not in chans["ACC"]
    assert chans["ACC"].size == 19


def test_bipolar_is_built_from_vertical_pairs():
    """A field that varies with depth survives a vertical pair and is cancelled
    by a same-row pair; a field common to every channel is cancelled by both."""
    rng = np.random.default_rng(0)
    n = 4_000
    idx = np.arange(0, 16)
    depths = geometry.channel_depths()[idx]
    common = rng.standard_normal(n)
    gradient = rng.standard_normal(n)                  # local, scales with depth
    block = np.zeros((n, config.N_CHANNELS))
    block[:, idx] = common[:, None] * 5 + gradient[:, None] * (depths / 20.0)[None, :]
    out = area_signals.reduce_block(block, {"DMS": idx})
    bip = out[("DMS", "bipolar")]
    assert abs(np.corrcoef(bip, gradient)[0, 1]) > 0.99
    assert abs(np.corrcoef(bip, common)[0, 1]) < 0.05


def test_read_notched_removes_mains_and_keeps_theta():
    t = np.arange(20_000) / FS
    theta = np.sin(2 * np.pi * 6 * t)
    mains = 5 * np.sin(2 * np.pi * 50 * t)
    dset = np.tile((theta + mains)[:, None], (1, 4))
    seg = area_signals.read_notched(dset, 8_000, 12_000)
    assert seg.shape == (4_000, 4)
    spec = np.abs(np.fft.rfft(seg[:, 0]))
    freqs = np.fft.rfftfreq(seg.shape[0], 1 / FS)
    at = lambda f: spec[np.argmin(np.abs(freqs - f))]  # noqa: E731
    assert at(50) < 0.02 * at(6)
    np.testing.assert_allclose(seg[:, 0], theta[8_000:12_000], atol=0.05)
