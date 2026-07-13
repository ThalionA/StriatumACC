"""Structural grid checks -- deliberately not an offset/alignment proof."""

import pytest

from striatum_lfp import align, config


@pytest.mark.parametrize(
    "mouse,expect",
    [(1212, 11_400_000), (614, 8_400_000), (727, 8_400_000), (731, 8_400_000)],
)
def test_alignment_real(mouse, expect):
    lfp = config.LFP_DIR / config.FILE_BY_MOUSE[mouse]
    if not lfp.exists() or not config.RAW_MAT[mouse].exists():
        pytest.skip("LFP/raw data absent")
    a = align.check_alignment(mouse)
    assert a.lfp_n_samples == expect
    assert a.spike_n_bins == expect          # equal length is necessary, not sufficient
    assert a.vr_max_ms <= a.lfp_n_samples     # behaviour fits inside the recording
    assert a.ok
