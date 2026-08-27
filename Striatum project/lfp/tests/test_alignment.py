"""Structural grid checks -- deliberately not an offset/alignment proof."""

import pytest

from striatum_lfp import align, cohort, config


def _lfp_path(mouse, probe="striatum"):
    return cohort.discover_lfp_files(config.LFP_DIR).get((mouse, probe))


@pytest.mark.parametrize("mouse", [523, 614, 624, 727, 730, 731, 822, 823,
                                   1105, 1106, 1201, 1206])
def test_grid_compatible_mice(mouse):
    """Every animal except 1212 exports exactly its ``binned_spikes`` length."""
    lfp = _lfp_path(mouse)
    if lfp is None or not config.RAW_MAT[mouse].exists():
        pytest.skip("LFP/raw data absent")
    a = align.check_alignment(mouse, lfp_path=lfp)
    assert a.lfp_n_samples == 8_400_000
    assert a.spike_n_bins == a.lfp_n_samples  # equal length is necessary, not sufficient
    assert a.vr_max_ms <= a.lfp_n_samples     # behaviour fits inside the recording
    assert a.ok


@pytest.mark.parametrize("probe", ["striatum", "visual"])
def test_1212_export_is_truncated_not_grid_compatible(probe):
    """1212 is the one animal whose export is shorter than its session.

    Its ``binned_spikes`` runs 11.4 M bins (190 min) but the 2026-08 export stops
    at 8.4 M (140 min), so ~41 min of behaviour has no LFP. The offset scan in
    ``scripts/run_lfp_identity.py`` places the export at offset 0, i.e. it is the
    truncated head of the same session rather than a different recording -- but
    the grid check must still fail, and trial-indexed analyses must exclude it.
    """
    lfp = _lfp_path(1212, probe)
    raw = config.raw_mat(1212, probe)
    if lfp is None or not raw.exists():
        pytest.skip("LFP/raw data absent")
    a = align.check_alignment(1212, lfp_path=lfp, raw_mat=raw)
    assert a.lfp_n_samples == 8_400_000
    assert a.spike_n_bins == 11_400_000
    assert not a.ok
