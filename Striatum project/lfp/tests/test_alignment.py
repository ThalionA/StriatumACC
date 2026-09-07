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
def test_1212_full_session_export(probe):
    """1212 was re-exported at full length on 2026-08-30/31; the gap is closed.

    The 2026-08 export stopped at 8.4 M samples against an 11.4 M-bin session, so
    the last ~41 min of behaviour -- the expert end -- had no LFP, and every
    trial-indexed result excluded it. The replacement runs the whole session and
    is grid-compatible on both probes. Kept as a regression test because the
    truncation was silent: index clipping made the short export look like a
    session that happened to end exactly on the last sample (see
    ``bandpower.truncated_trials``).
    """
    lfp = _lfp_path(1212, probe)
    raw = config.raw_mat(1212, probe)
    if lfp is None or not raw.exists():
        pytest.skip("LFP/raw data absent")
    a = align.check_alignment(1212, lfp_path=lfp, raw_mat=raw)
    assert a.lfp_n_samples == 11_400_000
    assert a.spike_n_bins == 11_400_000
    assert a.vr_max_ms <= a.lfp_n_samples
    assert a.ok
