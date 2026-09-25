"""Structural grid checks on the real exports -- not an offset/alignment proof.

Equal length and behaviour-inside-the-recording are necessary conditions for the
1 ms grid to be shared with ``binned_spikes``; they cannot detect a fixed offset.
Skipped wherever the voltage or the spike bundle is not on this machine.
"""

import numpy as np
import pytest

from striatum_lfp import analysis, cohort, config, reader


def _lfp_path(mouse, probe="striatum"):
    return cohort.discover_lfp_files(config.LFP_DIR).get((mouse, probe))


def _grid(mouse, probe="striatum"):
    lfp = _lfp_path(mouse, probe)
    if lfp is None or not config.raw_mat(mouse, probe).exists():
        pytest.skip("LFP/raw data absent")
    beh = analysis.read_behaviour(mouse, probe)
    vr_max_ms = int(round(float(np.max(beh["vr_times_s"])) * 1000.0))
    return reader.n_samples(lfp), beh["n_spike_bins"], vr_max_ms


@pytest.mark.parametrize("mouse", [523, 614, 624, 727, 730, 731, 822, 823,
                                   1105, 1106, 1201, 1206])
def test_grid_compatible_mice(mouse):
    """Every animal except 1212 exports exactly its ``binned_spikes`` length."""
    n_lfp, n_bins, vr_max_ms = _grid(mouse)
    assert n_lfp == 8_400_000
    assert n_bins == n_lfp           # equal length is necessary, not sufficient
    assert vr_max_ms <= n_lfp        # behaviour fits inside the recording


@pytest.mark.parametrize("probe", ["striatum", "visual"])
def test_1212_full_session_export(probe):
    """1212 was re-exported at full length on 2026-08-30/31; the gap is closed.

    The 2026-08 export stopped at 8.4 M samples against an 11.4 M-bin session, so
    the last ~41 min of behaviour -- the expert end -- had no LFP. Kept as a
    regression test because the truncation was silent: index clipping made the
    short export look like a session that ended exactly on its last sample (see
    ``bandpower.truncated_trials``).
    """
    n_lfp, n_bins, vr_max_ms = _grid(1212, probe)
    assert n_lfp == n_bins == 11_400_000
    assert vr_max_ms <= n_lfp
