"""Consistency contract for the real-data audit cache and headline summaries."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from striatum_lfp import config
from striatum_lfp.sanity import boolean_runs


def _rows(path):
    with path.open(newline="") as stream:
        return {int(row["mouse_id"]): row for row in csv.DictReader(stream)}


_SUMMARY = config.RESULTS_DIR / "sanity_summary.csv"
_TIMING = config.RESULTS_DIR / "sanity_timing_summary.csv"
_IDENTITY = config.RESULTS_DIR / "signal_identity_summary.csv"
_CACHE_PRESENT = all(
    path.exists()
    for path in (_SUMMARY, _TIMING, _IDENTITY)
) and all(
    (config.RESULTS_DIR / f"sanity_audit_{mouse}.npz").exists()
    for mouse in config.LFP_MICE
)


@pytest.mark.skipif(not _CACHE_PRESENT, reason="real-data audit cache not present")
@pytest.mark.parametrize("mouse", config.LFP_MICE)
def test_headline_integrity_summary_matches_full_scan_cache(mouse):
    row = _rows(_SUMMARY)[mouse]
    cache = np.load(config.RESULTS_DIR / f"sanity_audit_{mouse}.npz")
    zero = cache["window_exact_zero_fraction"] >= 0.99
    corridor = cache["world"] == 6
    dark = np.isfinite(cache["world"]) & ~corridor
    ordinary_rms = cache["window_median_channel_rms"][~zero]

    assert len(zero) == int(float(row["n_samples"])) // config.FS
    assert np.isclose(
        np.mean(cache["channel_exact_zero_fraction"]),
        float(row["exact_zero_fraction_all_values"]),
    )
    assert np.isclose(zero.mean(), float(row["zero_window_fraction_99pct"]))
    assert int(corridor.sum()) == int(row["corridor_window_count"])
    assert int(dark.sum()) == int(row["dark_window_count"])
    assert not zero[corridor].any()
    assert not zero[dark].any()
    assert np.isclose(np.median(ordinary_rms), float(row["median_rms_stored_units"]))

    runs = boolean_runs(zero)
    if mouse == 1212:
        assert len(runs) == 0
    else:
        assert len(runs) == 1
        assert runs[0, 1] == len(zero) - 1


@pytest.mark.skipif(not _CACHE_PRESENT, reason="real-data audit cache not present")
def test_timing_and_reference_claims_are_explicit_and_session_specific():
    timing = _rows(_TIMING)
    expected_peak_offsets = {614: 1.983, 727: 4.256, 731: 1.740}
    for mouse, expected in expected_peak_offsets.items():
        got = float(timing[mouse]["median_signed_voltage_peak_to_vr_sync_s"])
        assert np.isclose(got, expected, atol=0.002)

    identity = _rows(_IDENTITY)
    assert identity[1212]["area_mapping_status"] == "withheld_probe_identity_unverified"
    assert np.isnan(float(identity[1212]["dms_theta_median_r"]))
    assert float(identity[614]["strongest_peak_common_median_change_db"]) < -4.0
    assert abs(float(identity[614]["second_peak_common_median_change_db"])) < 0.2
    for mouse in (727, 731):
        assert abs(float(identity[mouse]["strongest_peak_common_median_change_db"])) < 0.2
        assert abs(float(identity[mouse]["second_peak_common_median_change_db"])) < 0.2


@pytest.mark.skipif(not _CACHE_PRESENT, reason="real-data audit cache not present")
def test_every_current_figure_has_compact_machine_readable_source_data():
    sanity_source = config.RESULTS_DIR / "sanity_figure_examples_and_psd.npz"
    identity_source = config.RESULTS_DIR / "signal_identity_figure_data.npz"
    assert sanity_source.exists()
    assert identity_source.exists()

    sanity = np.load(sanity_source)
    identity = np.load(identity_source)
    for mouse in config.LFP_MICE:
        assert sanity[f"{mouse}_ordinary_example_voltage"].shape == (4_000, 6)
        assert sanity[f"{mouse}_event_example_voltage"].shape == (4_000, 6)
        assert identity[f"{mouse}_stored_psd"].ndim == 1
        assert identity[f"{mouse}_common_median_referenced_psd"].shape == identity[
            f"{mouse}_stored_psd"
        ].shape
