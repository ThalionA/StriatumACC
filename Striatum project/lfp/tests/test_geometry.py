"""Ground-truth tests for channel -> depth -> area mapping.

Boundaries are the real values from RawData/Neuropixels_Depth_Data.csv (probe 1)
and Neuropixels_V1_Depth_Data.csv (probe 2), so these tests double as a check
that the CSV parse matches what the spike pipeline sees.
"""

import numpy as np

from striatum_lfp import geometry


def test_channel_depths_geometry():
    d = geometry.channel_depths()
    assert d.shape == (384,)
    assert d[0] == 0.0 and d[1] == 0.0          # first row at the tip
    assert d[2] == 20.0 and d[3] == 20.0        # second row, 20 um up
    assert d[382] == 3820.0 and d[383] == 3820.0
    assert np.all(np.diff(d) >= 0)              # monotone non-decreasing
    assert np.all(d == (np.arange(384) // 2) * 20.0)


def test_channel_area_masks_inclusive():
    # inclusive of both ends; just-outside is excluded
    depths = np.array([640.0, 650.0, 900.0, 1150.0, 1160.0])
    masks = geometry.channel_area_masks(depths, {"DMS": (650.0, 1150.0)})
    assert list(masks["DMS"]) == [False, True, True, True, False]


def test_load_boundaries_614_striatum():
    b = geometry.load_area_boundaries(614, probe="striatum")
    assert b == {"DMS": (650.0, 1150.0), "DLS": (0.0, 450.0), "ACC": (2000.0, 2500.0)}


def test_load_boundaries_731_blank_dls():
    b = geometry.load_area_boundaries(731, probe="striatum")
    assert "DLS" not in b                        # 731 has a blank DLS cell
    assert b["DMS"] == (500.0, 800.0)   # corrected 2026-08-10 (was 0-300)
    assert b["ACC"] == (2300.0, 3000.0)


def test_load_boundaries_1212_both_probes():
    s = geometry.load_area_boundaries(1212, probe="striatum")
    assert s["DMS"] == (500.0, 1100.0)
    assert s["DLS"] == (0.0, 300.0)
    assert s["ACC"] == (2300.0, 2700.0)
    v = geometry.load_area_boundaries(1212, probe="visual")
    assert v["V1"] == (1100.0, 2350.0)
    assert v["CA1"] == (200.0, 650.0)
    assert v["DG"] == (0.0, 200.0)


def test_unknown_probe_raises():
    import pytest

    with pytest.raises(ValueError):
        geometry.load_area_boundaries(614, probe="banana")


def test_end_to_end_614_channel_labels():
    depths = geometry.channel_depths()
    masks = geometry.channel_area_masks(depths, geometry.load_area_boundaries(614))
    assert masks["DMS"].sum() > 0 and masks["ACC"].sum() > 0 and masks["DLS"].sum() > 0
    # ACC (2000-2500 um) sits entirely above DLS (0-450 um) on the probe
    acc_idx = np.where(masks["ACC"])[0]
    dls_idx = np.where(masks["DLS"])[0]
    assert acc_idx.min() > dls_idx.max()
    # disjoint striatal ranges -> no channel double-labelled
    assert not np.any(masks["DMS"] & masks["ACC"])
    assert not np.any(masks["DMS"] & masks["DLS"])


# --- overlapping bands (added 2026-08-27 with the two-probe cohort) ----------

def test_touching_bands_go_to_the_later_area():
    """1206's probe-2 CSV has DG ending and CA1 starting at the same 1160 um.

    MATLAB assigns areas in the CSV column order V1, CA1, DG and lets the last
    write win (OrganiseStriatumDataIncV1.m:42-49, 162-179), so the shared depth
    belongs to DG -- not to both.
    """
    from striatum_lfp import geometry

    depths = np.array([1140.0, 1160.0, 1180.0])
    masks = geometry.channel_area_masks(
        depths, {"V1": (1880.0, 2800.0), "CA1": (1160.0, 1760.0), "DG": (400.0, 1160.0)}
    )
    assert masks["DG"].tolist() == [True, True, False]
    assert masks["CA1"].tolist() == [False, False, True]
    stacked = np.vstack([masks[a] for a in masks])
    assert stacked.sum(axis=0).max() == 1          # no channel labelled twice


def test_striatal_precedence_is_dms_then_dls_then_acc():
    from striatum_lfp import geometry

    depths = np.array([250.0, 650.0])
    masks = geometry.channel_area_masks(
        depths, {"DMS": (0.0, 700.0), "DLS": (200.0, 300.0), "ACC": (600.0, 900.0)}
    )
    assert masks["DLS"].tolist() == [True, False]
    assert masks["ACC"].tolist() == [False, True]
    assert masks["DMS"].tolist() == [False, False]


def test_non_overlapping_bands_are_unchanged():
    from striatum_lfp import geometry

    depths = np.array([100.0, 900.0, 2500.0])
    masks = geometry.channel_area_masks(
        depths, {"DMS": (700.0, 1200.0), "DLS": (0.0, 500.0), "ACC": (2100.0, 2600.0)}
    )
    assert masks["DLS"].tolist() == [True, False, False]
    assert masks["DMS"].tolist() == [False, True, False]
    assert masks["ACC"].tolist() == [False, False, True]
