"""Tests for the task/control cohort abstraction.

The control group is not a copy of the task group with different mouse numbers:
its probe-2 spike bundle is lowercase (``513_v1_raw.mat`` against the task's
``1105_V1_raw.mat``), its depth boundaries live in different CSVs, one of its
animals has an LFP file but is deliberately absent from the analysis list, and
its epoch windows are anchored to the TASK cohort's average learning point
because yoked controls have no learning point of their own.
"""

from __future__ import annotations

import pytest

from striatum_lfp import analysis, cohort, config, geometry


def test_both_cohorts_are_registered():
    assert set(config.COHORTS) == {"task", "control"}
    assert config.get_cohort("task").name == "task"
    assert config.get_cohort("control").name == "control"


def test_unknown_cohort_is_rejected():
    with pytest.raises(KeyError):
        config.get_cohort("control2")


def test_cohort_mouse_lists_match_the_organisers():
    # OrganiseStriatumDataIncV1.m:9 and OrganiseStriatumDataControlIncV1.m:20.
    assert config.get_cohort("task").mouse_ids == (
        523, 614, 624, 727, 730, 731, 822, 823,
        1105, 1106, 1201, 1206, 1212, 409, 418, 703)
    assert config.get_cohort("control").mouse_ids == (407, 513, 515, 817, 1205)


def test_probe_two_raw_suffix_differs_by_cohort():
    """The case trap: control is lowercase, task is uppercase."""
    assert config.get_cohort("task").v1_raw_suffix == "_V1_raw.mat"
    assert config.get_cohort("control").v1_raw_suffix == "_v1_raw.mat"


def test_raw_mat_paths_resolve_for_both_cohorts():
    task = config.raw_mat(1105, "visual", config.get_cohort("task"))
    ctrl = config.raw_mat(513, "visual", config.get_cohort("control"))
    assert task.name == "1105_V1_raw.mat" and task.exists()
    assert ctrl.name == "513_v1_raw.mat" and ctrl.exists()


# --- discovery ---------------------------------------------------------------

def test_control_discovery_finds_the_five_analysis_animals():
    ctrl = config.get_cohort("control")
    found = cohort.discover_lfp_files(ctrl.lfp_dir, ctrl.mouse_ids)
    assert {m for m, _ in found} == {407, 513, 515, 817, 1205}
    assert {(m, p) for m, p in found if p == "visual"} == {
        (513, "visual"), (515, "visual"), (817, "visual")}


def test_control_discovery_rejects_the_excluded_animal():
    """408 has an LFP export but is absent from the organiser's list."""
    ctrl = config.get_cohort("control")
    assert (ctrl.lfp_dir / "408_voltage_data_384ch.mat").exists()
    found = cohort.discover_lfp_files(ctrl.lfp_dir, ctrl.mouse_ids)
    assert 408 not in {m for m, _ in found}


def test_a_task_animal_id_is_not_accepted_as_a_control(tmp_path):
    (tmp_path / "727voltage_data_384ch.mat").write_bytes(b"")
    (tmp_path / "513_voltage_data_384ch.mat").write_bytes(b"")
    found = cohort.discover_lfp_files(tmp_path, config.get_cohort("control").mouse_ids)
    assert set(found) == {(513, "striatum")}


def test_parse_filename_is_scoped_to_the_given_mouse_list():
    assert cohort.parse_lfp_filename("513_voltage_data_384ch.mat", (513,)) == (513, "striatum")
    assert cohort.parse_lfp_filename("513_voltage_data_384ch.mat", (407,)) is None


# --- geometry ----------------------------------------------------------------

def test_control_depth_boundaries_come_from_the_control_csv():
    ctrl = config.get_cohort("control")
    b = geometry.load_area_boundaries(513, probe="striatum", cohort=ctrl)
    assert b["DMS"] == (1000.0, 1500.0)
    assert b["ACC"] == (2500.0, 3100.0)


def test_control_407_has_no_acc_band():
    ctrl = config.get_cohort("control")
    assert "ACC" not in geometry.load_area_boundaries(407, probe="striatum", cohort=ctrl)


def test_control_visual_probe_boundaries():
    ctrl = config.get_cohort("control")
    b = geometry.load_area_boundaries(817, probe="visual", cohort=ctrl)
    assert b["V1"] == (2800.0, 3650.0) and b["DG"] == (1100.0, 1650.0)


def test_task_and_control_boundaries_do_not_leak_into_each_other():
    with pytest.raises(KeyError):
        geometry.load_area_boundaries(513, probe="striatum",
                                      cohort=config.get_cohort("task"))
    with pytest.raises(KeyError):
        geometry.load_area_boundaries(727, probe="striatum",
                                      cohort=config.get_cohort("control"))


# --- learning points ---------------------------------------------------------

def test_controls_inherit_the_task_average_learning_point():
    """IntegratedAll_v1.m sets lp = avg_lp for every control animal."""
    task_lps = [v for v in analysis.cohort_learning_points().values() if v]
    expected = round(sum(task_lps) / len(task_lps))
    ctrl = analysis.cohort_learning_points(config.get_cohort("control"))
    assert set(ctrl) == {407, 513, 515, 817, 1205}
    assert set(ctrl.values()) == {expected}
    assert expected == 41            # as logged by CorridorVsDarkActivity 2026-08-27


def test_control_trial_counts_match_the_preprocessed_struct():
    counts = analysis.cohort_trial_counts(config.get_cohort("control"))
    assert [counts[m] for m in (407, 513, 515, 817, 1205)] == [209, 109, 132, 137, 145]


def test_control_behaviour_reads_the_same_vr_columns():
    beh = analysis.read_behaviour(513, "striatum", config.get_cohort("control"))
    assert beh["world"].max() > 6
    assert beh["position"].size == beh["trial"].size
    assert beh["n_spike_bins"] == 8_400_000


# --- shared results loader ---------------------------------------------------

def test_load_arms_returns_empty_for_a_missing_table():
    from striatum_lfp import results_io

    assert results_io.load_arms("evolution", "control2-does-not-exist") == []


def test_load_arms_coerces_numerics_but_keeps_labels():
    from striatum_lfp import results_io
    import numpy as np

    rows = results_io.load_arms("evolution", "task")
    if not rows:
        pytest.skip("task evolution table not generated")
    r = rows[0]
    assert isinstance(r["area"], str) and isinstance(r["band"], str)
    assert isinstance(r["z_corridor"], float)
    assert r["cohort"] == "task"
    assert np.isfinite(r["mouse_id"])


def test_hierarchical_uses_the_animal_as_the_unit():
    from striatum_lfp import results_io
    import numpy as np

    rows = [
        {"mouse_id": 1, "area": "DMS", "v": 1.0},
        {"mouse_id": 1, "area": "DMS", "v": 3.0},   # same animal: overwrites
        {"mouse_id": 2, "area": "DMS", "v": 5.0},
    ]
    mean, sem, n = results_io.hierarchical(rows, ("area",), "v")[("DMS",)]
    assert n == 2
    assert mean == pytest.approx(4.0)
    assert sem == pytest.approx(np.std([3.0, 5.0], ddof=1) / np.sqrt(2))


def test_hierarchical_drops_non_finite_values():
    from striatum_lfp import results_io

    rows = [{"mouse_id": 1, "area": "DMS", "v": float("nan")},
            {"mouse_id": 2, "area": "DMS", "v": 2.0}]
    assert results_io.hierarchical(rows, ("area",), "v")[("DMS",)][2] == 1
