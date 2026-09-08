"""The Python LFP constants must agree with MATLAB's `project_cfg.m`.

`project_cfg.m` is the project's single source of truth for areas, the spatial
grid, and the learning-point rule. The Python package mirrors those values as
module constants and says so in comments, but a comment cannot fail. When the
grid was re-cut from 2.5 cm to 5 cm on 2026-08-10 every consumer had to be
repointed by hand, and a Python mirror left behind would have gone on producing
plausible numbers on the wrong grid.

This test parses `project_cfg.m` directly and asserts the mirrors match, so a
change on the MATLAB side fails here instead of drifting silently. It reads the
file as text rather than launching MATLAB: the assignments are plain literals,
and the whole point is that this runs in the ordinary pytest sweep.

Created 2026-09-08.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from striatum_lfp import analysis, config

PROJECT_CFG = Path(__file__).resolve().parents[2] / "project_cfg.m"


def _scalars() -> dict[str, float]:
    """`cfg.<name> = <number>;` assignments, as floats."""
    text = PROJECT_CFG.read_text()
    pattern = re.compile(r"^\s*cfg\.(\w+)\s*=\s*(-?\d+(?:\.\d+)?)\s*;", re.MULTILINE)
    return {m.group(1): float(m.group(2)) for m in pattern.finditer(text)}


def _areas() -> tuple[str, ...]:
    """The `cfg.areas = {...}` cell array, in declaration order."""
    text = PROJECT_CFG.read_text()
    m = re.search(r"^\s*cfg\.areas\s*=\s*\{([^}]*)\}", text, re.MULTILINE)
    assert m, "cfg.areas not found in project_cfg.m"
    return tuple(re.findall(r"'([^']+)'", m.group(1)))


def test_project_cfg_is_readable():
    assert PROJECT_CFG.exists(), f"{PROJECT_CFG} not found"
    scalars = _scalars()
    for key in ("bin_size_au", "au_to_cm", "corridor_au", "max_bin",
                "lp_z_threshold", "lp_window", "lp_min_consecutive",
                "trials_per_epoch"):
        assert key in scalars, f"cfg.{key} missing from project_cfg.m"


@pytest.mark.parametrize("cfg_key, py_value, label", [
    ("au_to_cm", config.AU_TO_CM, "config.AU_TO_CM"),
    ("max_bin", config.Config().max_bin, "Config.max_bin"),
    ("lp_z_threshold", analysis.LP_Z_THRESHOLD, "analysis.LP_Z_THRESHOLD"),
    ("lp_window", analysis.LP_WINDOW, "analysis.LP_WINDOW"),
    ("lp_min_consecutive", analysis.LP_MIN_CONSECUTIVE, "analysis.LP_MIN_CONSECUTIVE"),
    ("trials_per_epoch", analysis.TRIALS_PER_EPOCH, "analysis.TRIALS_PER_EPOCH"),
])
def test_scalar_mirrors_match(cfg_key, py_value, label):
    matlab = _scalars()[cfg_key]
    assert py_value == matlab, (
        f"{label} = {py_value} but project_cfg.m cfg.{cfg_key} = {matlab}. "
        "project_cfg.m is the source of truth; update the Python mirror.")


def test_derived_grid_matches():
    """bin_size_cm and n_bins_full are derived in project_cfg; derive them the same way."""
    s = _scalars()
    bin_size_cm = s["bin_size_au"] * s["au_to_cm"]
    n_bins_full = s["corridor_au"] / s["bin_size_au"]
    assert analysis.BIN_SIZE_CM == bin_size_cm, (
        f"analysis.BIN_SIZE_CM = {analysis.BIN_SIZE_CM} but project_cfg derives "
        f"{bin_size_cm} cm/bin ({s['bin_size_au']} a.u. * {s['au_to_cm']} cm/a.u.)")
    assert config.Config().n_spatial_bins == n_bins_full, (
        f"Config.n_spatial_bins = {config.Config().n_spatial_bins} but project_cfg "
        f"derives {n_bins_full} bins")
    assert config.CORRIDOR_CM == s["corridor_au"] * s["au_to_cm"]


def test_area_list_matches():
    assert config.AREAS == _areas(), (
        f"config.AREAS = {config.AREAS} but project_cfg.m cfg.areas = {_areas()}")
    for area in config.AREAS:
        assert area in config.AREA_FIELD, f"{area} has no preprocessed-struct field name"
