"""Ground-truth tests for the distance-along-the-shank control.

The control exists to answer one question: is the cross-area LFP coupling
anything more than distance along the shank? The cross-area canonical
correlations fall monotonically with electrode separation (CA1-DG at 510 um
scores 0.93, ACC-DLS at 2044 um scores 0.55), which is the shape of a
volume-conducted field, and the within-area split-half "ceiling" is saturated at
0.97-0.999 so it discriminates nothing.

The test the meeting asked for is a matched-distance comparison: pairs the same
physical distance apart, within one area versus across an area boundary. These
tests pin the two outcomes that decide the answer, on synthetic fields where the
right answer is known by construction:

* a field whose correlation depends ONLY on separation must give within == across
  at matched separation (the null: coupling is distance, area labels add nothing);
* a field with an extra component shared inside one area must give
  within > across at matched separation (the alternative: an area-specific term).

Created 2026-09-17.
"""
from __future__ import annotations

import numpy as np
import pytest

from striatum_lfp import distance

PITCH = 20.0          # um between adjacent rows, as on the probe
N_BINS, N_TRIALS = 50, 40


def _depths(n_ch: int) -> np.ndarray:
    """Two channels per row, PITCH um apart, as `geometry.channel_depths` gives."""
    return np.repeat(np.arange(n_ch // 2) * PITCH, 2)[:n_ch].astype(float)


def _distance_field(depths, *, length_um=200.0, seed=0):
    """Channel signals whose correlation is EXACTLY a function of |dz|.

    Sampled from a multivariate normal whose covariance is the stationary kernel
    exp(-0.5 (dz / length_um)^2), so the field is translation-invariant by
    construction and there is no edge effect. An earlier version summed Gaussian-
    weighted sources over a finite depth range; channels near the ends of that
    range drew on fewer sources and were therefore MORE correlated with their
    neighbours, which showed up as a spurious within > across difference of 0.08
    and would have made this a null that is not null.
    """
    rng = np.random.default_rng(seed)
    dz = depths[:, None] - depths[None, :]
    cov = np.exp(-0.5 * (dz / length_um) ** 2)
    cov = cov + 1e-8 * np.eye(len(depths))            # keep it positive definite
    chol = np.linalg.cholesky(cov)
    sig = chol @ rng.standard_normal((len(depths), N_BINS * N_TRIALS))
    return np.exp(sig.reshape(len(depths), N_BINS, N_TRIALS))   # power is positive


def _areas(depths, boundary_um):
    """Two contiguous areas split at `boundary_um`."""
    return np.where(depths < boundary_um, "A", "B")


def _matched(res, lo, hi):
    """Mean r within and across, over pairs whose separation is in [lo, hi)."""
    sel = (res.separation_um >= lo) & (res.separation_um < hi)
    within = res.r_raw[sel & res.same_area]
    across = res.r_raw[sel & ~res.same_area]
    return within, across


def test_separation_is_the_absolute_depth_difference():
    depths = _depths(8)
    cube = _distance_field(depths)
    res = distance.pairwise_coupling(cube, depths, _areas(depths, 40.0))
    for k in range(res.separation_um.size):
        i, j = res.channel_i[k], res.channel_j[k]
        assert res.separation_um[k] == pytest.approx(abs(depths[i] - depths[j]))
    assert (res.channel_i < res.channel_j).all(), "each unordered pair appears once"
    n = len(depths)
    assert res.separation_um.size == n * (n - 1) // 2


def test_area_labels_are_carried_through():
    depths = _depths(12)
    areas = _areas(depths, 60.0)
    res = distance.pairwise_coupling(_distance_field(depths), depths, areas)
    for k in range(res.separation_um.size):
        i, j = res.channel_i[k], res.channel_j[k]
        assert res.same_area[k] == (areas[i] == areas[j])
        assert {res.area_i[k], res.area_j[k]} == {areas[i], areas[j]}


def test_unlabelled_channels_are_dropped():
    depths = _depths(12)
    areas = _areas(depths, 60.0).astype(object)
    areas[:4] = ""                                  # outside every area band
    res = distance.pairwise_coupling(_distance_field(depths), depths, areas)
    assert res.channel_i.size == 8 * 7 // 2
    assert (res.channel_i >= 4).all() and (res.channel_j >= 4).all()


def test_pure_distance_field_gives_within_equals_across():
    """The null. Correlation is a function of separation, so the boundary is invisible."""
    depths = _depths(80)
    areas = _areas(depths, depths[len(depths) // 2])
    res = distance.pairwise_coupling(_distance_field(depths, seed=1), depths, areas)
    # A narrow band, so the two classes are compared at all but identical
    # separations and any difference is an area effect rather than a distance one.
    within, across = _matched(res, 190.0, 211.0)
    assert within.size > 10 and across.size > 10, "need both classes at this separation"
    sep_w = res.separation_um[(res.separation_um >= 190) & (res.separation_um < 211) & res.same_area]
    sep_a = res.separation_um[(res.separation_um >= 190) & (res.separation_um < 211) & ~res.same_area]
    assert abs(sep_w.mean() - sep_a.mean()) < 5.0, "the two classes must be distance-matched"
    assert abs(within.mean() - across.mean()) < 0.05, (
        f"a distance-only field must not separate: within {within.mean():.3f} "
        f"vs across {across.mean():.3f}")


def test_area_specific_component_makes_within_exceed_across():
    """The alternative. An extra term shared inside area A must show up as within > across."""
    depths = _depths(80)
    boundary = depths[len(depths) // 2]
    areas = _areas(depths, boundary)
    cube = _distance_field(depths, seed=2)
    rng = np.random.default_rng(3)
    shared = rng.standard_normal((1, N_BINS, N_TRIALS))
    inside = depths < boundary
    cube = cube.copy()
    cube[inside] *= np.exp(1.5 * shared)            # same extra drive for all of A
    res = distance.pairwise_coupling(cube, depths, areas)
    within, across = _matched(res, 190.0, 211.0)
    assert within.mean() > across.mean() + 0.10, (
        f"an area-specific term must separate: within {within.mean():.3f} "
        f"vs across {across.mean():.3f}")


def test_residual_removes_a_purely_spatial_profile():
    """A cube with a shared spatial profile but no trial-to-trial covariation.

    `r_raw` sees the shared profile and is high; `r_residual` subtracts each
    channel's own mean profile first and must therefore find nothing.
    """
    rng = np.random.default_rng(4)
    depths = _depths(20)
    profile = np.abs(rng.standard_normal(N_BINS)) + 1.0
    indep = rng.standard_normal((len(depths), N_BINS, N_TRIALS)) * 0.02
    cube = np.exp(profile[None, :, None] + indep)
    res = distance.pairwise_coupling(cube, depths, _areas(depths, 100.0))
    assert res.r_raw.mean() > 0.9, "the shared spatial profile should dominate r_raw"
    assert abs(res.r_residual).mean() < 0.15, (
        f"r_residual should find no trial-to-trial structure, got "
        f"{abs(res.r_residual).mean():.3f}")


def test_summarise_bins_by_separation_and_class():
    depths = _depths(60)
    areas = _areas(depths, depths[30])
    res = distance.pairwise_coupling(_distance_field(depths, seed=5), depths, areas)
    rows = distance.summarise(res, bin_um=100.0)
    assert rows, "expected at least one summary row"
    for r in rows:
        assert r["n_pairs"] >= 1
        assert r["pair_class"] in {"within", "across"}
        assert r["sep_bin_lo_um"] % 100 == 0
        assert r["sep_bin_lo_um"] <= r["mean_separation_um"] < r["sep_bin_lo_um"] + 100
    # Counts must account for every pair exactly once.
    assert sum(r["n_pairs"] for r in rows) == res.separation_um.size


def test_matched_range_reports_where_both_classes_exist():
    depths = _depths(60)
    areas = _areas(depths, depths[30])
    res = distance.pairwise_coupling(_distance_field(depths, seed=6), depths, areas)
    lo, hi = distance.matched_separation_range(res)
    sel_w = res.same_area & (res.separation_um >= lo) & (res.separation_um <= hi)
    sel_a = ~res.same_area & (res.separation_um >= lo) & (res.separation_um <= hi)
    assert sel_w.any() and sel_a.any()
    # Beyond the upper end there must be no within-area pairs left to compare.
    assert not (res.same_area & (res.separation_um > hi)).any()


def _unbalanced_layout(n_ch=120, area_um=400.0):
    """Many thin areas, so within-area pairs are systematically CLOSER than across ones.

    This reproduces the geometry of the real probe: an area spans a few hundred
    microns, so within-area separations are capped while across-area separations
    run to the length of the shank. Restricting both classes to a shared range
    leaves them badly distance-mismatched.
    """
    depths = _depths(n_ch)
    areas = np.array([f"A{int(d // area_um)}" for d in depths], dtype=object)
    return depths, areas


def test_range_restriction_is_biased_by_the_separation_imbalance():
    """The bug this module was nearly shipped with, pinned as a test.

    On a distance-only field the truth is zero. A range-restricted comparison
    still reports a positive within - across, because within-area pairs inside
    the shared range are closer than across-area ones.
    """
    depths, areas = _unbalanced_layout()
    res = distance.pairwise_coupling(_distance_field(depths, seed=11), depths, areas)
    lo, hi = distance.matched_separation_range(res)
    in_range = (res.separation_um >= lo) & (res.separation_um <= hi)
    sep_w = res.separation_um[in_range & res.same_area].mean()
    sep_a = res.separation_um[in_range & ~res.same_area].mean()
    assert sep_a - sep_w > 50.0, "this layout must leave the classes distance-mismatched"
    naive = (res.r_raw[in_range & res.same_area].mean()
             - res.r_raw[in_range & ~res.same_area].mean())
    assert naive > 0.05, (
        "the range-restricted contrast should be visibly biased on a distance-only "
        f"field, got {naive:+.3f}")


def test_exact_matching_removes_that_bias():
    """The same field and layout, compared only at identical separations: no effect."""
    depths, areas = _unbalanced_layout()
    res = distance.pairwise_coupling(_distance_field(depths, seed=11), depths, areas)
    out = distance.exact_matched_contrast(res, min_pairs=5)
    assert out["n_separations"] >= 5, "need several separations with both classes"
    assert abs(out["d_raw"]) < 0.03, (
        f"exact matching must recover zero on a distance-only field, got {out['d_raw']:+.3f}")


def test_exact_matching_still_finds_a_real_area_term():
    """It must not be so conservative that it misses a true within-area component."""
    depths, areas = _unbalanced_layout()
    cube = _distance_field(depths, seed=12)
    rng = np.random.default_rng(13)
    inside = np.array([a == "A1" for a in areas])
    shared = rng.standard_normal((1, N_BINS, N_TRIALS))
    cube = cube.copy()
    cube[inside] *= np.exp(1.5 * shared)
    res = distance.pairwise_coupling(cube, depths, areas)
    out = distance.exact_matched_contrast(res, min_pairs=5)
    assert out["d_raw"] > 0.05, f"a real area term must survive, got {out['d_raw']:+.3f}"


# --- the across-animal test (moved out of the plot script, 2026-09-25) --------

def _matched_rows(values_by_mouse, band="theta", probes=("striatum",)):
    return [{"mouse_id": m, "probe": p, "band": band, "n_separations": 3, "d_raw": v}
            for m, v in values_by_mouse.items() for p in probes]


def test_contrast_stats_averages_probes_then_tests_across_animals():
    rows = _matched_rows({m: 0.1 for m in range(6)}, probes=("striatum", "visual"))
    out = distance.contrast_stats(rows, bands=("theta",))
    (r,) = out
    assert r["n_animals"] == 6                       # probes averaged, not counted twice
    assert r["p_raw"] == pytest.approx(2 / 2**6)
    assert r["reachable"]
    assert r["ci95_low"] == pytest.approx(0.1) and r["ci95_high"] == pytest.approx(0.1)


def test_contrast_stats_marks_an_unreachable_band():
    rows = _matched_rows({m: 0.1 for m in range(4)})
    (r,) = distance.contrast_stats(rows, bands=("theta",))
    assert not r["reachable"] and r["p_floor"] == pytest.approx(0.125)


def test_contrast_stats_skips_cells_without_matched_separations():
    rows = _matched_rows({m: 0.1 for m in range(6)})
    rows[0]["n_separations"] = 0
    (r,) = distance.contrast_stats(rows, bands=("theta",))
    assert r["n_animals"] == 5
