"""Tests for tom_cca.subspace_stats."""

from __future__ import annotations

import numpy as np
import pytest

from striatum_tcca import subspace_stats


class _MockCell:
    def __init__(self, lag_cc_per_dim, p_peak_cc_per_dim, ifi_per_dim,
                 skip_reason=None):
        self.lag_cc_per_dim = np.asarray(lag_cc_per_dim, dtype=float)
        self.p_peak_cc_per_dim = np.asarray(p_peak_cc_per_dim, dtype=float)
        self.ifi_per_dim = np.asarray(ifi_per_dim, dtype=float)
        self.skip_reason = skip_reason


def test_skipped_cell_returns_none():
    c = _MockCell(np.zeros((11, 3)), np.ones(3), np.zeros(3),
                  skip_reason="low_crossings")
    assert subspace_stats.cell_subspace_stats(c) is None


def test_n_sig_counts_dims_below_alpha():
    cc = np.full((11, 4), 0.4)
    cc[5] = [0.5, 0.5, 0.5, 0.5]                       # peak at lag 0 row
    p = np.array([0.01, 0.04, 0.2, 0.5])
    s = subspace_stats.cell_subspace_stats(
        _MockCell(cc, p, np.zeros(4)), alpha=0.05)
    assert s.n_sig == 2
    assert list(s.sig_mask) == [True, True, False, False]


def test_mi_all_vs_mi_sig_relationship():
    # Three dims with CCs 0.5, 0.5, 0.5 -- two of them significant.
    cc = np.full((1, 3), 0.5)
    p = np.array([0.01, 0.01, 0.5])
    s = subspace_stats.cell_subspace_stats(
        _MockCell(cc, p, np.array([0.1, 0.2, 0.3])), alpha=0.05)
    # MI = -0.5 * sum log(1 - 0.25) = -0.5 * sum log(0.75)
    expected_all = -0.5 * 3 * np.log(0.75)
    expected_sig = -0.5 * 2 * np.log(0.75)
    assert abs(s.mi_all - expected_all) < 1e-9
    assert abs(s.mi_sig - expected_sig) < 1e-9
    # mi_sig <= mi_all always (sig is a subset of all).
    assert s.mi_sig <= s.mi_all + 1e-9


def test_mi_sig_zero_when_no_dim_significant():
    cc = np.full((1, 2), 0.5)
    p = np.array([0.5, 0.5])
    s = subspace_stats.cell_subspace_stats(
        _MockCell(cc, p, np.array([0.1, 0.2])), alpha=0.05)
    assert s.n_sig == 0
    assert s.mi_sig == 0.0
    assert s.mi_all > 0


def test_ifi_weighted_dominated_by_high_cc_dim():
    cc = np.array([[0.9, 0.1]])
    p = np.array([0.01, 0.01])
    s = subspace_stats.cell_subspace_stats(
        _MockCell(cc, p, np.array([+1.0, -1.0])), alpha=0.05)
    # CC-weighted: (1 * 0.9 + -1 * 0.1) / (0.9 + 0.1) = 0.8
    assert abs(s.ifi_weighted - 0.8) < 1e-9
    # Simple mean would be 0; weighted mean is +0.8.
    assert abs(s.ifi_mean_all - 0.0) < 1e-9


def test_cc_clipped_below_one_to_avoid_log_blowup():
    cc = np.array([[0.999999]])
    p = np.array([0.001])
    s = subspace_stats.cell_subspace_stats(
        _MockCell(cc, p, np.array([0.0])))
    assert np.isfinite(s.mi_all)
    assert np.isfinite(s.mi_sig)


def test_negative_held_out_cc_is_floored_at_zero():
    cc = np.array([[-0.3, 0.5]])
    p = np.array([0.01, 0.01])
    s = subspace_stats.cell_subspace_stats(
        _MockCell(cc, p, np.array([0.1, 0.2])))
    # Negative held-out CC means "direction didn't generalise" -> 0 contribution.
    assert s.peak_cc_per_dim[0] == 0.0
    # MI from a 0-CC dim is 0; so mi_all should equal -0.5 * log(0.75).
    assert abs(s.mi_all - (-0.5 * np.log(0.75))) < 1e-9


def test_per_dim_long_form_emits_one_row_per_cell_dim():
    cc = np.full((1, 3), 0.5)
    p = np.array([0.01, 0.5, 0.01])
    cells = {("naive", 1): _MockCell(cc, p, np.array([0.1, 0.2, 0.3])),
             ("expert", 2): _MockCell(cc, p, np.array([-0.1, -0.2, -0.3]))}

    class _Result:
        animal_id = 1
        area_x = "CA1"
        area_y = "V1"
    r = _Result()
    r.cells = cells
    rows = subspace_stats.per_dim_long_form([r])
    assert len(rows) == 6                              # 2 cells * 3 dims
    sig_rows = [r for r in rows if r["sig"]]
    assert len(sig_rows) == 4                          # 2 cells * 2 sig dims
