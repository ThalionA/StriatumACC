"""Equivalence guard for the lagpairs back-port (2026-08-28).

``lagged._segment_lagged_pairs`` used to carry its own inline within-group pairing
loop; it now delegates to :mod:`striatum_tcca.lagpairs`, ported verbatim from
TomLearning ``tom_cca``. ``_ref_segment_lagged_pairs`` below is that old loop, frozen
exactly as it stood at ``lagged.py:145-168`` before the port. The tests assert the two
agree bit-for-bit, so the back-port cannot have changed a single lag-curve number.

Keep the reference frozen — it is the record of what the old code did, not code to
maintain.
"""

from __future__ import annotations

import numpy as np
import pytest

from striatum_tcca import lagged, lagpairs


def _ref_segment_lagged_pairs(Sx, Sy, groups, lag):
    """VERBATIM copy of the pre-2026-08-28 inline pairer (lagged.py:145-168)."""
    Xs, Ys, gs = [], [], []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        xt, yt = Sx[idx], Sy[idx]
        n = xt.shape[0]
        if n <= abs(lag) + 2:
            continue
        if lag >= 0:
            xp, yp = xt[: n - lag], yt[lag:]
        else:
            xp, yp = xt[-lag:], yt[: n + lag]
        Xs.append(xp); Ys.append(yp); gs.append(np.full(xp.shape[0], g))
    if not Xs:
        return None
    return np.vstack(Xs), np.vstack(Ys), np.concatenate(gs)


@pytest.fixture
def flat_scores():
    """Flat PCA-score stand-ins with UNEVEN trial lengths, some shorter than |lag|+2."""
    rng = np.random.default_rng(0)
    lens = [80, 30, 9, 4, 120, 55, 3]
    groups = np.concatenate([np.full(n, i) for i, n in enumerate(lens)])
    n, k = groups.size, 6
    Sx = rng.standard_normal((n, k))
    Sy = rng.standard_normal((n, k))
    return Sx, Sy, groups


@pytest.mark.parametrize("lag", list(range(-12, 13)))
def test_delegating_pairer_matches_the_frozen_inline_reference(flat_scores, lag):
    Sx, Sy, groups = flat_scores
    new = lagged._segment_lagged_pairs(Sx, Sy, groups, lag)
    ref = _ref_segment_lagged_pairs(Sx, Sy, groups, lag)
    if ref is None:
        assert new is None
        return
    assert new is not None
    for a, b in zip(new, ref):
        assert np.array_equal(a, b)


def test_no_pair_crosses_a_group_boundary(flat_scores):
    _, _, groups = flat_scores
    for lag in (-7, -1, 0, 5, 11):
        ix, iy = lagpairs.lag_pair_indices(groups, lag)
        assert np.array_equal(np.asarray(groups)[ix], np.asarray(groups)[iy])
        assert np.array_equal(iy, ix + lag)


def test_short_groups_are_dropped_by_the_min_extra_rule():
    groups = np.repeat([0, 1], [12, 5])          # group 1 has 5 bins
    ix, _ = lagpairs.lag_pair_indices(groups, lag=3)   # needs n > 3 + 2
    assert set(np.unique(groups[ix])) == {0}
    assert lagpairs.lag_local_index(5, 3) is None
    assert lagpairs.lag_local_index(6, 3) is not None


def test_returns_empty_when_no_group_is_long_enough():
    groups = np.repeat([0, 1], [4, 4])
    ix, iy = lagpairs.lag_pair_indices(groups, lag=9)
    assert ix.size == 0 and iy.size == 0
