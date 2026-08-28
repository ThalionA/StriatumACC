"""Within-group lag pairing — THE one place that pairs ``X[t]`` with ``Y[t + lag]``.

Sign convention (project-wide): **positive lag ⇒ X (the first-named area) leads Y** —
``X`` at bin ``t`` is paired with ``Y`` at bin ``t + lag``.

The running-state analyses concatenate the engaged bins of many trials into one flat
``(n_bins, k)`` matrix. Pairing must never straddle a trial (group) boundary, otherwise
the concatenation manufactures correlations at long lags. The lag-curve
functions (``lagged.heldout_lag_curve_flat_perdim`` and its CC1 slice) go through
:func:`lag_pair_indices`.

Ported verbatim from TomLearning ``tom_cca/lagpairs.py`` (2026-08-28) so the two projects
share one pairer; there it also serves ``lag_subspace`` and ``fixed_subspace``. It replaces
the inline copy that lived in ``lagged._segment_lagged_pairs`` — the two are bit-identical
(``tests/test_lagpairs.py``).

Guard: a group of ``n`` bins contributes only if ``n > |lag| + min_extra`` (default
``min_extra = 2``, the rule every live caller used).
"""

from __future__ import annotations

import numpy as np

DEFAULT_MIN_EXTRA = 2


def lag_local_index(n: int, lag: int, min_extra: int = DEFAULT_MIN_EXTRA):
    """Local pair indices for ONE contiguous group of ``n`` bins.

    Returns ``(ix, iy)`` with ``iy = ix + lag`` (both ascending), or ``None`` when the
    group is too short (``n <= |lag| + min_extra``).
    """
    lag = int(lag)
    if n <= abs(lag) + min_extra:
        return None
    if lag >= 0:
        return np.arange(0, n - lag), np.arange(lag, n)
    return np.arange(-lag, n), np.arange(0, n + lag)


def lag_pair_indices(groups, lag: int, min_extra: int = DEFAULT_MIN_EXTRA):
    """Global row indices pairing bin ``t`` with ``t + lag`` inside each group.

    ``groups`` is the per-bin group (trial) id, any dtype; the bins of a group are
    assumed contiguous and in time order (true for every caller: running bins are
    emitted trial by trial). Groups are visited in ``np.unique`` order and rows within a
    group ascend, so ``X[ix]`` reproduces the historical ``np.vstack`` order exactly.

    Returns ``(ix, iy)`` integer arrays (empty when no group is long enough).
    ``groups[ix] == groups[iy]`` always.
    """
    groups = np.asarray(groups)
    ixs, iys = [], []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        loc = lag_local_index(idx.size, lag, min_extra)
        if loc is None:
            continue
        ixs.append(idx[loc[0]])
        iys.append(idx[loc[1]])
    if not ixs:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    return np.concatenate(ixs), np.concatenate(iys)


def pair_within_groups(lag: int, groups, *arrays, min_extra: int = DEFAULT_MIN_EXTRA):
    """Convenience: ``(groups_paired, A_x, A_y, B_x, B_y, ...)`` for each array given.

    For every array ``A`` in ``arrays`` returns the leading-time rows ``A[ix]`` and the
    lagged rows ``A[iy]``; ``None`` arrays pass through as ``(None, None)``. Returns
    ``None`` if no group is long enough — the historical contract of the callers.
    """
    ix, iy = lag_pair_indices(groups, lag, min_extra)
    if ix.size == 0:
        return None
    out = [np.asarray(groups)[ix]]
    for A in arrays:
        if A is None:
            out.extend((None, None))
        else:
            out.extend((A[ix], A[iy]))
    return tuple(out)
