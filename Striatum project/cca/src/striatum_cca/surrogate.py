"""Surrogate significance for held-out canonical correlations (D7).

Significance is assessed on the **held-out** CC, per canonical dimension. The
earlier in-sample per-dimension test was miscalibrated: genuine signal in the
top dimensions shifts the whole in-sample CC spectrum upward, so a noise
dimension *j* was compared against the shuffle's dimension *j* (a lower-index,
larger value) and spuriously passed -- inflating the apparent subspace
dimensionality. The held-out CC of a noise dimension is ~0 regardless of its
index, so the held-out test is properly calibrated.

Null (cfg.null_type): "trials" permutes the trial correspondence between the
two areas (H&H -- tests trial-to-trial communication); "circshift" circularly
shifts each trial's bin axis (tests within-trial co-tuning, preserving spatial
autocorrelation). Either way the cross-validated CCA is recomputed and the real
held-out CC of each dimension compared to the shuffle distribution. The number
of dimensions passing is the communication-subspace dimensionality -- the
statistical n.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import core


def permute_trials(tensor: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle the trial axis (axis 0) -- breaks cross-area trial pairing (H&H)."""
    return tensor[rng.permutation(tensor.shape[0])]


def circshift_bins(tensor: np.ndarray, rng: np.random.Generator,
                   min_shift: int) -> np.ndarray:
    """Circularly shift each trial's bin axis by a random offset of at least
    ``min_shift`` bins.

    Breaks the within-trial bin alignment between the two areas while
    preserving each area's spatial autocorrelation -- a small shift would
    leave adjacent bins almost aligned, hence the minimum.
    """
    n_tr, n_bin, _ = tensor.shape
    out = np.empty_like(tensor)
    hi = n_bin - min_shift
    for t in range(n_tr):
        s = int(rng.integers(min_shift, hi + 1)) if hi > min_shift else 0
        out[t] = np.roll(tensor[t], s, axis=0)
    return out


def p_value(real: float, null: np.ndarray) -> float:
    """One-sided non-parametric p-value P(null >= real), +1/+1 corrected."""
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if null.size == 0:
        return np.nan
    return float((1 + np.sum(null >= real)) / (1 + null.size))


@dataclass
class NullResult:
    """Held-out-CC surrogate null and per-dimension significance for one epoch."""

    null_held_out: np.ndarray    # (n_shuffles, d) shuffle held-out CC per dim
    p_per_dim: np.ndarray        # (d,) one-sided p, real vs shuffle held-out CC
    n_significant: int           # significant communication-subspace dimensions


def build_null(
    scores_x: np.ndarray,
    scores_y: np.ndarray,
    real_held_out_cc: np.ndarray,
    cfg,
    alpha: float = 0.05,
) -> NullResult:
    """Per-dimension held-out-CC significance for one (animal, pair, epoch).

    Each surrogate permutes the trial correspondence and recomputes the
    5-fold cross-validated CCA; the real held-out CC of dimension *j* is
    compared to the shuffle distribution of held-out CC for dimension *j*.
    """
    rng = np.random.default_rng(cfg.surrogate_seed)
    real = np.atleast_1d(np.asarray(real_held_out_cc, dtype=float))
    d = real.shape[0]

    null = np.full((cfg.n_shuffles, d), np.nan)
    for s in range(cfg.n_shuffles):
        if cfg.null_type == "circshift":
            shuffled_y = circshift_bins(scores_y, rng, cfg.circshift_min_bins)
        else:
            shuffled_y = permute_trials(scores_y, rng)
        held_out = core.cca_cv(scores_x, shuffled_y, cfg).held_out_r
        m = min(d, held_out.shape[0])
        null[s, :m] = held_out[:m]

    p_per_dim = np.array([p_value(real[j], null[:, j]) for j in range(d)])
    return NullResult(
        null_held_out=null,
        p_per_dim=p_per_dim,
        n_significant=int(np.sum(p_per_dim < alpha)),
    )
