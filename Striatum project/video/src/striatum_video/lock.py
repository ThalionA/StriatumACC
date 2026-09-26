"""Evidence that video frame i is VR row i, block by block through a session."""

from itertools import pairwise

import numpy as np


def block_lock(video, vr, valid, n_blocks=12, max_shift=8):
    """Split the session into n_blocks; in each, correlate video[i] with
    vr[i + k] for k in [-max_shift, max_shift] over the frames where `valid`
    holds (for both indices). Returns {'best_shift', 'r_best', 'r_zero'}
    arrays, one per block. A locked session has best_shift ~ 0 in every block;
    a dropped or duplicated frame shows as a step in best_shift."""
    video, vr, valid = np.asarray(video, float), np.asarray(vr, float), np.asarray(valid, bool)
    n = video.size
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    best, r_best, r_zero = [], [], []
    for i0, i1 in pairwise(edges):
        lo, hi = i0 + max_shift, i1 - max_shift
        idx = np.arange(lo, hi)
        rs = {}
        for k in range(-max_shift, max_shift + 1):
            m = valid[idx] & valid[idx + k]
            rs[k] = np.corrcoef(video[idx][m], vr[idx + k][m])[0, 1] if m.sum() > 10 else np.nan
        k_best = max(rs, key=lambda k: -np.inf if np.isnan(rs[k]) else rs[k])
        best.append(k_best)
        r_best.append(rs[k_best])
        r_zero.append(rs[0])
    return {"best_shift": np.array(best), "r_best": np.array(r_best), "r_zero": np.array(r_zero)}
