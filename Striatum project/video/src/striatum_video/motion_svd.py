"""Motion SVD of the face, as in Stringer et al. 2019 (Science) / facemap, in
plain numpy (no PyTorch: facemap needs it only for keypoint tracking).

1. Per frame, |frame - previous frame| inside a face ROI, summed over
   factor x factor pixel blocks (spatial binning), stored exactly as uint16.
2. SVD of the mean-subtracted motion on a random subsample of frames: the
   right singular vectors are spatial motion masks.
3. Every frame's motion is projected on the masks -> k motion components per
   frame, used as covariates like any other movement signal.
"""

import numpy as np


def binned_motion(frames, roi, factor):
    """(n_frames, n_blocks) uint16: per frame, the sum of |diff| over each
    factor x factor block of the ROI (ROI trimmed to a multiple of factor).
    Frame 0 (no predecessor) is all zeros."""
    h, w = (roi.h // factor) * factor, (roi.w // factor) * factor
    rows, prev = [], None
    for frame in frames:
        cur = frame[roi.y:roi.y + h, roi.x:roi.x + w].astype(np.int16)
        if prev is None:
            rows.append(np.zeros((h // factor) * (w // factor), np.uint16))
        else:
            d = np.abs(cur - prev).reshape(h // factor, factor, w // factor, factor).sum((1, 3))
            rows.append(d.ravel().astype(np.uint16))  # <= factor^2 * 255 fits for factor <= 16
        prev = cur
    return np.stack(rows)


def fit_motion_svd(motion, n_components, n_sample, rng):
    """Spatial masks from a random subsample of frames (frame 0 excluded).
    Returns {'components': (n_blocks, k), 'singular_values', 'mean': (n_blocks,)}."""
    idx = rng.choice(np.arange(1, motion.shape[0]), size=min(n_sample, motion.shape[0] - 1), replace=False)
    x = motion[np.sort(idx)].astype(np.float64)
    mean = x.mean(0)
    _, s, vt = np.linalg.svd(x - mean, full_matrices=False)
    return {"components": vt[:n_components].T, "singular_values": s[:n_components], "mean": mean}


def project(motion, svd, chunk=20000):
    """(n_frames, k) motion components: (motion - mean) @ components."""
    out = np.empty((motion.shape[0], svd["components"].shape[1]))
    for i in range(0, motion.shape[0], chunk):
        out[i:i + chunk] = (motion[i:i + chunk].astype(np.float64) - svd["mean"]) @ svd["components"]
    return out
