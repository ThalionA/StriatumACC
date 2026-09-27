"""Motion SVD (Stringer et al. 2019 / facemap) on synthetic frames with two
independently moving patches: the top components must recover their time courses
and localise on their pixels."""

import numpy as np

from striatum_video.motion import Roi
from striatum_video.motion_svd import binned_motion, fit_motion_svd, project


def _two_patch_frames(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    frames = np.full((n, 40, 60), 50, np.uint8)
    # patch A (top-left) flickers with amplitude a_t; patch B (bottom-right) with b_t
    a = np.abs(np.convolve(rng.normal(size=n), np.ones(20) / 20, mode="same")) * 400
    b = np.abs(np.convolve(rng.normal(size=n), np.ones(50) / 50, mode="same")) * 600
    tex_a = rng.integers(0, 2, (10, 10))
    tex_b = rng.integers(0, 2, (10, 10))
    for t in range(n):
        sign = 1 if t % 2 else -1  # alternate so |frame diff| scales with amplitude
        frames[t, 5:15, 5:15] = np.clip(50 + sign * tex_a * a[t] / 2, 0, 255)
        frames[t, 25:35, 40:50] = np.clip(50 + sign * tex_b * b[t] / 2, 0, 255)
    return frames, a, b


def test_binned_motion_is_the_blockwise_sum_of_absolute_differences():
    f = np.zeros((2, 4, 4), np.uint8)
    f[1, :2, :2] = 3  # top-left 2x2 block changes by 3 in each of 4 pixels
    out = binned_motion(iter(f), Roi(0, 0, 4, 4), factor=2)
    assert out.dtype == np.uint16 and out.shape == (2, 4)
    np.testing.assert_array_equal(out[0], 0)          # frame 0: no predecessor -> 0
    np.testing.assert_array_equal(out[1], [12, 0, 0, 0])


def test_top_components_recover_independent_patch_time_courses():
    frames, a, b = _two_patch_frames()
    motion = binned_motion(iter(frames), Roi(0, 0, 60, 40), factor=5)
    svd = fit_motion_svd(motion, n_components=4, n_sample=2000, rng=np.random.default_rng(1))
    pcs = project(motion, svd)
    r = np.abs(np.corrcoef(np.column_stack([pcs[1:, :2], a[1:], b[1:]]).T)[:2, 2:])
    # each true course is captured by one of the top two components
    assert r.max(axis=0).min() > 0.9
    # and each component's spatial mask sits on one patch
    masks = svd["components"][:, :2].reshape(8, 12, 2)  # 40/5 x 60/5 blocks
    for k in range(2):
        m = np.abs(masks[..., k])
        top = np.unravel_index(np.argmax(m), m.shape)
        assert top in {(1, 1), (1, 2), (2, 1), (2, 2), (5, 8), (5, 9), (6, 8), (6, 9)}


def test_projection_of_the_mean_motion_is_zero():
    frames, _, _ = _two_patch_frames(500)
    motion = binned_motion(iter(frames), Roi(0, 0, 60, 40), factor=5)
    svd = fit_motion_svd(motion, n_components=3, n_sample=400, rng=np.random.default_rng(2))
    mean_row = np.round(svd["mean"]).astype(np.uint16)[None, :]
    assert np.all(np.abs(project(mean_row, svd)) < 0.5 * np.sqrt(svd["mean"].size))
