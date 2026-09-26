"""frame_features on synthetic frames whose motion is known."""

import cv2
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from striatum_video.motion import Roi, frame_features

WHOLE = Roi(0, 0, 10, 10)


def _me(frames, rois, wheel=None):
    h, w = frames[0].shape
    return frame_features(iter(frames), rois, wheel or Roi(0, 0, w, h))[0]


def _frames_with_moving_square(n_frames, moving_frames, shape=(60, 80)):
    """A bright 4x4 square that shifts one pixel right on each frame in
    `moving_frames` and stays still otherwise. Lives in the top-left quadrant."""
    frames = np.zeros((n_frames, *shape), dtype=np.uint8)
    x = 5
    for i in range(n_frames):
        if i in moving_frames:
            x += 1
        frames[i, 10:14, x:x + 4] = 200
    return frames


def _scrolling_texture(shifts_px, shape=(120, 60), seed=0):
    """Frames cut from a tall smoothed random texture that scrolls DOWN by
    shifts_px[i] pixels between frame i-1 and frame i."""
    rng = np.random.default_rng(seed)
    total = int(np.ceil(np.sum(np.abs(shifts_px)))) + shape[0] + 10
    tex = gaussian_filter(rng.random((total, shape[1])), 2)
    tex = (255 * (tex - tex.min()) / np.ptp(tex)).astype(np.uint8)
    pos = total - shape[0] - 5.0
    frames = []
    for s in [0.0, *shifts_px]:
        pos -= s
        i = round(pos)
        frames.append(tex[i:i + shape[0]])
    return np.stack(frames)


def test_motion_energy_is_zero_for_static_frames():
    frames = np.full((10, 30, 40), 77, dtype=np.uint8)
    me = _me(frames, {"all": Roi(0, 0, 40, 30)})
    assert np.isnan(me["all"][0])
    np.testing.assert_array_equal(me["all"][1:], 0.0)


def test_motion_energy_marks_exactly_the_moving_frames():
    moving = {3, 4, 8}
    me = _me(_frames_with_moving_square(12, moving), {"square": Roi(0, 0, 40, 30)})["square"]
    assert np.flatnonzero(me[1:] > 0).tolist() == [i - 1 for i in sorted(moving)]


def test_motion_outside_an_roi_does_not_leak_into_it():
    me = _me(_frames_with_moving_square(12, {3, 4, 8}), {"elsewhere": Roi(40, 30, 40, 30)})
    np.testing.assert_array_equal(me["elsewhere"][1:], 0.0)


def test_motion_energy_value_is_mean_absolute_difference():
    a = np.zeros((1, 10, 10), dtype=np.uint8)
    b = np.full((1, 10, 10), 10, dtype=np.uint8)
    b[0, :5] = 0  # half the pixels change by 10 -> mean |diff| = 5
    assert _me(np.concatenate([a, b]), {"r": WHOLE})["r"][1] == pytest.approx(5.0)


def test_uint8_frames_do_not_wrap_around():
    a = np.full((1, 10, 10), 250, dtype=np.uint8)
    b = np.full((1, 10, 10), 5, dtype=np.uint8)
    assert _me(np.concatenate([a, b]), {"r": WHOLE})["r"][1] == pytest.approx(245.0)


def test_roi_outside_the_frame_is_rejected():
    frames = np.zeros((3, 10, 10), dtype=np.uint8)
    with pytest.raises(ValueError):
        frame_features(iter(frames), {"r": Roi(5, 5, 10, 10)}, WHOLE)
    with pytest.raises(ValueError):
        frame_features(iter(frames), {}, Roi(5, 5, 10, 10))


def test_wheel_shift_recovers_known_signed_shifts():
    shifts = [0, 2, 5, 9, 0, 3, -4, 12]
    _, dy, dx, resp = frame_features(iter(_scrolling_texture(shifts)), {}, Roi(0, 0, 60, 120))
    assert np.isnan(dy[0])
    np.testing.assert_allclose(dy[1:], shifts, atol=0.5)
    np.testing.assert_allclose(dx[1:], 0, atol=0.5)
    assert np.all(resp[1:] > 0.3)


def test_wheel_shift_equals_an_isolated_pairwise_estimate():
    """Regression: OpenCV's phaseCorrelate windows its inputs IN PLACE, so a
    loop that reuses the previous frame's array double-windows it. Every
    per-frame shift must equal a fresh, isolated call on that frame pair."""
    frames = _scrolling_texture([2, 5, -3, 7])
    roi = Roi(0, 0, 60, 120)
    _, dy, dx, _ = frame_features(iter(frames), {}, roi)
    window = cv2.createHanningWindow((roi.w, roi.h), cv2.CV_64F)
    for i in range(1, len(frames)):
        (sx, sy), _ = cv2.phaseCorrelate(frames[i - 1].astype(np.float64),
                                         frames[i].astype(np.float64), window.copy())
        assert dy[i] == sy
        assert dx[i] == sx


def test_diff_maps_accumulate_per_label_and_skip_label_zero():
    from striatum_video.motion import diff_maps_by_label
    frames = np.zeros((5, 4, 4), dtype=np.uint8)
    frames[2, 0, 0] = 10   # frame 2 differs from 1 at (0,0) by 10; frame 3 differs from 2 by 10
    labels = np.array([1, 1, 2, 1, 0])  # frame 0 has no predecessor and is never counted
    sums, counts = diff_maps_by_label(iter(frames), labels, n_labels=2)
    np.testing.assert_array_equal(counts, [2, 1])       # label 1: frames 1, 3; label 2: frame 2
    assert sums[0][0, 0] == 10 and sums[1][0, 0] == 10  # frame 3 -> label 1, frame 2 -> label 2
    assert sums[0].sum() == 10 and sums[1].sum() == 10
