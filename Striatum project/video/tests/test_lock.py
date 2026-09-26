import numpy as np

from striatum_video.lock import block_lock


def _signal(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.convolve(rng.normal(size=n), np.ones(5) / 5, mode="same")


def test_locked_streams_give_zero_shift_everywhere():
    x = _signal(12000)
    out = block_lock(x, x + 0.1 * _signal(12000, 1), np.ones(12000, bool), n_blocks=6)
    assert np.all(out["best_shift"] == 0)
    assert np.all(out["r_zero"] > 0.9)


def test_a_dropped_frame_shows_as_a_step_in_the_best_shift():
    # The video lost one frame at index 6000: afterwards video[i] shows vr[i + 1].
    vr = _signal(12000)
    video = np.concatenate([vr[:6000], vr[6001:], [0.0]])
    out = block_lock(video, vr, np.ones(12000, bool), n_blocks=6)
    np.testing.assert_array_equal(out["best_shift"], [0, 0, 0, 1, 1, 1])


def test_invalid_frames_are_excluded():
    vr = _signal(6000)
    video = vr.copy()
    video[::7] = 100.0  # corrupt every 7th frame...
    valid = np.ones(6000, bool)
    valid[::7] = False  # ...and mark it invalid
    out = block_lock(video, vr, valid, n_blocks=3)
    assert np.all(out["r_zero"] > 0.99)
