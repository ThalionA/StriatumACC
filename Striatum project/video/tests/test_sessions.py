from datetime import datetime

import numpy as np
import pytest

from striatum_video.sessions import frame_times, parse_part_start


def test_parse_base_and_continuation_names():
    assert parse_part_start("Video_20241201181115289.avi") == datetime(2024, 12, 1, 18, 11, 15, 289000)
    assert parse_part_start("Video_20241201181115289_20241201_193759_0.avi") == datetime(2024, 12, 1, 19, 37, 59)
    with pytest.raises(ValueError):
        parse_part_start("Video_2024.avi")


def test_frame_times_are_the_vr_times_when_counts_match():
    vr = {"time": np.array([0.03, 0.06, 0.1])}
    np.testing.assert_array_equal(frame_times(vr, 3), vr["time"])


def test_frame_times_refuses_a_count_mismatch():
    with pytest.raises(ValueError, match="frame lock"):
        frame_times({"time": np.arange(5.0)}, 6)
