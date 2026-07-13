"""Ground-truth tests for LFP sanity diagnostics."""

import numpy as np

from striatum_lfp.sanity import (
    boolean_runs,
    common_median_reference,
    distance_to_nearest,
    frame_values_at_samples,
    median_welch_psd,
    robust_synchronous_peak_sample,
    robust_window_metrics,
    robust_zscore,
    signed_distance_to_nearest,
    summarise_fixed_windows,
    temporal_identity_metrics,
)


def test_robust_window_metrics_distinguishes_small_dense_signal_from_zeros():
    rng = np.random.default_rng(1)
    dense_microvolt_scale = rng.normal(0.0, 2e-5, size=(1_000, 4))
    mostly_zero = np.zeros((1_000, 4))
    mostly_zero[::100] = 1.0

    dense = robust_window_metrics(dense_microvolt_scale, fs_hz=1_000)
    sparse = robust_window_metrics(mostly_zero, fs_hz=1_000)

    assert dense.exact_zero_fraction == 0.0
    assert dense.median_channel_rms > 0.0
    assert sparse.exact_zero_fraction == 0.99
    # A large SD alone must not turn sparse impulses into continuous signal.
    assert sparse.nonzero_sample_fraction == 0.01


def test_metrics_are_scale_invariant_except_amplitude_fields():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2_000, 3))
    a = robust_window_metrics(x, fs_hz=1_000)
    b = robust_window_metrics(x * 1e-6, fs_hz=1_000)

    assert a.exact_zero_fraction == b.exact_zero_fraction
    assert np.isclose(a.spectral_flatness, b.spectral_flatness)
    assert np.isclose(a.common_mode_fraction, b.common_mode_fraction)
    assert np.isclose(a.median_channel_rms * 1e-6, b.median_channel_rms)


def test_frame_lookup_uses_latest_observation_not_future_frame():
    frame_ms = np.array([100, 200, 300])
    values = np.array([6, 12, 6])
    samples = np.array([99, 100, 150, 199, 200, 299, 300, 301])

    got = frame_values_at_samples(samples, frame_ms, values)

    assert np.isnan(got[0])
    np.testing.assert_array_equal(got[1:], [6, 6, 6, 12, 12, 6, 6])


def test_robust_zscore_resists_single_extreme_window():
    x = np.array([1.0, 1.1, 0.9, 1.0, 100.0])
    z = robust_zscore(x)
    assert np.max(np.abs(z[:4])) < 2.0
    assert z[-1] > 100.0


def test_fixed_window_summary_has_correct_counts_and_rms():
    x = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, -1.0],
            [1.0, -1.0],
        ]
    )
    summary = summarise_fixed_windows(x, window_samples=2)

    np.testing.assert_allclose(summary.exact_zero_fraction, [1.0, 0.0])
    np.testing.assert_allclose(summary.median_channel_rms, [0.0, 0.0])
    np.testing.assert_allclose(summary.median_channel_abs, [0.0, 1.0])


def test_fixed_window_summary_rejects_partial_window():
    x = np.zeros((5, 2))
    try:
        summarise_fixed_windows(x, window_samples=2)
    except ValueError as exc:
        assert "multiple" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_median_welch_psd_recovers_known_sine_peak():
    fs = 1_000
    time = np.arange(4_000) / fs
    base = np.sin(2 * np.pi * 8 * time)
    windows = np.stack(
        [np.column_stack([base, 0.5 * base]) for _ in range(5)], axis=0
    )

    frequencies, psd = median_welch_psd(windows, fs_hz=fs)

    peak_hz = frequencies[np.argmax(psd)]
    assert abs(peak_hz - 8.0) < 0.6
    assert psd.ndim == 1


def test_boolean_runs_returns_inclusive_bounds():
    runs = boolean_runs(np.array([False, True, True, False, True]))
    np.testing.assert_array_equal(runs, [[1, 2], [4, 4]])


def test_distance_to_nearest_handles_unsorted_reference():
    got = distance_to_nearest(np.array([9.0, 20.0]), np.array([25.0, 10.0, 0.0]))
    np.testing.assert_allclose(got, [1.0, 5.0])


def test_signed_distance_to_nearest_preserves_before_after_direction():
    got = signed_distance_to_nearest(
        np.array([9.0, 20.0, 27.0]), np.array([25.0, 10.0, 0.0])
    )
    np.testing.assert_allclose(got, [-1.0, -5.0, 2.0])


def test_robust_synchronous_peak_sample_finds_shared_transient():
    rng = np.random.default_rng(4)
    voltage = 0.1 * rng.normal(size=(30, 5))
    voltage[17, :] += np.array([4.0, 5.0, 6.0, 5.0, 4.0])
    voltage[8, 0] += 100.0  # one-channel spike must not define a shared event
    assert robust_synchronous_peak_sample(voltage) == 17


def test_common_median_reference_removes_shared_component():
    shared = np.arange(8.0)[:, None]
    local = np.array([[-1.0, 0.0, 1.0]])
    voltage = shared + local
    referenced = common_median_reference(voltage)
    np.testing.assert_allclose(np.median(referenced, axis=1), 0.0)
    np.testing.assert_allclose(referenced, np.repeat(local, 8, axis=0))


def test_temporal_identity_separates_lowpass_signal_from_white_noise():
    from scipy.signal import butter, sosfiltfilt

    rng = np.random.default_rng(3)
    white = rng.normal(size=(8, 4_000, 4))
    sos = butter(4, 80, btype="lowpass", fs=1_000, output="sos")
    lowpass = sosfiltfilt(sos, white, axis=1)

    white_metrics = temporal_identity_metrics(white, fs_hz=1_000)
    lowpass_metrics = temporal_identity_metrics(lowpass, fs_hz=1_000)

    assert abs(white_metrics.median_lag1_correlation) < 0.05
    assert lowpass_metrics.median_lag1_correlation > 0.8
    assert white_metrics.power_fraction_1_100hz < 0.3
    assert lowpass_metrics.power_fraction_1_100hz > 0.8
