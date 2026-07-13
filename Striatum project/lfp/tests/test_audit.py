"""Streaming audit tests on a tiny synthetic HDF5 export."""

import h5py
import numpy as np

from striatum_lfp.audit import audit_voltage_file


def test_audit_voltage_file_counts_zeros_and_windows_exactly(tmp_path):
    path = tmp_path / "tiny.mat"
    data = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1e-6, -1e-6],
            [2e-6, -2e-6],
            [0.0, 0.0],
            [3e-6, -3e-6],
            [4e-6, -4e-6],
            [5e-6, -5e-6],
        ],
        dtype=np.float32,
    )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data_to_save", data=data, chunks=(2, 2))

    result = audit_voltage_file(
        path, fs_hz=2, window_seconds=1, block_windows=2
    )

    assert result.n_samples == 8
    assert result.n_channels == 2
    assert result.exact_zero_count == 6
    assert result.total_value_count == 16
    assert result.finite_count == 16
    np.testing.assert_allclose(result.window_exact_zero_fraction, [1.0, 0.0, 0.5, 0.0])
    np.testing.assert_allclose(result.channel_exact_zero_fraction, [3 / 8, 3 / 8])
    assert np.all(result.window_median_channel_abs >= 0)
