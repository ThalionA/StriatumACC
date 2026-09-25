"""Ground-truth tests for LFP cohort discovery and the file-identity statistic."""

from __future__ import annotations

import numpy as np
import pytest

from striatum_lfp import cohort


# --- Filename -> (mouse, probe) ---------------------------------------------

@pytest.mark.parametrize(
    "name,expected",
    [
        ("1105_voltage_data_384ch.mat", (1105, "striatum")),
        ("523_voltage_data_384ch.mat", (523, "striatum")),
        ("727voltage_data_384ch.mat", (727, "striatum")),      # missing underscore
        ("1105_v1_voltage_data_384ch.mat", (1105, "visual")),
        ("1212_V1_voltage_data_384ch.mat", (1212, "visual")),  # case-insensitive
        ("voltage_data_384ch.mat", None),                      # unnamed / in-flight
        ("voltage_data_384ch 2.mat", None),                    # superseded June copy
        ("lfp_mapping.txt", None),
        (".DS_Store", None),
        ("1105_raw.mat", None),
    ],
)
def test_parse_lfp_filename(name, expected):
    assert cohort.parse_lfp_filename(name) == expected


def test_parse_rejects_unknown_mouse_id():
    """A numeric prefix that is not a task mouse is not silently accepted."""
    assert cohort.parse_lfp_filename("9999_voltage_data_384ch.mat") is None


def test_discover_lfp_files(tmp_path):
    for name in [
        "1105_voltage_data_384ch.mat",
        "1105_v1_voltage_data_384ch.mat",
        "727voltage_data_384ch.mat",
        "voltage_data_384ch.mat",
        "lfp_mapping.txt",
    ]:
        (tmp_path / name).write_bytes(b"")
    found = cohort.discover_lfp_files(tmp_path)
    assert set(found) == {(1105, "striatum"), (1105, "visual"), (727, "striatum")}
    assert found[(1105, "visual")].name == "1105_v1_voltage_data_384ch.mat"


def test_discover_reports_unmatched(tmp_path):
    (tmp_path / "voltage_data_384ch.mat").write_bytes(b"")
    (tmp_path / "823_voltage_data_384ch.mat").write_bytes(b"")
    found, skipped = cohort.discover_lfp_files(tmp_path, return_skipped=True)
    assert set(found) == {(823, "striatum")}
    assert "voltage_data_384ch.mat" in skipped


# --- Binning ----------------------------------------------------------------

def test_bin_mean_exact_on_known_ramp():
    x = np.arange(12, dtype=float).reshape(12, 1)
    out = cohort.bin_mean(x, 4)
    assert out.shape == (3, 1)
    np.testing.assert_allclose(out[:, 0], [1.5, 5.5, 9.5])


def test_bin_mean_drops_incomplete_tail():
    x = np.ones((10, 2))
    assert cohort.bin_mean(x, 4).shape == (2, 2)


def test_bin_mean_rejects_nonpositive_width():
    with pytest.raises(ValueError):
        cohort.bin_mean(np.ones((4, 1)), 0)


# --- Per-column Pearson r ----------------------------------------------------

def test_pearson_columns_matches_numpy_corrcoef():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 7))
    y = rng.normal(size=500)
    got = cohort.pearson_columns(X, y)
    want = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(7)])
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_pearson_columns_known_ground_truth():
    y = np.linspace(0, 1, 200)
    X = np.column_stack([y, -y, np.ones_like(y)])
    got = cohort.pearson_columns(X, y)
    np.testing.assert_allclose(got[:2], [1.0, -1.0], atol=1e-12)
    assert np.isnan(got[2])          # zero-variance column -> undefined, not 0


def test_pearson_columns_length_mismatch():
    with pytest.raises(ValueError):
        cohort.pearson_columns(np.ones((10, 2)), np.ones(9))


# --- Coupling score ----------------------------------------------------------

def test_coupling_score_separates_matched_from_mismatched():
    """Synthetic ground truth: a shared latent drives env and its OWN mua only."""
    rng = np.random.default_rng(1)
    n, n_ch = 4000, 32
    latent = rng.normal(size=n)
    env = latent[:, None] * 0.6 + rng.normal(size=(n, n_ch))
    mua_matched = latent + rng.normal(size=n) * 0.5
    mua_other = rng.normal(size=n)
    matched = cohort.coupling_score(env, mua_matched)
    mismatched = cohort.coupling_score(env, mua_other)
    assert matched > 5 * mismatched
    assert matched > 0.2


def test_coupling_score_ignores_dead_channels():
    rng = np.random.default_rng(2)
    n = 2000
    latent = rng.normal(size=n)
    env = np.column_stack([latent + rng.normal(size=n) * 0.1, np.zeros(n)])
    score = cohort.coupling_score(env, latent)
    assert np.isfinite(score) and score > 0.8


# --- Column normalisation ----------------------------------------------------

def test_column_normalise_neutralises_a_universal_correlator():
    """A candidate that scores high against every file must not win a row.

    File 0's true match is candidate 0 (0.30). Candidate 2 is an artefactual
    "universal correlator" scoring 0.50 against everything, so it wins the raw
    argmax and loses it after normalisation.
    """
    raw = np.array([
        [0.30, 0.02, 0.50],
        [0.02, 0.30, 0.50],
        [0.02, 0.02, 0.52],
    ])
    assert np.argmax(raw[0]) == 2                      # raw statistic is fooled
    norm = cohort.column_normalise(raw)
    assert np.argmax(norm[0]) == 0
    assert np.argmax(norm[1]) == 1


def test_column_normalise_leaves_a_clean_matrix_diagonal():
    raw = np.array([[0.30, 0.02], [0.02, 0.30]])
    norm = cohort.column_normalise(raw)
    assert np.argmax(norm[0]) == 0 and np.argmax(norm[1]) == 1


def test_column_normalise_handles_an_all_zero_column():
    raw = np.array([[0.3, 0.0], [0.1, 0.0]])
    norm = cohort.column_normalise(raw)
    assert np.isnan(norm[:, 1]).all()
    assert np.isfinite(norm[:, 0]).all()
