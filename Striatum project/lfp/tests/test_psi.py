"""Ground-truth tests for the phase-slope index.

The 2026-09-09 meeting asked whether a direction of communication can be read
from the LFP. The obstacle is the thing the distance control established on
2026-09-17: cross-area coupling on this probe is a function of separation and
nothing else, i.e. one shared field seen by many electrodes. A shared field
produces INSTANTANEOUS mixing, and every amplitude- or lag-based measure reports
a confident spurious answer on instantaneously mixed signals.

The phase-slope index (Nolte et al. 2008) is the measure chosen because it is
blind to instantaneous mixing by construction: mixing real sources with real
coefficients makes the coherency real, and PSI is the imaginary part of a product
of coherencies, so it vanishes. These tests pin exactly that, plus the sign
convention and the case that matters -- a genuine lag hidden underneath heavy
mixing.

Created 2026-09-17.
"""
from __future__ import annotations

import numpy as np
import pytest

from striatum_lfp import psi

FS = 1000.0
BAND = (8.0, 40.0)          # wide enough to hold many frequency bins
NPERSEG = 1024
N = 400_000                  # ~400 s, so the jackknife has plenty of segments


def _pink(n, rng, *, alpha=1.0):
    """A 1/f^alpha noise source, so the tests run on LFP-like spectra."""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1 / FS)
    scale = np.ones_like(f)
    scale[1:] = 1.0 / f[1:] ** (alpha / 2)
    return np.fft.irfft(spec * scale, n=n)


def test_independent_signals_give_no_direction():
    rng = np.random.default_rng(0)
    x, y = _pink(N, rng), _pink(N, rng)
    out = psi.phase_slope_index(x, y, fs=FS, band=BAND, nperseg=NPERSEG)
    assert abs(out["z"]) < 3.0, f"independent signals should give z~0, got {out['z']:.2f}"


def test_instantaneous_mixing_gives_no_direction():
    """The volume-conduction case: two real sources, real mixing, no delay.

    This is the decisive property. An amplitude correlation on these two sensors
    is large -- they share both sources -- but there is no lag anywhere, so a
    direction-of-communication measure must report nothing.
    """
    rng = np.random.default_rng(1)
    u, v = _pink(N, rng), _pink(N, rng)
    s1 = 0.8 * u + 0.6 * v
    s2 = 0.5 * u + 0.9 * v
    assert abs(np.corrcoef(s1, s2)[0, 1]) > 0.5, "the sensors should be strongly mixed"
    out = psi.phase_slope_index(s1, s2, fs=FS, band=BAND, nperseg=NPERSEG)
    assert abs(out["z"]) < 3.0, (
        f"instantaneous mixing must not look directed, got z = {out['z']:.2f}")


def test_a_leading_signal_gives_a_positive_index():
    """Sign convention, pinned: psi(x, y) > 0 means x LEADS y."""
    rng = np.random.default_rng(2)
    lag = 8                                        # samples, i.e. 8 ms
    src = _pink(N + lag, rng)
    x = src[lag:]                                  # x is the earlier copy
    y = src[:-lag] + 0.3 * _pink(N, rng)           # y repeats it `lag` later
    out = psi.phase_slope_index(x, y, fs=FS, band=BAND, nperseg=NPERSEG)
    assert out["z"] > 5.0, f"a clear lag should be strongly detected, got z = {out['z']:.2f}"
    assert out["psi"] > 0, "x leads y, so the index must be positive"


def test_the_index_flips_with_the_direction():
    rng = np.random.default_rng(3)
    lag = 8
    src = _pink(N + lag, rng)
    x, y = src[lag:], src[:-lag] + 0.3 * _pink(N, rng)
    fwd = psi.phase_slope_index(x, y, fs=FS, band=BAND, nperseg=NPERSEG)
    rev = psi.phase_slope_index(y, x, fs=FS, band=BAND, nperseg=NPERSEG)
    assert np.sign(fwd["psi"]) == -np.sign(rev["psi"])
    assert fwd["psi"] == pytest.approx(-rev["psi"], rel=1e-6)


def test_a_real_lag_survives_heavy_instantaneous_mixing():
    """The case this analysis actually faces.

    A genuine delayed interaction between two areas, plus a large shared field
    that reaches both electrodes with no delay. The shared field dominates the
    amplitude correlation; PSI must still recover the lag and its direction.
    """
    rng = np.random.default_rng(4)
    lag = 8
    src = _pink(N + lag, rng)
    a = src[lag:]                                   # area A
    b = src[:-lag] + 0.4 * _pink(N, rng)            # area B, driven by A at 8 ms
    field = _pink(N, rng)                           # the shared, zero-lag field
    s1 = a + 2.0 * field
    s2 = b + 2.0 * field
    out = psi.phase_slope_index(s1, s2, fs=FS, band=BAND, nperseg=NPERSEG)
    assert out["z"] > 3.0, (
        f"a real lag under heavy mixing must survive, got z = {out['z']:.2f}")
    assert out["psi"] > 0, "A drives B, so the index must stay positive"


def test_band_selection_changes_nothing_when_the_lag_is_broadband():
    rng = np.random.default_rng(5)
    lag = 8
    src = _pink(N + lag, rng)
    x, y = src[lag:], src[:-lag] + 0.3 * _pink(N, rng)
    lo = psi.phase_slope_index(x, y, fs=FS, band=(8.0, 20.0), nperseg=NPERSEG)
    hi = psi.phase_slope_index(x, y, fs=FS, band=(20.0, 40.0), nperseg=NPERSEG)
    assert lo["psi"] > 0 and hi["psi"] > 0, "a broadband lag should show in both bands"


def test_too_few_frequency_bins_is_refused_not_guessed():
    """A band holding fewer than three bins cannot support a slope."""
    rng = np.random.default_rng(6)
    x, y = _pink(20_000, rng), _pink(20_000, rng)
    with pytest.raises(ValueError, match="frequency bins"):
        psi.phase_slope_index(x, y, fs=FS, band=(4.0, 5.0), nperseg=256)


def test_too_few_segments_is_refused_not_guessed():
    """The z-score comes from a jackknife over segments; it needs segments."""
    rng = np.random.default_rng(7)
    x, y = _pink(2_048, rng), _pink(2_048, rng)
    with pytest.raises(ValueError, match="segments"):
        psi.phase_slope_index(x, y, fs=FS, band=BAND, nperseg=1024, min_segments=8)


def test_windows_never_straddle_a_snippet_boundary():
    """Segmenting per snippet must yield fewer windows than segmenting the join.

    The difference is exactly the windows that would have spanned two trials, whose
    two halves come from different moments of behaviour and whose phase relationship
    is therefore arbitrary. An earlier version concatenated first and segmented
    after, which silently included them.
    """
    rng = np.random.default_rng(8)
    snips = [_pink(5_000, rng) for _ in range(6)]
    per_snippet = psi.count_segments(snips, NPERSEG)
    joined = psi.count_segments([np.concatenate(snips)], NPERSEG)
    assert per_snippet < joined, "per-snippet segmentation must drop the straddling windows"

    freqs, Sxy, _, _ = psi.segment_spectra(snips, snips, fs=FS, nperseg=NPERSEG)
    assert Sxy.shape[0] == per_snippet


def test_a_lag_is_recovered_from_per_trial_snippets():
    """The driver's real calling pattern: many short trials rather than one long run."""
    rng = np.random.default_rng(9)
    lag = 8
    xs, ys = [], []
    for _ in range(40):
        src = _pink(10_000 + lag, rng)
        xs.append(src[lag:])
        ys.append(src[:-lag] + 0.3 * _pink(10_000, rng))
    out = psi.phase_slope_index(xs, ys, fs=FS, band=BAND, nperseg=NPERSEG)
    assert out["z"] > 5.0, f"the lag should survive trial-wise segmentation, z = {out['z']:.2f}"
    assert out["psi"] > 0


def test_short_snippets_are_dropped_not_padded():
    rng = np.random.default_rng(10)
    good = [_pink(6_000, rng) for _ in range(10)]
    withshort = good + [_pink(100, rng)]
    a = psi.phase_slope_index(good, good, fs=FS, band=BAND, nperseg=NPERSEG)
    b = psi.phase_slope_index(withshort, withshort, fs=FS, band=BAND, nperseg=NPERSEG)
    assert a["n_segments"] == b["n_segments"], "a too-short snippet must contribute nothing"


def test_bipolar_uses_non_overlapping_pairs_not_a_telescoping_mean():
    """`mean(diff(x))` collapses to one wide pair; the derivation must not do that."""
    rng = np.random.default_rng(11)
    x = rng.standard_normal((2_000, 8))
    got = psi.bipolar_derivation(x)
    telescoped = np.mean(np.diff(x, axis=1), axis=1)
    wide = (x[:, -1] - x[:, 0]) / (x.shape[1] - 1)
    assert np.allclose(telescoped, wide), "the trap: mean(diff) IS the wide pair"
    assert not np.allclose(got, telescoped), "the derivation must not telescope"
    expected = np.mean(x[:, 0::2] - x[:, 1::2], axis=1)
    assert np.allclose(got, expected)


def test_bipolar_cancels_a_common_field():
    """A signal present identically on every channel must vanish."""
    rng = np.random.default_rng(12)
    # Unit variance, so the field-to-local-noise ratio below is the one intended;
    # _pink returns an arbitrary scale and an unnormalised field made the mean
    # correlate only 0.81 with it, which tested the fixture rather than the code.
    field = _pink(4_000, rng)
    field = field / field.std()
    x = np.tile(field[:, None], (1, 6))
    assert np.allclose(psi.bipolar_derivation(x), 0.0, atol=1e-9)

    # With local signal added, bipolar must suppress the field far more than the mean.
    local = rng.standard_normal((4_000, 6)) * 0.2
    y = x + local
    mono = y.mean(axis=1)
    bip = psi.bipolar_derivation(y)
    assert abs(np.corrcoef(mono, field)[0, 1]) > 0.9, "the mean should still see the field"
    assert abs(np.corrcoef(bip, field)[0, 1]) < 0.2, "bipolar should not"


def test_bipolar_drops_an_unpaired_last_channel():
    rng = np.random.default_rng(13)
    x = rng.standard_normal((500, 7))
    assert np.allclose(psi.bipolar_derivation(x), psi.bipolar_derivation(x[:, :6]))


def test_shuffled_null_is_centred_on_zero_for_a_real_lag():
    """Mismatching trials must destroy the lag it was built to find."""
    rng = np.random.default_rng(14)
    lag = 8
    xs, ys = [], []
    for _ in range(30):
        src = _pink(8_000 + lag, rng)
        xs.append(src[lag:])
        ys.append(src[:-lag] + 0.3 * _pink(8_000, rng))
    obs = psi.phase_slope_index(xs, ys, fs=FS, band=BAND, nperseg=NPERSEG)
    null = psi.trial_shuffled_null(xs, ys, fs=FS, band=BAND, nperseg=NPERSEG,
                                   n_shuffles=12, seed=1)
    assert null["n_shuffles"] >= 10
    z_null = (obs["psi"] - null["null_mean"]) / null["null_sd"]
    assert z_null > 3.0, f"a real lag must beat its own shuffled null, got {z_null:.2f}"
    assert abs(null["null_mean"]) < abs(obs["psi"]), "the null must sit nearer zero"


def test_shuffled_null_matches_the_observed_value_when_there_is_no_interaction():
    """Independent areas: the observed index is just another draw from the null."""
    rng = np.random.default_rng(15)
    xs = [_pink(8_000, rng) for _ in range(30)]
    ys = [_pink(8_000, rng) for _ in range(30)]
    obs = psi.phase_slope_index(xs, ys, fs=FS, band=BAND, nperseg=NPERSEG)
    null = psi.trial_shuffled_null(xs, ys, fs=FS, band=BAND, nperseg=NPERSEG,
                                   n_shuffles=12, seed=2)
    z_null = (obs["psi"] - null["null_mean"]) / null["null_sd"]
    assert abs(z_null) < 3.0, f"no interaction should not beat the null, got {z_null:.2f}"


def test_the_jackknife_is_the_calibrated_statistic_not_the_shuffle():
    """Which variance estimate is right, measured rather than assumed.

    The jackknife divides by a spread estimated over segments that overlap by
    half, which is a good reason to suspect it. The mismatched-trial surrogate
    looks like the principled alternative. On no-interaction data with a strong
    shared instantaneous field -- the situation this analysis is actually in --
    it is the other way round: the surrogate is anti-conservative, because every
    trial shares the same slow field so a mismatched pair stays coupled.

    A calibrated z has sd ~ 1 across independent cells with no interaction.
    """
    zj, zs = [], []
    for c in range(10):
        rng = np.random.default_rng(500 + c)
        xs, ys = [], []
        for _ in range(25):
            field = _pink(4_000, rng)
            xs.append(_pink(4_000, rng) + 2.0 * field)
            ys.append(_pink(4_000, rng) + 2.0 * field)
        obs = psi.phase_slope_index(xs, ys, fs=FS, band=(15.0, 30.0), nperseg=NPERSEG)
        nul = psi.trial_shuffled_nulls(xs, ys, fs=FS, bands={"b": (15.0, 30.0)},
                                       nperseg=NPERSEG, n_shuffles=10, seed=c)["b"]
        zj.append(obs["z"])
        zs.append((obs["psi"] - nul["null_mean"]) / nul["null_sd"])
    sd_jack = float(np.std(zj))
    sd_shuf = float(np.std(zs))
    assert sd_jack < 2.0, f"the jackknife z should be roughly calibrated, sd = {sd_jack:.2f}"
    assert sd_shuf > sd_jack, (
        "the shuffled surrogate must be the wider, anti-conservative one: "
        f"sd {sd_shuf:.2f} vs jackknife {sd_jack:.2f}")


# --- the across-animal direction test (moved out of the plot script) --------

def test_direction_stats_one_row_per_pair_band_reference_with_bh():
    rows = []
    for m in range(8):
        for ref in ("monopolar", "bipolar"):
            rows.append({"mouse_id": m, "area_a": "DMS", "area_b": "ACC", "band": "theta",
                         "reference": ref, "epoch": "All",
                         "z": 2.0 if ref == "monopolar" else (-1) ** m * 0.5})
    out = psi.direction_stats(rows, bands=("theta",))
    by_ref = {r["reference"]: r for r in out}
    assert set(by_ref) == {"monopolar", "bipolar"}
    assert by_ref["monopolar"]["p_raw"] == pytest.approx(2 / 2**8)
    assert by_ref["bipolar"]["p_raw"] > 0.5
    assert by_ref["monopolar"]["n_animals"] == 8
    assert "p_fdr" in by_ref["bipolar"]
