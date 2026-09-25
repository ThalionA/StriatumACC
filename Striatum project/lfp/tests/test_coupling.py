"""Ground-truth tests for same-frequency and theta-gamma coupling.

Everything here is shaped by what 2026-09-17 established: cross-area coupling on
this probe is a shared field (the distance control), and no direction survives
removing it (the phase-slope index). So every BETWEEN-area coupling measure has
to be checked against the case where two electrodes simply see one source, and
the measure that matters is the one that returns nothing there.

Two families:

* **Same-frequency amplitude coupling** -- raw envelope correlation, which a
  shared field inflates, against the orthogonalised version (Hipp et al. 2012),
  which removes the zero-lag component first.
* **Theta-gamma phase-amplitude coupling** (Tort's modulation index) -- within
  one signal, and between two.

Created 2026-09-17.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import hilbert, sosfiltfilt

from striatum_lfp import coupling, filtering

FS = 1000.0
N = 60_000                       # 60 s
THETA = (4.0, 8.0)
GAMMA = (30.0, 80.0)


def _pink(n, rng):
    w = rng.standard_normal(n)
    sp = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1 / FS)
    sc = np.ones_like(f)
    sc[1:] = 1.0 / f[1:] ** 0.5
    return np.fft.irfft(sp * sc, n=n)


def _pac_signal(n, rng, *, depth=1.0, f_gamma=55.0):
    """Gamma whose amplitude follows the phase of a NOISE-DRIVEN theta rhythm.

    The theta carrier is filtered pink noise, not a sine. That matters for more
    than realism: with a pure sine, a time-shift surrogate cannot destroy
    phase-amplitude coupling at all -- rolling the amplitude by any lag just
    moves the preferred phase and leaves the modulation as strong as it was, so
    every surrogate scores as high as the observed value and the test returns
    p = 1. Real theta drifts in frequency, so a shift genuinely decouples it.
    (Found by that exact failure, 2026-09-17.)

    ``depth=0`` gives theta and gamma that coexist with no relationship, which is
    the null these tests need as well as the effect.
    """
    sos_t = filtering.design_band_sos(THETA, fs=int(FS))
    theta = sosfiltfilt(sos_t, _pink(n, rng))
    theta = theta / (theta.std() or 1.0)
    phase = np.angle(hilbert(theta))
    envelope = 1.0 + depth * (1.0 + np.cos(phase - np.pi / 3)) / 2.0
    t = np.arange(n) / FS
    gamma = envelope * np.sin(2 * np.pi * f_gamma * t)
    return theta + 0.5 * gamma + 0.3 * _pink(n, rng)


# --------------------------------------------------------------------------
# instantaneous phase
# --------------------------------------------------------------------------

def test_band_phase_recovers_a_known_oscillation():
    t = np.arange(N) / FS
    x = np.sin(2 * np.pi * 6.0 * t)
    sos = filtering.design_band_sos(THETA, fs=int(FS))
    phase = filtering.band_phase(x, sos)
    # Unwrapped phase of a 6 Hz sine advances by 2*pi*6 rad per second.
    slope = np.polyfit(t[1000:-1000], np.unwrap(phase)[1000:-1000], 1)[0]
    assert slope == pytest.approx(2 * np.pi * 6.0, rel=0.02)
    assert phase.min() >= -np.pi - 1e-9 and phase.max() <= np.pi + 1e-9


# --------------------------------------------------------------------------
# same-frequency amplitude coupling
# --------------------------------------------------------------------------

def test_shared_field_inflates_raw_but_not_orthogonalised_correlation():
    """The decisive case. One source, two electrodes, no interaction."""
    rng = np.random.default_rng(0)
    source = _pink(N, rng)
    a = source + 0.3 * _pink(N, rng)
    b = source + 0.3 * _pink(N, rng)
    sos = filtering.design_band_sos(GAMMA, fs=int(FS))
    raw = coupling.envelope_correlation(a, b, sos)
    orth = coupling.orthogonalised_envelope_correlation(a, b, sos)
    assert raw > 0.4, f"a shared field should inflate the raw correlation, got {raw:.3f}"
    assert abs(orth) < 0.15, (
        f"orthogonalising must remove the zero-lag component, got {orth:.3f}")


def test_genuine_lagged_comodulation_survives_orthogonalisation():
    """A real amplitude interaction, with a lag, must not be thrown away."""
    rng = np.random.default_rng(1)
    lag = 25                                     # 25 ms
    drive = np.abs(_pink(N + lag, rng))
    carrier_a = _pink(N + lag, rng)
    carrier_b = _pink(N, rng)
    sos = filtering.design_band_sos(GAMMA, fs=int(FS))
    a = drive[lag:] * carrier_a[lag:]
    b = drive[:-lag] * carrier_b                 # same drive, delayed, own carrier
    orth = coupling.orthogonalised_envelope_correlation(a, b, sos)
    assert orth > 0.05, f"a lagged co-modulation should survive, got {orth:.3f}"


def test_independent_signals_give_no_amplitude_coupling():
    rng = np.random.default_rng(2)
    sos = filtering.design_band_sos(GAMMA, fs=int(FS))
    orth = coupling.orthogonalised_envelope_correlation(_pink(N, rng), _pink(N, rng), sos)
    assert abs(orth) < 0.1


def test_orthogonalised_correlation_is_symmetric():
    rng = np.random.default_rng(3)
    a, b = _pink(N, rng), _pink(N, rng)
    sos = filtering.design_band_sos(GAMMA, fs=int(FS))
    ab = coupling.orthogonalised_envelope_correlation(a, b, sos)
    ba = coupling.orthogonalised_envelope_correlation(b, a, sos)
    assert ab == pytest.approx(ba, abs=1e-12), "the measure averages both directions"


# --------------------------------------------------------------------------
# theta-gamma phase-amplitude coupling
# --------------------------------------------------------------------------

def test_modulation_index_finds_injected_pac():
    """Judged against the no-PAC case and its own surrogates, not a fixed cut.

    Tort's MI has no natural scale -- what counts as "large" depends on the
    bandwidths, the record length and the noise floor -- so an absolute threshold
    would be testing the fixture's amplitude rather than the estimator.
    """
    rng = np.random.default_rng(4)
    with_pac = coupling.modulation_index_from_signal(
        _pac_signal(N, rng, depth=1.0), None, fs=FS, phase_band=THETA, amp_band=GAMMA)
    without = coupling.modulation_index_from_signal(
        _pac_signal(N, rng, depth=0.0), None, fs=FS, phase_band=THETA, amp_band=GAMMA)
    assert with_pac > 3 * without, (
        f"injected PAC should stand well clear of none: {with_pac:.4f} vs {without:.4f}")
    out = coupling.modulation_index_with_surrogates(
        _pac_signal(20_000, rng, depth=1.0), None, fs=FS, phase_band=THETA,
        amp_band=GAMMA, n_surrogates=60, seed=1)
    assert out["p"] < 0.05, f"and should beat its own surrogates, p = {out['p']:.3f}"


def test_modulation_index_is_near_zero_without_pac():
    rng = np.random.default_rng(5)
    x = _pac_signal(N, rng, depth=0.0)
    mi = coupling.modulation_index_from_signal(x, x, fs=FS, phase_band=THETA,
                                               amp_band=GAMMA)
    assert mi < 0.005, f"theta and gamma with no relationship should give ~0, got {mi:.4f}"


def test_modulation_index_grows_with_modulation_depth():
    rng = np.random.default_rng(6)
    mis = [coupling.modulation_index_from_signal(
               _pac_signal(N, rng, depth=d), None, fs=FS,
               phase_band=THETA, amp_band=GAMMA)
           for d in (0.0, 0.5, 1.5)]
    assert mis[0] < mis[1] < mis[2], f"MI should be monotone in depth, got {mis}"


def test_between_area_pac_is_spurious_when_one_source_reaches_both():
    """Two electrodes on one PAC-carrying source show 'between-area' PAC.

    This is the trap for the between-area measure, and the reason the analysis is
    run on bipolar signals: nothing here is an interaction between two areas, yet
    theta phase from one electrode predicts gamma amplitude at the other.
    """
    rng = np.random.default_rng(7)
    source = _pac_signal(N, rng, depth=1.5)
    a = source + 0.2 * _pink(N, rng)
    b = source + 0.2 * _pink(N, rng)
    mi = coupling.modulation_index_from_signal(a, b, fs=FS, phase_band=THETA,
                                               amp_band=GAMMA)
    assert mi > 0.01, (
        "the point of this test is that a shared source DOES produce apparent "
        f"between-area PAC; got MI = {mi:.4f}")


def test_surrogate_test_is_calibrated_on_data_without_pac():
    """False-positive rate of the time-shift surrogate, measured not assumed.

    The lesson from the phase-slope work the same day: a surrogate that looks
    principled can be anti-conservative, and the only way to know is to run it on
    data built with no effect and count. A calibrated test rejects ~5% of the time.
    """
    hits = 0
    trials = 20
    for c in range(trials):
        rng = np.random.default_rng(900 + c)
        x = _pac_signal(20_000, rng, depth=0.0)
        out = coupling.modulation_index_with_surrogates(
            x, x, fs=FS, phase_band=THETA, amp_band=GAMMA,
            n_surrogates=40, seed=c)
        if out["p"] < 0.05:
            hits += 1
    rate = hits / trials
    assert rate <= 0.25, (
        f"the surrogate test rejected {rate:.0%} of no-PAC cases; a calibrated "
        "test should sit near 5% and this one is anti-conservative")


def test_surrogate_test_detects_real_pac():
    rng = np.random.default_rng(8)
    x = _pac_signal(20_000, rng, depth=1.5)
    out = coupling.modulation_index_with_surrogates(
        x, x, fs=FS, phase_band=THETA, amp_band=GAMMA, n_surrogates=40, seed=0)
    assert out["p"] < 0.05
    assert out["z"] > 2.0


def test_phase_bins_partition_the_circle():
    """A uniform amplitude distribution over phase must give exactly zero MI."""
    rng = np.random.default_rng(9)
    phase = rng.uniform(-np.pi, np.pi, 200_000)
    amp = np.ones_like(phase)
    assert coupling.modulation_index(phase, amp) == pytest.approx(0.0, abs=1e-12)


def test_trial_derangement_moves_every_trial():
    for n in (2, 3, 10, 40):
        order = coupling.trial_derangement(n, seed=n)
        assert sorted(order) == list(range(n))
        assert all(order[i] != i for i in range(n))


def test_trial_derangement_is_reproducible():
    assert coupling.trial_derangement(12, seed=3) == coupling.trial_derangement(12, seed=3)
