"""Ground-truth tests for band-power extraction.

Known sinusoids of known amplitude/frequency must produce the right band
envelopes; a band that contains a tone recovers its amplitude, a band that does
not stays near zero. Zero-phase is checked by an amplitude-modulated signal whose
envelope must peak at the modulation peak (no lag).
"""

import numpy as np

from striatum_lfp import features
from striatum_lfp.config import BANDS

FS = 1000
INTERIOR = slice(500, -500)  # drop filtfilt edge transients


def _env(x, band):
    return features.band_envelope(x, features.design_band_sos(band, fs=FS))


def test_pure_tone_recovers_amplitude_in_its_band():
    n = 5000
    t = np.arange(n) / FS
    amp = 2.0
    x = amp * np.sin(2 * np.pi * 6 * t)  # 6 Hz -> theta (4-8)
    env_theta = _env(x, BANDS["theta"])[INTERIOR]
    # envelope of a pure in-band tone ~ its amplitude
    assert abs(env_theta.mean() - amp) < 0.15 * amp
    # and vastly exceeds an out-of-band band
    env_beta = _env(x, BANDS["beta"])[INTERIOR]
    assert env_theta.mean() > 10 * env_beta.mean()


def test_two_tone_splits_into_the_right_bands():
    n = 6000
    t = np.arange(n) / FS
    x = 1.0 * np.sin(2 * np.pi * 6 * t) + 1.5 * np.sin(2 * np.pi * 40 * t)  # theta + low-gamma
    e_theta = _env(x, BANDS["theta"])[INTERIOR].mean()
    e_lowg = _env(x, BANDS["low_gamma"])[INTERIOR].mean()
    e_beta = _env(x, BANDS["beta"])[INTERIOR].mean()  # nothing at 15-30
    assert abs(e_theta - 1.0) < 0.2
    assert abs(e_lowg - 1.5) < 0.3
    assert e_beta < 0.2 * min(e_theta, e_lowg)


def test_broadband_captures_more_than_a_narrow_band():
    rng = np.random.default_rng(3)
    n = 20000
    x = rng.standard_normal(n)  # white -> power in every band
    e_broad = _env(x, BANDS["broadband"])[INTERIOR].mean()
    e_theta = _env(x, BANDS["theta"])[INTERIOR].mean()
    assert e_broad > e_theta  # 1-100 Hz spans far more than 4-8 Hz


def test_zero_phase_envelope_peaks_at_modulation_peak():
    n = 6000
    t = np.arange(n) / FS
    carrier = np.sin(2 * np.pi * 20 * t)  # beta carrier
    bump = np.exp(-0.5 * ((t - 3.0) / 0.15) ** 2)  # AM peak at t = 3.0 s
    env = _env(bump * carrier, BANDS["beta"])
    peak_t = t[np.argmax(env)]
    assert abs(peak_t - 3.0) < 0.02  # < 20 ms: no causal lag


def test_extract_bands_returns_all_bands_shape_preserved():
    x = np.random.default_rng(0).standard_normal((4000, 3))  # (time, channels)
    out = features.extract_bands(x)
    assert set(out) == set(BANDS)
    for env in out.values():
        assert env.shape == x.shape
        assert np.all(env >= 0)


def test_square_smooth_alternative_tracks_hilbert():
    n = 5000
    t = np.arange(n) / FS
    x = 2.0 * np.sin(2 * np.pi * 6 * t)
    sos = features.design_band_sos(BANDS["theta"], fs=FS)
    hil = features.band_envelope(x, sos, method="hilbert")[INTERIOR].mean()
    sq = features.band_envelope(x, sos, method="square_smooth", smooth_samples=50)[INTERIOR].mean()
    # sqrt(mean power) of a sinusoid amplitude A is A/sqrt(2); hilbert gives A
    assert abs(hil - 2.0) < 0.3
    assert abs(sq - 2.0 / np.sqrt(2)) < 0.3
