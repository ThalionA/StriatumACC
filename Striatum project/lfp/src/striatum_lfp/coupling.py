"""Same-frequency amplitude coupling and theta-gamma phase-amplitude coupling.

Written after the two results of 2026-09-17, which govern every choice here:
cross-area coupling on this probe is a function of electrode separation and
nothing else (the distance control), and no direction of interaction survives
removing the far field (the phase-slope index). One shared field, seen by many
electrodes.

That makes the naive versions of both measures untrustworthy BETWEEN areas:

* A raw amplitude-envelope correlation between two electrodes in one field is
  large and means nothing. The orthogonalised version (Hipp et al., *Nat
  Neurosci* 15:884, 2012) removes the part of one signal that is in phase with
  the other before taking envelopes, which is exactly the instantaneous leakage.
* Phase-amplitude coupling is worse, because it is spurious in a way that looks
  specific: if one source carrying theta-gamma coupling reaches two electrodes,
  then theta phase at one predicts gamma amplitude at the other, and the number
  is a genuine measurement of a single region seen twice. There is no clever
  statistic for this -- only a reference that cancels the far field (bipolar) and
  the discipline of reporting the within-area value beside the between-area one.

Both families are therefore computed on whatever signal the caller passes, and
the drivers pass bipolar derivations as the primary and monopolar alongside, so
the two can be compared the way they were for the phase-slope index.

Filtering and envelopes come from :mod:`striatum_lfp.filtering`; nothing is
re-implemented here.

Created 2026-09-17.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert, sosfiltfilt

from . import filtering

#: Phase bins for the modulation index. 18 bins of 20 degrees is Tort's default.
N_PHASE_BINS = 18


# --------------------------------------------------------------------------
# same-frequency amplitude coupling
# --------------------------------------------------------------------------

def envelope_correlation(x: np.ndarray, y: np.ndarray, sos: np.ndarray) -> float:
    """Pearson r between the band-limited amplitude envelopes of ``x`` and ``y``.

    The naive measure. Reported only as the comparison for the orthogonalised
    one: two electrodes in a shared field produce a large value here with no
    interaction of any kind.
    """
    ex = filtering.band_envelope(np.asarray(x, float), sos)
    ey = filtering.band_envelope(np.asarray(y, float), sos)
    if np.std(ex) == 0 or np.std(ey) == 0:
        return np.nan
    return float(np.corrcoef(ex, ey)[0, 1])


def _orthogonalise(target: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Amplitude of ``target`` after removing its component in phase with ``reference``.

    ``imag(target * conj(reference) / |reference|)`` is the part of ``target``
    perpendicular to ``reference`` in the complex plane at each instant. Anything
    that reached both sensors instantaneously lies along ``reference`` and is
    removed; anything with a phase lag survives.
    """
    mag = np.abs(reference)
    with np.errstate(divide="ignore", invalid="ignore"):
        perp = np.imag(target * np.conj(reference) / np.where(mag > 0, mag, np.nan))
    return np.abs(perp)


def orthogonalised_envelope_correlation(x: np.ndarray, y: np.ndarray,
                                        sos: np.ndarray) -> float:
    """Amplitude coupling with the zero-lag (shared-field) component removed.

    Symmetrised by averaging the two orthogonalisation directions, since
    orthogonalising y against x and x against y are not the same operation and
    neither is privileged.
    """
    ax = hilbert(sosfiltfilt(sos, np.asarray(x, float)))
    ay = hilbert(sosfiltfilt(sos, np.asarray(y, float)))
    out = []
    for target, reference in ((ay, ax), (ax, ay)):
        perp = _orthogonalise(target, reference)
        ref_amp = np.abs(reference)
        ok = np.isfinite(perp) & np.isfinite(ref_amp)
        if ok.sum() < 10 or np.std(perp[ok]) == 0 or np.std(ref_amp[ok]) == 0:
            continue
        out.append(np.corrcoef(perp[ok], ref_amp[ok])[0, 1])
    return float(np.mean(out)) if out else np.nan


# --------------------------------------------------------------------------
# theta-gamma phase-amplitude coupling (Tort's modulation index)
# --------------------------------------------------------------------------

def modulation_index(phase: np.ndarray, amplitude: np.ndarray,
                     n_bins: int = N_PHASE_BINS) -> float:
    """Tort's modulation index: how far mean amplitude per phase bin is from flat.

    The mean amplitude in each phase bin is normalised to a distribution and
    compared with the uniform one by Kullback-Leibler divergence, scaled to
    ``[0, 1]``. Exactly zero when amplitude does not depend on phase.
    """
    phase = np.asarray(phase, float)
    amplitude = np.asarray(amplitude, float)
    ok = np.isfinite(phase) & np.isfinite(amplitude)
    if ok.sum() < n_bins * 10:
        return np.nan
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    idx = np.clip(np.digitize(phase[ok], edges) - 1, 0, n_bins - 1)
    sums = np.bincount(idx, weights=amplitude[ok], minlength=n_bins)
    counts = np.bincount(idx, minlength=n_bins)
    if (counts == 0).any():
        return np.nan
    means = sums / counts
    total = means.sum()
    if total <= 0:
        return np.nan
    p = means / total
    # KL against uniform, normalised by log(n_bins) so the range is [0, 1].
    nz = p > 0
    kl = np.log(n_bins) + float(np.sum(p[nz] * np.log(p[nz])))
    return float(kl / np.log(n_bins))


def modulation_index_from_signal(phase_signal: np.ndarray,
                                 amp_signal: np.ndarray | None, *,
                                 fs: float, phase_band: tuple[float, float],
                                 amp_band: tuple[float, float],
                                 n_bins: int = N_PHASE_BINS) -> float:
    """MI taking theta phase from one signal and gamma amplitude from another.

    ``amp_signal=None`` means "the same signal", i.e. within-area coupling.
    Passing two different signals gives the between-area measure, which carries
    the shared-source caveat in this module's docstring.
    """
    if amp_signal is None:
        amp_signal = phase_signal
    p_sos = filtering.design_band_sos(phase_band, fs=int(fs))
    a_sos = filtering.design_band_sos(amp_band, fs=int(fs))
    phase = filtering.band_phase(np.asarray(phase_signal, float), p_sos)
    amp = filtering.band_envelope(np.asarray(amp_signal, float), a_sos)
    return modulation_index(phase, amp, n_bins=n_bins)


def modulation_index_with_surrogates(phase_signal: np.ndarray,
                                     amp_signal: np.ndarray | None, *,
                                     fs: float, phase_band: tuple[float, float],
                                     amp_band: tuple[float, float],
                                     n_surrogates: int = 200, seed: int = 0,
                                     n_bins: int = N_PHASE_BINS) -> dict:
    """MI against a time-shift surrogate distribution.

    The amplitude series is rolled by a random offset relative to the phase
    series, which destroys any phase-amplitude relationship while leaving both
    signals' own spectra and autocorrelation untouched. The shift is drawn from
    the middle of the record so a small roll cannot leave the two nearly aligned.

    The false-positive rate of this test is measured, not assumed -- see
    ``tests/test_coupling.py``. That check exists because the same day's
    phase-slope work produced a surrogate that looked principled and was
    anti-conservative by a factor of two.
    """
    if amp_signal is None:
        amp_signal = phase_signal
    p_sos = filtering.design_band_sos(phase_band, fs=int(fs))
    a_sos = filtering.design_band_sos(amp_band, fs=int(fs))
    phase = filtering.band_phase(np.asarray(phase_signal, float), p_sos)
    amp = filtering.band_envelope(np.asarray(amp_signal, float), a_sos)

    observed = modulation_index(phase, amp, n_bins=n_bins)
    n = amp.size
    if not np.isfinite(observed) or n < 4 * n_bins:
        return {"mi": observed, "mi_surrogate_mean": np.nan, "z": np.nan,
                "p": np.nan, "n_surrogates": 0}

    rng = np.random.default_rng(seed)
    lo, hi = n // 10, n - n // 10
    shifts = rng.integers(lo, hi, size=n_surrogates)
    surr = np.array([modulation_index(phase, np.roll(amp, int(s)), n_bins=n_bins)
                     for s in shifts])
    surr = surr[np.isfinite(surr)]
    if surr.size < 10:
        return {"mi": observed, "mi_surrogate_mean": np.nan, "z": np.nan,
                "p": np.nan, "n_surrogates": int(surr.size)}
    sd = surr.std(ddof=1)
    # One-sided: PAC can only push the distribution away from uniform. The +1s
    # keep p above zero, so a cell can never be reported as p = 0 on 200 draws.
    p = (1.0 + float((surr >= observed).sum())) / (surr.size + 1.0)
    return {
        "mi": float(observed),
        "mi_surrogate_mean": float(surr.mean()),
        "z": float((observed - surr.mean()) / sd) if sd > 0 else np.nan,
        "p": float(p),
        "n_surrogates": int(surr.size),
    }


def trial_derangement(n: int, seed: int = 0) -> list[int]:
    """A permutation of ``range(n)`` that moves every trial (a derangement).

    The calibration null for PAC on concatenated trials: phase from trials in
    order, amplitude from the SAME trials re-paired so no trial meets itself.
    Each trial's statistics, the concatenation boundaries and the session's slow
    drift all survive; only the within-trial phase-amplitude pairing is gone.
    The fraction of cells this null calls significant is the real false-positive
    rate the observed rate must be read against.
    """
    if n < 2:
        raise ValueError("a derangement needs at least two trials")
    rng = np.random.default_rng(seed)
    while True:
        order = rng.permutation(n)
        if np.all(order != np.arange(n)):
            return [int(i) for i in order]
