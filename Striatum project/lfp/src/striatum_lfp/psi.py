"""Phase-slope index: direction of interaction, blind to instantaneous mixing.

Why this measure and not another. The distance control (2026-09-17) established
that cross-area LFP coupling on this probe is a function of electrode separation
and nothing else: one shared field, seen by many electrodes. A shared field
reaches every electrode at the same instant, and instantaneous mixing is exactly
what breaks the usual directional measures -- a lagged cross-correlation or a
Granger fit on two sensors that share a source will return a confident number
that describes the mixing, not an interaction.

The phase-slope index (Nolte et al., *Phys Rev Lett* 100:234101, 2008) is
constructed so that it cannot do that. Mixing real sources with real coefficients
leaves the coherency real; PSI is the imaginary part of a product of coherencies
at neighbouring frequencies, so it vanishes for any purely instantaneous mixture,
however strong. What survives is a consistent phase slope across frequency, which
is what a true conduction delay produces.

    PSI = Im( sum_f  conj(C(f)) * C(f + df) )

over the frequency bins of a band, with ``C`` the complex coherency. The sign
convention is pinned by test: **PSI > 0 means the first signal LEADS the second**.

The raw index has no natural scale, so it is reported as a z-score against a
jackknife over segments -- the normalisation Nolte proposes, and the one that
makes values comparable across animals, bands and area pairs.

What PSI does NOT rescue: it is insensitive to instantaneous mixing, not to a
shared source with a delay. A third region driving both areas with different
conduction delays produces a real phase slope and PSI will report it as a direct
interaction. That confound needs anatomy or a third recording site, not a better
statistic, and any result from this module carries it.

Created 2026-09-17.
"""
from __future__ import annotations

import numpy as np

#: Fewest frequency bins a band needs before a slope means anything.
MIN_BINS = 3


def _as_snippets(sig) -> list[np.ndarray]:
    """Accept a single array or a list of per-trial snippets, return a list."""
    if isinstance(sig, np.ndarray) and sig.ndim == 1:
        return [sig.astype(float)]
    return [np.asarray(s, dtype=float) for s in sig]


def segment_spectra(x, y, *, fs: float, nperseg: int, noverlap: int | None = None):
    """Per-segment cross- and auto-spectra with a Hann taper.

    ``x`` and ``y`` may be single arrays or matching LISTS of per-trial snippets.
    Each snippet is segmented INDEPENDENTLY and the segments pooled: a window is
    never allowed to straddle two trials, because the two halves would come from
    different moments of behaviour and their phase relationship would be
    arbitrary. Snippets shorter than one window contribute nothing rather than
    being zero-padded, since padding invents phase.

    Spectra are returned per segment, not averaged, because the jackknife that
    sets the scale of the index needs to drop one segment at a time.
    """
    xs, ys = _as_snippets(x), _as_snippets(y)
    if len(xs) != len(ys):
        raise ValueError(f"x and y must have the same number of snippets; "
                         f"got {len(xs)} and {len(ys)}")
    if noverlap is None:
        noverlap = nperseg // 2
    step = nperseg - noverlap
    win = np.hanning(nperseg)

    X_all, Y_all = [], []
    for xi, yi in zip(xs, ys):
        if xi.shape != yi.shape:
            raise ValueError(f"paired snippets must match; got {xi.shape} and {yi.shape}")
        if xi.size < nperseg:
            continue
        n_seg = 1 + (xi.size - nperseg) // step
        idx = (np.arange(n_seg) * step)[:, None] + np.arange(nperseg)[None, :]
        xw = (xi[idx] - xi[idx].mean(axis=1, keepdims=True)) * win
        yw = (yi[idx] - yi[idx].mean(axis=1, keepdims=True)) * win
        X_all.append(np.fft.rfft(xw, axis=1))
        Y_all.append(np.fft.rfft(yw, axis=1))
    if not X_all:
        raise ValueError(f"no snippet is as long as one {nperseg}-sample window")
    X = np.concatenate(X_all, axis=0)
    Y = np.concatenate(Y_all, axis=0)
    freqs = np.fft.rfftfreq(nperseg, 1.0 / fs)
    return freqs, X * np.conj(Y), np.abs(X) ** 2, np.abs(Y) ** 2


def _psi_from_means(Sxy: np.ndarray, Sxx: np.ndarray, Syy: np.ndarray) -> float:
    """PSI from band-restricted mean spectra (1-D over frequency)."""
    denom = np.sqrt(Sxx * Syy)
    with np.errstate(divide="ignore", invalid="ignore"):
        coh = np.where(denom > 0, Sxy / denom, 0.0 + 0.0j)
    return float(np.imag(np.sum(np.conj(coh[:-1]) * coh[1:])))


def phase_slope_index(x: np.ndarray, y: np.ndarray, *, fs: float,
                      band: tuple[float, float], nperseg: int = 1024,
                      noverlap: int | None = None,
                      min_segments: int = 8) -> dict:
    """Phase-slope index between ``x`` and ``y`` over ``band``, with a jackknife z.

    Positive means ``x`` leads ``y``. Returns the raw index, the jackknife
    standard deviation, the z-score, and the counts that produced them, so a
    caller can refuse a cell on its own terms rather than trusting a bare number.
    """
    freqs, Sxy, Sxx, Syy = segment_spectra(x, y, fs=fs, nperseg=nperseg,
                                           noverlap=noverlap)
    n_seg = Sxy.shape[0]
    if n_seg < min_segments:
        raise ValueError(
            f"jackknife needs at least {min_segments} segments, got {n_seg}; "
            "use a shorter nperseg or a longer signal")

    sel = (freqs >= band[0]) & (freqs <= band[1])
    n_bins = int(sel.sum())
    if n_bins < MIN_BINS:
        raise ValueError(
            f"band {band} holds only {n_bins} frequency bins at nperseg={nperseg}; "
            f"a phase slope needs at least {MIN_BINS}. Lengthen nperseg.")

    bxy, bxx, byy = Sxy[:, sel], Sxx[:, sel], Syy[:, sel]
    full = _psi_from_means(bxy.mean(0), bxx.mean(0), byy.mean(0))

    # Delete-one jackknife over segments. Sums are kept so each leave-one-out
    # mean is a subtraction rather than a re-reduction over the whole stack.
    sum_xy, sum_xx, sum_yy = bxy.sum(0), bxx.sum(0), byy.sum(0)
    loo = np.array([
        _psi_from_means((sum_xy - bxy[i]) / (n_seg - 1),
                        (sum_xx - bxx[i]) / (n_seg - 1),
                        (sum_yy - byy[i]) / (n_seg - 1))
        for i in range(n_seg)
    ])
    sd = float(np.sqrt((n_seg - 1) / n_seg * np.sum((loo - loo.mean()) ** 2)))
    return {
        "psi": full,
        "psi_sd": sd,
        "z": float(full / sd) if sd > 0 else np.nan,
        "n_segments": n_seg,
        "n_freq_bins": n_bins,
        "band_lo_hz": float(band[0]),
        "band_hi_hz": float(band[1]),
    }


def count_segments(snippets, nperseg: int, noverlap: int | None = None) -> int:
    """How many windows the snippets yield once each is segmented on its own.

    Strictly fewer than segmenting their concatenation, which is the point: the
    difference is the windows that would have straddled a trial boundary.
    """
    if noverlap is None:
        noverlap = nperseg // 2
    step = nperseg - noverlap
    return int(sum(1 + (np.asarray(s).size - nperseg) // step
                   for s in _as_snippets(snippets) if np.asarray(s).size >= nperseg))


def bipolar_derivation(channels: np.ndarray) -> np.ndarray:
    """Mean of NON-OVERLAPPING column-pair differences: (samples,) from (samples, n).

    Columns come in pairs -- ``(0, 1), (2, 3), ...`` -- and each pair contributes
    ``x[:, 2k] - x[:, 2k + 1]``. Which channels form a pair is decided by the
    caller (``geometry.vertical_pairs`` via ``area_signals.reduce_block``); this
    function only does the arithmetic.

    Differencing two nearby electrodes cancels the far field to first order,
    which is what makes a bipolar derivation worth computing on a probe whose
    cross-area coupling is a shared field.

    It has to be non-overlapping pairs. Averaging `np.diff` over channels looks
    like the same thing and is not: the sum of consecutive differences
    telescopes, so `mean(diff(x))` is exactly `(x[-1] - x[0]) / (n - 1)` -- one
    wide pair spanning the whole area, resting on two channels and cancelling far
    less of the field than a local pair. Verified identical to that wide pair
    before this helper existed (2026-09-17).

    An odd last column is dropped, since it has no partner.
    """
    x = np.asarray(channels, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"expected (samples, channels); got {x.shape}")
    n = x.shape[1] - (x.shape[1] % 2)
    if n < 2:
        raise ValueError(f"need at least two channels for a bipolar pair; got {x.shape[1]}")
    return np.nanmean(x[:, 0:n:2] - x[:, 1:n:2], axis=1)


def trial_shuffled_nulls(x, y, *, fs: float, bands: dict, nperseg: int,
                         n_shuffles: int = 20, seed: int = 0,
                         noverlap: int | None = None) -> dict:
    """Mismatched-trial surrogate. **Do not use this as a null on these data.**

    Kept because it was tried, measured and found wanting, and the measurement is
    worth not repeating. Pairing area A's trial *i* with area B's trial *j* looks
    like it should destroy any genuine interaction while preserving everything
    else. On this recording it does not: every trial of a session shares the same
    slow field, so a mismatched pair stays coupled, the spread across shufflings
    is far narrower than the true sampling variability, and dividing by it
    inflates z.

    Measured on 40 synthetic cells built with NO interaction and a strong shared
    instantaneous field, 60 trials each (2026-09-17):

    ========================  ==========  ===============
    statistic                 sd(z)       |z| > 2
    ========================  ==========  ===============
    jackknife over segments   1.12        10%  (expect 5%)
    this shuffled surrogate   2.48        38%  (expect 5%)
    ========================  ==========  ===============

    So the jackknife -- whose overlapping segments were the reason to distrust it
    -- is the calibrated one, and this surrogate is anti-conservative by a factor
    of about two in z. Use ``phase_slope_index``'s ``z``.
    """
    xs, ys = _as_snippets(x), _as_snippets(y)
    n = len(xs)
    empty = {b: {"null_mean": np.nan, "null_sd": np.nan, "n_shuffles": 0} for b in bands}
    if n < 2:
        return empty
    rng = np.random.default_rng(seed)
    vals: dict[str, list[float]] = {b: [] for b in bands}
    for _ in range(n_shuffles):
        # A derangement, so no trial is paired with itself.
        perm = rng.permutation(n)
        for k in range(n):
            if perm[k] == k:
                swap = (k + 1) % n
                perm[k], perm[swap] = perm[swap], perm[k]
        xa, yb = [], []
        for k in range(n):
            m = min(xs[k].size, ys[perm[k]].size)
            if m < nperseg:
                continue
            xa.append(xs[k][:m])
            yb.append(ys[perm[k]][:m])
        if not xa:
            continue
        try:
            freqs, Sxy, Sxx, Syy = segment_spectra(xa, yb, fs=fs, nperseg=nperseg,
                                                   noverlap=noverlap)
        except ValueError:
            continue
        for name, edges in bands.items():
            sel = (freqs >= edges[0]) & (freqs <= edges[1])
            if int(sel.sum()) < MIN_BINS:
                continue
            vals[name].append(_psi_from_means(Sxy[:, sel].mean(0), Sxx[:, sel].mean(0),
                                              Syy[:, sel].mean(0)))
    out = {}
    for name in bands:
        arr = np.array(vals[name], dtype=float)
        if arr.size < 3:
            out[name] = {"null_mean": np.nan, "null_sd": np.nan, "n_shuffles": int(arr.size)}
        else:
            out[name] = {"null_mean": float(arr.mean()),
                         "null_sd": float(arr.std(ddof=1)),
                         "n_shuffles": int(arr.size)}
    return out


def trial_shuffled_null(x, y, *, fs: float, band: tuple[float, float],
                        nperseg: int, n_shuffles: int = 20, seed: int = 0,
                        noverlap: int | None = None) -> dict:
    """Single-band wrapper around :func:`trial_shuffled_nulls`."""
    return trial_shuffled_nulls(x, y, fs=fs, bands={"band": band}, nperseg=nperseg,
                                n_shuffles=n_shuffles, seed=seed,
                                noverlap=noverlap)["band"]


def per_animal_z(rows, *, pair, band, ref, epoch) -> dict[int, float]:
    """``{mouse: mean PSI z}`` for one area pair x band x reference x epoch."""
    by_mouse: dict[int, list[float]] = {}
    for r in rows:
        if ((r["area_a"], r["area_b"]) != tuple(pair) or r["band"] != band
                or r["reference"] != ref or r["epoch"] != epoch):
            continue
        if np.isfinite(r["z"]):
            by_mouse.setdefault(int(r["mouse_id"]), []).append(float(r["z"]))
    return {m: float(np.mean(v)) for m, v in by_mouse.items()}


def direction_stats(rows, *, bands, epoch: str = "All") -> list[dict]:
    """Is there a consistent direction across animals? One row per pair x band x reference.

    Exact sign-flip on the per-animal jackknife z (the animal is the unit), BH
    over pairs x bands within each reference, with the test's floor carried so
    an untestable pair (three hippocampal animals) is marked, not read as null.
    """
    from . import stats

    pairs = sorted({(r["area_a"], r["area_b"]) for r in rows})
    out = []
    for ref in ("monopolar", "bipolar"):
        cells = []
        for pair in pairs:
            for band in bands:
                vals = np.array(list(per_animal_z(rows, pair=pair, band=band, ref=ref,
                                                  epoch=epoch).values()))
                if not vals.size:
                    continue
                floor = stats.sign_flip_floor(vals.size)
                cells.append({"area_a": pair[0], "area_b": pair[1], "band": band,
                              "reference": ref, "epoch": epoch, "n_animals": int(vals.size),
                              "mean_z": float(vals.mean()),
                              "sem_z": float(vals.std(ddof=1) / np.sqrt(vals.size))
                              if vals.size > 1 else np.nan,
                              "p_raw": stats.sign_flip_test(vals), "p_floor": floor,
                              "reachable": stats.can_reach(floor)})
        if cells:
            adj, rej = stats.fdr_bh(np.array([c["p_raw"] for c in cells]))
            for c, a, k in zip(cells, adj, rej):
                c["p_fdr"], c["survives_fdr"] = float(a), bool(k)
        out += cells
    return out
