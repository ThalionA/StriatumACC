"""Coupling as a function of electrode separation along the shank.

The control the 2026-09-09 meeting asked for. Every cross-area coupling number
this package reports is confounded with distance: DMS, DLS and ACC sit on one
shank, so "different area" and "further apart" are the same axis. Held-out CC1
falls monotonically with separation -- CA1-DG at 510 um scores 0.93, DLS-DMS at
778 um 0.75, ACC-DMS at 1336 um 0.61, ACC-DLS at 2044 um 0.55 -- which is what a
volume-conducted field looks like and not what area-specific communication has to
look like. The within-area split-half ceiling that was supposed to bracket this
is saturated at 0.97-0.999 and so discriminates nothing.

The test that separates the two: take pairs of channels the SAME distance apart,
and ask whether it matters that the distance crosses an area boundary.

* If coupling is a function of separation alone, within-area and across-area
  pairs agree at matched separation and the area labels add nothing.
* If there is an area-specific term, they part company at matched separation.

Two coupling measures are reported for every pair, because they answer different
objections:

``r_raw``
    Pearson r between the two channels' log band power over all (spatial bin,
    trial) samples. This is the quantity the CCA arm is built on, so it is the
    one the standing caveat is about.
``r_residual``
    The same after subtracting each channel's own mean spatial profile, so the
    shared spatial tuning -- which largely tracks running speed -- cannot
    manufacture a correlation. What is left is trial-to-trial co-fluctuation.

Nothing here is re-referenced: the caches are notched raw voltage, where volume
conduction is at its strongest. That is deliberate -- this measures the thing the
caveat is about.

Created 2026-09-17.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairCoupling:
    """One row per unordered channel pair, all arrays the same length."""

    channel_i: np.ndarray
    channel_j: np.ndarray
    separation_um: np.ndarray
    same_area: np.ndarray
    area_i: np.ndarray
    area_j: np.ndarray
    r_raw: np.ndarray
    r_residual: np.ndarray


def _corr_rows(mat: np.ndarray) -> np.ndarray:
    """Pearson correlation between every pair of rows of ``mat`` (n_ch, n_samples).

    Written out rather than using ``np.corrcoef`` so that a channel with zero
    variance yields NaN instead of a warning and a garbage 1.0.
    """
    mat = mat - mat.mean(axis=1, keepdims=True)
    sd = np.sqrt((mat ** 2).sum(axis=1))
    sd[sd == 0] = np.nan
    unit = mat / sd[:, None]
    return unit @ unit.T


def pairwise_coupling(cube: np.ndarray, depths: np.ndarray,
                      area_of_channel) -> PairCoupling:
    """Coupling and separation for every pair of LABELLED channels.

    ``cube`` is ``(n_channels, n_bins, n_trials)`` band power in linear units;
    it is log-transformed here, as everywhere else in this package, because band
    power is strongly right-skewed and a Pearson r on raw power is dominated by
    its largest samples. Channels whose area label is empty are dropped: they sit
    outside every depth band in the CSV and have no area to compare.
    """
    cube = np.asarray(cube, dtype=float)
    depths = np.asarray(depths, dtype=float)
    areas = np.asarray(area_of_channel, dtype=object)
    if cube.ndim != 3:
        raise ValueError(f"cube must be (n_ch, n_bins, n_trials); got {cube.shape}")
    if not (cube.shape[0] == depths.size == areas.size):
        raise ValueError("cube, depths and area labels disagree on the channel count")

    keep = np.array([a not in ("", None) for a in areas])
    idx = np.flatnonzero(keep)
    if idx.size < 2:
        empty_f, empty_i, empty_b, empty_o = (np.array([], float), np.array([], int),
                                              np.array([], bool), np.array([], object))
        return PairCoupling(empty_i, empty_i, empty_f, empty_b, empty_o, empty_o,
                            empty_f, empty_f)

    sub = cube[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_power = np.log(np.where(sub > 0, sub, np.nan))
    # A channel with any non-finite sample would poison the whole correlation, so
    # fill per channel with that channel's own mean rather than dropping trials.
    ch_mean = np.nanmean(log_power, axis=(1, 2), keepdims=True)
    log_power = np.where(np.isfinite(log_power), log_power, ch_mean)

    n_ch = idx.size
    raw = log_power.reshape(n_ch, -1)
    # Residual: remove each channel's own mean spatial profile, so a profile
    # shared across channels (largely a speed profile) cannot create coupling.
    resid = (log_power - log_power.mean(axis=2, keepdims=True)).reshape(n_ch, -1)

    c_raw = _corr_rows(raw)
    c_res = _corr_rows(resid)

    ii, jj = np.triu_indices(n_ch, k=1)
    gi, gj = idx[ii], idx[jj]
    return PairCoupling(
        channel_i=gi,
        channel_j=gj,
        separation_um=np.abs(depths[gi] - depths[gj]),
        same_area=(areas[gi] == areas[gj]),
        area_i=areas[gi],
        area_j=areas[gj],
        r_raw=c_raw[ii, jj],
        r_residual=c_res[ii, jj],
    )


def matched_separation_range(res: PairCoupling) -> tuple[float, float]:
    """The separation range where BOTH within- and across-area pairs exist.

    Within-area separations are capped by how thick the area is, so the two
    classes only overlap over part of the range. Comparing outside the overlap
    compares a distance effect with an area effect, which is the confound this
    module exists to avoid.
    """
    if res.separation_um.size == 0:
        return (np.nan, np.nan)
    w = res.separation_um[res.same_area]
    a = res.separation_um[~res.same_area]
    if w.size == 0 or a.size == 0:
        return (np.nan, np.nan)
    return (float(max(w.min(), a.min())), float(min(w.max(), a.max())))


def summarise(res: PairCoupling, *, bin_um: float = 100.0,
              extra: dict | None = None) -> list[dict]:
    """Aggregate pairs into separation bins x {within, across}.

    One row per (separation bin, class). The per-pair arrays are far too large to
    write out -- 384 channels is 73,536 pairs per band per file -- and the figure
    only ever needs the binned means.
    """
    rows: list[dict] = []
    if res.separation_um.size == 0:
        return rows
    bin_lo = np.floor(res.separation_um / bin_um) * bin_um
    for lo in np.unique(bin_lo):
        for cls, mask in (("within", res.same_area), ("across", ~res.same_area)):
            sel = (bin_lo == lo) & mask
            if not sel.any():
                continue
            row = {
                "sep_bin_lo_um": float(lo),
                "sep_bin_hi_um": float(lo + bin_um),
                "pair_class": cls,
                "n_pairs": int(sel.sum()),
                "mean_separation_um": float(res.separation_um[sel].mean()),
                "mean_r_raw": float(np.nanmean(res.r_raw[sel])),
                "median_r_raw": float(np.nanmedian(res.r_raw[sel])),
                "mean_r_residual": float(np.nanmean(res.r_residual[sel])),
                "median_r_residual": float(np.nanmedian(res.r_residual[sel])),
            }
            row.update(extra or {})
            rows.append(row)
    return rows

def exact_matched_contrast(res: PairCoupling, *, min_pairs: int = 10,
                           boundary: tuple[str, str] | None = None) -> dict:
    """Within minus across at EXACTLY equal separation, averaged over separations.

    Restricting both classes to a shared separation RANGE is not the same as
    matching their separation DISTRIBUTIONS, and on this probe the difference
    decides the answer. Measured on the task cohort (2026-09-17): inside the
    shared range, within-area pairs averaged 367 um apart and across-area pairs
    583 um, and the per-cell separation imbalance correlated with the coupling
    difference at r = -0.63 -- so a range-restricted "within > across" of +0.07
    was mostly the within pairs being closer, not an area boundary.

    Channel depths lie on an exact 20 um grid, so the confound can be removed
    outright rather than modelled: compare the two classes only at IDENTICAL
    separation values, then average those differences. Separations are weighted
    equally, so a separation with many pairs cannot dominate.

    ``boundary=(A, B)`` restricts the contrast to one boundary: across = A-B
    pairs, within = pairs inside A or inside B. Pooled, a striatal DMS-DLS
    boundary and a cortico-striatal one dilute each other, and the matched
    separations mostly probe the nearer (striatal) one.
    """
    out: dict = {"n_separations": 0, "n_pairs_used": 0}
    if res.separation_um.size == 0:
        return out
    within_cls, across_cls = res.same_area, ~res.same_area
    if boundary is not None:
        a_, b_ = boundary
        ai, aj = res.area_i.astype(str), res.area_j.astype(str)
        within_cls = res.same_area & np.isin(ai, boundary)
        across_cls = ((ai == a_) & (aj == b_)) | ((ai == b_) & (aj == a_))
    diffs_raw: list[float] = []
    diffs_res: list[float] = []
    seps_used: list[float] = []
    n_used = 0
    for s in np.unique(res.separation_um):
        at = res.separation_um == s
        w, a = at & within_cls, at & across_cls
        if w.sum() < min_pairs or a.sum() < min_pairs:
            continue
        diffs_raw.append(float(np.nanmean(res.r_raw[w]) - np.nanmean(res.r_raw[a])))
        diffs_res.append(float(np.nanmean(res.r_residual[w]) - np.nanmean(res.r_residual[a])))
        seps_used.append(float(s))
        n_used += int(w.sum() + a.sum())
    if not diffs_raw:
        return out
    out.update({
        "n_separations": len(seps_used),
        "n_pairs_used": n_used,
        "sep_min_um": min(seps_used),
        "sep_max_um": max(seps_used),
        "d_raw": float(np.mean(diffs_raw)),
        "d_raw_median": float(np.median(diffs_raw)),
        "d_residual": float(np.mean(diffs_res)),
        "d_residual_median": float(np.median(diffs_res)),
        "frac_separations_positive": float(np.mean(np.array(diffs_raw) > 0)),
    })
    return out


def contrast_stats(matched: list[dict], *, bands, field: str = "d_raw",
                   boundary: str = "all") -> list[dict]:
    """Within - across at identical separation, tested across animals, per band.

    An animal's probes are averaged first (the animal is the unit); cells with no
    matched separation are skipped. Exact sign-flip per band, BH over bands, and
    a t-based 95 % CI across animals -- a null here is only as informative as
    that interval is narrow, so the interval travels with the p.
    """
    from scipy.stats import t as t_dist

    from . import stats

    rows = []
    for band in bands:
        by_mouse: dict[int, list[float]] = {}
        for r in matched:
            v = r.get(field, np.nan)
            if r["band"] != band or not r.get("n_separations", 0) or not np.isfinite(v):
                continue
            by_mouse.setdefault(int(r["mouse_id"]), []).append(float(v))
        vals = np.array([np.mean(v) for v in by_mouse.values()])
        if not vals.size:
            continue
        n = vals.size
        sem = float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        half = float(t_dist.ppf(0.975, n - 1) * sem) if n > 1 else np.nan
        floor = stats.sign_flip_floor(n)
        rows.append({"band": band, "boundary": boundary, "field": field, "n_animals": n,
                     "mean": float(vals.mean()), "sem": sem,
                     "ci95_low": float(vals.mean()) - half if n > 1 and sem > 0
                     else float(vals.mean()),
                     "ci95_high": float(vals.mean()) + half if n > 1 and sem > 0
                     else float(vals.mean()),
                     "p_raw": stats.sign_flip_test(vals), "p_floor": floor,
                     "reachable": stats.can_reach(floor)})
    if rows:
        adj, rej = stats.fdr_bh(np.array([r["p_raw"] for r in rows]))
        for r, a, k in zip(rows, adj, rej):
            r["p_fdr"], r["survives_fdr"] = float(a), bool(k)
    return rows
