"""Unique contribution of movement to single-unit firing, beyond position.

Per animal: rows = usable trials x 5 cm bins (NaN-free); every selected unit
(cca's area masks, FS excluded) is fitted at once by encoding.cv_delta_r2.
Both models include slow-drift terms (trial_drift_basis, ~1 per 25 trials), so
trends shared by firing and movement over tens of trials are not credited to
movement. Null: whole-trial circular shifts of the movement block (N_SHUFFLES
offsets spread over 10-90% of the session), which keep each covariate's slow
structure; a unit counts as modulated if its dR2 exceeds the 95th percentile of
its own null. (A first run with an exchangeable within-bin trial shuffle and no
drift terms flagged 50-94% of units with median dR2 ~0.003: shared drift.) Held-out dR2
is also reported inside each learning epoch.

Firing rates: spatial_binned_fr_all via striatum_cca.dataio (raw-trial order;
its empty bins match the video bins exactly at shift 0 in all four animals,
checked 2026-09-26).

Usage: python scripts/run_movement_encoding.py 1105 1106 1201 1206
Writes results/movement_encoding.npz (per-unit table) and prints a summary.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "cca" / "src"), str(ROOT.parent / "lfp" / "src")]
from striatum_cca import dataio
from striatum_cca.config import Config
from striatum_lfp import config as lfp_config

from striatum_video.encoding import (
    _r2,
    circular_shift_trials,
    cv_delta_r2,
    trial_drift_basis,
)

MOVEMENT_COVS = ("vr_speed", "me_wheel", "me_whiskers", "me_mouth", "me_spout", "lick_frac")
VR_COVS = ("vr_speed", "lick_frac")
VIDEO_COVS = ("me_wheel", "me_whiskers", "me_mouth", "me_spout")
SVD_COVS = tuple(f"svd_{k}" for k in range(1, 11))  # face motion SVD (run_motion_svd.py)
COVARIATES = MOVEMENT_COVS + SVD_COVS
# contrast -> (covariates in BOTH models, covariates tested)
CONTRASTS = {"movement": ((), MOVEMENT_COVS), "video_beyond_vr": (VR_COVS, VIDEO_COVS),
             "face_svd_beyond_vr": (VR_COVS, SVD_COVS)}
AREAS = ("DMS", "DLS", "ACC", "V1", "CA1")
EPOCHS = ("Naive", "Intermediate", "Expert")
N_SHUFFLES = 50
MIN_RATE_HZ = 0.02  # project fr_threshold


def session_rows(animal, binned):
    """(trial, bin) rows of usable trials where every unit's FR and every covariate is finite."""
    trials = binned["usable"]
    trials = trials[trials < animal.spatial_fr.shape[0]]
    t_idx, b_idx = np.meshgrid(trials, np.arange(animal.n_bins), indexing="ij")
    t_idx, b_idx = t_idx.ravel(), b_idx.ravel()
    fr = animal.spatial_fr[t_idx, b_idx, :]
    mov = np.column_stack([binned[c][t_idx, b_idx] for c in COVARIATES])
    ok = np.isfinite(fr).all(1) & np.isfinite(mov).all(1)
    return t_idx[ok], b_idx[ok], fr[ok], mov[ok]


def contrast(y, bins, trials, test_cov, nuisance, n_bins, epoch_trials):
    """dR2 of adding test_cov to (position + nuisance), its circular-shift null,
    the null's empirical FPR, and held-out dR2 inside each epoch."""
    n_tr = np.unique(trials).size
    r2_base, r2_full, pred = cv_delta_r2(y, bins, test_cov, trials, n_bins=n_bins, nuisance=nuisance)
    shifts = np.unique(np.linspace(0.1 * n_tr, 0.9 * n_tr, N_SHUFFLES).astype(int))
    null = np.empty((shifts.size, y.shape[1]))
    for k, sh in enumerate(shifts):
        _, r2_null, _ = cv_delta_r2(y, bins, circular_shift_trials(test_cov, bins, trials, sh), trials,
                                    n_bins=n_bins, nuisance=nuisance)
        null[k] = r2_null - r2_base
    delta = r2_full - r2_base
    null95 = np.quantile(null, 0.95, axis=0)
    # Calibration on the real data: treat each null shift as if it were the real
    # alignment; the fraction of units it flags against the remaining shifts is
    # the procedure's empirical false-positive rate (nominal 5%).
    fpr = np.mean([(null[j] > np.quantile(np.delete(null, j, axis=0), 0.95, axis=0)).mean()
                   for j in range(null.shape[0])])
    ep = {}
    for e, idx in epoch_trials.items():
        m = np.isin(trials, idx)
        ep[e] = (_r2(y[m], pred["full"][m]) - _r2(y[m], pred["pos"][m])) if m.sum() > 50 else np.full(y.shape[1], np.nan)
    return {"r2_base": r2_base, "delta": delta, "null95": null95, "modulated": delta > null95,
            "null_fpr": np.full(y.shape[1], fpr), **{f"delta_{e}": v for e, v in ep.items()}}


def analyse(session, animal, cfg):
    binned = np.load(ROOT / "results" / f"{session}_binned.npz")
    trials, bins, fr, mov = session_rows(animal, binned)
    units, areas = [], []
    for area in AREAS:
        u = dataio.select_units(animal, area, cfg)
        units += u.tolist()
        areas += [area] * u.size
    units, areas = np.array(units, int), np.array(areas)
    keep = fr[:, units].mean(0) >= MIN_RATE_HZ
    units, areas = units[keep], areas[keep]
    y = fr[:, units]
    n_tr = np.unique(trials).size
    drift = trial_drift_basis(trials, n_basis=max(3, n_tr // 25))
    col = {c: i for i, c in enumerate(COVARIATES)}
    epoch_trials = {e: binned[f"epoch_{e}"] for e in EPOCHS}
    out = {"session": np.full(units.size, int(session)), "unit": units, "area": areas}
    for name, (base_cov, test_cov) in CONTRASTS.items():
        nuisance = drift
        if base_cov:
            # z-scored on all rows (a mean/SD-only leak across folds, negligible);
            # the tested covariates are z-scored per training fold inside cv_delta_r2
            b = mov[:, [col[c] for c in base_cov]]
            nuisance = np.hstack([drift, (b - b.mean(0)) / b.std(0)])
        res = contrast(y, bins, trials, mov[:, [col[c] for c in test_cov]], nuisance, animal.n_bins, epoch_trials)
        out |= {f"{name}__{k}": v for k, v in res.items()}
        print(f"{session} {name}: {n_tr} trials, {units.size} units, null FPR {100 * res['null_fpr'][0]:.1f}%")
    return out


def main(sessions):
    animals = dataio.load_animals()
    ids = list(lfp_config.TASK.mouse_ids)
    cfg = Config()
    rows = [analyse(s, animals[ids.index(int(s))], cfg) for s in sessions]
    table = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    np.savez(ROOT / "results" / "movement_encoding.npz", **table)
    for name in CONTRASTS:
        print(f"\n== {name}: area animal n %modulated median dR2 | held-out dR2 median E-N  E-I")
        for area in AREAS:
            for s in sessions:
                m = (table["area"] == area) & (table["session"] == int(s))
                if m.sum() == 0:
                    continue
                g = {k.split("__")[1]: v[m] for k, v in table.items() if k.startswith(name + "__")}
                print(f"{area:4s} {s} {m.sum():3d}  {100 * g['modulated'].mean():4.0f}%  {np.median(g['delta']):+.4f} | "
                      f"{np.nanmedian(g['delta_Expert'] - g['delta_Naive']):+.4f} "
                      f"{np.nanmedian(g['delta_Expert'] - g['delta_Intermediate']):+.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])
