"""Movement encoding at the temporal arm's resolution (cfg.temporal_bin_ms,
20 ms) instead of 5 cm bins.

Spikes: striatum_cca.dataio.area_tensor(bin_mode="temporal"), i.e. exactly the
temporal CCA arm's per-trial corridor bins (over-long, disengaged traversals are
already emptied there). Covariates: per-frame signals put on those bins by
striatum_video.temporal (spike column 0 = 2nd corridor VR row), entered at lags
LAGS_MS. Base model: position (5 cm one-hot of the interpolated position) +
slow-drift terms + lagged |VR speed| and lick. Tested blocks: lagged face motion
SVD (10) and lagged ROI ME (4).

Null: the tested block is circularly shifted along the concatenated timeline by
N_SHIFTS offsets spread over 10-90% of it (keeps its autocorrelation, breaks its
alignment to the spikes). The empirical FPR is estimated as in the 5 cm driver.

Usage: python scripts/run_temporal_encoding.py 1105 1106 1201 1206
Writes results/temporal_encoding.npz and prints a summary.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "cca" / "src"), str(ROOT.parent / "lfp" / "src")]
from striatum_cca import dataio
from striatum_cca.config import Config
from striatum_lfp import config as lfp_config

from striatum_video.binning import position_bin_index, spatial_bin_edges
from striatum_video.confounds import temporal_confound
from striatum_video.encoding import _r2, cv_delta_r2, trial_drift_basis
from striatum_video.signals import ME_ROIS, frame_signals
from striatum_video.temporal import lag_within_trial

AREAS = ("DMS", "DLS", "ACC", "V1", "CA1")
EPOCHS = ("Naive", "Intermediate", "Expert")
LAGS_MS = (-200, -100, 0, 100, 200)
N_SVD = 10
N_SHIFTS = 50
MIN_RATE_HZ = 0.02


def build_rows(session, animal, cfg):
    """Rows = 20 ms bins of usable, non-empty corridor traversals."""
    vr, t_ms, sig = frame_signals(session, N_SVD)
    binned = np.load(ROOT / "results" / f"{session}_binned.npz")
    per_area = {a: dataio.area_tensor(animal, a, cfg) for a in AREAS}
    units = np.concatenate([idx for _, idx in per_area.values()]).astype(int)
    areas = np.concatenate([[a] * idx.size for a, (_, idx) in per_area.items()])
    spikes = [np.concatenate(trial, axis=1) for trial in zip(*(act for act, _ in per_area.values()))]
    names = tuple(sig)
    per_trial = temporal_confound(vr, t_ms, sig, names, [t.shape[0] for t in spikes], cfg.temporal_bin_ms, (0,))
    keep = [t for t in binned["usable"] if t < len(spikes) and spikes[t].shape[0] > 0]
    y = np.concatenate([spikes[t] for t in keep]).astype(float)
    cols = np.concatenate([per_trial[t] for t in keep])
    cov = {n: cols[:, i] for i, n in enumerate(names) if n != "x"}
    pos = position_bin_index(cols[:, names.index("x")], spatial_bin_edges())
    tri = np.concatenate([np.full(spikes[t].shape[0], t) for t in keep])
    return y, cov, pos, tri, units, areas, binned


def lagged_block(cov, names, trials, bin_ms):
    cols = []
    for n in names:
        for lag in LAGS_MS:
            cols.append(lag_within_trial(cov[n], trials, round(lag / bin_ms)))
    return np.column_stack(cols)


def contrast(y, bins, trials, test, nuisance, epoch_trials):
    r2_base, r2_full, pred = cv_delta_r2(y, bins, test, trials, n_bins=50, nuisance=nuisance)
    shifts = np.linspace(0.1, 0.9, N_SHIFTS) * y.shape[0]
    null = np.empty((N_SHIFTS, y.shape[1]))
    for k, sh in enumerate(shifts.astype(int)):
        _, r2_null, _ = cv_delta_r2(y, bins, np.roll(test, sh, axis=0), trials, n_bins=50, nuisance=nuisance)
        null[k] = r2_null - r2_base
    delta = r2_full - r2_base
    null95 = np.quantile(null, 0.95, axis=0)
    fpr = np.mean([(null[j] > np.quantile(np.delete(null, j, axis=0), 0.95, axis=0)).mean()
                   for j in range(N_SHIFTS)])
    ep = {}
    for e, idx in epoch_trials.items():
        m = np.isin(trials, idx)
        ep[e] = _r2(y[m], pred["full"][m]) - _r2(y[m], pred["pos"][m]) if m.sum() > 500 else np.full(y.shape[1], np.nan)
    return {"r2_base": r2_base, "delta": delta, "null95": null95, "modulated": delta > null95,
            "null_fpr": np.full(y.shape[1], fpr), **{f"delta_{e}": v for e, v in ep.items()}}


def analyse(session, animal, cfg):
    y, cov, pos, trials, units, areas, binned = build_rows(session, animal, cfg)
    ok = pos >= 0
    y, pos, trials = y[ok], pos[ok], trials[ok]
    cov = {k: v[ok] for k, v in cov.items()}
    rate = y.mean(0) / (cfg.temporal_bin_ms / 1000)
    keep = rate >= MIN_RATE_HZ
    y, units, areas = y[:, keep], units[keep], areas[keep]
    n_tr = np.unique(trials).size
    drift = trial_drift_basis(trials, n_basis=max(3, n_tr // 25))
    base = lagged_block(cov, ("vr_speed", "lick_frac"), trials, cfg.temporal_bin_ms)
    nuisance = np.hstack([drift, (base - base.mean(0)) / base.std(0)])
    epoch_trials = {e: binned[f"epoch_{e}"] for e in EPOCHS}
    out = {"session": np.full(units.size, int(session)), "unit": units, "area": areas}
    for name, names in {"face_svd_beyond_vr": tuple(f"svd_{k + 1}" for k in range(N_SVD)),
                        "roi_me_beyond_vr": tuple(f"me_{r}" for r in ME_ROIS)}.items():
        res = contrast(y, pos, trials, lagged_block(cov, names, trials, cfg.temporal_bin_ms), nuisance, epoch_trials)
        out |= {f"{name}__{k}": v for k, v in res.items()}
        print(f"{session} {name}: {n_tr} trials, {y.shape[0]} bins of {cfg.temporal_bin_ms} ms, "
              f"{units.size} units, null FPR {100 * res['null_fpr'][0]:.1f}%, "
              f"{100 * res['modulated'].mean():.0f}% modulated", flush=True)
    return out


def main(sessions):
    animals = dataio.load_animals()
    ids = list(lfp_config.TASK.mouse_ids)
    cfg = replace(Config(), bin_mode="temporal")
    rows = [analyse(s, animals[ids.index(int(s))], cfg) for s in sessions]
    table = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    np.savez(ROOT / "results" / "temporal_encoding.npz", **table)
    for name in ("face_svd_beyond_vr", "roi_me_beyond_vr"):
        print(f"\n== {name} @ {cfg.temporal_bin_ms} ms: area animal n %modulated median dR2 | held-out E-N E-I")
        for area in AREAS:
            for s in sessions:
                m = (table["area"] == area) & (table["session"] == int(s))
                if m.sum() == 0:
                    continue
                g = {k.split("__")[1]: v[m] for k, v in table.items() if k.startswith(name + "__")}
                print(f"{area:4s} {s} {m.sum():3d}  {100 * g['modulated'].mean():4.0f}%  {np.median(g['delta']):+.5f} | "
                      f"{np.nanmedian(g['delta_Expert'] - g['delta_Naive']):+.5f} "
                      f"{np.nanmedian(g['delta_Expert'] - g['delta_Intermediate']):+.5f}")


if __name__ == "__main__":
    main(sys.argv[1:])
