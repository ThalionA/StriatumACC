"""Does face motion change with learning beyond running speed?

Per animal and feature (mouth, whisker ME):
  1. SpeedModel (ME | |VR speed|) fit on bins of usable trials OUTSIDE the three
     epochs, then applied to the epoch trials -> speed residual per bin.
  2. Zone mean per trial (pre-reward 75-125 cm, reward zone 125-169 cm), then
     Expert - Naive and Expert - Intermediate, in grey levels and in units of the
     fit-set SD of per-trial zone means (d).
  3. ROI validity: partial r(ME, lick fraction | speed) over fit-set bins.
  4. Drift: the still-frame ME floor, first vs last third of usable trials, and
     per epoch. The contrasts are repeated after subtracting each trial's floor.
n = 4 animals: exact tests floor at p = 0.125, so no p-values; signs per animal.

Usage: python scripts/run_face_learning.py 1105 1106 1201 1206
Writes results/face_learning.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from striatum_video.speed_control import SpeedModel, partial_corr

FEATURES = ("me_mouth", "me_whiskers", "me_spout")
ZONES = {"pre_reward": (15, 25), "reward_zone": (25, 34)}  # bin index ranges, 5 cm bins
EPOCHS = ("Naive", "Intermediate", "Expert")


def zone_means(x, trials, zone):
    lo, hi = ZONES[zone]
    return np.nanmean(x[trials][:, lo:hi], axis=1)


def analyse(session):
    d = np.load(ROOT / "results" / f"{session}_binned.npz")
    ep = {e: d[f"epoch_{e}"] for e in EPOCHS}
    fit = np.setdiff1d(d["usable"], np.concatenate(list(ep.values())))
    speed = d["vr_speed"]
    out = {"n_fit_trials": int(fit.size)}
    usable = d["usable"]
    third = max(1, usable.size // 3)
    for f in FEATURES:
        model = SpeedModel.fit(speed[fit].ravel(), d[f][fit].ravel())
        resid = d[f] - model.predict(speed)
        floor = d[f"still_{f}"]
        resid_floor = resid - (floor - np.nanmean(floor[fit]))[:, None]
        r = {"partial_r_licks_given_speed": partial_corr(d[f][fit].ravel(), d["lick_frac"][fit].ravel(),
                                                         speed[fit].ravel()),
             "raw_r_licks": float(np.corrcoef(*[a[np.isfinite(a) & np.isfinite(b)] for a, b in
                                                ((d[f][fit].ravel(), d["lick_frac"][fit].ravel()),
                                                 (d["lick_frac"][fit].ravel(), d[f][fit].ravel()))])[0, 1]),
             "raw_r_speed": float(np.corrcoef(*[a[np.isfinite(a) & np.isfinite(b)] for a, b in
                                                ((d[f][fit].ravel(), speed[fit].ravel()),
                                                 (speed[fit].ravel(), d[f][fit].ravel()))])[0, 1]),
             "floor_first_third": float(np.nanmedian(floor[usable[:third]])),
             "floor_last_third": float(np.nanmedian(floor[usable[-third:]])),
             "floor_by_epoch": {e: float(np.nanmedian(floor[ep[e]])) for e in EPOCHS}}
        for z in ZONES:
            sd = np.nanstd(zone_means(resid, fit, z))
            m = {e: float(np.nanmean(zone_means(resid, ep[e], z))) for e in EPOCHS}
            mf = {e: float(np.nanmean(zone_means(resid_floor, ep[e], z))) for e in EPOCHS}
            raw = {e: float(np.nanmean(zone_means(d[f], ep[e], z))) for e in EPOCHS}
            spd = {e: float(np.nanmean(zone_means(speed, ep[e], z))) for e in EPOCHS}
            r[z] = {"resid": m, "resid_floor_corrected": mf, "raw": raw, "speed": spd, "fit_sd": float(sd),
                    "E_minus_N_d": (m["Expert"] - m["Naive"]) / sd,
                    "E_minus_I_d": (m["Expert"] - m["Intermediate"]) / sd,
                    "E_minus_N_d_floorcorr": (mf["Expert"] - mf["Naive"]) / sd,
                    "E_minus_I_d_floorcorr": (mf["Expert"] - mf["Intermediate"]) / sd}
        out[f] = r
    return out


def main(sessions):
    res = {s: analyse(s) for s in sessions}
    (ROOT / "results" / "face_learning.json").write_text(json.dumps(res, indent=2))
    for f in FEATURES:
        print(f"\n== {f}")
        print("animal  fit_n  partial_r(licks|speed) raw_r  r(speed)  floor 1st->last third (%)")
        for s, r in res.items():
            x = r[f]
            ch = 100 * (x["floor_last_third"] / x["floor_first_third"] - 1)
            print(f"{s:6s} {r['n_fit_trials']:5d}  {x['partial_r_licks_given_speed']:+.3f}"
                  f"                {x['raw_r_licks']:+.3f}  {x['raw_r_speed']:+.3f}  {x['floor_first_third']:.2f}->{x['floor_last_third']:.2f} ({ch:+.0f}%)")
        for z in ZONES:
            print(f"  {z}: d = (Expert - X) / fit-set SD of trial zone means; speed E-N (a.u./s)")
            for s, r in res.items():
                x = r[f][z]
                print(f"   {s}: E-N {x['E_minus_N_d']:+.2f} (floor-corr {x['E_minus_N_d_floorcorr']:+.2f}) | "
                      f"E-I {x['E_minus_I_d']:+.2f} (floor-corr {x['E_minus_I_d_floorcorr']:+.2f}) | "
                      f"speed E-N {x['speed']['Expert'] - x['speed']['Naive']:+.1f}")


if __name__ == "__main__":
    main(sys.argv[1:])
