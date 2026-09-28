"""Band power across the corridor and across trials, per area -- the maps
behind plot_lfp_position_trial.py.

For each band-power cache (``results/lfp_band_trials_<cohort>/*.npz``) and each
area with >= min_sites channels: ``arms.area_position_trial_map``, i.e. log10
power z-scored per channel over corridor + dark (the within-session scale) and
averaged over the area's channels, as a 50-bin x trial map; once raw and once
with each channel's within-trial log-speed component removed (as the evolution
arm does). Trials, learning point, disengagement point and epochs come from
``trials.SessionTrials`` restricted to the trials the cache covers.

Run from ``Striatum project/lfp``::

    /opt/anaconda3/bin/python scripts/run_lfp_position_trial.py --cohort task

Writes ``results/lfp_position_trial_<cohort>/<mouse>_<probe>.npz`` (small).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from striatum_lfp import analysis, arms, config, trials  # noqa: E402

MIN_SITES = config.DEFAULT.min_sites
EPOCHS = ("Naive", "Intermediate", "Expert")


def one_file(path: Path, cohort_name: str, out_dir: Path) -> str:
    z = np.load(path, allow_pickle=False)
    mouse, probe = int(z["mouse_id"]), str(z["probe"])
    bands = [str(b) for b in z["bands"]]
    corridor = z["corridor"].astype(np.float64)       # (band, channel, 50, trial)
    dark = z["dark"].astype(np.float64)
    session = trials.sessions_for(cohort_name)[mouse].with_data(z["good_trials"])
    speed = analysis.bin_speed_cm_s(z["corridor_bin_start_ms"], z["corridor_bin_stop_ms"])
    with np.errstate(divide="ignore", invalid="ignore"):
        log_speed = np.log10(speed)
    log_speed[~np.isfinite(log_speed)] = np.nan

    areas = [a for a in config.AREAS if z[f"is_{a.lower()}"].sum() >= MIN_SITES]
    n_bins, n_trials = corridor.shape[2], corridor.shape[3]
    maps = np.full((len(areas), len(bands), n_bins, n_trials), np.nan, np.float32)
    maps_resid = np.full_like(maps, np.nan)
    for ai, area in enumerate(areas):
        mask = z[f"is_{area.lower()}"]
        for bi in range(len(bands)):
            maps[ai, bi] = arms.area_position_trial_map(corridor[bi, mask], dark[bi, mask])
            maps_resid[ai, bi] = arms.area_position_trial_map(corridor[bi, mask], dark[bi, mask],
                                                              speed_covariate=log_speed)

    good_raw = np.flatnonzero(session.matlab_good)
    lp_raw = (int(good_raw[session.lp - 1])
              if session.lp is not None and session.lp - 1 < good_raw.size else -1)
    epochs = session.epochs()
    out = out_dir / f"{mouse}_{probe}.npz"
    np.savez(out, mouse_id=mouse, probe=probe, cohort=cohort_name, areas=np.array(areas),
             bands=np.array(bands), maps=maps, maps_speed_resid=maps_resid,
             speed_cm_s=speed.astype(np.float32), usable=session.usable(),
             lp_good=-1 if session.lp is None else session.lp, lp_raw=lp_raw,
             lp_source=session.lp_source, dp_raw=session.dp,
             n_channels=np.array([int(z[f"is_{a.lower()}"].sum()) for a in areas]),
             **{f"epoch_{e}": epochs[e] for e in EPOCHS})
    return f"{mouse}_{probe}: {len(areas)} areas {areas}, {session.usable().size} usable trials, LP raw {lp_raw}"


def main() -> None:
    p = argparse.ArgumentParser()
    config.add_cohort_argument(p)
    args = p.parse_args()
    in_dir = config.RESULTS_DIR / f"lfp_band_trials_{args.cohort}"
    out_dir = config.RESULTS_DIR / f"lfp_position_trial_{args.cohort}"
    out_dir.mkdir(exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for path in sorted(in_dir.glob("*.npz")):
            print(one_file(path, args.cohort, out_dir), flush=True)


if __name__ == "__main__":
    main()
