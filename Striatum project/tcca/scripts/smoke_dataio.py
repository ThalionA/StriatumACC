"""Stage 0 smoke: load real striatum data through the temporal data layer.

Verifies the 1 ms corridor read, derived velocity, the running gate, cohort
classification and epoch/period logic on the actual preprocessed_data*.mat.

Run:  python3 scripts/smoke_dataio.py [--group task|control]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from striatum_tcca import config, dataio  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="task", choices=["task", "control"])
    args = ap.parse_args()
    cfg = config.DEFAULT

    path = config.DATA_BY_GROUP[args.group]
    print(f"data file: {path}\n  exists: {path.is_file()}")
    if not path.is_file():
        print("  -> file missing; cannot smoke. Point me at the right path.")
        return

    animals = dataio.load_animals(group=args.group)
    entries, yoked = dataio.classify_cohort(animals, cfg)
    learners = [a for a in animals if entries[a.animal_id].role == "learner"]
    print(f"\n{len(animals)} animals | "
          f"{len(learners)} learners | yoked LP = {yoked}")

    print("\nper-animal (FS-excluded unit counts per area):")
    hdr = "  id  role        rawLP  nTr  cp   " + " ".join(f"{a:>4}" for a in config.AREAS)
    print(hdr)
    for a in animals:
        e = entries[a.animal_id]
        counts = " ".join(f"{len(dataio.select_units(a, ar, cfg)):>4}" for ar in config.AREAS)
        cp = "nan" if np.isnan(a.change_point) else f"{a.change_point:.0f}"
        print(f"  {a.animal_id:>2}  {e.role:<10}  {str(e.raw_lp):>5}  {a.n_trials:>3}  "
              f"{cp:>3}  {counts}")

    # Pick the first learner with two well-populated striatal areas and stream it.
    pair = ("DMS", "ACC")
    target = next(
        (a for a in learners
         if all(len(dataio.select_units(a, ar, cfg)) >= cfg.min_units for ar in pair)),
        None,
    )
    if target is None:
        print(f"\nno learner has >= {cfg.min_units} units in both {pair}")
        return

    print(f"\n--- streaming animal {target.animal_id}, engaged period, "
          f"{cfg.temporal_bin_ms} ms bins, sigma {cfg.gaussian_sd_ms} ms ---")
    s = dataio.load_temporal_streams(target, cfg, period="engaged")
    in_trial = ~np.isnan(s.trial_idx)
    run = dataio.running_mask(s, cfg)
    n_trials_seen = len(np.unique(s.trial_idx[in_trial]))
    med_speed = float(np.median(s.vel[in_trial])) if in_trial.any() else float("nan")
    print(f"  total bins: {s.spikes.shape[0]} over {n_trials_seen} traversals")
    print(f"  median in-trial speed: {med_speed:.1f} cm/s")
    print(f"  running-gate (>= {cfg.velocity_thresh_cm_s} cm/s) retains "
          f"{100 * run.mean():.0f}% of bins")

    for ar in pair:
        act, vel, tidx, idx = dataio.area_activity(target, ar, cfg, period="engaged")
        z_run = act[run]
        print(f"  {ar}: activity {act.shape}, {len(idx)} units, "
              f"z-mean|running={np.nanmean(z_run):+.3f} z-std|running={np.nanstd(z_run):.3f}")

    rng = dataio.period_trial_range(target, "disengaged")
    print(f"  disengaged trial range: {rng}  "
          f"({'present' if rng[1] > rng[0] else 'none — animal engaged throughout'})")


if __name__ == "__main__":
    main()
