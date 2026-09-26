# Results

Written by `scripts/run_lfp_pipeline.sh`. Tables listed in `../.gitignore` as
whitelisted are committed (they carry every number in `NOTES.md`); caches
(`lfp_band_trials_*`, `lfp_arms_moving_depth_*`, `lfp_psd_*`) and per-trial tables
are regenerable and not.

- `lfp_inventory_*`, `lfp_identity_*` — per-file audit; filename vs spiking.
- `lfp_band_trials_<cohort>/<mouse>_<probe>.npz` — band power per channel × bin ×
  raw trial, the input to everything downstream (not committed).
- `lfp_bandpower_summary_*`, `lfp_bandpower_validation_*` — extraction summary;
  trials vs MATLAB's good mask and bin spans vs MATLAB `durations`.
- `lfp_arms_<arm>_<cohort>.csv` — evolution, decoding, reliability, moving
  reliability (+ `_epochs`), CCA, behaviour; `*_stats_*` are the across-animal
  tests (exact sign-flip, BH), with each test's floor and reachability.
- `lfp_group_contrast.csv` — task vs control per cell, exact permutation, BH within
  arm × metric; `primary` marks the pre-registered test.
- `lfp_distance_*` — separation-matched within/across coupling, per boundary;
  `lfp_psi_*` — phase-slope index and its direction test; `lfp_coupling_*` — PAC
  and envelope coupling with the re-paired-trial null (`p_trial_repaired_null`).
- `lfp_probe2_clock_audit.csv` — the probe-2 clock measurement.
- `pipeline_*.log` / `*.out` — run logs. `_archive_july/` — outputs of deleted July
  code, provenance only.
