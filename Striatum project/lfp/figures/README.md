# Figures

Every figure here is written by `scripts/run_lfp_pipeline.sh` (plots step) from the
tables in `results/`, as a `.svg` + `.png` pair; titles are computed from those
tables. Suffix `_task` / `_control` = cohort; no suffix = both cohorts.

| figure | script | what |
|---|---|---|
| `lfp_cohort_overview_*`, `lfp_session_integrity_*`, `lfp_spectra_*`, `lfp_depth_by_frequency_*`, `lfp_identity_matrix_*` | `plot_lfp_inventory.py` | per-file audit and filename identity |
| `lfp_evolution_log_*` (primary), `lfp_evolution_z_*`, `lfp_evolution_speed_residual_*`, `lfp_evolution_fraction_*`, `lfp_evolution_speed_*` | `plot_lfp_arms.py` | Naive → Inter → Expert, change from each animal's Naive |
| `lfp_decoding_*`, `lfp_reliability_*`, `lfp_cca_*`, `lfp_cca_vs_distance_*` | `plot_lfp_arms.py` | decoding vs null, split-half reliability, CCA vs null, CC1 vs separation |
| `lfp_reliability_moving_*` (`_session`, `_depth`, `_vs_units`) | `plot_lfp_arms.py` | the unit pipeline's moving-window metric on band power |
| `lfp_task_vs_control`, `lfp_dls_theta_task_vs_control` | `plot_lfp_task_vs_control.py` | group contrasts |
| `lfp_evolution_z_task_vs_control`, `lfp_reliability_moving_*_task_vs_control` | `plot_lfp_combined.py` | both cohorts on one axis |
| `lfp_distance_control` | `plot_lfp_distance_control.py` | within vs across area at identical separation |
| `lfp_psi_direction` | `plot_lfp_psi.py` | phase-slope index, monopolar vs vertical bipolar |
| `lfp_position_trial_<mouse>_<cohort>[_speedresid]` | `plot_lfp_position_trial.py` | one animal: trial × position heatmaps per area × band, with running speed |
| `lfp_position_profiles_<cohort>[_speedresid]`, `lfp_trial_evolution_<cohort>[_speedresid]` | `plot_lfp_position_trial.py` | epoch position profiles, and power vs trial relative to LP (animal = unit; descriptive) |
| `lfp_probe2_clock_audit` | `audit_probe2_clock.py` (not in the pipeline) | which clock the probe-2 data are on |

`_archive_pre_2026-09-25/` holds figures no current script produces (the July
audit, the pre-cohort-split names, the quarantined learning and absolute-threshold
sets). Provenance only; do not present them.
