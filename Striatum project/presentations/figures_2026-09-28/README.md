# Meeting 2026-09-28: figures and tables

Built by `presentations/collect_meeting_2026-09-28.py`. It only copies files;
each figure comes from its own pipeline. The script refuses a figure that is
missing, or older than the analysis it belongs to. Images and tables are
git-ignored; this README and the collector are tracked.

- **A.** Video and movement: the `video/` subproject, 2026-09-25/26.
- **B.** Spike CCA, partial variant: the fix that moved partialling inside the
  CV folds (`cca/NOTES.md` round 18).
- **C.** LFP and information theory, re-run on the fixed pipeline (commit
  `3610f9e`, 2026-09-26; `lfp/README.md` "Where things stand").

## A. Video: movement on the neural clock

| file | take-away |
|---|---|
| `A1_video_frame_lock` | The top camera is triggered once per VR frame, so frame i = VR row i = `VR_times_synched(i)`. Video wheel shift vs VR displacement gives r ≥ 0.93 at zero shift in all 12 blocks of 1105/1106/1201/1206. The drift is under 1 frame, and no frames are dropped. |
| `A2_video_lick_maps` | Licking moves pixels on the **spout** (1.5–3 grey levels). Running moves the face and wheel by ~20. The hand-drawn mouth box sits in the running region: partial r with licks given speed is ≤ 0.13. |
| `A3_video_motion_svd_masks` | Face motion SVD (Stringer 2019 method). SVD 1 (50–68% of variance) is global motion. The next components are whisker pad, snout, paws and one mouth/spout pattern. |
| `A4_video_binned_1201`, `_1206` | Motion energy per trial × 5 cm bin, on the same grid as `spatial_binned_fr_all`: MATLAB's durations are reproduced to 2e-15 s. Mouth and whisker ME dip where the mouse stops, so they track running. |
| `A5_movement_encoding_5cm` | Per-unit cross-validated ΔR² beyond position and slow drift, against a circular-shift null (empirical false-positive rate ~7%). Movement modulates 24–84% of units, but the median ΔR² is < 0.01 in 14/15 area × animal cells. Video, as ROIs or SVD, adds ~nothing beyond VR speed + licks. |
| (20 ms, no figure) | At the temporal arm's 20 ms bins, face SVD beyond VR modulates 52–79% of units (5 cm: 12–52%). The per-unit ΔR² stays < 0.01 except 1106 DMS (0.021). |
| `A6_cca_movement_removed` | Cross-area CCA with movement regressed out, both arms, 18 animal-pairs, each against 10 shifted controls. CC1 falls below all controls in 24–44% of animal-pair-epochs (chance 9%). The median drop is ≤ 0.010, with a few large pair-specific drops (1106 DMS–DLS 0.82 → 0.52). The naive → expert picture is unchanged. |

**Caveat.** n = 4 animals with video, 3 of them learners, so these are
direction checks, not tests. No learning effect appears anywhere in A.

## B. Spike CCA: partialling moved inside the CV folds

The committed partial CCA regressed other areas out over ALL of an epoch's
samples before cross-validation, which inflated held-out CC. On independent
synthetic areas, 16 random confounds added +0.013. It is now partialled inside
each fold; see `cca/NOTES.md` round 18. The cca plotters emit PNG only.

| file | take-away |
|---|---|
| `B1_..._BEFORE_fix`, `B2_..._AFTER_fix` | The same Stage-2 panel before and after. Held-out partial CC1 falls by a median 0.0096 like-for-like (64% of epochs fall, most at low CC). Significant dimensions: 686 → 616. |
| `B3_cca_partial_subspace_dim` | Communication-subspace dimensionality after the fix. |
| `B4_cca_partial_directionality` | IFI directionality after the fix. **Fragile:** V1–DMS intermediate IFI ≠ 0 is lost (p 0.014 → 0.104), V1–ACC naive IFI is new (0.34 → 0.0027), and several means change sign. |
| `B5_cca_plain_vs_partial` | Plain vs partial CC1 per pair (`run_partial.py --fresh`). |
| `tables/B_epoch_stats_partial_{BEFORE,AFTER}_fix.csv` | Committed epoch statistics before and after. The per-animal rm-ANOVA is n.s. for every pair in both. |

**Held:** striatal-triangle and V1–ACC partial CC > 0 in every epoch (animal
t-test p ≤ 0.027), and no learning effect. **Lost:** V1–DMS expert CC > 0
(p 0.031 → 0.080). **Open:** on real epochs, regressing out pure noise still
nudges held-out CC1 up by ~0.011 (synthetic: n.s.). Compare partial CC only
against same-dimensionality controls.

## C. LFP and information theory (other session, re-run 2026-09-26)

Claims below are from `lfp/README.md` and commit `3610f9e`. C1, C2, C3 and C9
were also inspected here.

| file | take-away |
|---|---|
| `C1_lfp_task_vs_control` | Task vs yoked Control 1. The pre-registered evolution contrast differs in **0 cells**, as does decoding. Spatial-profile reliability is higher in task animals. Task animals also run far more stereotypically (speed-profile r 0.97 vs 0.54). **The panel titles say 0/24, 12/24 raw and 11/24 after speed; the LFP README says 0/30, 15/30 and 13/30** (the README apparently includes the total-power band). Settle which count is quoted. |
| `C2_lfp_dls_theta_task_vs_control` | The 2026-08-28 DLS-theta dissociation does not survive (group × epoch p_FDR 0.584). **The title is clipped on the right**; the speed-residualised p is in `tables/C_lfp_task_vs_control_contrast.csv`. |
| `C3_lfp_evolution_log_task` | Within task animals: DLS theta falls (Δ −0.060 log10, p_FDR 0.029), in the dark ITI too; DMS beta rises (+0.029, p_FDR 0.018). Nothing else survives. |
| `C4_lfp_decoding_task` | Position decodes from band power in all 15 striatal/ACC cells, but R² above null is only 0.01–0.08. |
| `C5_lfp_reliability_task_vs_control` | The unit pipeline's moving-window reliability metric on band power, both cohorts. |
| `C6_lfp_cca_task`, `C7_lfp_cca_vs_distance_task`, `C8_lfp_distance_control` | LFP CCA vs null. CC1 falls with electrode separation within animals, and within- vs across-area at identical separation shows the shared far field. |
| `C9_lfp_psi_direction` | Phase-slope index: 0/24 bipolar cells have a direction consistent across animals. Monopolar and bipolar disagree (r = −0.25, sign agrees in 45%). |
| `C10_spike_mi_overview`, `C11_lfp_mi_overview` | Information theory. "Beyond speed" does not survive (8/9 features at or below the speed floor), and no spike-MI learning change survives correction. |
| `tables/C_lfp_pac_coupling_epochs_task.csv` | Theta–gamma PAC in 66–78% of cells vs a 2–6% calibrated null (descriptive). No current pipeline figure exists; the 2026-09-18 meeting figure predates the re-run. |

## Deliberately not included

- **`figures_2026-09-18/`.** Drawn from pre-re-run tables, and its titles
  hard-code claims since retracted (e.g. `02_information_beyond_speed`).
- **`tcca/`.** Last updated 2026-09-18 and already shown then.
- **`lfp/figures/_archive_pre_2026-09-25/`.** Per its README, provenance only.
