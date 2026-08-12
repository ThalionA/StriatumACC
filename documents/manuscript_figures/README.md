# Manuscript figure panels

Auto-collected by `collect_manuscript_figures.py`. Each panel is a COPY, so
regenerating `figures/` does not disturb the manuscript set. Re-run the script
to refresh. Panels are candidates for the layout — several do not yet show what
the narrative claims, and three of the analyses are refuted; see
`FIGURE_PLAN_AUDIT.md` before writing any caption.

**40 panels collected · 6 have no figure yet · 0 expected file missing**

## Collected

| Panel | Source | Files | Shows |
|---|---|---|---|
| `Fig1B_lick_heatmap_task` | `Striatum project/figures/integrated_01_task_spatial_lick_rate_heatmaps.png` | svg+png | Learning heatmap, task |
| `Fig1B_lick_heatmap_control1` | `Striatum project/figures/integrated_02_control_1_spatial_lick_rate_heatmaps.png` | svg+png | Learning heatmap, control 1 |
| `Fig1B_lick_heatmap_control2` | `Striatum project/figures/integrated_03_control_2_temporal_lick_rate_heatmaps.png` | svg+png | Learning heatmap, control 2 |
| `Fig1B_performance_zerror_task` | `Striatum project/figures/integrated_04_task_group_z_scored_lick_errors.png` | svg+png | Performance quantification (z-scored lick error) -> epochs |
| `Fig1B_performance_by_epoch` | `Striatum project/figures/integrated_06_z_scored_errors_by_epoch.png` | svg+png | Performance by epoch (defines Naive/Intermediate/Expert) |
| `Fig1C_behavioural_stability_epochs` | `Striatum project/figures/integrated_07_behavioral_stability_evolution_epochs.png` | svg+png | Increasing stability of licks and velocity |
| `Fig1C_behavioural_evolution_yoked` | `Striatum project/figures/integrated_08_behavioural_evolution_across_yoked_epochs.png` | svg+png | Behavioural evolution across yoked epochs, all 3 groups |
| `Fig1E_RL_fit_quality` | `Striatum project/rl_model/figures/fig_real_fit_quality.png` | png | RL fit quality per mouse (see caveat: the CV null is exposure-confounded) |
| `Fig1E_RL_lick_profiles` | `Striatum project/rl_model/figures/fig_real_lick_profiles.png` | png | RL model per-epoch lick profiles vs data |
| `Fig2C_activity_by_area_epoch_task` | `Striatum project/figures/spatiotemporal_03_task_raw_fr_hierarchical_spatial.png` | svg+png | Evolution of activity across epochs by area, task |
| `Fig2C_activity_by_area_epoch_control` | `Striatum project/figures/spatiotemporal_11_control_raw_fr_hierarchical_spatial.png` | svg+png | Same, control |
| `Fig2C_activity_by_area_and_type_task` | `Striatum project/figures/spatiotemporal_25_task_raw_fr_pooled_increasers_spatial.png` | svg+png | Activity by area x cell type (MSN/FS/TAN/RS) |
| `Fig2D_neural_stability_hierarchical` | `Striatum project/figures/integrated_09_stability_allgroups_hierarchical_zscored.png` | svg+png | Trial-to-trial reliability across epochs, animal-level, 3 groups |
| `Fig2E_position_decoding_evolution` | `Striatum project/figures/integrated_11_ml_decoding_evolution_yoked.png` | svg+png | Position decoding across epochs, 3 groups |
| `Fig2E_decoding_error_profile` | `Striatum project/figures/integrated_12_spatial_decoding_error_profile_across_corridor.png` | svg+png | Decoding error along the corridor |
| `Fig2E_decoding_certainty_profile` | `Striatum project/figures/integrated_13_spatial_certainty_profile_across_corridor.png` | svg+png | Decoder certainty (normalised entropy) along the corridor |
| `Fig3A_TCA_components_rank5` | `Striatum project/figures/tca_components_rank5.png` | svg+png | TCA components (neuron / spatial / trial factors) |
| `Fig3A_TCA_components_rank4` | `Striatum project/figures/tca_components_rank4.png` | svg+png | TCA components at the BIC-selected rank |
| `Fig3A_TCA_rank_diagnostics` | `Striatum project/figures/tca_bic_diagnostics.png` | svg+png | Rank selection diagnostics |
| `Fig3B_TCA_task_vs_control_trialfactors` | `Striatum project/figures/tca_task_vs_control_trialfactors.png` | svg+png | Ensemble activity evolution, task vs control (the LP-slot control) |
| `Fig3C_ensemble_area_composition` | `Striatum project/figures/ensemble_04_area_composition_per_ensemble.png` | svg+png | Anatomical distribution of ensembles |
| `Fig3C_ensemble_type_composition` | `Striatum project/figures/ensemble_05_neuron_type_composition_of_ensembles.png` | svg+png | Cell-type composition of ensembles |
| `Fig3C_ensemble_area_type_composition` | `Striatum project/figures/ensemble_09_joint_area_and_neuron_type_composition_of_ensembles.png` | svg+png | Joint area x cell-type composition |
| `Fig3D_ensemble_ablation_delta` | `Striatum project/figures/ensemble_58_decoding_ko_delta.png` | svg+png | In-silico ensemble ablation (leave-one-trial-out decoder) |
| `Fig3D_ensemble_single_decoders` | `Striatum project/figures/ensemble_59_decoding_single_boxplot.png` | svg+png | Single-ensemble decoders |
| `Fig3E_RL_latents` | `Striatum project/rl_model/figures/fig_example_latents.png` | png | RL internal variables on synthetic ground truth (value, RPE, precision) |
| `Fig3E_RL_latent_recovery` | `Striatum project/rl_model/figures/fig_latent_recovery.png` | png | Latent recovery on synthetic ground truth (a gate, not a result; synthetic so its date is independent of the real fits) |
| `Fig3F_neural_encoding_of_latents` | `Striatum project/rl_model/figures/fig_neural_encoding.png` | png | Per-neuron encoding of value/RPE/precision (rerun 2026-08-12, all 16 mice) |
| `Fig4B_canoncorr_across_learning_spatial` | `Striatum project/cca/figures/stage2_comm_strength_committed_partial.png` | png | Canonical correlation across learning, spatial arm |
| `Fig4B_canoncorr_across_learning_temporal` | `Striatum project/tcca/figures/grid_strength_b25.png` | svg+png | Canonical correlation across learning, temporal arm (25 ms) |
| `Fig4C_information_flow_windows` | `Striatum project/tcca/figures/grid_ifi_windows.png` | svg+png | Information-flow index vs integration window |
| `Fig4C_lag_curves` | `Striatum project/tcca/figures/grid_lagcurves.png` | svg+png | CC1 lag curves (symmetry = no directional flow) |
| `Fig4C_directionality_spatial` | `Striatum project/cca/figures/directionality_partial.png` | png | Directionality, spatial arm |
| `Fig4D_gini_across_learning_temporal` | `Striatum project/tcca/figures/grid_gini.png` | svg+png | Gini across learning (corrected partner-dependent metric) |
| `Fig4D_gini_across_learning_spatial` | `Striatum project/cca/figures/stage3_gini_committed_partial.png` | png | Gini across learning, spatial arm (committed partial config) |
| `Fig4F_subspace_dimensionality` | `Striatum project/cca/figures/stage2_subspace_dim_committed_partial.png` | png | Number of significant canonical dimensions per pair x epoch |
| `Fig4G_membership_overlap` | `Striatum project/cca/figures/stage3_membership_overlap_committed_partial.png` | png | Cross-epoch membership overlap (Jaccard) of the subspace |
| `SuppS1_bin_size_comparison` | `Striatum project/figures/compare_bin_sizes.png` | svg+png | 2.5 cm vs 5 cm spatial binning decision |
| `SuppS2_TCA_balance_comparison` | `Striatum project/figures/tca_balance_comparison.png` | svg+png | TCA with vs without per-area unit balancing |
| `SuppS3_CCA_contrasts` | `Striatum project/tcca/figures/grid_contrasts.png` | svg+png | Temporal CCA: plain vs partial, FS, bin size |

## No figure yet (illustration or unbuilt analysis)

| Panel | Why |
|---|---|
| `Fig1A_behaviour_schematic` | Behaviour schematic + control groups (illustration; no code) |
| `Fig1D_optimality` | Optimality vs reward rate (legacy/optimality_analysis.m is orphaned) |
| `Fig2A_probe_track_atlas` | Neuropixels track + Allen Atlas (histology; no code in repo) |
| `Fig2B_recording_examples` | Example rasters/waveforms (no live code) |
| `Fig4A_CCA_schematic` | CCA schematic (illustration; no code) |
| `Fig4E_network_schematic` | Network schematic (illustration; no code) |
