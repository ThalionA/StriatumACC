#!/usr/bin/env python3
"""Collect the current best candidate figure for each manuscript panel.

Panels follow the handwritten four-figure layout. Figures are COPIED (svg+png
where both exist) into documents/manuscript_figures/ under panel-coded names,
so the manuscript set is stable even as figures/ is regenerated.

Panels with no candidate yet are reported, not invented.
"""
from __future__ import annotations
import shutil
from pathlib import Path

ROOT = Path("/Users/theoamvr/Desktop/Experiments/StriatumACC")
PROJ = ROOT / "Striatum project"
FIG = PROJ / "figures"
TFIG = PROJ / "tcca" / "figures"
CFIG = PROJ / "cca" / "figures"
RFIG = PROJ / "rl_model" / "figures"
OUT = ROOT / "documents" / "manuscript_figures"

# panel code -> (source file | None, what the panel is meant to show)
PANELS: list[tuple[str, Path | None, str]] = [
    # ---------------- Figure 1: behaviour ----------------
    ("Fig1A_behaviour_schematic",            None,
     "Behaviour schematic + control groups (illustration; no code)"),
    ("Fig1B_lick_heatmap_task",              FIG / "integrated_01_task_spatial_lick_rate_heatmaps.png",
     "Learning heatmap, task"),
    ("Fig1B_lick_heatmap_control1",          FIG / "integrated_02_control_1_spatial_lick_rate_heatmaps.png",
     "Learning heatmap, control 1"),
    ("Fig1B_lick_heatmap_control2",          FIG / "integrated_03_control_2_temporal_lick_rate_heatmaps.png",
     "Learning heatmap, control 2"),
    ("Fig1B_performance_zerror_task",        FIG / "integrated_04_task_group_z_scored_lick_errors.png",
     "Performance quantification (z-scored lick error) -> epochs"),
    ("Fig1B_performance_by_epoch",           FIG / "integrated_06_z_scored_errors_by_epoch.png",
     "Performance by epoch (defines Naive/Intermediate/Expert)"),
    ("Fig1C_behavioural_stability_epochs",   FIG / "integrated_07_behavioral_stability_evolution_epochs.png",
     "Increasing stability of licks and velocity"),
    ("Fig1C_behavioural_evolution_yoked",    FIG / "integrated_08_behavioural_evolution_across_yoked_epochs.png",
     "Behavioural evolution across yoked epochs, all 3 groups"),
    ("Fig1D_optimality",                     None,
     "Optimality vs reward rate (legacy/optimality_analysis.m is orphaned)"),
    ("Fig1E_RL_fit_quality",                 RFIG / "fig_real_fit_quality.png",
     "RL fit quality per mouse (see caveat: the CV null is exposure-confounded)"),
    ("Fig1E_RL_lick_profiles",               RFIG / "fig_real_lick_profiles.png",
     "RL model per-epoch lick profiles vs data"),

    # ---------------- Figure 2: anatomy + activity ----------------
    ("Fig2A_probe_track_atlas",              None,
     "Neuropixels track + Allen Atlas (histology; no code in repo)"),
    ("Fig2B_recording_examples",             None,
     "Example rasters/waveforms (no live code)"),
    ("Fig2C_activity_by_area_epoch_task",    FIG / "spatiotemporal_03_task_raw_fr_hierarchical_spatial.png",
     "Evolution of activity across epochs by area, task"),
    ("Fig2C_activity_by_area_epoch_control", FIG / "spatiotemporal_11_control_raw_fr_hierarchical_spatial.png",
     "Same, control"),
    ("Fig2C_activity_by_area_and_type_task", FIG / "spatiotemporal_25_task_raw_fr_pooled_increasers_spatial.png",
     "Activity by area x cell type (MSN/FS/TAN/RS)"),
    ("Fig2D_neural_stability_hierarchical",  FIG / "integrated_09_stability_allgroups_hierarchical_zscored.png",
     "Trial-to-trial reliability across epochs, animal-level, 3 groups"),
    ("Fig2E_position_decoding_evolution",    FIG / "integrated_11_ml_decoding_evolution_yoked.png",
     "Position decoding across epochs, 3 groups"),
    ("Fig2E_decoding_error_profile",         FIG / "integrated_12_spatial_decoding_error_profile_across_corridor.png",
     "Decoding error along the corridor"),
    ("Fig2E_decoding_certainty_profile",     FIG / "integrated_13_spatial_certainty_profile_across_corridor.png",
     "Decoder certainty (normalised entropy) along the corridor"),

    # ---------------- Figure 3: TCA / ensembles ----------------
    ("Fig3A_TCA_components_rank5",           FIG / "tca_components_rank5.png",
     "TCA components (neuron / spatial / trial factors)"),
    ("Fig3A_TCA_components_rank4",           FIG / "tca_components_rank4.png",
     "TCA components at the BIC-selected rank"),
    ("Fig3A_TCA_rank_diagnostics",           FIG / "tca_bic_diagnostics.png",
     "Rank selection diagnostics"),
    ("Fig3B_TCA_task_vs_control_trialfactors", FIG / "tca_task_vs_control_trialfactors.png",
     "Ensemble activity evolution, task vs control (the LP-slot control)"),
    ("Fig3C_ensemble_area_composition",      FIG / "ensemble_04_area_composition_per_ensemble.png",
     "Anatomical distribution of ensembles"),
    ("Fig3C_ensemble_type_composition",      FIG / "ensemble_05_neuron_type_composition_of_ensembles.png",
     "Cell-type composition of ensembles"),
    ("Fig3C_ensemble_area_type_composition", FIG / "ensemble_09_joint_area_and_neuron_type_composition_of_ensembles.png",
     "Joint area x cell-type composition"),
    ("Fig3D_ensemble_ablation_delta",        FIG / "ensemble_58_decoding_ko_delta.png",
     "In-silico ensemble ablation (leave-one-trial-out decoder)"),
    ("Fig3D_ensemble_single_decoders",       FIG / "ensemble_59_decoding_single_boxplot.png",
     "Single-ensemble decoders"),
    ("Fig3E_RL_latents",                     RFIG / "fig_example_latents.png",
     "RL internal variables on synthetic ground truth (value, RPE, precision)"),
    ("Fig3E_RL_latent_recovery",             RFIG / "fig_latent_recovery.png",
     "Latent recovery on synthetic ground truth (a gate, not a result; synthetic so its date is independent of the real fits)"),
    ("Fig3F_neural_encoding_of_latents",     RFIG / "fig_neural_encoding.png",
     "Per-neuron encoding of value/RPE/precision (rerun 2026-08-12, all 16 mice)"),

    # ---------------- Figure 4: communication ----------------
    ("Fig4A_CCA_schematic",                  None,
     "CCA schematic (illustration; no code)"),
    ("Fig4B_canoncorr_across_learning_spatial", CFIG / "stage2_comm_strength_committed_partial.png",
     "Canonical correlation across learning, spatial arm"),
    ("Fig4B_canoncorr_across_learning_temporal", TFIG / "grid_strength_b25.png",
     "Canonical correlation across learning, temporal arm (25 ms)"),
    ("Fig4C_information_flow_windows",       TFIG / "grid_ifi_windows.png",
     "Information-flow index vs integration window"),
    ("Fig4C_lag_curves",                     TFIG / "grid_lagcurves.png",
     "CC1 lag curves (symmetry = no directional flow)"),
    ("Fig4C_directionality_spatial",         CFIG / "directionality_partial.png",
     "Directionality, spatial arm"),
    ("Fig4D_gini_across_learning_temporal",  TFIG / "grid_gini.png",
     "Gini across learning (corrected partner-dependent metric)"),
    ("Fig4D_gini_across_learning_spatial",   CFIG / "stage3_gini_committed_partial.png",
     "Gini across learning, spatial arm (committed partial config)"),
    ("Fig4F_subspace_dimensionality",        CFIG / "stage2_subspace_dim_committed_partial.png",
     "Number of significant canonical dimensions per pair x epoch"),
    ("Fig4G_membership_overlap",              CFIG / "stage3_membership_overlap_committed_partial.png",
     "Cross-epoch membership overlap (Jaccard) of the subspace"),
    ("Fig4E_network_schematic",              None,
     "Network schematic (illustration; no code)"),

    # ---------------- Supplementary / methods ----------------
    ("SuppS1_bin_size_comparison",           FIG / "compare_bin_sizes.png",
     "2.5 cm vs 5 cm spatial binning decision"),
    ("SuppS2_TCA_balance_comparison",        FIG / "tca_balance_comparison.png",
     "TCA with vs without per-area unit balancing"),
    ("SuppS3_CCA_contrasts",                 TFIG / "grid_contrasts.png",
     "Temporal CCA: plain vs partial, FS, bin size"),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    copied, missing_src, no_candidate = [], [], []

    for code, src, desc in PANELS:
        if src is None:
            no_candidate.append((code, desc))
            continue
        if not src.exists():
            missing_src.append((code, str(src.relative_to(ROOT)), desc))
            continue
        n = 0
        for ext in (".png", ".svg"):
            cand = src.with_suffix(ext)
            if cand.exists():
                shutil.copy2(cand, OUT / f"{code}{ext}")
                n += 1
        copied.append((code, str(src.relative_to(ROOT)), n, desc))

    lines = ["# Manuscript figure panels",
             "",
             "Auto-collected by `collect_manuscript_figures.py`. Each panel is a COPY, so",
             "regenerating `figures/` does not disturb the manuscript set. Re-run the script",
             "to refresh. Panels are candidates for the layout — several do not yet show what",
             "the narrative claims, and three of the analyses are refuted; see",
             "`FIGURE_PLAN_AUDIT.md` before writing any caption.",
             "",
             f"**{len(copied)} panels collected · {len(no_candidate)} have no figure yet · "
             f"{len(missing_src)} expected file missing**",
             "",
             "## Collected", "",
             "| Panel | Source | Files | Shows |", "|---|---|---|---|"]
    for code, src, n, desc in copied:
        lines.append(f"| `{code}` | `{src}` | {'svg+png' if n == 2 else 'png'} | {desc} |")
    lines += ["", "## No figure yet (illustration or unbuilt analysis)", "",
              "| Panel | Why |", "|---|---|"]
    for code, desc in no_candidate:
        lines.append(f"| `{code}` | {desc} |")
    if missing_src:
        lines += ["", "## Expected source missing (regenerate)", "",
                  "| Panel | Expected path |", "|---|---|"]
        for code, src, desc in missing_src:
            lines.append(f"| `{code}` | `{src}` |")
    (OUT / "README.md").write_text("\n".join(lines) + "\n")

    print(f"collected {len(copied)} panels into {OUT.relative_to(ROOT)}")
    print(f"  no candidate yet : {len(no_candidate)}")
    print(f"  source missing   : {len(missing_src)}")
    for code, src, desc in missing_src:
        print(f"    MISSING {code}  <- {src}")


if __name__ == "__main__":
    main()
