# StriatumACC — Project Audit & Priority List

## 2026-08-12 — Cell types fixed; full regeneration; manuscript panel set

**Cell-type classification was broken two ways in series** and is now fixed
(`classify_neuron_types.m` + test). (1) Both organisers hardcoded
`final_nt_p2 = NaN` and never loaded `<mouse>_v1_neurontype2025.mat`, which
has existed all along — so V1/CA1/DG had no cell type at all. (2) ACC was
labelled with the striatal four-way rule, so cortical cells came out
MSN/TAN/UIN and code 5 (RS) was never assigned. A THIRD, independent fault
sat in the plotting: `SpatioTemporalActivityEvolution` remapped every label
`>3` to 4 while its panels ask for `target_types=[1 2 3 5]`, erasing RS at
draw time regardless of the data. Rule now: DMS/DLS keep MSN/FSN/TAN/UIN;
ACC/V1/CA1/DG split FS (<0.4 ms peak-trough) vs RS (everything else).
Result: ACC 76 FS/623 RS, V1 42/222, CA1 17/116, DG 7/11 (11-16% FS, the
right ballpark for cortex/hippocampus).

**Full chain regenerated** (organisers → preprocessing → TCA → ensembles →
spatiotemporal). Two pre-existing crashes had to be cleared first: both
preprocessing scripts aborted on undefined variables
(`dimensionality_stim_all`, `pca_stim_dimensionality_*`) — caches are saved
before those points, so the data was never affected, but any scripted chain
died there.

**Deliverables.** `presentations/StriatumUpdate_20260811.pptx` — 271 slides,
15 white section dividers, 256 figures, no text on figure slides.
`documents/manuscript_figures/` — 37 panels mapped onto the handwritten
four-figure layout (tracked in git). Six panels have no figure (1A/2A/2B/4A/4E
illustrations; 1D needs `legacy/optimality_analysis.m` resurrected).

**Standing caveat:** panels are placed by position in the layout, not by
whether they support the claim. Fig 2D/2E and the Fig 4 set are refuted or
null — see `documents/FIGURE_PLAN_AUDIT.md`.

## 2026-08-11 — Figure-plan audit (adversarially verified) + tcca epoch grid

**Audit.** Every panel of the handwritten manuscript plan verified against
current code/artefacts: `documents/FIGURE_PLAN_AUDIT.md` (8 verifier + 8
adversarial-skeptic agents; 0 verdicts overturned, 15 modified). Headlines:
Fig 4's learning framing is refuted by our own artefacts (recast as static
architecture + local refinement); the RL lick-channel CV headline (+0.60
nats/bin) is NOT citable — the saturated null lacks the model's dwell-time
exposure term (fix = cheap null re-scoring); the TCA arm is unrunnable until
Run_TCA_pipeline reruns (tca_outputs.mat deleted); Figs 1–2 have sound cores
but zero citable numbers on disk. Cohort for Methods: DMS 394 / DLS 522 /
ACC 699 (the July report's 1540/367/490/683 matches no artefact).

**tcca grid** (priors registered → scored, PREDICTIONS.md 2026-08-11): the
strength and IFI nulls are robust across bin {25,10 ms} × FS {excl,incl} ×
{partial,plain}, IFI at every integration window to ±250 ms. Surprise (P3
reversed): plain CCA < partial FS-excluded — the triangle's coupling is
pair-specific, not shared drive from other recorded areas; partialling is
denoising. FS-incl uplift exists only in the plain frame. Partner-dependent
`gini_pearson` (metric fix per audit §6) is also flat → Fig 4d null survives
the correction. Details: `Striatum project/tcca/NOTES.md` top entry.

## 2026-08-10 — Depth-boundary corrections, control2 rebuild, control-V1, 5 cm standardisation

Cross-checked all depth CSVs against Zihao's readme (INCR volume, his Aug-3
corrections): fixed task 731 (DMS 0–300 → 500–800; old band was internal
capsule), control2 1011 (whole striatal row was a copy-paste of 624's),
1103 (DMS 850–1050, phantom DLS removed), 317 (DLS end 100 → 200). Task id
507 = session 0705 (Vishal MMDD swap), deliberately excluded. All three
organised products regenerated post-fix (task 16, control1 5, control2 6).

Control1 now carries V1/CA1/DG via new `OrganiseStriatumDataControlIncV1.m`
(+ `RawDataControl/Neuropixels_V1_Depth_Data_control.csv` — NB RawDataControl/
is gitignored, so the corrected control depth CSVs exist only on disk).
Control2 organiser rebuilt out of Legacy/: it had NO save() and a broken CSV
path, so `all_data_control2.mat` had never existed; now saves, sessions
316/317/1011/1103/1107_M1/1107_M2 (string mouseids; VR trial = row 10).
`PreprocessStriatumControl2` fr_threshold 0.1 → 0.02, moved to root.

Bin size standardised to 5 cm (bin_size_au=4, 50 bins) after a registered,
scored comparison (PREDICTIONS.md 2026-08-10): 5 cm beat 2.5 cm on split-half
reliability in 50/50 animal-area pairs, no detectable sub-5cm structure
anywhere (kernel `bin_comparison_metrics.m`, test passes 10 seeds). 2.5 cm
emission dropped; `project_cfg.m` re-gridded (max_bin 30, landmarks 20/25) and
every consumer repointed (tcca/cca/lfp configs, IntegratedAll, Run_TCA now
reads project_cfg; cca lag knobs halved to preserve cm). Discovered: MI v2 /
Nonlinear_Epoch_Decoding / CrossSpatialBinDecoding always hardcoded 5 cm
velocity — 2.5 cm-era velocity outputs were 2× too high; correct as of the
switch. Knobs re-set: mi_pool_win 3, mi_max_lag 10, epoch max_lag 10.
Products now saved directly into `processed_data/`. Headless MATLAB works:
`/Applications/MATLAB_R2026a.app/bin/matlab -batch`.

OPEN: rerun all downstream (CEBRA export→fit, cca stage2/sweep, MI, decoding,
TCA→SpatioTemporal/ensemble, tcca both groups, IntegratedAll_v1,
summary_numbers) — /stats-rigor the regenerated cca headline numbers (sample
budget halved). Stale 2.5 cm results parked (~6 GB cca/results s2p5 pickles,
cebra_data exports) — delete once reruns land. Control2 624 row unused;
1107_M2 depths flagged "??" by Zihao — treat its counts as provisional.

## 2026-07-13 — LFP audit validation hardened

Rechecked the complete LFP package and corrected two figure/code errors: state
histograms now exclude periodic events, and LF diagnostics report the unscaled
difference ratio against a white-noise reference. Raw event peaks reproduce the
documented VR-sync offsets, but common-median referencing disproved the blanket
peak claim: it removes 4.54 dB of 614's ~154 Hz peak, not the persistent ~74 Hz
peak. Learning outputs are quarantined; 57 tests and figure-source caches now
enforce the current integrity claims. Exact LFP↔VR alignment remains the blocker.

## 2026-07-12 — LFP integrity audit (current gate)

The four `RawData/LFP/voltage_data_384ch*.mat` exports were re-audited from every
stored value after an absolute-unit threshold produced a false “99% empty” diagnosis.
Voltage is continuous throughout behaviour; zeros are terminal padding only. However,
the exports are broadband-dominated, contain deterministic 60 s/5 s high-amplitude
events, and 614/727/731 have narrow ~75 and ~151 Hz contamination (the former invalidates
the planned 30–80 Hz low-gamma band). 1212 is qualitatively different and must remain
separate. Common-median referencing removes much of 614's ~154 Hz peak but not the
~74 Hz peak or either peak in 727/731. Exact source band, gain, filtering/resampling and voltage↔VR offset are absent
from repository provenance. Downstream learning/position/CCA analyses are therefore
gated. Full evidence and corrected figures: `Striatum project/lfp/NOTES.md` and
`Striatum project/lfp/figures/sanity_audit_overview_v2.{png,svg}`.

_Audit completed 2026-05-07. Covers all `.m` files in active code paths, the `Preprocessing/` directory, root-level utilities, the entire `Striatum project/Legacy/` folder, and the three Python CEBRA scripts._

## Update log

**2026-05-07 first pass:**

- Filename collision fixed in `ProcessStriatumControl.m` (lines 727 and 783 now write to `preprocessed_data_control.mat`).
- `fr_threshold` aligned to `0.02 Hz` across both Task and Control preprocessing.
- SEM copy-paste fixed in `ensemble_analysis.m` (lines 600-602 now use the matching `bad_pre`/`bad_post` arrays).
- Active code consolidated into `Striatum project/`. 46 `.m` files moved up (5 from `Preprocessing/`, 41 load-bearing helpers from `Legacy/` including three transitive deps). 26 orphan `.m` files plus figure artifacts remain in `Legacy/`.
- CEBRA pipeline rewritten. New `save_for_cebra.m` exports multi-dim labels (position, lick_rate, lick_errors, velocity, LP). New `cebra_analysis.py` does multi-dim contrastive learning with per-fold StandardScaler, trial-wise train/test split, held-out ridge decoder of position as the CEBRA evaluation metric, per-area refits, and multi-session consistency scoring. Outputs land in both `.npz` and `.mat`.

**2026-05-07 second pass — done by user, not me:**

- Removed the orphan root utilities `V_allData_1ms.m` and `lick_correction_Vishal.m`. Kept `Synch_NP_VR.m`, `ReadChannel.m`, `ReadMeta.m` (still used to produce raw NPx files); also kept `raw_data_bin.m` for now.
- Renamed `Legacy/` → `legacy/` (lowercase).
- Moved `MutualInformationStriatum.m` (v1 plug-in MI) and `StriatumTaskControl_IntegratedAnalysis.m` (subset of `IntegratedAll_v1.m`) into `legacy/`. Active versions of MI (v2 with bias correction) and the integrated analysis (v1 with three groups) remain in the active folder.
- Committed all of the above to `main`.

**2026-05-08 seventh pass — CA1 + DG added; area handling generalised:**

CA1 (3 mice: 1212, 1206, 1201) and DG (same 3 mice) added. `RawData/Neuropixels_V1_Depth_Data.csv` extended with `CA1 Start`, `CA1 End`, `DG Start`, `DG End` columns. Mice 1106 and 1105 have only RHP/SC/LP annotations in `V1_depth.txt`, no explicit CA1 or DG, so those rows are blank.

Generalisation work to make future area additions cheap:

- `project_cfg.m` — `cfg.areas`, `cfg.area_field_map`, `cfg.area_colors`, `cfg.area_pairs` all extended. The new addition recipe is documented inline: edit the CSV + add an entry here. 15 area pairs now (n=6).
- `is_area_safe.m` — generalised version of `is_v1_safe.m`. Handles any area name. `is_v1_safe` retained as a thin wrapper so existing callers don't break.
- `OrganiseStriatumDataIncV1.m` — refactored to parse arbitrary `<Area>Start` / `<Area>End` columns from the CSV via header-name inspection. The probe-2 area assignment now loops over whichever areas the CSV declares. Per-area mean FR fields (`average_<Area>_fr`) and the visualisation block also generalised.
- `reorganize_spikes_by_area.m` — now driven by `cfg.areas` (or an optional `area_order` arg). No more hardcoded V1 strcmp.
- `ProcessStriatumTask.m` — adds `is_ca1` and `is_dg` to `preprocessed_data`. The deeper per-area dimensionality / cross-area correlation blocks (V1-specific) are NOT extended for CA1/DG; they're nice-to-have plotting and not load-bearing for the main analyses. Easy to extend later if needed.
- `Run_TCA_pipeline.m` — area lists, area_field_map, and per-area colour palette now pulled from `project_cfg()`. The TCA + plotting flags are unchanged.
- `IntegratedAll_v1.m` — `areas` extended to `{dms, dls, acc, v1, ca1, dg, all}` (n=7); `cond_names` adds `'No-CA1'` and `'No-DG'` (n=8 conditions); area_colors and cond_colors extended; section-12 unit summary prints CA1/DG counts and FR.
- `CCA_striatum_spatial_v2.m` — network layout extended with positions for CA1/DG.
- `SpatioTemporalActivityEvolution.m` — area-only panel sections (lines ~1275/1410/1539) extended to 6 areas. Cell-type-stratified panels (lines 893/1060/1937) intentionally left at 3 areas (V1/CA1/DG have no MSN/FSN/TAN classification — expanding those panels would add empty columns).
- `summary_numbers.m`, `neurontype_classification.m` — extended.

`buildCombinedTensor.m` was already cfg-driven via `cfg.area_field_map`, so the V1 rescue logic now automatically rescues `is_ca1` and `is_dg` too.

**Required reruns to use CA1 + DG everywhere:**

1. `OrganiseStriatumDataIncV1` (regenerates `all_data.mat` with CA1/DG units in `final_areas` + `final_spikes`)
2. `ProcessStriatumTask` (regenerates `preprocessed_data.mat` with `is_ca1` + `is_dg` populated)
3. `save_for_cebra` (regenerates `cebra_data/` with CA1/DG label rows — already cfg-driven)
4. `Run_TCA_pipeline` → `ensemble_analysis` (TCA pipeline)
5. The downstream analyses (`CCA_striatum_spatial_v2`, `MutualInformationStriatum_v2`, `Nonlinear_Epoch_Decoding`, `CrossSpatialBinDecoding`, `IntegratedAll_v1`, `SpatioTemporalActivityEvolution`)

Adding *another* area in future (e.g. CA3 or LGN if those become 3+ mice): edit `Neuropixels_V1_Depth_Data.csv` (add 2 columns) and `project_cfg.m` (one entry each in `cfg.areas`, `cfg.area_field_map`, `cfg.area_colors`, plus pairs in `cfg.area_pairs` if cross-area analyses are wanted). Everything driven by `cfg` will pick up the new area; the few scripts that still hardcode area lists (`IntegratedAll_v1` Section 6/7/9, `SpatioTemporal` target_areas) need a one-line append.

**2026-05-07 sixth pass — held-out CCA:**

New helper `held_out_canoncorr.m`. Splits samples 50/50 (seeded), fits `canoncorr` on the train half to learn projections `(A, B)`, projects the held-out half through those projections, and reports the correlation of the projected variates. Returns held-out canonical correlations alongside in-sample for direct comparison.

Wired into `CCA_striatum_spatial_v2.m`:

- Both the trial-wise loop (line ~241) and the bin-wise loop (line ~285) now compute *both* in-sample and held-out canonical correlations. New storage arrays: `cca_tr_held`, `cca_tr_held_shuff`, `cca_bin_held`, `cca_bin_held_shuff` (mirrors of the existing ones).
- Shuffles use the same train/test split as the real call (via shared seed) so the null is comparable.
- New `group_results` fields: `all_bins_corr_held`, `all_bins_corr_held_shuff`, `trial_corr_{early,pre,post}_held{,_shuff}`.

The in-sample fields are kept so the new run is directly comparable to the old. Any new figures should plot the held-out values; the in-sample is biased upward and should be flagged as such.

**2026-05-07 fifth pass — statistical rigour for SpatioTemporal:**

Two new helpers in `Striatum project/`:

- `fdr_correct.m` — Benjamini-Hochberg / Benjamini-Yekutieli / Holm correction with NaN-preservation and proper monotone enforcement. Returns adjusted p-values, significance mask, and the surviving raw-p threshold.
- `annotate_ks_panels_fdr.m` — applies BH-FDR across a row of KS-test panels in a tiled figure and writes the corrected p-values into each panel's text annotation. Centralised the boilerplate that used to repeat at four sites in SpatioTemporalActivityEvolution.

Two correctness improvements to `SpatioTemporalActivityEvolution.m`:

1. **Modulation classifier de-circularised**. Increaser/Decreaser/Maintainer labels are now defined on a *first-half* split of the naive (`1:2`) and expert (`21:25`) epochs. The disjoint second half (`naive_test = 3`, `expert_test = 26:30`) is exposed to downstream callers as held-out trials. Previously labels and tests both used the full epochs, which combined with whole-session z-scoring forced the per-neuron mean to zero and made an Increaser a Decreaser elsewhere by accounting identity. Descriptive plots over full epochs are still drawn (the labels just no longer enter the test by construction).
2. **FDR correction added to all four KS-test grids**. Per panel, the script now stores `(p_raw, ks_stat)` rather than drawing the significance star inline; after the per-figure area loop, `annotate_ks_panels_fdr` applies BH across the row of panels and writes both raw and FDR-corrected p-values into the panel annotation. Significance stars are now triggered by `p_FDR < 0.05` rather than `p_raw < 0.05`. With four panels per figure (DMS / DLS / ACC / V1) the correction is mild but mathematically defensible.

Same patterns are still pending in `IntegratedAll_v1.m` Section 11 (36 cross-modal scatter tests with hard `p < 0.05` highlighting). Apply `fdr_correct` there next time you touch that section.

**2026-05-07 fourth pass — efficiency + modularity refactor (Phases 1-5):**

Six new reusable helpers in `Striatum project/`:

- `find_learning_points.m` — single LP finder. START-of-qualifying-window convention. Configurable window / min-consecutive / threshold. Replaces six near-identical inline LP loops across IntegratedAll, MI v2, save_for_cebra, CrossSpatial, Nonlinear_Epoch, and the legacy script.
- `epoch_indices.m` — Naive / Intermediate / Expert indices given LP and n_trials. Supports both `lp` and `lp1` Expert-start conventions for backward-compat with MI v2's slicing.
- `is_v1_safe.m` — defensive accessor returning a false-mask for animals lacking a V1 probe. Used by every V1-aware caller.
- `loo_ridge_press.m` — closed-form leave-one-out ridge via the PRESS identity `y_loo_i = (y_hat_i - h_ii * y_i) / (1 - h_ii)`. Runs in `O(p^3 + n*p^2)` instead of `O(n*p^3)`. ~10x speedup for Nonlinear_Epoch_Decoding, CrossSpatialBinDecoding, and IntegratedAll's lick decoder. Drop-in replacement.
- `batch_triu_corr_mean.m` — vectorised mean-of-upper-triangle Pearson r across the third dimension of a `[cells x bins x w]` tensor, using `pagemtimes`. ~50-100x faster than the per-cell `corrcoef` loop in IntegratedAll Section 6.
- `project_cfg.m` — single source of truth for project-wide constants: areas, area-pair list, area colours, learning-point parameters, epoch geometry, landmark bins, decoder hyperparameters, fr_threshold, neuron-type column, RNG seed, default data paths.

LP-convention reconciliation: standardised on START-of-qualifying-window (the dominant convention used by IntegratedAll, processTaskData, save_for_cebra, the legacy learning_points_task script). Two callers preserved their non-standard behaviour without changing analysis windows:

- `MutualInformationStriatum_v2.m` keeps its END-of-window LP by post-shifting the helper's output (`lp_end = lp_start + window - 1`).
- `CrossSpatialBinDecoding.m` and `Nonlinear_Epoch_Decoding.m` keep their stricter "7 strictly consecutive" rule via `lp_window=7, lp_min_consecutive=7`.

So no analysis numbers should shift due to LP refactoring.

Closed-form PRESS — small caveats worth knowing:

- `loo_ridge_press` standardises features using full-fit (not per-fold) statistics. This introduces `O(1/n)` train/test leakage in the standardisation step that wasn't present in the original Nonlinear_Epoch_Decoding (which standardised per-fold). For `n=200-300` trials the bias is ~0.3-0.5% — well below the noise floor — but it's there. CrossSpatialBinDecoding and IntegratedAll's lick decoder already used full-fit standardisation, so the PRESS replacement is faithful for those.
- The PRESS implementation includes an unpenalised intercept by default; the original Nonlinear/CrossSpatial decoders did not have an explicit intercept (because z-scoring made the response approximately mean-zero). Negligible effect.

Vectorisation + parallelisation:

- `IntegratedAll_v1.m` Section 6 trial-correlation loop replaced with one `batch_triu_corr_mean` call per trial. Was `for c (1000) x for win (300)` per group, now O(trials).
- `MutualInformationStriatum_v2.m` outer animal loop converted to `parfor`. Preallocated `mi_results(n_animals).win_centers = []` so workers can write into the indexed struct safely.

Shuffle counts bumped from 1 to `cfg.n_shuffles = 25`:

- `CrossSpatialBinDecoding.m` now stores `shuff_mean`, `shuff_std`, and `empirical_p` matrices alongside the original `moving_r_shuff` (which now holds just the first sample for backwards-compat). One-sided empirical p-values via `mean(shuffles >= real)`.
- `Nonlinear_Epoch_Decoding.m` similarly stores `r_shuff_mean`, `r_shuff_std`, `empirical_p` per (target_b, source_b, epoch). GPR pathway extracted into a `local_gpr_loo` helper at the bottom of the file.

Project-wide cfg adoption:

`CrossSpatialBinDecoding.m`, `Nonlinear_Epoch_Decoding.m`, `MutualInformationStriatum_v2.m`, and `CCA_striatum_spatial_v2.m` now start with `cfg = project_cfg();` and only override script-specific fields. Removed the hardcoded absolute path `/Users/theoamvr/...` from the CCA script (now uses `./CCA_Results/`).

Phase 6 (split processing from plotting in IntegratedAll, SpatioTemporal, ProcessStriatum*) is **deferred** to a focused follow-up session. The helpers extracted in Phases 1-5 already do most of the modularity work — the residual surgery is mostly cosmetic reorganisation and benefits from a fresh, dedicated session.

**2026-05-07 third pass — V1 propagation done:**

Data state: `preprocessed_data.mat` was regenerated on Apr 24 11:13, after the V1-aware `OrganiseStriatumDataIncV1.m` (Apr 22) and `ProcessStriatumTask.m` (Apr 24 08:56). So `is_v1` is already in `preprocessed_data`. Confirm with `arrayfun(@(s) sum(s.is_v1), preprocessed_data)`.

Code changes (V1 wired into every active analysis script):

- `CrossSpatialBinDecoding.m` — V1 added to `cfg.regions` and color list. `is_v1` plumbed into `clean_data` with a defensive fallback for animals without a V1 probe.
- `Nonlinear_Epoch_Decoding.m` — same as above.
- `MutualInformationStriatum_v2.m` — V1 added to `cfg.regions`; six area pairs total now (V1-DMS, V1-DLS, V1-ACC added to the existing three).
- `CCA_striatum_spatial_v2.m` — V1 added to `cfg.areas_to_include`, `cfg.area_field_map`, the area-pair list (six pairs), and the network layout (V1 placed at the bottom as sensory input).
- `summary_numbers.m` — V1 unit count and V1 firing rate added to the task block. Also fixed the `size(x,2)` bug at line 1 that was silently returning 1's instead of unit counts. Helper `getOrFalse` added at the file bottom.
- `SpatioTemporalActivityEvolution.m` — area-only panels (lines 1231/1366/1495) already had V1 from a prior edit. Cell-type-stratified panels (lines 849/1016/1895) intentionally left at 3 areas because V1 has no MSN/FSN/TAN classification.
- `neurontype_classification.m` — V1 added to the per-area unit-count bar (V1 units fall in the Unclassified bucket since the MSN/FSN/TAN scheme is striatum-specific).
- `IntegratedAll_v1.m` — biggest change. V1 added to:
  - Section 6 stability analysis: `areas` extended to `{'dms','dls','acc','v1','all'}` (n_areas=5), masks struct, hier-/pooled- arrays, `area_colors` with purple V1.
  - Section 7 decoding: `cond_names` adds `'No-V1'` so the ablation knockout test now includes V1 dropout.
  - Section 9 spatiotemporal-by-area: `areas` extended to `{'DMS','DLS','ACC','V1'}`, masks include V1.
  - Section 11 cross-modal scatters: layout expanded from 3×4 to 3×n_area_cols (5 columns, holding DMS, DLS, ACC, V1, All Units). All `(:,4)` references for "All Units" replaced with `(:, all_col)`.
  - Section 12 unit summary: prints V1 unit count, V1 firing rate, and "n/N animals have V1 probe" so it's visible at a glance which mice contributed V1.
  - New local helper `is_v1_safe(s)` defends against animals without an `is_v1` field, so the file still runs cleanly on Control / Control2 groups.

Stale generated files that need to be rerun against the V1-extended scripts:

- `processed_data/cross_spatial_decoding_results.mat`
- `processed_data/nonlinear_epoch_decoding.mat`
- `processed_data/ridge_epoch_decoding_results.mat`, `gpr_epoch_decoding_results.mat`
- `processed_data/striatal_cca_group_results.mat`
- The MI cache (computed on first run of `MutualInformationStriatum_v2.m`)
- Any saved figure SVGs that don't include a V1 panel

Control data: `preprocessed_data_control.mat` and `preprocessed_data_control2.mat` predate the `fr_threshold = 0.02` alignment. Strictly speaking they should be regenerated for parity, but it's not blocking V1 work since control mice don't have V1 probes anyway.

Remaining priority list items unchanged below.

---

## Lab meeting log

**2026-05-25 — striatum project (with Mme.).** Next meeting: **2026-06-04, 09:30** (← tomorrow as of this entry).

Raw points as recorded:

- Still decoding in disengaged?
- Subspace comm with decreasing units only.
- Try to extend to temporal bins. → Able to see finer changes in first 3 trials.
- See how temporal chunks are dealt with in Buzsáki paper. → velocity threshold (?).

Interpretation, mapped to the codebase (action items tracked as block **M** in §6).

> **Implementation target for the CCA items is the Python `striatum_cca` package** (`Striatum project/cca/src/striatum_cca/`), per the methods contract `~/Documents/ResearchVault/Methods/CCA_HH_Adapted.md` §6. The MATLAB `CCA_striatum_spatial_v3.m` is the retired v5 instantiation — *sequestered, do not extend.* The vault meeting note is `Projects/Striatum/Meetings/2026-05-25-Striatum-Meeting.md`; tasks live in `Striatum-Tasks.md`.

1. **Decoding in the disengaged state.** The pipeline currently *removes* disengaged data: `dataio.py` truncates the trial axis at `change_point_mean` and the temporal path excludes over-long traversals (`config.temporal_max_trial_ms`); per CCA_HH_Adapted §1.3 (departure #1) there is deliberately **no engagement filter** because epochs already partition by state. The question is the complement: is position still decodable, and is the inter-area subspace still aligned, *during* disengagement? Plan: split trials engaged/disengaged from `change_point_mean`, then run the position decoder and area-pair CCA on the disengaged segment and compare held-out accuracy / canonical correlations against engaged. Retained signal ⇒ state-independent code; collapse ⇒ it tracks engagement. *Quickest first answer: stop excluding the disengaged tail and decode it separately.*

2. **Communication subspace restricted to Decreaser units.** The modulation classifier in `SpatioTemporalActivityEvolution.m` (line ~293) labels each neuron Increaser (1) / Decreaser (2) / Maintainer (3) on a held-out trial half. The ask is to recompute the inter-area communication subspace using **only Decreaser units** (`modulation_class == 2`) on each side of the area pair — do the units that *lose* activity with learning carry the cross-area communication? Plan: export `modulation_class` as a per-unit label, add a unit-subset filter to the `striatum_cca` loader (`dataio.py`/`config.py`), and contrast Decreasers-only vs all-units vs Increasers-only. Watch the unit-count floor — Decreasers-only may leave too few units in some area×animal cells for a stable `canoncorr`; log/skip cells below a minimum.

3. **Temporal binning (vs spatial).** Most analyses bin along the corridor (spatial alignment). **`striatum_cca` already exposes a temporal mode** (`config.bin_mode = "temporal"`, `temporal_bin_ms = 20`, added round 8) that re-bins 1 ms spike counts into time bins from corridor onset with no window/clipping. So this is *run-and-analyse*, not build-from-scratch: run `bin_mode="temporal"` focused on the **first 3 trials** to resolve the fast early-learning changes ("able to see finer changes in first 3 trials"), tuning `temporal_bin_ms`.

4. **Buzsáki temporal-chunking method (literature).** Check how the Buzsáki-lab paper segments time into chunks — the suspicion is a **velocity threshold** gates the chunking (cf. `simulatePOMDP_striatum.m:19`, `v_lick = 10 cm/s` "velocity when licking/engaged"). Find the paper (queued in vault `papers_to_read.md`), read its segmentation method, and write the recipe into `~/Documents/ResearchVault/Methods/` so the temporal-bin work in item 3 can adopt it. Resolve the "(?)" on whether velocity is the actual gating variable.

---

## TL;DR

The codebase is functional but is carrying significant cruft from rapid iteration. The biggest concrete risks are (a) a filename-collision bug in `ProcessStriatumControl.m` that silently overwrites task data, (b) a load-bearing `fr_threshold` mismatch (0.02 vs 0.05 Hz) between Task and Control preprocessing that biases group comparisons, and (c) several scientific holes — no held-out cross-validation in the TCA rank selection, no held-out canonical correlations in CCA, and uncorrected multiple comparisons in cross-modal scatters and bin-wise statistics. Beyond bugs, there is heavy duplication: `IntegratedAll_v1.m` is a strict superset of `StriatumTaskControl_IntegratedAnalysis.m`; `OrganiseStriatumDataIncV1.m` supersedes `OrganiseStriatumData.m`; `MutualInformationStriatum_v2.m` supersedes its v1; the `Legacy/` folder mixes 24 truly-orphan files with about 20 helpers that the active pipeline still depends on. The folder name itself is misleading and worth fixing. CEBRA was scaffolded but never wired into the main pipeline; it is worth resurrecting with position rather than lick errors as the contrastive label.

---

## 1 · Repository structure as it actually exists

The `Striatum project/Legacy/` directory contains a mixture of three distinct kinds of file. About two dozen are genuinely orphaned — never referenced from anywhere active. Roughly twenty are *load-bearing helpers* called by `ProcessStriatumTask.m`, `ProcessStriatumControl.m`, `Run_TCA_pipeline.m`, `CCA_striatum_spatial_v2.m`, or `ensemble_analysis.m` (`compute_firing_rates`, `cut_data_per_trial`, `extract_binned_spikes`, `find_change_points`, `separate_dark_and_corridor_periods`, `calculate_lick_precision`, `spatial_binning`, `decode_position`, `decode_position_mld`, `processTaskData`, `processControlData`, `runTCAAnalysis`, `tca_with_bic_extended`, `filterDataByArea`, `decode_ensemble_ablation`, the `plot*` factor visualisers). The remaining ~10 are plotting/visualisation helpers that are not strictly necessary but are still being called.

The active entry-point scripts at `Striatum project/`:

- `Run_TCA_pipeline.m` — TCA pipeline (loads preprocessed data, builds combined tensor, fits CP-NMU, plots)
- `ensemble_analysis.m` — runs after Run_TCA_pipeline (interprets neuron-factor matrix as ensemble assignments, decoding ablation)
- `summary_numbers.m` — reporting (unit counts, FR by area, FR by neuron type)
- `IntegratedAll_v1.m` — three-group analysis (Task / Control1 / Control2), the canonical figure-producing script
- `StriatumTaskControl_IntegratedAnalysis.m` — two-group earlier version, ~70% byte-identical to v1
- `SpatioTemporalActivityEvolution.m` — 2100-line single-script omnibus; spatial/temporal evolution figures
- `CCA_striatum_spatial_v2.m` — pairwise CCA (DMS-DLS, DMS-ACC, DLS-ACC) trial-wise and bin-wise
- `MutualInformationStriatum.m` (v1, plug-in MI) and `MutualInformationStriatum_v2.m` (Miller-Madow corrected, zero-aware bins)
- `Nonlinear_Epoch_Decoding.m` — ridge / GPR cross-bin decoding by epoch
- `CrossSpatialBinDecoding.m` — trial-resolved cross-bin ridge decoding
- `simulatePOMDP_striatum.m` — standalone forward simulation of a belief-state RL agent
- `neurontype_classification.m` — small plotting script
- `buildCombinedTensor.m`, `estimate_trialwise_variance.m`, `reorganize_spikes_by_area.m` — helpers used by active code

`Preprocessing/`: `OrganiseStriatumData.m`, `OrganiseStriatumDataIncV1.m` (V1-extended), `OrganiseStriatumDataControl.m`, `ProcessStriatumTask.m` (77 KB), `ProcessStriatumControl.m` (79 KB).

Root utilities: `ReadChannel.m` and `ReadMeta.m` (used by `Synch_NP_VR.m`); `Synch_NP_VR.m` (NPx/VR pulse alignment, run once per recording); `V_allData_1ms.m`, `raw_data_bin.m`, `lick_correction_Vishal.m` (orphaned, hardcoded paths).

CEBRA: `Legacy/cebra_test.py`, `Legacy/cebra_multianimal.py`, `Legacy/cebra_single_multi_comparison.py`, `Legacy/save_for_cebra.m` — none referenced from active code.

---

## 2 · Code that is genuinely superseded or orphaned

### 2.1 Orphan files in `Legacy/` (never called by anything)

Confirmed via grep across all `.m` files. Safe to delete (git preserves history):

`BehaviourOnlyAnalysis.m`, `GPFA_striatum.m`, `OrganiseStriatumDataControl2.m`, `PreprocessStriatumControl2.m`, `ProcessStriatumData.m`, `ProcessStriatumModular.m`, `Run_DMS_ACC_Model.m`, `SummaryProcessingPlotting.m` (122 KB!), `all_task_striatum_plot.m`, `beliefMDP.m`, `calculate_accuracy.m`, `calculate_nll.m`, `cosine_stability_analysis.m`, `ensemble_pca.m`, `joint_DMS_ACC_analysis.m`, `optimality_analysis.m`, `peth_striatum.m`, `save_for_cebra.m`, `striatum_cca.m`, `striatum_plots.m`, `striatum_umap.m`, `supermouse_tca.m`, `tca_with_bic.m`, `temporal_alignment_events.m`. Also `learning_points_task.m` (the active scripts that name it actually inline the equivalent loop rather than calling the file).

### 2.2 Active files superseded by newer counterparts

| Old | Replacement | Recommendation |
|---|---|---|
| `Preprocessing/OrganiseStriatumData.m` | `Preprocessing/OrganiseStriatumDataIncV1.m` | Delete the older file once you confirm IncV1 is used everywhere; the V1 path falls through cleanly when no V1 probe exists |
| `Striatum project/MutualInformationStriatum.m` (plug-in MI) | `MutualInformationStriatum_v2.m` (Miller-Madow + zero-aware bins) | Move v1 to `Legacy/` and rename v2 to drop the suffix. v1 will return upward-biased MI with 5-shuffle bias estimates and should not be re-run |
| `StriatumTaskControl_IntegratedAnalysis.m` | `IntegratedAll_v1.m` (3-group superset, ~70% byte-identical for the shared two-group logic) | Move the two-group file to `Legacy/` once you confirm no figures are still being pulled from it |
| `Legacy/striatum_cca.m` | `CCA_striatum_spatial_v2.m` | Already in Legacy; v2 supersedes it in every dimension (CV, shuffles, learning-point yoking) |

### 2.3 Root utilities to clean

- `Synch_NP_VR.m`, `ReadChannel.m`, `ReadMeta.m` — keep. These run once per Neuropixels recording to produce the synced `*_raw.mat` files consumed by `OrganiseStriatumDataIncV1`.
- `raw_data_bin.m` — has hardcoded `E:\visual_learning` Windows path, depends on `readNPY` and an Excel `info.xls` workflow you've moved past. Delete.
- `V_allData_1ms.m`, `lick_correction_Vishal.m` — orphaned, never referenced. Delete.

### 2.4 Folder rename

`Striatum project/Legacy/` is misleading because about half its contents are load-bearing for the active pipeline. After 2.1 and 2.2 are done, rename the helpers that remain into `Striatum project/src/` (or your preferred name) and reserve `Legacy/` strictly for archived code that will not be called again.

---

## 3 · Bugs and correctness risks (sorted by severity)

### 3.1 Critical — fix immediately

**(a) `ProcessStriatumControl.m` overwrites task data.** Lines 727 and 783 of `Preprocessing/ProcessStriatumControl.m` call `save('preprocessed_data.mat', ...)`. That is the same filename `ProcessStriatumTask.m` saves to. Running Control after Task silently destroys the task preprocessed file. The first save in Control (line ~290) correctly uses `preprocessed_data_control.mat`; the later two writes in the plotting/analysis sections do not. This is a one-line fix per call site and should be the first thing done.

**(b) `fr_threshold` mismatch between Task and Control.** `ProcessStriatumTask.m` filters units at `fr_threshold = 0.02 Hz`; `ProcessStriatumControl.m` filters at `0.05 Hz`. Whatever the right answer is, having different filters silently biases every group comparison downstream. Decide and align.

**(c) Latent reference to undefined variable in both processing files.** Task line ~434 references `dimensionality_stim_all`, Control line ~297 references `pca_stim_dimensionality_all`. Neither is actually saved inside the per-animal loop; both rely on the loop's last-iteration scalar still being in scope. This breaks if the loop early-exits or if anyone adds a `clear` between the loop and the post-loop block.

**(d) Control file uses variables before they exist.** `ProcessStriatumControl.m` lines 408-434 reference `first_idx` and `rest_idx` which are only defined later (lines 443+). On a fresh run this errors.

### 3.2 Important — fix soon

**Trial-alignment index space.** In both `ProcessStriatumTask.m` and `ProcessStriatumControl.m`: `n_trials` is set, then overwritten after a `goodTrials` filter. `change_point_mean` is computed against pre-filter `trialDurations_vr` while everything downstream uses post-filter trial indexing. Silent off-by-one risk.

**`interp1(..., 'extrap')` for NPx-VR alignment.** `OrganiseStriatumData*.m` and the per-trial code all use `interp1(npx_time, idx, vr_time, 'nearest', 'extrap')` with no bounds check. Any VR time outside the NPx recording is snapped to a boundary index without warning. Replace with explicit `assert(min(vr_time) >= npx_time(1) && max(vr_time) <= npx_time(end))` or a logged warning.

**`buildCombinedTensor.m` silently drops mice that lack the required `lp ± 10` window.** Only an `fprintf` is emitted. At minimum, log the dropped animals into a returned struct so downstream code knows what was excluded.

**Trial-shuffle baselines run with no seed.** `IntegratedAll_v1.m`, `StriatumTaskControl_IntegratedAnalysis.m`, `Run_TCA_pipeline.m` (TCA inits and per-area subsampling), `Nonlinear_Epoch_Decoding.m`, `CrossSpatialBinDecoding.m`, `MutualInformationStriatum*.m`, `tsne` calls — none seed `rng`. Runs are not reproducible. Set a `cfg.seed` once and propagate.

**SEM copy-paste bug in `ensemble_analysis.m`.** Around lines 600-602 the SEM for the "bad pre" and "bad post" traces both use `sem(ensemble_activity_good_post{iensemble})` rather than the corresponding `bad_pre`/`bad_post` arrays — error bars on the red and magenta traces are wrong.

**No CV in TCA rank selection.** `tca_with_bic_extended.m` computes BIC on the same data the model was fit to, with reconstruction error reported as the *minimum* over 25 inits (an order statistic that biases toward overfitting). And the `Run_TCA_pipeline` then overrides BIC's choice with a manual `best_n_factors = 5` anyway, making the BIC machinery decorative. Replace with held-out-entry CV (mask random tensor entries, refit, score reconstruction on masked entries).

**No held-out CCA.** All canonical correlations in `CCA_striatum_spatial_v2.m` are in-sample. The shuffle null partially controls for chance but does not address the inflation of `r` due to fitting on the same data being scored.

**Decoder leakage / NaN→0 in `IntegratedAll_v1.m` line 734.** `activity(isnan(activity)) = 0` is applied to the entire LOO tensor before the held-out trial is scored, biasing predictions toward zero in the test fold. The two-group file does this per-bin which is honest; v1 should match.

**NaN handling on initial epoch trials.** In several files `valid_N = sum(~isnan(data(:,1,1)))` assumes the first epoch's first trial is populated for every valid animal, which is false when an animal's `lp` is near a boundary and an epoch is partially clipped.

### 3.3 Statistical / scientific correctness

**Uncorrected multiple comparisons everywhere.** `IntegratedAll_v1.m` section 11 fires 36 cross-modal Pearson tests with no FDR adjustment, then highlights significant panels. `SpatioTemporalActivityEvolution.m` runs 24-cell KS-test grids per panel and stamps significance stars on each. The bin-wise skewness plots use the SEM patch as if it were a significance test. Add Holm or BH-FDR correction across panel-level tests and a cluster-based correction for bin-wise stats.

**Pseudo-replication / independence violations.** Pooled scatter correlations in section 11 of `IntegratedAll_v1.m` treat trials from the same mouse as independent. Cluster-bootstrap by mouse, or report mouse-clustered standard errors, or fit mixed-effects models.

**Modulation-class circularity.** `SpatioTemporalActivityEvolution.m` defines Increaser/Decreaser based on naive-vs-expert delta of session-z-scored activity, then plots and tests on the same trials. Because z-scoring forces the per-neuron session mean to zero, an Increaser must be a Decreaser elsewhere by accounting identity. Define labels on a held-out half and test on the disjoint half.

**Single-shuffle nulls.** `Nonlinear_Epoch_Decoding.m` and `CrossSpatialBinDecoding.m` use one shuffle per cell — variance of the null is unestimated. Raise to ≥100, store the distribution, and report a percentile or empirical p-value rather than the shuffle mean.

**Plug-in MI estimator with 5-shuffle bias correction (`MutualInformationStriatum.m`).** The bias dominates the signal at this sample size. v2 fixes this with Miller-Madow correction; another reason to retire v1.

**Unregularised CCA on PC scores in `CCA_striatum_spatial_v2.m`.** `n_components` is variance-thresholded and grows when sessions have rich spectra; combined with the only-sufficiency check `sum(valid_bins) < max(nc1,nc2) + 5`, you can hit near-rank-deficient regimes where CC1 inflates. Either fix `n_components` or use ridge/regularised CCA.

**Generalised-variance fallback hides failures.** `try/catch` around `pagesvd` in the preprocessing files swallows errors and falls back to a per-trial `svd` loop without logging. `log(sv.^2)` produces `-Inf` for low-rank trials. Surface failures explicitly.

### 3.4 Plotting hygiene against project rules

- Heavy processing is intermixed with plotting in every preprocessing file; about 80% of `ProcessStriatumTask.m` and `ProcessStriatumControl.m` is plotting that should not live in `Preprocessing/`.
- Several plots in `SpatioTemporalActivityEvolution.m` and `IntegratedAll_v1.m` lack y-axis units (Hz on raw FR, cm/s on velocity). The CLAUDE.md rule about axis labels with units is being broken.
- `xline` epoch markers `0, 3, 10, 20` in `SpatioTemporalActivityEvolution.m` are decoupled from the actual `epoch_trials` definition. Change one, the other drifts silently.

---

## 4 · Performance / speedup opportunities

| Where | What | Expected win |
|---|---|---|
| `tca_with_bic_extended.m` | The `parfor` over inits is commented out. With 25 inits × 7 factor counts × ~200 iterations of `cp_nmu` this is the dominant cost of the TCA pipeline | Near-linear speedup (5-8× on a typical workstation) |
| `Nonlinear_Epoch_Decoding.m` and `CrossSpatialBinDecoding.m` | LOO ridge refits the solver from scratch per held-out trial; closed-form LOO via the hat-matrix (PRESS) reduces this to O(1) extra per fold | ~10× |
| `ensemble_analysis.m` lines 1066-1092 and 156-188 | Triple-nested for-loops doing per-trial column correlations; can be vectorised as one batched matrix multiply | 50-100× on the loop body |
| `IntegratedAll_v1.m` LOO Poisson decoder | `setdiff(1:n_tr, t_test)` rebuilt 75,000+ times per animal; replace with a precomputed logical mask and update the rate vector incrementally | ~5× |
| `ProcessStriatumTask/Control.m` | Trial-to-trial correlation block has an `n × t × corr` triple loop that can be reshaped and computed in one `corr` call | 10-20× |
| `ProcessStriatumTask/Control.m` | `pagesvd` already used; the same is not done for `cov`/`pca` — the per-animal PCA on dark/stim activity is computed twice (once in the loop, once after) | Removes a redundant pass |
| `ProcessStriatumTask/Control.m` | `slice_spikes` arrayfun re-slices `final_spikes` 3-4 times per animal; cache one cell-of-trials and mask | Modest, but cleans up |
| `MutualInformationStriatum*.m` | `parfor ianimal` is the easy win; for the `(w_targ, w_source, u)` triple loop, batch `accumarray` over a units dimension | 5-10× |
| `SpatioTemporalActivityEvolution.m` | The z-score loop runs 7+ times in different sections; collapse to one preamble | Cuts ~150 lines and recomputes |
| `IntegratedAll_v1.m` `cat` in loops | Pooled accumulation via `cat(1, …, …)` inside a loop is quadratic; preallocate or push to cell and `cat` once | Significant on larger cohorts |

---

## 5 · Analysis improvements and extensions

### 5.1 Standardise across analysis files

The repo currently uses three different epoch-around-LP conventions: `lp:(lp+9)` (CCA), `(lp+1):(lp+10)` (decoding/MI), `(lp-10):(lp-1)` paired with `lp:(lp+9)` (Integrated). Pick one, write a single `epoch_indices(lp, n_trials, trials_per_epoch)` helper, and call it from every script. Same for the constants: `n_bins = 50`, `bin_size = 4`, landmark bins `(20, 25)`, the `(:,5)` neuron-type column index, the velocity factor `(4*1.25)`, the `lp_window` and `lp_threshold` should all live in a single `cfg.m` consumed by every entry point.

### 5.2 Promote real source code out of `Legacy/`

Create `src/` (or your preferred name) and move the load-bearing helpers there. The set is finite and traceable from the existing dependency graph: `compute_firing_rates`, `cut_data_per_trial`, `extract_binned_spikes`, `find_change_points`, `separate_dark_and_corridor_periods`, `calculate_lick_precision`, `spatial_binning`, `compute_trial_metrics`, `decode_position`, `decode_position_mld`, `processTaskData`, `processControlData`, `runTCAAnalysis`, `tca_with_bic_extended`, `filterDataByArea`, `decode_ensemble_ablation`, plus the factor `plot*` helpers used by `Run_TCA_pipeline`. Then split each preprocessing file into a thin orchestrator that calls these.

### 5.3 Refactor the preprocessing pair into one parameterised function

`ProcessStriatumTask.m` and `ProcessStriatumControl.m` are 95% byte-identical for their first ~290 lines. Extract that into `preprocess_session(all_data, cfg)` parameterised by condition and `fr_threshold`. The ~1700 lines of plotting per file split out into a separate `plot_preprocessing_diagnostics.m` (or several scripts) under a `figures/` script tree.

### 5.4 Refactor `IntegratedAll_v1.m`

The big payoff is extracting four pure functions: `find_learning_points`, `epoch_indices`, `decode_position_poisson` (LOO), `decode_lick_pattern_ridge`. Plus a `compute_unit_stability` for the trial-trial-correlation sections. After that the script collapses to ~150 lines of orchestration and is testable.

### 5.5 Split `SpatioTemporalActivityEvolution.m`

This 2100-line, 104 KB file is the prime candidate for a real split. The obvious decomposition is:

1. `src/preprocess/build_zscored_tensor.m` — one canonical z-score with a documented choice (whole-session vs per-trial vs per-epoch), with a held-out-trials variant for label definition
2. `src/classify/classify_modulation.m` — Increaser/Decreaser/Maintainer with label definition on a separate trial half
3. `src/stats/population_skewness.m`, `src/stats/distribution_ks.m` — vectorised stats with FDR correction
4. Three small plotting scripts: `plot_spatiotemporal_evolution.m`, `plot_distributions.m`, `plot_scatter_kde.m`

### 5.6 Add cross-validation everywhere it's missing

- TCA: held-out-entry CV for rank selection in `tca_with_bic_extended.m`. Drop the manual `best_n_factors = 5` override or comment why it stays.
- CCA: train/test split or k-fold canonical correlations. Fit `(A, B)` on train, project test, correlate. Report held-out CC1 alongside in-sample.
- All ridge decoders: per-fold z-scoring, closed-form LOO PRESS, and `lambda` selected by inner CV rather than fixed at 1.0.
- All shuffle nulls: ≥100 shuffles, store the distribution, report empirical p-values.

### 5.7 Add tests for the load-bearing helpers

The CLAUDE.md TDD rule applies most usefully to: `cut_data_per_trial`, `extract_binned_spikes`, `separate_dark_and_corridor_periods`, `calculate_lick_precision`, `spatial_binning`, `find_change_points`. Synthetic data with known firing rates, known reward zones, known trial counts and known change points. These are the functions where a silent off-by-one in trial alignment would corrupt every downstream analysis.

### 5.8 CEBRA — what's there and how to make it useful

The three Python files plus `save_for_cebra.m` are an early scaffold. They train CEBRA-Time, CEBRA-Behavior, and CEBRA-Hybrid models on lick-error labels (single-animal) and compute multi-session consistency across 8 mice (multi-animal). All three live in `Legacy/` and are not referenced by anything active.

Reasons to revive: CEBRA's contrastive learning produces a stable low-dimensional embedding conditioned on continuous behaviour. It complements TCA in important ways — CEBRA gives a per-timepoint embedding while TCA gives a per-trial factorisation. Multi-session CEBRA with consistency analysis is a particularly strong test of cross-mouse alignment of the neural code.

What needs to change before the CEBRA path produces a publishable result:

- Use **position** (continuous along bins) as the primary CEBRA-Behavior label, not lick errors. Position is the meaningful behavioural axis on this task and what every other analysis decodes against.
- Add lick rate and velocity as secondary labels in a hybrid setup.
- The current `train_test_split` standardises on the full cleaned data before splitting — the scaler sees test data. Move `StandardScaler.fit()` to train only.
- The multi-session script trains for 1000 iterations, the single-session for 10000-15000. Bring multi-session up to ≥10000 or it will not converge.
- Benchmark the embedding by training a linear decoder of position from the embedding on a held-out trial set, and compare the decoding accuracy against the existing ridge / Poisson decoders. This is the canonical CEBRA evaluation.
- Compare embeddings across learning epochs (Naive / Intermediate / Expert) — does the manifold change shape with learning?
- Compute multi-session consistency *separately for DMS, DLS, ACC* to ask whether the cross-area code aligns more strongly within-area than across-area.
- Save the embeddings and consistency scores to `data/processed/cebra/` so they can be plotted from MATLAB or Python without re-fitting.

### 5.9 Scientific extensions worth considering

- **Per-fold ridge λ.** All decoders use `λ = 1.0`. This is almost certainly suboptimal for some areas/epochs; nested-CV λ selection is cheap and would tighten effect sizes.
- **GPR's `try/catch` silently substituting train-mean** in `Nonlinear_Epoch_Decoding.m` should at minimum log how often it fires per area/epoch — could be informing real differences.
- **Mixed-effects modelling** for the cross-modal scatters in `IntegratedAll_v1.m` section 11. Right now pooled correlations have both inflated N (within-mouse non-independence) and no multiple-comparisons correction.
- **Cluster-based permutation tests** for spatial bin × trial inference, to replace per-bin uncorrected SEM patches.
- **Trial-shuffle preserving spatial structure.** Current shuffles permute trials wholesale; consider also a within-spatial-bin trial shuffle to test whether trial-to-trial structure (not just temporal autocorrelation) is what's carrying the signal.
- **Compare CP-NMU TCA against CP-ALS** with sign awareness, to test whether the non-negativity constraint is doing scientific work or just stabilising fits.

---

## 6 · Prioritised task list

Ordered roughly by `(scientific risk × user-facing impact) / effort`. **High-priority items first.**

### M — lab-meeting follow-ups (from 2026-05-25; next meeting 2026-06-04 09:30)

See the Lab meeting log above for full context. These are scientific directions, not bug fixes. **CCA work goes in the Python `striatum_cca` package** (`Striatum project/cca/src/striatum_cca/`); the MATLAB `CCA_striatum_spatial_v3.m` is legacy-sequestered (`Methods/CCA_HH_Adapted.md` §6) — do not extend it.

- **M1. Decoding in the disengaged state.** Split trials into engaged/disengaged via `change_point_mean`, then run position decoding + area-pair CCA on the disengaged segment and compare held-out accuracy / canonical correlations against engaged. The pipeline currently *removes* disengaged data (`dataio.py` truncation; temporal path drops over-long traversals via `temporal_max_trial_ms`). *Quickest first answer: stop excluding the disengaged tail and decode it separately.*
- **M2. Communication subspace, Decreaser units only.** Export `modulation_class` (from `SpatioTemporalActivityEvolution.m`, line ~293) as a per-unit label; add a unit-subset filter to the `striatum_cca` loader (`dataio.py`/`config.py`); run Decreasers-only (`==2`) vs all-units vs Increasers-only. Log/skip area×animal cells with too few units for a stable `canoncorr`.
- **M3. Temporal-bin run (first 3 trials).** *Already implemented* — set `config.bin_mode = "temporal"` (round 8; `temporal_bin_ms`, `temporal_max_trial_ms`) and run focused on the first 3 trials to resolve finer learning dynamics. Tune `temporal_bin_ms`; this is run-and-analyse, not build.
- **M4. Buzsáki temporal-chunking method (lit review).** Find the Buzsáki paper (queued in vault `papers_to_read.md`), read its time-chunk segmentation (likely a velocity threshold — cf. `simulatePOMDP_striatum.m:19`), and write the recipe to `~/Documents/ResearchVault/Methods/`. Feeds M3.

### P0 — fix today

1. **Fix `ProcessStriatumControl.m` filename collision** (lines 727 and 783). Change `'preprocessed_data.mat'` → `'preprocessed_data_control.mat'`. Single-character fix, prevents silent task-data destruction.
2. **Resolve `fr_threshold` mismatch** between Task (0.02) and Control (0.05). One number, one decision, both files updated.
3. **Set a single `rng(seed)` at the top of every script** that uses random state (TCA inits, shuffles, subsampling, t-SNE). Use `cfg.seed`. Make runs reproducible.

### P1 — fix this week

4. **Cross-validate the TCA rank.** Replace the BIC-on-training in `tca_with_bic_extended.m` with held-out-entry CV. Remove the manual `best_n_factors = 5` override or document why it stays.
5. **Fix the SEM copy-paste bug** in `ensemble_analysis.m` (~lines 600-602). Currently the bad-pre and bad-post error bars are computed from the good-post array.
6. **Decoder NaN→0 leakage** in `IntegratedAll_v1.m` line 734. Match the per-bin handling in `StriatumTaskControl_IntegratedAnalysis.m`.
7. **Fix the `change_point_mean` index-space mismatch** in both `ProcessStriatumTask.m` and `ProcessStriatumControl.m` (computed in pre-filter trial space, used in post-filter space).
8. **Fix the latent variable references** in both processing files (`dimensionality_stim_all` / `pca_stim_dimensionality_all` not actually saved in the loop; `first_idx`/`rest_idx` referenced before defined in Control).
9. **Replace `interp1(..., 'extrap')` with a bounds check** in NPx-VR alignment so silent clock drift surfaces as an error.

### P2 — fix this month

10. **Promote load-bearing helpers out of `Legacy/`** into `src/`. The set is enumerated above. Rename the `Legacy/` folder to mean only what it says.
11. **Delete the 24 orphan `Legacy/` files** plus `learning_points_task.m` (also effectively orphan). Plus `V_allData_1ms.m`, `raw_data_bin.m`, `lick_correction_Vishal.m` at the root. Use a feature branch; git preserves history.
12. **Move superseded files to `Legacy/`**: `OrganiseStriatumData.m` (superseded by IncV1), `MutualInformationStriatum.m` (superseded by v2), `StriatumTaskControl_IntegratedAnalysis.m` (superseded by IntegratedAll_v1).
13. **Rename `MutualInformationStriatum_v2.m`** to drop the `_v2` suffix once v1 is archived.
14. **Add multiple-comparisons correction** (BH-FDR) to `IntegratedAll_v1.m` section 11 and to all KS-test grids in `SpatioTemporalActivityEvolution.m`.
15. **Closed-form LOO PRESS** for ridge in `Nonlinear_Epoch_Decoding.m`, `CrossSpatialBinDecoding.m`, and the LOO Poisson loop in `IntegratedAll_v1.m`. ~10× speedup for free.
16. **Enable `parfor`** on TCA inits (`tca_with_bic_extended.m`), on the per-animal MI loop, and on the outer trial/bin loops in `CCA_striatum_spatial_v2.m`. Seed RNG per worker.
17. **Cluster-bootstrap or mixed-effects** for the cross-modal scatters in section 11.
18. **Vectorise the ensemble-correlation triple loops** in `ensemble_analysis.m`.

### P3 — refactor / extend (this quarter)

19. **Extract `preprocess_session(all_data, cfg)`** from `ProcessStriatumTask.m` and `ProcessStriatumControl.m`. Split the post-line-290 plotting into separate scripts.
20. **Split `SpatioTemporalActivityEvolution.m`** into 3 plotting scripts and 3-4 src modules with tests.
21. **Extract `find_learning_points`, `epoch_indices`, `decode_position_poisson`, `decode_lick_pattern_ridge`** as pure functions; refactor `IntegratedAll_v1.m` to call them.
22. **Single shared `cfg.m`** consumed by every entry-point. All hardcoded magic numbers (n_bins, bin_size, landmark bins, neuron-type column, velocity factor, lp_window, lp_threshold, paths) live there.
23. **Add tests** for `cut_data_per_trial`, `extract_binned_spikes`, `separate_dark_and_corridor_periods`, `calculate_lick_precision`, `spatial_binning`, `find_change_points` against synthetic data.
24. **Define modulation classes (`SpatioTemporalActivityEvolution.m`) on held-out trials** to remove the circularity.
25. **Replace bin-wise SEM-as-significance plots** with cluster-based permutation tests.
26. **Fix the `xline(0,3,10,20)` epoch markers** to be derived from `epoch_trials` so they don't drift.
27. **Standardise the epoch-around-LP convention** across CCA, decoding, MI, and Integrated scripts.

### P4 — extensions (when ready)

28. **Resurrect the CEBRA pipeline** with position as the primary contrastive label, multi-area consistency analysis, and a held-out linear-decoder benchmark. See section 5.8.
29. **Per-fold ridge λ via nested CV** in all decoders.
30. **Compare CP-NMU vs CP-ALS** with sign-aware factor matching.
31. **Add CEBRA-vs-TCA-vs-PCA benchmark** on held-out trial decoding.
32. **Within-bin trial shuffle** as a second null in addition to the wholesale trial shuffle.
33. **Add a `summary_numbers_v1`-style first-line bug review** — line 1 of `summary_numbers.m` uses `size(x, 2)` on `is_dms` to count units; if `is_dms` is stored as a column vector this returns 1 (not n_units). Compare against the correct `length(is_dms)` used at line 47.

---

## 7 · Notes on what is already good

To avoid only pointing at problems: the project has a lot going for it. The TCA pipeline structure (`Run_TCA_pipeline` → `buildCombinedTensor` → `runTCAAnalysis` → `ensemble_analysis`) is a sensible decomposition. `MutualInformationStriatum_v2.m` already implements bias correction and zero-aware binning, which is more rigorous than most published analyses of this kind. `CCA_striatum_spatial_v2.m` is a serious analysis with shuffle nulls and learning-point yoking, and the precession-index idea is genuinely interesting. `Nonlinear_Epoch_Decoding.m` correctly z-scores per-fold (a leakage trap most code falls into). `buildCombinedTensor.m` is defensive about field mismatches between Task and Control. The CLAUDE.md project conventions are clear and well-considered. The orphan-file count (24) and the redundancy patterns (v1/v2, IntegratedAll vs StriatumTaskControl) are normal accumulation; the project is not in unusual shape, just ready for a cleanup pass.
