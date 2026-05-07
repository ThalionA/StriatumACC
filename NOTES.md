# StriatumACC — Project Audit & Priority List

_Audit completed 2026-05-07. Covers all `.m` files in active code paths, the `Preprocessing/` directory, root-level utilities, the entire `Striatum project/Legacy/` folder, and the three Python CEBRA scripts._

## Update log

**2026-05-07 (later same day):** acted on the user's choices.

- Filename collision fixed in `ProcessStriatumControl.m` (lines 727 and 783 now write to `preprocessed_data_control.mat`).
- `fr_threshold` aligned to `0.02 Hz` across both Task and Control preprocessing.
- SEM copy-paste fixed in `ensemble_analysis.m` (lines 600-602 now use the matching `bad_pre`/`bad_post` arrays).
- **Active code consolidated into `Striatum project/`.** The `Preprocessing/` subfolder and the load-bearing helpers from `Legacy/` were promoted up. Forty-six `.m` files moved in total: 5 from `Preprocessing/` and 41 from `Legacy/` (including the three transitive deps `scalar_to_quantile`, `fit_tca_model`, `cp_nmu_ortho`). The `Preprocessing/` directory is now empty (the sandbox could not delete it; do `rmdir "Striatum project/Preprocessing"` from your terminal). Twenty-six orphan `.m` files plus all figure artifacts remain in `Legacy/` per the user's instruction to leave orphans in place.
- **CEBRA pipeline rewritten.** The three Legacy CEBRA scripts (`cebra_test.py`, `cebra_multianimal.py`, `cebra_single_multi_comparison.py`) and the old `save_for_cebra.m` are superseded by:
  - `Striatum project/save_for_cebra.m` — exporter that writes per-mouse `.mat` files with multi-dimensional behavioural labels (position, per-bin lick rate, per-trial lick errors, velocity, learning point, area labels, neuron types).
  - `Striatum project/cebra_analysis.py` — unified pipeline with multi-dim contrastive labels, **per-fold StandardScaler** (fixes the train/test leakage in the legacy code), trial-wise train/test split, **held-out ridge decoder of position from the embedding** (the canonical CEBRA evaluation), per-area refits (DMS-only / DLS-only / ACC-only), multi-session model with consistency scores, and outputs in both `.npz` (Python) and `.mat` (MATLAB) so existing plotting code can consume them.
- Remaining items from the priority list below are unchanged.

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
