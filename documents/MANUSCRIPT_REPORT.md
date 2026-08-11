# StriatumACC — Manuscript Consolidation Report

**Date:** 2026-07-30 · **Scope:** every `.m` and `.py` file in the repository (≈34 k lines MATLAB, ≈25 k lines Python), all sub-project logs, all result artefacts on disk, and the manuscript draft.

**Companion documents**
- [`MANUSCRIPT_INVENTORY.md`](MANUSCRIPT_INVENTORY.md) — all 290 analyses, one structured entry each (question, method as coded, statistic, inferential unit, outputs, soundness).
- [`MANUSCRIPT_RESULTS.md`](MANUSCRIPT_RESULTS.md) — all 115 recorded outcomes with verbatim values, sources, and whether the backing artefact still exists.

---

## 1. Bottom line

There is a real manuscript draft — [`Striatum-ACC paper.docx`](Striatum-ACC%20paper.docx) — with a clear two-stage narrative and a written Methods section. **The narrative is good and the data can probably support it. But almost none of its four figures is currently backed by a citable number, and its central quantitative claim rests on a measurement artefact that runs in the same direction as the claim.**

Three findings dominate everything else:

1. **Running speed is inside the dependent variable, and the estimator is biased.** Spatial firing rate = spikes ÷ dwell-time, and dwell-time = bin width ÷ speed. Occupancy is computed as *last-minus-first* VR sample — `(k−1)·dt` for `k` samples — instead of `k·dt`, inflating every rate by `k/(k−1)`, i.e. **17–50 %**, and the inflation is *speed-dependent*. The paper's own Fig 1C is "inverse relationship of licks and velocity", and the paper's premise is that mice speed up as they learn. So the paper's headline effects — stability rises, decoding improves, ensembles emerge — are each exactly what a stereotyping speed profile would produce through the denominator alone. A repository-wide grep for speed matching, speed regression, or speed as a covariate returns **nothing**.

2. **Nothing the manuscript needs has a recorded outcome.** Position decoding (Fig 2B/C), stability (Fig 2A), TCA rank and ensemble statistics (Fig 3), the lick decoder, mutual information, and the ensemble ablation (Fig 4C) have **no numbers anywhere** — not in a log, not in a CSV, not in a `.mat`. Every existing figure came from a live MATLAB workspace and cannot be regenerated. The root `.gitignore` excludes `*.csv *.pkl *.npz *.png *.svg *.mat *.txt`, so essentially no result is tracked. The one exception is `popsim/data/generated/recovery_benchmark.json`.

3. **The best work in the repository is not in the paper.** The `cca/` + `tcca/` communication-subspace arms (≈300 tests, ground-truth validated by `popsim`, pre-registered in `PREDICTIONS.md`) have delivered a well-powered, cross-method-replicated **null**. The manuscript does not mention them. Conversely the paper's four figures rest on the three arms carrying the most severe defects.

### Do these five things first

| # | Action | Effort |
|---|---|---|
| 1 | Fix the occupancy denominator in `spatial_binning.m:45`, regenerate both preprocessing caches, and report the size of the change | half a day + rerun |
| 2 | Fix the 2× velocity error (`(4*1.25)` against 2.5 cm bins) in `MutualInformationStriatum_v2.m:52`, `Nonlinear_Epoch_Decoding.m:37`, `CrossSpatialBinDecoding.m:33`, `save_for_cebra.m:29`, and `bin_size=5` in both decoders | 1 hour |
| 3 | Add an **LP-shuffled null** as the standard reference for every "tracks learning" claim | 1 day, applies to every arm |
| 4 | Persist every analysis output to CSV with a provenance stamp (git SHA, cfg hash, date, `fr_threshold`); un-ignore results in `.gitignore` | half a day |
| 5 | Restore the leave-one-out template in `decode_ensemble_ablation.m:60-69` (Fig 4C currently trains on its test trials) | 15 minutes |

---

## 2. The manuscript as drafted vs what the code supports

Draft title (preferred of three): *"A Stable Spatial Map in the Striato-Cingulate Network Precedes the Emergence of Goal-Directed Action Ensembles"*

Narrative: a two-stage process — rapid stabilisation of a network-wide spatial representation, followed by slower emergence of functionally-specific ensembles that map onto RL computational variables and drive optimal behaviour.

**The paper is DMS / DLS / ACC only.** This is the right call and it resolves a lot: V1 (5 animals), CA1 (3), and DG (3, 18 units total, one animal contributing a single unit) cannot support group inference and are correctly absent from the draft. Keep them out.

### Figure-readiness

| Panel | Claim | Code | Verdict |
|---|---|---|---|
| **1A** corridor schematic | — | — | ✅ illustration |
| **1B** lick heatmap | licks restrict to VZ→RZ window | `IntegratedAll_v1.m` S5, `plotAlignedBehavior.m` | ✅ **strongest panel in the paper.** `plotAlignedBehavior.m` already uses animal-level SEM with a correct per-slot non-NaN count |
| **1C** licks vs velocity | speed/lick trade-off | `IntegratedAll_v1.m:324,441`, `ensemble_analysis.m:643` | 🔴 **velocity is wrong by 2×** in four live scripts; and the rate denominator bias is speed-dependent. Fix before plotting |
| **1D** strategies → reward rate, optimum | optimality argument | `legacy/optimality_analysis.m` (833 lines) | 🔴 **orphaned in `legacy/`.** The normative-optimum panel has no live code. Either resurrect or drop the optimality claim |
| **1E** RL agent reproduces behaviour | model matches licking | `rl_model/` (v5 fits, 16 mice) | 🟢 **the one arm with real recorded outcomes.** Lick channel: held-out CV gain **+0.57 nats/bin, 16/16 mice positive**; per-epoch lick change reproduced for 11/16 (mean Δr 0.70). Velocity is the weak channel (6/16, Δr 0.37; held-out CV −0.07). Report the lick channel, be honest about velocity |
| **2A** trial-to-trial stability rises | spatial map stabilises early | `IntegratedAll_v1.m` S6 | 🟠 decoder-free and conceptually clean, but the null is **one unseeded `randperm`** plotted as a mean±SEM band over *units*, so no p-value is attachable; edge windows (3 and 4 trials) fall inside Naive only, making Naive noisier by construction. Publish the **Hierarchical** (animal-level) variant only |
| **2B** decoding accuracy improves | spatial code sharpens | `IntegratedAll_v1.m` S7A | 🟢 **the decoder core is sound** — genuine leave-one-trial-out, RMSE normalised by bin count, entropy by log₂(bins) so 100-bin and 50-bin groups are comparable, and a matched tuning-shuffle. Three fixes: train the template *within* epoch (currently session-global, so epoch differences partly measure similarity to a session average), move z-scoring inside the fold, and fix `plot_decoder_accuracy_vs_chance.m` (both live call sites pass unit count where `n_pos_bins` is expected, drawing chance at 1/50 instead of 1/100) |
| **2C** decoding vs learning rate | dissociation | `visualize_performance_correlations.m` | 🟠 supplementary at best; needs the best-bootstrap selection removed and animal-level inference |
| **3A** ensembles via TCA | population components | `Run_TCA_pipeline.m`, `tca_with_bic_extended.m` | 🔴 **rank is hardcoded.** `Run_TCA_pipeline.m:1020-1021` sets `best_n_factors = 5` unconditionally *after* BIC selection, then re-plots the diagnostics figure with the xline labelled "Selected = 5". BIC cannot select a rank anyway: n = 1630×50×30 = 2.45 M tensor entries treated as independent, so the penalty is negligible and BIC is a monotone transform of RSS — and the RSS fed to it is the **minimum over 25 inits**, an order statistic biased more at higher rank. Declare rank 5 a priori and delete the BIC figure, or wire in `tca_with_cv.m` (which already implements held-out refits and is simply not called) |
| **3B** ensembles have distinct spatial tuning | — | `plotSpatialFactors.m` | 🟠 plausible; needs animal-level inference |
| **3C** ensembles emerge with different time courses | the slow stage | `ensemble_analysis.m` | 🔴 **circular**: the tensor's trial axis is built as slots 1:10, lp−10:lp−1, lp+1:lp+10, so pre/post-LP blocks differ in behaviour *by construction*, and no LP-shuffled null exists anywhere in the repository |
| **3D** anatomical distribution | area specialisation | `plotNeuronFactorsByArea.m` | 🔴 one-way ANOVA on 52–568 **units** pooled across 13 mice with no animal term. `plotFactorHeatmapsByMouseArea.m` already computes the per-mouse means that should replace it |
| **3E** cell-type composition | MSN/FSN/TAN | `neurontype_classification.m` | 🔴 **striatal criteria applied to cortical units.** MSN/FSN/TAN/UIN are assigned by hardcoded thresholds to every probe-1 unit including all ACC units. TCA cell-type panels apply the labels to the whole tensor with no striatum-only mask. Restrict to DMS/DLS and say so in every panel title |
| **4A** RL latents | value, RPE, precision | `rl_model/` | 🟢 latents recover well (value 0.996, RPE 0.994, lick_rate 0.999, precision 0.938). Caveat: `precision` keeps only ~6 % of its variance after per-bin demeaning, so it is structurally uninformative as a spatial regressor — the project already flags this |
| **4B** ensembles encode computational variables | the mechanistic bridge | `rl_model/rl_model/neural_encoding.py` | 🔴 **the analysis exists but has never been run on current latents.** `rl_model/UNDERSTANDING.md`'s own Won't-Do says "the neural regression itself (Fig 4 analysis) — downstream". `encoding_v5/`/`v6/` hold per-mouse `.npz` but no dR² table. **This is the paper's mechanistic claim and it is unbuilt** |
| **4C** ensemble ablation | ensembles are necessary | `decode_ensemble_ablation.m` | 🔴 **trains on its test trials.** At lines 60-69 the correct LOO line is commented out — `% mu = mean(tensor(idx,:,trn),3,'omitnan');` — and replaced by `mu = mean(tensor(idx,:,21:30),3,'omitnan');`. `trn` is computed on the line above and never used. So for test trials 21:30 the trial is inside its own template, and for trials 1:20 an expert template is applied to naive trials. **The learning effect and the leakage point the same way.** `cfg.cv_type='leave-one-out'` and `cfg.kfolds=5` have no implementing branch |

**Summary: 3 panels green, 4 amber, 8 red.** The two unbuilt-or-broken panels that matter most are **4B** (the mechanistic bridge, never run) and **4C** (leakage). The narrative's second stage — "ensembles map onto computational variables" — is currently an aspiration.

---

## 3. The pipeline

```
SpikeGLX .lf.bin + VirMEN .csv
   └─ Synch_NP_VR.m           pulse-matched piecewise-affine clock alignment → VR_times_synched
   └─ raw_data_bin.m          1 ms spike counts, Phy 'good' units only    → RawData/<id>_raw.mat  [saved MANUALLY]
        └─ OrganiseStriatumDataIncV1.m   depth→area assignment from 2 CSVs → all_data.mat
             └─ ProcessStriatumTask.m    trial cutting, spatial binning, LP detection, fr_threshold 0.02 Hz
                  → processed_data/preprocessed_data{2p5cm,5cm}.mat        ← the hub every analysis reads
                       ├─ Run_TCA_pipeline → buildCombinedTensor → cp_nmu_ortho → ensemble_analysis
                       ├─ IntegratedAll_v1.m  (Fig 1B/1C, 2A, 2B, and a 46-test scatter grid)
                       ├─ SpatioTemporalActivityEvolution.m  (modulation classes, KS grids w/ BH-FDR)
                       ├─ Nonlinear_Epoch_Decoding / CrossSpatialBinDecoding / MutualInformation_v2
                       ├─ cca/ (Python, spatial CCA)  and  tcca/ (Python, temporal CCA)
                       ├─ rl_model/ (Python+JAX, belief-state actor-critic)
                       └─ save_for_cebra → cebra_analysis.py
```

**Geometry, as actually coded.** Corridor 200 a.u. = 250 cm (1 a.u. = 1.25 cm). Landmark (VZ) at 80 a.u. = **100 cm**; reward zone 100–135 a.u. = **125–169 cm**. Bins: 2 a.u. = 2.5 cm → 100 bins (also a legacy 4 a.u./5 cm → 50-bin preprocessing, which TCA still uses).

Two unit errors I verified directly:
- `project_cfg.m:73-74` — `cfg.visual_zone_cm = 80` and `cfg.reward_zone_cm = 100` are **a.u. values labelled cm**. `ProcessStriatumTask.m:10-13` has it right (`reward_zone_start_cm = 125`). Any panel captioned "reward zone at 100 cm" is off by 25 %.
- `cfg.max_bin = 60` is commented "RZ + ~10 bins", but bin 60 = 120 a.u. and the reward zone runs 100–135 a.u. — the truncation **clips the last third of the reward zone**.
- Also `cca/UNDERSTANDING.md:30` states "200 cm corridor"; it is 250 cm.
- And `raw_data_bin.m:58-60` uses reward-zone end 130 a.u. where `ProcessStriatumTask.m` uses 135.

**Cohort.** 16 task mice, 1540 units (DMS 367, DLS 490, ACC 683; MSN 1095, FSN 136, TAN 119). 14 reach a learning point (mean LP ≈ 41); 12 disengage by trial 111. Control 1 = blank-corridor habituation, 6 mice, *spatial*. Control 2 = dark habituation, stationary, 7 mice, *temporal* — and it rests on a **2025-01-28 cache** with no V1/CA1/DG, no cell types, and a different firing-rate floor.

**Per-area animal coverage for the analysed (learner) cohort:** DMS 13, DLS 10, ACC 14, V1 4, CA1 2. `IntegratedAll_v1.m` S12 prints counts over all 16 including non-learners, **overstating V1 by 63 % and CA1 by 156 %** relative to the cohort in the figures. A Methods table copied from that printout would be wrong.

---

## 4. The fourteen arms

Detail for each is in [`MANUSCRIPT_INVENTORY.md`](MANUSCRIPT_INVENTORY.md). Severity across all 290 analyses: **45 fatal, 142 major, 91 minor, 12 ok**; status: 208 active, 21 superseded, 21 legacy, 24 scaffold, 8 gated, 8 decorative.

| Arm | What it does | State |
|---|---|---|
| **Preprocessing** (`Synch_NP_VR`, `raw_data_bin`, `Organise*`, `ProcessStriatum*`) | raw → area-assigned, spatially-binned rates | Foundational and **unreproducible**: `raw_data_bin.m` never saves its own outputs (manual `save`), its four dependencies (`readNPY`, `gaussFilter`, `gaussFilter_border`, `interpolate_binned_values`) are **absent from the repo**, and its `minsend=140` constant contradicts the archive (1212 has 1.14e7 columns ⇒ was run at 190). `Synch_NP_VR.m` has three unguarded extrapolation paths and reuses a stale drift rate `r` across loop iterations |
| **Per-session processing** | dimensionality, stability, decoding calls | `calc_dim` cumsums PCA's *coefficient* matrix instead of `explained`, so every per-area dimensionality is stored as `0×0` empty. The stim-vs-dark t-test references `dimensionality_stim_all`, never defined |
| **TCA** | supermouse CP-NMU, rank, factors | Rank hardcoded; BIC decorative; RNG unseeded (`rng(0)` commented out at line 3); the per-area balancing `randperm` picks *which* 52 units the whole decomposition is fitted to, once, unseeded. `tca_outputs.mat` is stale in six verified ways |
| **Ensembles** | argmax assignment, ablation | Ablation leaks (§2, Fig 4C). t-SNE colours points by the argmax of the very coordinates being embedded |
| **IntegratedAll_v1** | the canonical three-group figure script | Contains the paper's best panels (S5, S6-Hierarchical, S7A) *and* its worst (S11: 46 uncorrected Pearson tests on trials pooled across animals from overlapping 5-trial windows, one regression line fitted across two epoch clouds, significance encoded as an axis line-width) |
| **SpatioTemporal** | modulation classes, KS grids | **The one arm that got multiple comparisons right** — BH-FDR via `annotate_ks_panels_fdr.m` at four sites — and the classifier was properly de-circularised onto disjoint trial halves. No recorded outcomes though |
| **MATLAB CCA** (v2, v3) | inter-area canonical correlation | **Delete both.** v2 has zero residualisation (grep-verified), so it measures shared position tuning, not communication. v3 is correctly specified but calls `v5_residualise`/`v5_pca`/`v5_cca`, which do not exist — it cannot run |
| **Decoding + MI** | cross-bin ridge/GPR, mutual information | MI significance is the 95th percentile of **25** shuffles, per unit per window, uncorrected across thousands of tests. MI uses its own LP convention (`lp+9`), so its "expert" epoch is nine trials later than every other script's. Cross-area MI is computed on **PC1 of raw rate** — the position ramp — and read as "flow", but MI is symmetric |
| **Models / legacy** | POMDP forward sim, optimality | `simulatePOMDP_striatum.m` is forward simulation only, never fitted. `optimality_analysis.m` (Fig 1D) is orphaned in `legacy/` |
| **`cca/`** (Python) | spatial communication subspaces | **Methodologically the best arm.** Residualisation, per-epoch PCA with k from the sample budget, 5-fold whole-trial CV, held-out CC1 as effect size with in-sample used only as the permutation statistic, two nulls, non-learners retained as a yoked control, D1–D12 design record, 135 tests |
| **`tcca/`** (Python) | temporal replication | 165 tests, numeric layer ported verbatim from a validated sibling, bin width chosen on evidence, reproduced a lost cohort run exactly, then used it to *retract* its sibling's claim. See §6 for a correction |
| **`lfp/`** | 384-ch voltage integrity | **Exemplary self-gating.** Refuses to decode or run CCA until provenance is established; quarantined its own prior outputs; corrected its own false "99 % empty" diagnosis; withheld area labels for the one animal with unverified probe identity. Currently a methods/negative result, not a figure |
| **`rl_model/`** | belief-state actor-critic | Real recorded outcomes, honest parameter-recovery gates, a correctness-fix log. The neural-encoding bridge (Fig 4B) is built but never run on current latents; the real model ladder was never run |
| **`popsim/`** | ground-truth validation | **10/10 scenarios recovered**, including `mediated` (0.87 → 0.24 under partial CCA) and `common_input` (0.74 → 0.19), and a bridge that drives the *real* `striatum_cca` code on simulated data. The only git-tracked result table in the repository. Cite it in Methods |
| **CEBRA** | contrastive embedding | Config and `.npz` tracked; but conditioning a contrastive embedding on behaviour and then evaluating it by decoding position is close to circular as a change-detector. Drop |

---

## 5. Blocking issues, ranked

Twelve issues were classed blocking-for-publication. Ranked by how much of the paper each one moves.

1. **Running speed inside the dependent variable, nowhere controlled** (§1). Affects Fig 1C, 2A, 2B, 3C — the whole positive half.
2. **No LP-shuffled null.** Every learning axis is defined by the LP; the Expert window in `IntegratedAll_v1.m` S4.3 *begins at* the LP, which is by definition the first sustained sub-threshold trial, so the Naive→Expert behavioural step is partly selection on the plotted quantity. Nothing distinguishes "trial factor 3 rises after the LP" from "any quantity with a monotone session trend, aligned to a change point chosen from a correlated behavioural variable, shows a step there". Cheapest high-value fix in this review.
3. **Fig 4C trains on its test trials** (§2). Fifteen-minute fix, but until then the ablation result is uninterpretable.
4. **TCA rank hardcoded while the figure says BIC selected it** (§2, Fig 3A). This is a false methods statement in a figure.
5. **Naive-vs-Expert confounds learning with time-in-session, satiety and electrode drift.** Expert sits 30–80 trials later in a 2–3 hour session. There is **no ISI-violation, amplitude, presence-ratio, contamination or drift metric anywhere** in the pipeline; unit inclusion is Phy manual label + depth interval + `fr_threshold` (which removes 2 of 2039 units and is therefore inert). The depth gate silently discards ~half the Phy-good clusters (animal 523: 268 → 133) with no accounting. The natural control — the 5 s dark ITI, already cached as `temp_binned_dark_fr` — is barely used.
6. **Inferential unit is wrong in most headline analyses.** Animal-level n is 13–14. Unit-, trial-, bin-, dimension- and bootstrap-level replication appear throughout; with n in the hundreds any nonzero difference reaches p < 0.05.
7. **Cached artefacts predate the code that made them.** `tca_outputs.mat` (2026-05-14) records `analysis_mode='task_only'`, zero control animals, DG still included, 1630 unbalanced units (predating the subsampling block), an input file that no longer exists, and an un-normalised model. **No number traceable to it is safe to cite.** The task preprocessing cache (2026-05-23) predates `rng(42,'twister')` being added to `ProcessStriatumTask.m:8` on 2026-05-24, so its 1000-draw lick shuffles — and therefore *every learning point and epoch boundary* — came from an unseeded run. `ProcessStriatumControl.m` has no `rng` call at all.
8. **Three LP conventions and three epoch conventions coexist.** Project rule → 14 learners, mean LP 41. The `window=7`-on-truncated-trace rule in `Nonlinear_Epoch_Decoding.m:44` / `CrossSpatialBinDecoding.m:42` → **8 learners, mean LP 52**, with animal 1 moving 22 → 85. MI v2 adds +9. So the same mouse is a learner in one figure and a non-learner in another, and "expert" means different trials in different panels. Also: Naive is absolute (1:10) while Intermediate is LP-relative with **no overlap guard** — mouse 823 (LP 14) has Naive and Intermediate sharing 7 of 10 trials.
9. **Controls are no-task baselines, not controls for any claim the paper makes.** Both are between-animal; they differ from Task in cue, reward, learning and (Control 2) locomotion simultaneously; and having no LP they cannot control for time-in-session or drift *within* task animals, which is exactly what the Naive-vs-Expert contrast needs. Caption them as a task-requirement baseline. Note Control 1's cache contains **446 V1/CA1/DG units that are silently included in every "all units" computation while invisible to area analyses**, which is why No-V1 ablation curves are byte-identical to All in the control columns.
10. **`S11` cross-modal grid is circular and uncorrected.** A pure between-epoch mean shift manufactures a strong pooled correlation with no within-epoch relationship. Drop as an inferential figure. Its SVGs on disk date from 2026-05-07 — before CA1 existed in the project.
11. **Cell-type labels from striatal criteria applied to cortical and hippocampal units** (§2, Fig 3E).
12. **Single unseeded permutation plotted as a mean±SEM band** (§2, Fig 2A). One draw gives no null variance, so no test can be attached.

---

## 6. A correction the project needs: `tcca`'s residualisation

`tcca/README.md` and the 2026-07-28 `tcca/NOTES.md` entry both describe the temporal arm as **"residual + partial CCA"**, and the entry's causal explanation for why round 8's temporal reorientation vanished is that *"a large part of round 8's temporal rotation was plausibly shared position/time tuning, which residual CCA removes."*

**The code does not do trial-mean residualisation.** I verified this directly:
- `cfg.subtract_trial_mean` appears **only** at `config.py:154`. It is read nowhere in `src/` or `scripts/`.
- `core.residualise` is called **only** from `tests/test_core.py`.
- The live path is `run_epochs → runner → subspace_window.window_subspace`, which does `partial.partial_out(X, Z)` and per-unit z-scoring — nothing else.

This is a defensible design consequence, not a bug: in the temporal representation traversals have variable length and are concatenated into a flat timeline, so there is no common trial×bin axis to average over. But three things follow.

- **The empirical result stands.** Rotation-minus-floor at animal level (n=10): mean +1.92°, median +1.03°, Wilcoxon p=0.38, positive mean carried by one animal (A10 +22.4°; without it −0.35°). I re-verified this from `epoch_cross.csv`: striatal triangle 78/140 = 55.7 %, n=10 mean +1.922, median +1.027, range −9.96 (A6) to +22.44 (A10). That is a measurement, and at n=10 the one-sided Wilcoxon floor is 0.001, so it is a **powered null, not a p-floor**.
- **The explanation does not.** Shared position/time tuning is removed only insofar as the third-area regression absorbs it — and position tuning shared between DMS and DLS is removed only if ACC also carries it and ACC is in Z. The actual differences between round 8 and this run are held-out CV vs in-sample permutation, partial-out vs none, animal-level vs pair-config aggregation, and 25 ms vs 40/20 ms bins. The most parsimonious reading is that **honest cross-validation and the correct inferential unit** account for 90 % → n.s., which is a cleaner and more useful methodological lesson than "residualisation removed it".
- **Fix the docs and the dead knob** before either arm is written up, and rewrite the Methods sentence for the temporal arm as: per-unit z-scoring over the engaged reference, plus partialling out every other simultaneously recorded area; trial-mean residualisation is not applicable to the temporal representation.

---

## 7. What is solid

Being fair — this is a well-run project by the standards of what it does with its own errors.

- **`popsim` ground-truth validation.** Coupling specified at the latent level so the subspace is exactly known; 10 scenarios including `mediated`, `common_input`, `rotated_subspace`, `noise_correlation`; 10/10 recovered; a bridge that drives the *real* analysis code. `noise_correlation` is the instructive one: population CCA 0.999 while latent coupling is 0.197. This is what makes the CCA null credible rather than merely underpowered.
- **The `cca/` design** (D1–D12 with a versioned edit log): held-out CC1 as effect size, in-sample only as the permutation statistic with the ~2× bias documented, whole-trial folds, missing bins dropped rather than imputed with the reason given, non-learners retained as a yoked control.
- **`epoch_stats.py`** is careful code: RM-ANOVA with epoch as a within-animal factor plus Holm step-down, and a `min_n=6` guard on `wilcoxon_vs0` with the reason in the docstring (signed-rank cannot reach two-sided p < 0.0625 at n < 6).
- **The `PREDICTIONS.md` practice.** Falsifiable priors registered before consequential runs, then scored honestly — including the lesson that a falsifier had been set *at the chance rate* (P = 0.75 under an exchangeable null, so ≥7/9 could not discriminate in either direction).
- **The LP fix.** The old `movsum` rule returned a window-*start* trial that could itself be above threshold (z = +1.28 for animal 3). Fixed in both languages and verified to give identical LPs for all 16 animals — independent reimplementation with cross-language agreement.
- **The `lfp/` arm's conduct** (§4). Correcting a confident wrong claim with a full error record is exactly right.
- **BH-FDR is wired in where the design most needs it** — four sites in `SpatioTemporalActivityEvolution.m`. The gap is reuse, not absence.
- **Three panels already use the right inferential unit** and should be promoted: `plotFactorHeatmapsByMouseArea.m`, `plotAlignedBehavior.m`, `visualize_trial_evolution.m` (which correctly collapses bootstrap resamples *within* animal before the across-animal SEM).
- **Correct negative decisions:** dropping DG; restricting group inference to the striatal triangle; recording the original "DMS leads early, DLS later" hypothesis as unsupported rather than quietly reframing it.

---

## 8. The orphan asset: there is a second paper here

The `cca/` + `tcca/` + `popsim/` arms answer a different question from the draft, and answer it well:

> Residual inter-areal communication in the striato-cingulate network is real but modest (held-out CC1 ≈ 0.10–0.30, roughly half of learner animals individually significant), and across learning it does **not** change in strength, does **not** acquire a direction (IFI ≈ 0), does **not** de-sparsify (Gini flat 0.4–0.5), and does **not** reorient.

Every one of those is animal-level, cross-validated, and replicated across an independent temporal implementation. Paired with the draft's local-refinement result, "learning refines local representations without reorganising inter-areal communication" is a publishable, falsifiable claim.

Two reporting rules from that arm must propagate before anything is quoted:
- **Lead with medians.** DLS-ACC intermediate: mean 0.134 vs **median 0.017** — the typical animal shows essentially no communication. DMS-DLS expert: 0.190 vs 0.092.
- **Never pool a proportion across pairs.** All-pairs above-floor = 61 %, p < 0.001 — manufactured by eight pairs resting on **one animal each** (CA1-*, DG-*) that score 83–100 % by noise. Backing animals: DMS-ACC 10, DMS-DLS 7, DLS-ACC 7, V1-DMS/V1-ACC 3, V1-DLS/CA1-V1 2, the other eight **1**.

Three further issues to fix in that arm before it is written up: `aggregate.per_animal_matrix` drops animals lacking a significant dimension in *any* epoch (selection on the dependent variable — DMS-ACC 9→6, DMS-DLS 7→5, CA1-DMS 1→0), the primary null (circshift) was chosen partly because it returned 2.6× more significant dimensions (533 vs 201) and IFI signs flip between nulls, and `cca/RESULTS.md` carries its own staleness banner and must not be a Methods source.

---

## 9. Gaps: never run, or lost

**Never run** — `analyze_epochs.py` (the driver that would apply `paired_stats`/`mixed_effects`); the real RL model ladder (only a 3-mouse synthetic demo exists); the RL neural-encoding run on current latents (Fig 4B); CEBRA (promised tables never produced); and all four 2026-05-25 lab-meeting asks — engaged-vs-disengaged decoding (M1), Decreaser-only subspace (M2), first-3-trials temporal binning (M3), Buzsáki chunking recipe (M4).

**Phantom** — `cca/NOTES.md` round 17 and `ResearchVault/Methods/CCA_HH_Adapted.md` §6.2 both describe a complete running-state temporal CCA arm with "169 tests". `segments.py`, `lagged_temporal.py` and `run_temporal_runstate.py` **do not exist on disk**. `tcca/NOTES.md` already flags this; the vault Methods note still asserts it and needs correcting.

**Lost** — every LFP integrity summary CSV (`sanity_summary.csv`, `sanity_timing_summary.csv`, `signal_identity_summary.csv`) is absent from disk while `lfp/results/README.md` declares them the defensible outputs; the audit's headline numbers survive only in prose. `recovery_v7.npz` is absent although `DONE_v7` exists, so the v7 recovery numbers have no backing array. The four `tcca` CSVs remain git-untracked despite `.gitignore` negation rules written specifically to protect them, and their producing script `run_epochs.py` is modified and uncommitted.

**Unowned** — `Auto_Reports/` (2 pptx) and `cosyne2025/figures/` have no producer anywhere in the repository.

---

## 10. Prioritised actions

**P0 — before any figure is regenerated**
1. Occupancy denominator (`spatial_binning.m:45`) → `k·dt`; regenerate both caches; report the effect size of the fix (17–50 %, speed-dependent — a result in its own right).
2. Velocity 2× error in four live scripts; `bin_size=5` → 2.5 in both decoders; fix `plot_decoder_accuracy_vs_chance.m` call sites.
3. Seed everything from `cfg.seed` (currently read by nothing); add `rng` to `ProcessStriatumControl.m`; regenerate caches so learning points are reproducible.
4. Restore the LOO template in `decode_ensemble_ablation.m`.
5. Persist every output to CSV with a provenance stamp; un-ignore results in `.gitignore`.

**P1 — before submission**
6. LP-shuffled null (≥1000 draws, seeded) as the standard reference for every "tracks learning" claim.
7. Speed-matched Naive-vs-Expert contrast; publish the matched version as primary. Add per-bin speed and trial duration as competing regressors.
8. One LP rule and one epoch convention in `project_cfg`, enforced by deleting the inline reimplementations; add an overlap guard and a per-animal overlap column.
9. Move all headline inference to animal level; publish Hierarchical variants only; per-animal points on every panel.
10. Unit quality/stability metrics (presence ratio, amplitude drift, ISI violations) + the dark-ITI drift control; report depth-gate attrition.
11. Re-run TCA end to end; declare rank a priori or wire in `tca_with_cv.m`; regenerate `tca_outputs.mat`.
12. Re-run S12 restricted to learners, to CSV, with per-area animal counts.
13. Build Fig 4B: land the two-timescale redesign, pass the recovery gate, then regress latents against activity with speed/duration/lick rate as competitors, cross-validated within animal, tested at animal level. **If this is not built, scale the claim back from "ensembles encode computational variables" to "activity is refined across learning" and drop Fig 4.**

**P2 — hygiene**
14. Delete `CCA_striatum_spatial_v2.m` and `v3.m`; drop cross-area MI §8–12; drop CEBRA; drop S11 as inferential; drop the t-SNE and behavioural-stability-vs-precision panels.
15. Fix `calc_dim` (`explained`, not coefficients) or drop relative dimensionality.
16. Restrict cell-type analyses to DMS/DLS with a stated mask.
17. Fix the `project_cfg.m` a.u./cm labels and `max_bin`; reconcile the 130 vs 135 a.u. reward-zone end.
18. Correct the `tcca` residualisation description and remove the dead knob; correct the vault's round-17 claim; mark `cca/RESULTS.md` superseded.
19. Delete stale SVGs so nothing on disk predates the current cohort.

---

## 11. Coverage and caveats of this report

Built from a 14-reader survey of every `.m` and `.py` file, two results-harvest passes over all logs and artefacts, and three independent critics (statistical, theoretical, completeness). 3.5 M tokens of subagent reading; 290 analyses catalogued; 115 recorded outcomes; 72 artefacts; 49 critique findings.

Known limits:
- Severity ratings are reader judgement, not verified reruns. Claims I verified myself are marked as such in §2, §3 and §6; everything else traces to a file and line in the inventory and should be spot-checked before it enters the manuscript.
- No analysis was re-executed. Where a number is quoted it is quoted from a log or a CSV, with reproducibility status attached in `MANUSCRIPT_RESULTS.md`.

---

## 12. Addendum — completeness audit (corrected run)

The 14-reader survey covered **235 of 266 code files**. A dedicated completeness pass enumerated all code, diffed against the files actually read, and swept for statistical machinery the survey never saw. Everything below I re-verified myself with the command shown.

### 12.1 Four whole analysis families are missing from the inventory

Confirmed by grepping the generated inventory: `tucker_als` **0 hits**, `HOSVD` **0**, `nnmf` **0**, `robustfit` **0**, `ttest2` **0**, `"coding dimension"` **0**. These are not obscure corners — three of them have figures sitting on disk.

| Family | Where | Why it matters |
|---|---|---|
| **Tucker decomposition** (`tucker_als`, ranks 10×5×2), **HOSVD with 5-fold CV** (`tenmat`/`ttm`/`svd`, lines 511–576), **NMF on licks and velocity** (`nnmf`, 1–5 components with a reconstruction-error curve) | `legacy/ProcessStriatumData.m` (699 lines, never read) | **This is the provenance of the TCA rank choice.** Three alternative decompositions were tried — and the HOSVD arm is the one place in the whole repository where a tensor rank was selected by *genuine cross-validation* (`cv_errors_hosvd`, 5 folds, line 511). Fig 3A's "rank 5" has an evidential history that the inventory does not record, and it may well justify the a priori choice §5.4 recommends |
| **Coding-dimension family** — 5 variants: lick-error median split with per-spatial-bin `ttest2`, early-vs-late, per-area with **d′**, similarity to a **learning-point PCA prototype** via per-bin cosine similarity, and CD relative to disengagement with **`robustfit`** R² | `legacy/SummaryProcessingPlotting.m` (2934 lines, the largest unaudited file, 30 sections) | **16 figures exist on disk** — `codingdimension_animal{2,3,4,7}.png`, `cd_area_animal1-6.png`, `cd_area_diseng_animal1-6.png` (I listed them). A co-author could mistake these for current results. The learning-point-prototype cosine-similarity idea is also a genuinely different and arguably better operationalisation of "the spatial map stabilises" than Fig 2A's trial-to-trial correlation |
| **Fano factors** (single-unit and population) plus a **shuffle-controlled inter-area correlation trio** (across trials, across bins, all-pairwise per trial) | `legacy/striatum_plots.m` (1162 lines, never read) | `figures/FF_spatial_taskcontrols.png` and `figures/popFF_animal3_prepost.png` are on disk. The inventory's only two "Fano" hits are `popsim` simulation parameters, not analyses |
| **Contrast psychometrics** — sigmoid fits over contrasts [1, 0.25, 0.1, 0.01] for 15 behavioural measures including corridor-entry-to-first-lick latency, mean inter-lick interval, grating first-500 ms deceleration, minimum grating velocity, trial duration | `plotPooledContrastFigures.m` — **deleted** in commit `7dcf51f` "Reorganisation of folders" | I recovered it: `git show 7dcf51f^:"Striatum project/plotPooledContrastFigures.m"` returns **198 lines**. Nothing in the current codebase computes any of these measures. If the paper makes any claim about fine-grained behavioural kinematics, this is the only code that ever did it |

Recover the deleted file with:

```bash
git show '7dcf51f^:Striatum project/plotPooledContrastFigures.m' > "Striatum project/legacy/plotPooledContrastFigures.m"
```

Also worth recovering for Methods provenance: `V_allData_1ms.m` (deleted in `5bb33ec`) is the original 1 ms collation script and **the origin of the `1 a.u. = 1.25 cm` constant and the 1 ms grid the entire current pipeline assumes**.

### 12.2 There is no MATLAB test coverage at all

`CLAUDE.md` and `AGENTS.md` both state *"Tests: `tests/` via `run_v5_tests.m`"*. I checked: there is **no MATLAB `tests/` directory**, and `git log --all -- "*run_v5_tests*"` is **empty** — the file has never existed in this repository's history. Every Python sub-package is well tested (`cca` 143, `tcca` 165, collected by `pytest`); **all ≈34 k lines of MATLAB, including every analysis behind Figures 1–4, are untested.** Correct the orientation note before anyone relies on it.

Related: `python3 -m pytest cca tcca` from `Striatum project/` **fails with 7 collection errors** ("import file mismatch") because seven test filenames are duplicated across the two packages with no `__init__.py`. The suites can only be run separately.

### 12.3 Half of `tcca` is tested but never applied to data

I traced transitive reachability from the only live driver. **9 of 18 modules are never reached:**

```
reachable:  config, core, dataio, lagged, membership, partial, runner, subspace, subspace_window
never run:  crosspair, early_trials, kcca_window, kernel_cca, mixed_effects,
            paired_stats, subspace_stats, surrogate, trajectory
```

`tcca/scripts/` contains exactly `run_epochs.py` and `smoke_dataio.py`. The seven drivers `tcca/NOTES.md` promises — `analyze_epochs.py`, `run_engagement.py`, `run_trajectory.py`, `run_ifi_windows.py`, `run_transition.py`, `run_early_trials.py`, `run_kcca.py` — **do not exist**.

The consequence is specific and matters: **`mixed_effects` and `paired_stats` are the modules that would supply the formal animal-level statistics**, and they have never touched data. Every Stage-2 cohort claim rests on an in-driver Wilcoxon. `surrogate.py` is dead too — `subspace_window.py:128` implements its own inline `np.roll` null instead. So the report's §8 recommendation ("use the LMM path already present") is a *build*, not a *run*.

### 12.4 Whole result domains have no artefact on disk

`processed_data/` contains exactly: `all_data{,_control,_control2}.mat`, `cross_spatial_decoding_results.mat`, `preprocessed_data{2p5cm,5cm}.mat`, `preprocessed_data_control{2,2p5cm,5cm}.mat`, `tca_outputs.mat`.

Absent, though the code that writes them declares them:
- `shannon_mi_results.mat`, `cross_spatial_mi_results.mat`, `cross_area_mi_results.mat`, `pid_shared_info_results.mat` — **the entire mutual-information and PID domain has never produced a saved result.**
- `nonlinear_epoch_decoding.mat`, `ridge_epoch_decoding_results.mat`, `gpr_epoch_decoding_results.mat` — **`Nonlinear_Epoch_Decoding.m` has never produced a saved artefact in either decoder mode.**
- `striatal_cca_group_results.mat` — the MATLAB CCA group result.
- Every LFP summary CSV that `lfp/results/README.md` calls the defensible output (`sanity_summary.csv`, `sanity_timing_summary.csv`, `signal_identity_summary.csv`, `decode_summary.csv`, `cca_summary.csv`, `learning_evolution_summary.csv`). `lfp/results/` holds only 6 `.npz` and the README. **The audit's headline integrity numbers survive only as prose in `NOTES.md`.**

This strengthens §1's point: these are not stale results, they are *absent* results. Any of these analyses that reaches the manuscript must be run from scratch.

### 12.5 `cca/` round 17 is fiction — retract it

`cca/NOTES.md` round 17 and `ResearchVault/Methods/CCA_HH_Adapted.md` §6.2 describe a complete running-state temporal CCA arm. On disk: `segments.py`, `lagged_temporal.py`, `run_temporal_runstate.py`, their tests, `bin_mode="temporal_runstate"`, `prepare_pair_temporal`, `analyse_pair_temporal` and `sweep._temporal_runstate` are **all absent**. The claimed "169 tests" is actually **143**. What *did* land is the velocity-derivation data layer (`AU_TO_CM`, `velocity_thresh_cm_s`, `trial_velocity`, `build_temporal_streams`, `area_running_activity`, `test_velocity.py`).

`tcca/NOTES.md:189-190` already says this. **The vault Methods note still asserts it** and is the more dangerous copy, because the smoke numbers quoted there (DMS→ACC peak CC1 0.27–0.38 vs DLS–ACC 0.03–0.11) come from code that does not exist and cannot be re-derived. Mark both retracted.

### 12.6 Two smaller items

**An n=8 behaviour-only cohort is invisible to the project.** `BehaviourOnly/` holds 8 `.mat` files, and **three of those mice (1215, 1217, 1219) do not appear in `RawData/` at all**. Its only consumer, `legacy/BehaviourOnlyAnalysis.m`, is on the orphan list. `rl_model/UNDERSTANDING.md` explicitly wants these mice for "Fig 1/2 behavioural panels" and pipeline validation — a larger, partly disjoint behavioural cohort is sitting unused.

**Nothing in `cosyne2025/` is reproducible as drawn.** 21 `.eps` files, no code anywhere in that path, and it is gitignored. They use a **four-phase Early/Middle/Late/Stop (or Naive/Middle/Trained/Stop) design that survives nowhere in the codebase**, and two of their metrics have no implementation: *lick ratio in reward zone* (the current metric, `calculate_lick_precision.m`, is a different quantity — summed squared distance of pre-RZ licks against a 1000-draw uniform null) and *cumulative reward* (`vr_reward` exists as a raw channel but no code integrates it). Two files have no extension at all. Treat as historical conference artefacts.

**On figure provenance generally:** only 7 `.m` files in the repository write figures. Matching every figure basename against every string literal in every code file (including `sprintf` and `regexprep`-sanitised variants), **262 of 488 basenames in `figures/` and 83 of 113 in `legacy/Figures/` have no producing string anywhere in the code** — they were saved by hand from figure windows. `GroupPrecession.eps` cannot have been produced by any code here: nothing in the repo writes `.eps`.

### 12.7 Documentation corrections owed

- `CLAUDE.md` / `AGENTS.md`: remove the `run_v5_tests.m` / MATLAB `tests/` claim (§12.2).
- `CLAUDE.md`: the modulation-classifier split is documented as "naive 1:2 / expert 21:25, tests naive 3 / expert 26:30". The code (`SpatioTemporalActivityEvolution.m:268-274`) gives `naive_label` = **trial 1 only** and `naive_test = 2:3`. Disjointness holds; the documented indices do not match.
- `NOTES.md:170` / `CLAUDE.md`: the "`_control.mat`/`_control2.mat` predate the `fr_threshold` alignment" warning is right for Control 2 but **wrong for Control 1** — `ProcessStriatumControl.m:32` uses 0.02; only `legacy/PreprocessStriatumControl2.m:19` still uses 0.1 Hz.
- `cca/NOTES.md` round 17 + vault `CCA_HH_Adapted.md` §6.2: mark retracted (§12.5).
- `NOTES.md` §5.1–5.5 names `cfg.m`, `build_zscored_tensor.m`, `classify_modulation.m` and six others. These are **deferred Phase-6 refactor proposals, not missing code** — worth labelling so a reader does not hunt for them.

### 12.8 Effect on the priorities in §10

Nothing in §1–§11 is overturned. Three additions:

- **P1, new:** recover `plotPooledContrastFigures.m` and decide whether the paper makes any behavioural-kinematics claim. If yes, that code is the only implementation.
- **P1, new:** read `legacy/ProcessStriatumData.m` lines 484–646 before finalising Fig 3A. The HOSVD cross-validated rank selection may already justify rank 5 — which would convert §5.4 from "a false methods statement" into "an a priori choice with recorded evidence".
- **P2, new:** decide the coding-dimension family's status. It has 16 figures on disk and no inventory entry; either bring it in as a supplementary operationalisation of map stabilisation, or delete the figures so nothing unowned sits in `figures/`.

Survey coverage after this pass: **266 of 266 code files accounted for.** The 21 unread Python files are test suites, which belong in the inventory as correctness evidence rather than as findings — two of them encode manuscript-relevant guarantees (`tcca/tests/test_core.py::test_cv_in_sample_is_biased_above_held_out_on_noise` is the empirical basis for the report-held-out rule; `tcca/tests/test_mixed_effects.py` encodes the identifiability guards that refuse an LMM below 4 animals).
