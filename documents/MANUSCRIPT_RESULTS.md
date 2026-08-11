# StriatumACC — Recorded Outcomes

Every quantitative result located in the repository, its source, and whether the artefact backing it
still exists. Harvested from all sub-project `NOTES.md`/`RESULTS.md`/`UNDERSTANDING.md`, `PREDICTIONS.md`,
result CSV/JSON/npz files, and git history. **Values are verbatim.**

Reproducibility key: `yes` = artefact on disk and regenerable · `partially` = artefact present but config superseded ·
`NO` = artefact absent, lost, or the claim was withdrawn.

**115 recorded findings · 72 result artefacts · 60 gaps.**

> ⚠️ Read `MANUSCRIPT_REPORT.md` §3 before quoting any number here. Several are superseded by later,
> better-controlled runs within the same arm, and several are means on right-skewed n≤7 samples.

---


## 1. Spatial CCA (cca)

### ⚠️ SPATIAL CCA (round 1-2, 5 cm bins, residual, FS-excluded, in-sample-permutation significance) — held-out CC1 by pair and epoch, learner-group means

**Value.** DMS-DLS 0.27 / 0.34 / 0.26 (n=8); DMS-ACC 0.18 / 0.11 / 0.17 (n=10); DLS-ACC 0.13 / 0.10 / 0.14 (n=7) [naive/intermediate/expert]; V1 pairs 0.03-0.15 mostly n.s.; CA1 pairs n=1-2 anecdotal

*Source:* `Striatum project/cca/NOTES.md:86-90; identical table in cca/RESULTS.md:142-146` · *Reproducible:* partially — superseded config; stage2_fs_excluded.pkl / stage2_main.pkl still on disk (cca/results/, 23 May) but gitignored; the 5 cm input preprocessed_data5cm.mat is present

### ⚠️ Fraction of learner mice with CC1 significantly above the trial-shuffle null (round 1-2)

**Value.** DMS-DLS 4-5 of 8; DMS-ACC 4-5 of 10; DLS-ACC 2-3 of 7

*Source:* `Striatum project/cca/NOTES.md:88-90; cca/RESULTS.md:148-150` · *Reproducible:* partially (pkl on disk, gitignored)

### ⚠️ In-sample CC1 inflation vs held-out (round 1-2)

**Value.** held-out ≈0.25 vs in-sample ≈0.55; 'much weaker than the in-sample CC1 (0.4-0.9)'; 'biased high by roughly two- to four-fold'

*Source:* `Striatum project/cca/NOTES.md:189-190; cca/RESULTS.md:152-153, 96-97, 236-237` · *Reproducible:* partially — in-sample machinery deleted at round 6 (cca/NOTES.md:306-307), so this number cannot be recomputed with current code

### ✅ FS-included vs FS-excluded held-out CC1 agreement

**Value.** r = 0.89 across matched cells

*Source:* `Striatum project/cca/NOTES.md:106-107; cca/RESULTS.md:212-214` · *Reproducible:* yes-ish — stage2_fs_included.pkl + stage2_fs_excluded.pkl on disk (gitignored)

### ⚠️ Directionality (round 1-2): lag curves and group IFI

**Value.** lag curves 'peak sharply at 0 cm and fall off roughly symmetrically'; group IFI ≈ 0, no epoch significantly different from 0

*Source:* `Striatum project/cca/NOTES.md:98-104; cca/RESULTS.md:163-170` · *Reproducible:* partially (pkl on disk)

### ❌ Per-dimension IFI pooled over significant subspace dimensions (round 2)

**Value.** 357 significant canonical dimensions (striatal learner pairs); mean IFI +0.023; one-sample t p = 0.18; IFI 0.005-0.03 across lag-integration windows |lag| ≤ 1..5, all SEM bars touching 0

*Source:* `Striatum project/cca/NOTES.md:182-190; cca/RESULTS.md:222-229` · *Reproducible:* no — round-2 significance test was later shown to over-call (round 6, cca/NOTES.md:293-307); the 357 count is an artefact of the buggy in-sample per-dimension test

### ❌ Subspace dimensionality (round 2, buggy in-sample test)

**Value.** bimodal; mean 4.76 significant canonical dimensions per epoch; many epochs 0-1, tail to ~20

*Source:* `Striatum project/cca/NOTES.md:187-189; cca/RESULTS.md:225-227` · *Reproducible:* no — superseded by round-6 fix

### ❌ Significance over-calling bug and its correction (round 6)

**Value.** DMS-DLS had ~136 'significant' subspace dimensions across the cohort under the in-sample test; diagnostic animal 2 DMS-DLS in-sample n_sig = 13 vs held-out n_sig = 3; after the fix dimensionality drops to ~1.5-2.4 per (animal, epoch)

*Source:* `Striatum project/cca/NOTES.md:293-307` · *Reproducible:* no — the in-sample code path was removed

### ⚠️ Signal vs residual CCA (round 2 factorial)

**Value.** signal CCA ≈0.33 vs residual ≈0.25 for DMS-DLS at expert

*Source:* `Striatum project/cca/NOTES.md:186-188; cca/RESULTS.md:232-234` · *Reproducible:* partially (stage2_resid_*/stage2_signal_* pkls on disk)

### ✅ Corrected learning-point rule (round 3) — the old MATLAB movsum rule returned a window-start trial that was itself above threshold

**Value.** under the old rule the returned LP trial had z = +1.28 (animal 3), -0.77 (animal 2), -0.14 (animal 15); corrected LPs: animal 3 → 32 (was 29); every learner ≥ 22; identical LPs to the fixed MATLAB rule for all 16 animals

*Source:* `Striatum project/cca/NOTES.md:199-210` · *Reproducible:* yes — fix lives in find_learning_points.m / processTaskData.m; per-animal LPs recomputable

### ⚠️ Held-out CC1 after the LP fix (round 3, expert epoch)

**Value.** DMS-DLS 0.265, DMS-ACC 0.175, DLS-ACC 0.138 ('LP shifts were only 1-3 trials')

*Source:* `Striatum project/cca/NOTES.md:221-224` · *Reproducible:* partially

### ✅ Missing-bin (NaN) fraction in spatially binned FR

**Value.** 5 cm file: 0.01-3.3% NaN; 2.5 cm file: up to 33% file-wide, 0-0.2% within analysis epochs for most animals, 23% for the worst (animal 10)

*Source:* `Striatum project/cca/NOTES.md:236-241 (2.5 cm); cca/UNDERSTANDING.md:397-399 (5 cm)` · *Reproducible:* yes — both .mat files on disk (preprocessed_data5cm.mat, preprocessed_data2p5cm.mat)

### ⚠️ Held-out CC1 at 2.5 cm vs 5 cm bins (round 4, residual, FS-excl, expert)

**Value.** 2.5 cm: DMS-DLS 0.24, DMS-ACC 0.11, DLS-ACC 0.08 — vs 5 cm 0.27 / 0.18 / 0.14 ('finer bins did not raise the canonical correlations')

*Source:* `Striatum project/cca/NOTES.md:251-257` · *Reproducible:* partially (pkls on disk)

### ✅ Learning point is non-deterministic across preprocessing runs (un-seeded 1000-shuffle lick-precision baseline)

**Value.** animal 8 came out LP 16 on the 5 cm file and LP 14 on the 2.5 cm file for identical behaviour; animal 8's z-scored lick error: trial 9 = -3.2, trials 14-16 ≈ -2 to -3.2. Fixed by rng(42,'twister') in ProcessStriatumTask.m

*Source:* `Striatum project/cca/NOTES.md:261-274` · *Reproducible:* yes — but only after a preprocessing re-run; the caches on disk predate the seed fix

### ✅ LP-criterion sensitivity (min_consecutive 7 vs 8)

**Value.** raising to 8 cleans animal 8 (lp 16 → 20) but reclassifies animals 5 & 6 as non-learners and destabilises animal 11 (lp 84 → 152); default kept at 7

*Source:* `Striatum project/cca/UNDERSTANDING.md:144-147` · *Reproducible:* yes

### ✅ Cohort size per area pair — projected (v1 spec) vs realised (Stage 1, FS-excluded)

**Value.** Projected usable n: DMS-ACC 13; DMS-DLS 9, DLS-ACC 9; V1-DMS 4, V1-ACC 4, V1-DLS 3; CA1-DMS/DLS/ACC/V1 2 each; partial triplet n=9. Realised learner fits: DMS-DLS 8, DMS-ACC 10, DLS-ACC 7, V1-DMS 3, V1-DLS 2, V1-ACC 3, CA1-DMS/DLS/ACC 1 each, CA1-V1 2

*Source:* `Striatum project/cca/UNDERSTANDING.md:38-49 (projected), 398-408 (realised)` · *Reproducible:* yes

### ✅ Animal 13 (a CA1 learner) loses all striatal areas to FS exclusion

**Value.** >50% of its DMS/DLS/ACC units classify as fast-spiking; 'the CA1 arm is effectively a single-animal anecdote (animal 11)'

*Source:* `Striatum project/cca/UNDERSTANDING.md:405-408` · *Reproducible:* yes

### ✅ Per-epoch sample budget

**Value.** 5 cm: 10 trials x 50 bins = 500 samples; 2.5 cm: 10 trials x 100 bins = 1000 nominal samples; k reaches the cap of 30

*Source:* `Striatum project/cca/UNDERSTANDING.md:50-51; cca/NOTES.md:247-249; cca/RESULTS.md:58-60` · *Reproducible:* yes

### ❌ Dropping the intermediate epoch enlarged the cohort (round 7)

**Value.** learner pairs rise from 39 to 42 (residual, FS-excluded)

*Source:* `Striatum project/cca/NOTES.md:355-361` · *Reproducible:* no — 3 epochs restored at round 10

### ❌ Round-7 held-out CC1, learners, residual FS-excluded, two epochs (naive / expert)

**Value.** DMS-DLS 0.25/0.29, DMS-ACC 0.17/0.13, DLS-ACC 0.14/0.08; 'no clear naive→expert change in communication strength (unpaired p n.s. for the well-powered pairs)'

*Source:* `Striatum project/cca/NOTES.md:405-411` · *Reproducible:* no — round-7 pkls flagged for deletion (cca/NOTES.md:413-419); stage2_res_fsexcl.pkl still on disk

### ✅ 27-config spatial parameter sweep (round 8) — subspace REORIENTATION above the within-epoch split-half floor, per pair

**Value.** 'holds in 19-27 of 27 configs for every pair'. Recomputed exactly from cca/figures/sweep_summary.csv: DMS-DLS 27/27, V1-ACC 27/27, V1-DLS 27/27, DMS-ACC 26/27, V1-DMS 26/27, CA1-DLS 26/26, CA1-ACC 24/27, DLS-ACC 23/27, CA1-V1 21/27, CA1-DMS 19/27

*Source:* `Striatum project/cca/NOTES.md:440-446; verified against cca/figures/sweep_summary.csv (270 rows, 27 tags)` · *Reproducible:* yes — sweep_summary.csv on disk (gitignored)

### ✅ 27-config sweep — communication STRENGTH change (naive vs expert) is at chance

**Value.** 'significant in ≤3/27 configs for every pair'. Recomputed: DMS-DLS 3/27, DMS-ACC 3/27, DLS-ACC 1/27, V1-DMS 2/16, V1-ACC 2/20, CA1-ACC 1/8, CA1-V1 1/19

*Source:* `Striatum project/cca/NOTES.md:446-448; verified against cca/figures/sweep_summary.csv` · *Reproducible:* yes

### ✅ Sweep enrichment of p<0.05 for naive-vs-expert strength across all sweep cells (chance = 0.05)

**Value.** highest cells: pair DMS-ACC 0.1403 (enrichment 1.66, n=991), CA1-ACC 0.1274 (1.51), k_rule fixed30/samples15 0.1237 (1.46); lowest: DMS-DLS 0.0447 (0.53), k_rule fixed3 0.0 (0.0)

*Source:* `Striatum project/cca/figures/sweep_enrichment_p_naive_vs_expert.csv` · *Reproducible:* yes (on disk, gitignored)

### ✅ Sweep enrichment of p<0.05 for IFI (window 3) vs 0

**Value.** V1-DLS 0.10 (2.2, n=10), V1-ACC 0.0785 (1.73), DLS-ACC 0.071 (1.56); residual 0.0562 vs signal 0.034; CA1-ACC 0.0058 (0.13)

*Source:* `Striatum project/cca/figures/sweep_enrichment_p_ifi_w3.csv` · *Reproducible:* yes

### ✅ Round-8 TEMPORAL arm (40 ms / 20 ms, signal CCA, 50 surrogates) — subspace reorientation

**Value.** '18 of 20 pair-config cells above the floor'. Verified from cca/figures/sweep_summary_temporal.csv: 18/20 with angle_minus_floor>0; the two failures are CA1-V1 at t40 (-0.0762) and V1-ACC at t20 (-0.0085); largest CA1-ACC t40 +0.398 rad

*Source:* `Striatum project/cca/NOTES.md:457-463; verified in cca/figures/sweep_summary_temporal.csv (20 rows)` · *Reproducible:* yes — but the arm was signal CCA with no partialling and in-sample permutation; superseded by tcca 2026-07-28 (see below)

### ❌ Round-8 temporal arm compute limits

**Value.** 10 ms 'computationally intractable in the sandbox (one pair > 45 s at 200 surrogates)'; ran 40 ms and 20 ms at 50 surrogates (p-resolution 0.02); disengaged traversals >60 s on a 250 cm corridor excluded

*Source:* `Striatum project/cca/NOTES.md:454-462` · *Reproducible:* n/a (a compute record)

### ✅ Round-9 honest corrections — two V1 'findings' were two-mouse artefacts

**Value.** the min_units=6 / LP-8 / samples-15 config gave V1-ACC strength p ≈ 0.02-0.03 on 2 animals; reverting to LP-7 (n=3) pushed it to p ≈ 0.26 and the V1-DMS IFI flip vanished. V1-DMS / V1-ACC = 2 learners at every min_units; LP-7 → 3 learners on V1 pairs, LP-8 → 2

*Source:* `Striatum project/cca/NOTES.md:483-491` · *Reproducible:* yes (sweep_summary.csv)

### ⚠️ trials_per_epoch 15 rejected

**Value.** no extra significant dims (134 → 129) and cohort cost (33 → 27 learner pairs)

*Source:* `Striatum project/cca/NOTES.md:491-493` · *Reproducible:* partially — stage2_committed_tpe15.pkl on disk

### ✅ Circshift vs trial-permutation null — significant subspace dimensions pooled over all pairs x epochs

**Value.** trial-perm = 201, circshift = 533 (≈2.6x). Per-pair/epoch detail in null_comparison.csv, e.g. DMS-DLS naive nsig 15 → 47; DMS-ACC intermediate 19 → 58; DLS-ACC expert 5 → 31

*Source:* `Striatum project/cca/NOTES.md:508-517; cca/figures/null_comparison.csv` · *Reproducible:* yes — null_comparison.csv + stage2_committed_{trials,circshift}.pkl on disk (gitignored)

### ✅ Circshift admits weaker dimensions — mean held-out CC of the significant pool falls

**Value.** DMS-DLS naive 0.245 → 0.139 (trials → circshift). Full table: DMS-DLS int 0.3153→0.1674, expert 0.2353→0.1648; DMS-ACC naive 0.1719→0.1179; DLS-ACC expert 0.1121→0.0873

*Source:* `Striatum project/cca/NOTES.md:524-528; cca/figures/null_comparison.csv` · *Reproducible:* yes

### ✅ IFI sign flips between nulls on several cells

**Value.** V1-ACC expert +0.18 → -0.07 (trials → circshift); csv gives ifi_trials 0.1801 vs ifi_circshift -0.0701

*Source:* `Striatum project/cca/NOTES.md:527-528; cca/figures/null_comparison.csv` · *Reproducible:* yes

### ⚠️ Within-area subspace similarity across an area's partners (round 11)

**Value.** mean pairwise |cosine| ≈ 0.2-0.4, vs a random-vector baseline ≈ 0.15-0.25 for 10-30-unit areas → 'partner-specific, not one shared read-out'

*Source:* `Striatum project/cca/NOTES.md:570-577; commit c34ca62` · *Reproducible:* partially — stage3 pkls on disk; no CSV of the similarity matrices

### ⚠️ Partial CCA, full conditioning (round 11) — cohort

**Value.** 99 cells from 9 animals (only animals with ≥3 areas qualify); partial CC1 ≈ plain CC1 for every pair, 'often slightly higher'; the confound regression is fitted on all samples, not cross-validated

*Source:* `Striatum project/cca/NOTES.md:580-588` · *Reproducible:* partially — partial_committed.pkl on disk

### ⚠️ Partial CCA on the DMS/DLS/ACC triplet (round 1-2, naive epoch)

**Value.** DMS-DLS|ACC 0.30→0.30; DMS-ACC|DLS 0.22→0.25; DLS-ACC|DMS 0.14→0.12; n=7 animals with all three areas

*Source:* `Striatum project/cca/NOTES.md:145-149; cca/RESULTS.md:202-208` · *Reproducible:* partially — partial.pkl / partial_z0.pkl / partial_z1.pkl on disk

### ⚠️ Partial-CCA variant of the full committed pipeline (round 12) — cohort

**Value.** 33 pairs qualify (a pair needs the animal to have ≥1 other recorded area)

*Source:* `Striatum project/cca/NOTES.md:600-606` · *Reproducible:* partially — stage2_committed_circshift_partial.pkl on disk

### ⚠️ FS inclusion enlarges the committed cohort (round 13)

**Value.** plain 37 → 48 pairs; partial 33 → 46 pairs; IFI(|lag| ≤ 10 bins) 'essentially unchanged'

*Source:* `Striatum project/cca/NOTES.md:629-633; commit 25908b6` · *Reproducible:* partially — *_fsincl pkls on disk

### ✅ Committed-config epoch ANOVA on held-out CC (round 15/16, partial pipeline, per significant dimension)

**Value.** 'no epoch effect on CC — every per-dimension ANOVA p > 0.1'. Exact: DMS-DLS F=0.081 p=0.923 (n_dim 50/48/47, n_animals 5); DMS-ACC F=1.192 p=0.306 (51/57/52, n=7); DLS-ACC F=1.934 p=0.148 (50/56/35, n=4); V1-DMS p=0.100; V1-ACC p=0.630; CA1-DMS p=0.640; CA1-ACC p=0.579; CA1-V1 p=0.723

*Source:* `Striatum project/cca/NOTES.md:680-686; verbatim from cca/figures/epoch_stats_partial.csv` · *Reproducible:* yes — epoch_stats_partial.csv + stage2_committed_circshift_partial.pkl on disk (both gitignored)

### ✅ Committed-config epoch ANOVA on IFI (|lag| ≤ 10 bins), per significant dimension

**Value.** DMS-DLS F=3.528 p=0.03198, Tukey intermediate-vs-expert p=0.02611 (naive-inter 0.214, naive-expert 0.596); CA1-ACC p=0.06998; CA1-V1 p=0.06257; DMS-ACC p=0.1838; DLS-ACC p=0.12474; V1-ACC p=0.18185. Per-animal RM-ANOVA finds nothing significant (DMS-DLS F=1.258 p=0.335 at n=5)

*Source:* `Striatum project/cca/NOTES.md:682-686; commit e92968d; verbatim from cca/figures/epoch_stats_partial.csv` · *Reproducible:* yes

### ✅ Committed-config held-out CC vs 0, per pair x epoch, both aggregation units (partial pipeline)

**Value.** DMS-DLS: per-dim mean 0.14268/0.14868/0.15409 (Wilcoxon p=0.0 all three, n_dims 50/48/47); per-animal mean 0.15397/0.13967/0.18001 (t p=0.00189/0.03134/0.02695, n=5). DMS-ACC per-animal 0.12036/0.09602/0.11531 (p=6e-05/3e-05/0.00077, n=7). DLS-ACC per-animal 0.10728/0.09382/0.09266 (p=0.00642/0.00134/0.00118, n=4). V1-ACC per-animal 0.12123/0.10215/0.12065 (p=0.00552/0.00126/0.03623, n=3). CA1-* n=1, no test

*Source:* `Striatum project/cca/figures/epoch_vs0_partial.csv (all rows)` · *Reproducible:* yes — the only place the committed config's headline CC magnitudes are written down

### ✅ Committed-config IFI vs 0, per pair x epoch (partial pipeline, per-animal t-test)

**Value.** DMS-DLS 0.0081 (p=0.9176) / -0.0175 (0.7391) / 0.1212 (0.0664), n=5; DMS-ACC 0.0467 (0.4507) / -0.061 (0.052) / -0.025 (0.5233), n=7; DLS-ACC -0.002 (0.9658) / -0.0667 (0.0594) / -0.0672 (0.3523), n=4; V1-ACC 0.124 (0.0935) / -0.2525 (0.3592) / 0.1119 (0.7374), n=3

*Source:* `Striatum project/cca/figures/directionality_partial.csv; same numbers per-dim in epoch_vs0_partial.csv` · *Reproducible:* yes

### ✅ Committed-config peak lag (partial pipeline)

**Value.** peak_lag_median = 0.0 bins for EVERY pair; peak_lag_mean DMS-DLS +0.5586 bins (+1.3966 cm), DMS-ACC -0.9812 (-2.4531 cm), DLS-ACC -0.6383 (-1.5957 cm), CA1-ACC -1.9444 (-4.8611 cm); n_sig_dims_pooled DMS-DLS 145, DMS-ACC 160, DLS-ACC 141, V1-DMS 39, V1-ACC 43, CA1-DMS 14, CA1-ACC 18, CA1-V1 21, V1-DLS 0, CA1-DLS 0

*Source:* `Striatum project/cca/figures/directionality_partial.csv` · *Reproducible:* yes

### ✅ Round-8-era 2-epoch committed sweep cell (sweep_committed_results.csv) — CC and angle-minus-floor per pair, FS excl/incl

**Value.** DMS-DLS excl cc_naive 0.2049 cc_expert 0.1381 d_cc -0.0668 p=0.3056, ifi_w3 -0.1052 p=0.33, angle_minus_floor 0.369; DMS-ACC excl 0.1701/0.1223 p=0.1886, angle 0.2335; DLS-ACC excl 0.1562/0.1060 p=0.4685, angle 0.2567; V1-ACC excl 0.0994/0.1377 p=0.0311, ifi_w3 0.2236 p=0.0386, angle 0.1681; CA1-DMS excl angle -0.4435

*Source:* `Striatum project/cca/figures/sweep_committed_results.csv` · *Reproducible:* yes

### ⚠️ Principal angles / subspace reorientation, round-3 era (radians)

**Value.** cross-epoch principal angle of the dominant direction ≈ 1.3-1.4 rad, above the within-epoch split-half floor ≈ 1.0-1.2 rad; DMS-ACC and DLS-ACC naive→expert significantly above floor (paired t-test)

*Source:* `Striatum project/cca/NOTES.md:134-137; cca/RESULTS.md:176-182` · *Reproducible:* partially — stage3.pkl / stage3_z*.pkl on disk; config since changed (round 14)

### ⚠️ Membership overlap (Jaccard) and weight sparsity (Gini), round-3 era

**Value.** cross-pair member-set Jaccard ≈ 0.22-0.31 (chance ≈ 0.25 for top-quartile sets); cross-epoch naive-vs-expert Jaccard ≈ 0.22; Gini ≈ 0.6-0.8, mild increase naive→expert for striatal pairs, e.g. DMS-DLS 0.67 → 0.76

*Source:* `Striatum project/cca/NOTES.md:139-143; cca/RESULTS.md:186-198` · *Reproducible:* partially (stage3 pkls on disk)

### ❌ d_sub decision evidence

**Value.** at d_sub=3 the within-epoch split-half principal angle was 'already near-orthogonal' — 2nd/3rd canonical dimensions not estimable from 10-trial epochs → d_sub = 1

*Source:* `Striatum project/cca/NOTES.md:127-131; cca/UNDERSTANDING.md:359-365` · *Reproducible:* no — no stored split-half-at-d3 artefact

### ❌ Common-unit (member vs non-member) spatial activity profiles

**Value.** 'Member and non-member profiles share spatial structure; member units sit modestly higher in amplitude in several pairs but are not a starkly distinct subpopulation' — no numbers given

*Source:* `Striatum project/cca/NOTES.md:548-552` · *Reproducible:* figures only (PNG/SVG, gitignored)

### ❌ Round-17 'Arm A' running-state temporal CCA smoke on real data (2 animals)

**Value.** median running speed 19-27 cm/s; the 2 cm/s gate retains 75-82% of bins; DMS→ACC peak CC1 ~0.27-0.38 vs DLS-ACC ~0.03-0.11; '169 tests; ruff clean'

*Source:* `Striatum project/cca/NOTES.md:760-764; also asserted in ResearchVault Methods/CCA_HH_Adapted.md §6.2 Arm A note` · *Reproducible:* NO — the claimed code does not exist: cca/src/striatum_cca/segments.py, lagged_temporal.py and cca/scripts/run_temporal_runstate.py are all absent from disk (verified). tcca/NOTES.md:188-192 states the round-17 claim is false and the work was superseded by tcca/

### ✅ Committed spatial CCA configuration (frozen, round 14)

**Value.** partial CCA; 2.5 cm spatial bins (100-bin preprocessing); residual (per-(bin,unit) trial mean subtracted); whole-engaged-period per-unit z-scoring applied before residualisation; FS units excluded, min_units 6; 3 epochs x 10 trials; LP criterion 7; PCs by samples rule, 15 samples per PC, k_cap 30; 5-fold whole-trial CV; held-out-CC per-dimension significance p < 0.05; circshift null, min shift 15 bins, n_shuffles 250 (raised from 200); d_sub 1; learners only; animal 8 forced non-learner

*Source:* `Striatum project/cca/NOTES.md:636-661; cca/UNDERSTANDING.md:248-256; commit f3df867` · *Reproducible:* yes — config.DEFAULT in cca/src/striatum_cca/config.py

### ❌ Cross-project methods-contract locked parameters (NOT what the striatum pipeline actually runs)

**Value.** bin_width 25 ms; lag_range ±200 ms = ±8 bins; n_lags 17; cv_folds 10; n_shuffles 50; sig_threshold = mean + 3·sd of shuffle peak; lagged smoothing Gaussian 3 bins σ=1; central_lag_window ±0.3 s (±12 bins); min_paired_samples 5; min_block_bins 12 (=300 ms); k = min(k_units, floor(N_samples/100), 30)

*Source:* `ResearchVault/Methods/CCA_HH_Adapted.md §3.1 (lines 136-166)` · *Reproducible:* n/a — contract. Note the striatum pipeline uses 5-fold CV, 250 circshift surrogates, samples-per-PC 15, and a held-out-CC p<0.05 threshold, i.e. it departs from this contract on cv_folds, n_shuffles, sig_threshold and the k rule

### ✅ cca test-suite growth (a proxy the notes report per round)

**Value.** 35 (Stage 1) → 53 → 67 (Stage 3) → 74 → 75 → 77 → 78 → 81 → 87 → 88 → 99 → 107 → 114 → 130 → 135 tests, ruff clean throughout

*Source:* `Striatum project/cca/NOTES.md:28, 179, 218, 249, 317, 403, 465, 562, 589, 618, 634, 687, 708; commits a9e682c, 8abae5e` · *Reproducible:* yes (pytest)

### ✅ Across the full CCA hyperparameter sweep, the fraction of configurations reaching p<0.05 for the naive-vs-expert canonical-correlation contrast barely exceeds the 5% chance rate — i.e. the effect is not robust to analysis choices

**Value.** frac_p<0.05 by pair: DMS-ACC 0.1403 (n_cells=991, enrichment_vs_chance 1.66); CA1-ACC 0.1274 (212, 1.51); CA1-V1 0.0900 (600, 1.06); V1-ACC 0.0857 (525, 1.01); V1-DMS 0.0769 (442, 0.91); DLS-ACC 0.0535 (804, 0.63); DMS-DLS 0.0447 (985, 0.53); V1-DLS 0.0000 (4, 0.0). By k_rule: fixed30 0.1237 (558, 1.46), samples15 0.1237 (558, 1.46), samples40 0.1188 (446, 1.40), samples25 0.1086 (534, 1.28), fixed10 0.0795 (365, 0.94), fixed20 0.0639 (501, 0.76), var95 0.0615 (504, 0.73), var85 0.0609 (394, 0.72), var75 0.0530 (321, 0.63), fixed5 0.0167 (240, 0.20), fixed3 0.0000 (142, 0.0). By bin: 2.5cm 0.0969 (2311, 1.15) vs 5cm 0.0719 (2252, 0.85). By cca: residual 0.0952 (2405, 1.13) vs signal 0.0728 (2158, 0.86). By z: on 0.0995 (2291, 1.18) vs off 0.0695 (2272, 0.82). By fs: incl 0.0896 (2546, 1.06) vs excl 0.0783 (2017, 0.93).

*Source:* `Striatum project/cca/figures/sweep_enrichment_p_naive_vs_expert.csv (2026-05-24 09:57)` · *Reproducible:* Yes — CSV on disk, and the 2210 underlying .pkl remain in cca/results/. Both are gitignored and exist in a single copy.

### ✅ The same enrichment audit for the information-flow-index (IFI) directionality metric shows essentially NO enrichment over chance for the striatal pairs that carry the main claim

**Value.** frac_p<0.05 by pair: V1-DLS 0.1000 (n_cells=10, enrichment 2.2); V1-ACC 0.0785 (662, 1.73); DLS-ACC 0.0710 (887, 1.56); DMS-ACC 0.0517 (1026, 1.14); DMS-DLS 0.0393 (1018, 0.86); CA1-V1 0.0277 (759, 0.61); V1-DMS 0.0158 (569, 0.35); CA1-ACC 0.0058 (346, 0.13); CA1-DMS 0.0000 (20, 0.0). By cca: residual 0.0562 (2741, 1.23) vs signal 0.0340 (2556, 0.75). By bin: 2.5cm 0.0536 (2666, 1.18) vs 5cm 0.0372 (2631, 0.82).

*Source:* `Striatum project/cca/figures/sweep_enrichment_p_ifi_w3.csv (2026-05-24 09:57)` · *Reproducible:* Yes — CSV on disk, gitignored, single copy.

### ✅ Canonical correlations are reliably above zero in every striatal/cortical pair with data, at both dimension and animal level — this is the one CCA result that survives both levels of analysis

**Value.** cc, animal-level t-test vs 0: DMS-DLS naive 0.1456 p=0.00712 (n=5), intermediate 0.16448 p=0.01807, expert 0.18333 p=0.03492; DMS-ACC naive 0.11463 p=1e-05 (n=6), intermediate 0.09711 p=3e-05, expert 0.12844 p=0.01439; DLS-ACC naive 0.10394 p=0.0009 (n=5), intermediate 0.09700 p=0.00013, expert 0.09491 p=0.00046; V1-ACC naive 0.11860 p=0.00743 (n=3), intermediate 0.09843 p=0.00712, expert 0.11357 p=0.06213; CA1-V1 naive 0.10335 p=0.00938 (n=2), expert 0.09951 p=0.03844. Dimension-level Wilcoxon vs 0 gives p=0.0 for all DMS-DLS / DMS-ACC / DLS-ACC cc cells (n_dims 31-58).

*Source:* `Striatum project/cca/figures/epoch_vs0_plain.csv (2026-06-02 10:37)` · *Reproducible:* Yes — CSV on disk, gitignored. Underlying stage2 .pkl present.

### ✅ Every statistically significant epoch effect in the CCA analysis is significant ONLY at the pseudoreplicated dimension level; the matched animal-level test is non-significant in all cases

**Value.** DLS-ACC ifi: anova_dim_F=5.29931 p=0.00626 and tukey_ni_p=0.00497 (dims n=41/48/31) BUT rm_anova_F=0.74034 p=0.50699 with all posthoc p=0.79647/1.0/1.0 (animals n=5). DMS-ACC ifi: anova_dim_p=0.0269, tukey_ie_p=0.04531 BUT rm_anova_p=0.50548 (n=6). DMS-DLS ifi (partial): anova_dim_p=0.03198, tukey_ie_p=0.02611 BUT rm_anova_p=0.33483 (n=5). DLS-ACC ifi trend_dim_p=0.0274 (slope -0.05648) BUT trend_animal_p=0.64775 (slope -0.01764). No pair shows any significant animal-level rm-ANOVA: all rm_anova_p lie in 0.33-0.69.

*Source:* `Striatum project/cca/figures/epoch_stats_plain.csv and epoch_stats_partial.csv (2026-06-02 10:37)` · *Reproducible:* Yes — both CSVs on disk, gitignored.

### ✅ The choice of null model materially changes both the number of significant dimensions and the effect size, with the circular-shift null yielding more 'significant' dimensions at lower correlation than the trial-shuffle null

**Value.** DMS-DLS naive: nsig_trials=15 vs nsig_circshift=47; cc_trials=0.2413 vs cc_circshift=0.1429 (n_learn=7). DMS-DLS intermediate 13 vs 44, cc 0.3153 vs 0.1674. DMS-DLS expert 24 vs 52, cc 0.2353 vs 0.1648. DMS-ACC naive 14 vs 51, cc 0.1719 vs 0.1179 (n_learn=9). DMS-ACC intermediate 19 vs 58, cc 0.1367 vs 0.0977. DLS-ACC intermediate 25 vs 48, cc 0.1170 vs 0.0989 (n_learn=6). DLS-ACC expert 5 vs 31, cc 0.1121 vs 0.0873.

*Source:* `Striatum project/cca/figures/null_comparison.csv (2026-05-24 15:45)` · *Reproducible:* Yes — CSV on disk, gitignored.

### ✅ The committed-configuration CCA results give scattered uncorrected p<0.05 hits whose sign is inconsistent across area pairs and across the FS-interneuron inclusion switch

**Value.** p_naive_vs_expert: DLS-ACC fs=incl 0.0097 (d_cc=-0.0638) but fs=excl 0.4685 (d_cc=-0.0502); DMS-ACC incl 0.0227 (d_cc=-0.0787) but excl 0.1886 (-0.0478); CA1-V1 incl 0.0164 (d_cc=+0.1163) but excl 0.5556 (+0.0332); V1-ACC excl 0.0311 (+0.0383) and incl 0.0186 (+0.0553); DMS-DLS incl 0.1775, excl 0.3056. p_ifi_w3: DMS-ACC incl 0.0266 (ifi_w3=-0.1714), V1-ACC excl 0.0386 (ifi_w3=+0.2236). Note d_cc is NEGATIVE (expert < naive) for all three striatal-ACC/DLS pairs but POSITIVE for CA1-V1 and V1-ACC. 20 pair x fs cells, no multiple-comparison correction.

*Source:* `Striatum project/cca/figures/sweep_committed_results.csv (2026-05-24 10:26)` · *Reproducible:* Yes — CSV on disk, gitignored; stage2_committed_*.pkl present in cca/results/.

### ✅ Two directionality p-values reach significance only on n=2 animals, i.e. at the small-n t-test floor

**Value.** V1-DMS naive ifi = -0.0942 with p_naive_vs0_ttest = 0.0068 on n_animals=2. CA1-V1 trend_slope = -0.1048 with trend_p = 0.0204 on n_animals=2 (and its epoch-level p values are 0.4622/0.688/0.1497, all non-significant). CA1-ACC and CA1-DMS rows report ifi values with BLANK p-values because n_animals=1. V1-DLS and CA1-DLS have n_animals=0 and are entirely empty.

*Source:* `Striatum project/cca/figures/directionality_plain.csv (2026-06-02 10:37)` · *Reproducible:* Yes — CSV on disk, gitignored.


## 2. Temporal CCA (tcca)

### ✅ TEMPORAL CCA (tcca, 25 ms, residual+partial, circshift null, held-out whole-trial CV) — cohort size

**Value.** 125 cells / 11 learners; animals 3 and 15 skipped (too few run-trials for disjoint epochs)

*Source:* `Striatum project/tcca/NOTES.md:16-20, 84-86; PREDICTIONS.md:30` · *Reproducible:* yes — verified directly: epoch_metrics.csv has 125 rows, 11 animals (1,2,4,5,6,7,9,10,11,13,14), all role=learner

### ✅ tcca held-out cc1, striatal triangle, mean per epoch (naive/int/expert)

**Value.** DMS-DLS 0.25/0.30/0.19; DMS-ACC 0.17/0.19/0.19; DLS-ACC 0.22/0.13/0.30. Recomputed from CSV: DMS-DLS 0.2482/0.2972/0.1899 (n=7/7/7); DMS-ACC 0.1704/0.1850/0.1888 (n=9/10/10); DLS-ACC 0.2158/0.1343/0.3008 (n=7/7/7)

*Source:* `Striatum project/tcca/NOTES.md:88-90; verified against tcca/results/epoch_metrics.csv` · *Reproducible:* yes — CSV on disk (but untracked despite NOTES claiming committed)

### ✅ tcca held-out cc1 — MEDIANS diverge badly from the means on a right-skewed n=7 (reporting trap)

**Value.** DMS-DLS naive 0.248 vs median 0.148; DMS-DLS expert 0.190 vs 0.092; DLS-ACC intermediate 0.134 vs median 0.017. Recomputed medians: DMS-DLS 0.1479/0.2681/0.0922; DMS-ACC 0.2136/0.1865/0.1360; DLS-ACC 0.1817/0.0170/0.2007

*Source:* `Striatum project/tcca/NOTES.md:44-49; PREDICTIONS.md:59-63; verified against epoch_metrics.csv` · *Reproducible:* yes

### ⚠️ tcca epoch effect on cc1

**Value.** 'No significant epoch change for any pair (all paired Wilcoxon n.s.; n=7 → p-floor 0.016)'

*Source:* `Striatum project/tcca/NOTES.md:91-92` · *Reproducible:* partially — in-driver Wilcoxon, no stats table written; analyze_epochs.py not yet built

### ✅ tcca IFI, Gini and n_sig at cohort level

**Value.** IFI ≈ 0 (±0.05), no epoch trend; Gini_x ~flat 0.4-0.5, 'no de-sparsification'; n_sig sensible at cohort level (1-3). Reproduction check: IFI mean +0.000, sd 0.09; Gini_x median 0.42; n_sig median 1, max exactly 12. Recomputed from CSV: IFI mean +0.0000 sd 0.090, Gini_x median 0.416, n_sig median 1 max 12

*Source:* `Striatum project/tcca/NOTES.md:17-19, 95-97; PREDICTIONS.md:33-35; verified against epoch_metrics.csv` · *Reproducible:* yes

### ❌ tcca bin-width evidence (10 ms vs 25 ms)

**Value.** 10 ms too sparse per-cell: A2 DMS-ACC naive cc1 = -0.17 with n_sig=6, k=20 PCs on ~10 trials, per-dim held-out CC swings ±0.6. 25 ms clean: DMS-DLS peaks at intermediate 0.15 → 0.40 → 0.10 (2-animal smoke). --max-lag 0 auto-scales the IFI window to ±50 ms

*Source:* `Striatum project/tcca/NOTES.md:68-78` · *Reproducible:* no — smoke output not saved

### ✅ The 2-animal 'DMS-DLS intermediate peak' did NOT survive the cohort

**Value.** per-animal inconsistent: A1 peaks intermediate, A10 monotonic up, A5 monotonic down

*Source:* `Striatum project/tcca/NOTES.md:92-94` · *Reproducible:* yes (epoch_metrics.csv is per-animal)

### ✅ TEMPORAL subspace reorientation does NOT survive residual+partial CCA with held-out CV (the headline 2026-07-28 result)

**Value.** animal as the unit (n=10): mean rot−floor = +1.92°, median +1.03°, Wilcoxon p=0.38, t p=0.26. Positive mean carried by one animal (A10 +22.4°); without it mean = -0.35°. Per-animal range -10.0° to +22.4°. Side-level (pseudoreplicated) striatal triangle 78/140 = 56%, p=0.10. Round 8's signal-CCA arm reported 18/20 = 90%

*Source:* `Striatum project/tcca/NOTES.md:26-41; PREDICTIONS.md:37-46` · *Reproducible:* yes — verified exactly from epoch_cross.csv: striatal-triangle 78/140 = 55.7%; n=10 animals mean +1.922 median +1.027 min -9.96 (A6) max +22.44 (A10); mean without A10 = -0.357

### ✅ The n=10 animal-level rotation test is a powered null, not a p-floor

**Value.** at n=10 the one-sided Wilcoxon p-floor is 0.001

*Source:* `Striatum project/tcca/NOTES.md:34-36; PREDICTIONS.md:43-46` · *Reproducible:* yes (analytic)

### ✅ Pooling the above-floor proportion across pairs is an artefact

**Value.** all-pairs = 61% above floor, p<0.001, manufactured by eight pairs resting on ONE animal each (CA1-*, DG-*) that score 83-100% by noise. Verified from epoch_cross.csv: all-pairs 151/248 = 60.9%; per pair CA1-ACC 6/6=100%, CA1-DMS 6/6=100%, DG-CA1 6/6=100%, DG-DMS 6/6=100%, DG-ACC 5/6=83%, DG-DLS 5/6=83%, CA1-DLS 4/6=67%, DG-V1 4/6=67%, DMS-ACC 38/56=68%, DLS-ACC 26/42=62%, V1-ACC 11/18=61%, V1-DMS 11/18=61%, V1-DLS 5/12=42%, DMS-DLS 14/42=33%, CA1-V1 4/12=33%

*Source:* `Striatum project/tcca/NOTES.md:47-51; PREDICTIONS.md:54-58; verified against epoch_cross.csv` · *Reproducible:* yes

### ✅ Backing animal count per pair (tcca 25 ms run) — must be attached to every panel

**Value.** DMS-ACC 10, DMS-DLS 7, DLS-ACC 7, V1-DMS 3, V1-ACC 3, V1-DLS 2, CA1-V1 2, the other eight pairs 1 each. Verified: DMS-ACC {1,2,4,5,6,7,9,10,11,14}; DMS-DLS and DLS-ACC {1,2,4,5,10,11,14}; V1-DMS/V1-ACC {9,10,11}; V1-DLS {10,11}; CA1-V1 {11,13}; CA1-*/DG-* = animal 11 only

*Source:* `Striatum project/tcca/NOTES.md:50-51; verified against epoch_metrics.csv` · *Reproducible:* yes

### ✅ tcca cross-epoch membership turnover (Jaccard)

**Value.** not stated in prose. Computed from epoch_cross.csv: jaccard_x median 0.285 (mean 0.262), jaccard_y median 0.220 (mean 0.261) — consistent with the spatial pipeline's ≈0.22 near-chance figure

*Source:* `Striatum project/tcca/results/epoch_cross.csv (124 rows)` · *Reproducible:* yes

### ✅ Per-animal learning points used by the tcca 25 ms run

**Value.** A1 22, A2 44, A4 53, A5 54, A6 36, A7 34, A9 44, A10 67, A11 84, A13 23, A14 39 (animals 3, 8, 12, 15, 16 absent: 3 and 15 skipped, 8/12/16 non-learners)

*Source:* `Striatum project/tcca/results/epoch_metrics.csv (lp column)` · *Reproducible:* yes

### ✅ tcca dataio real-data smoke

**Value.** 16 animals, 13 learners, yoked LP 43; animal 1 → 161,934 bins / 124 traversals at 10 ms, median running speed 19.2 cm/s, gate keeps 75% of bins, z-scoring exact; V1/CA1/DG arms sparse (1-4 animals)

*Source:* `Striatum project/tcca/NOTES.md:183-186` · *Reproducible:* yes — scripts/smoke_dataio.py + preprocessed_data2p5cm.mat on disk

### ✅ tcca test-suite sizes

**Value.** Stage 0: 25 synthetic-ground-truth tests; Stage 1: 161 tests (25 data-layer + 136 numeric, ~32 s); Stage 2: 165 green

*Source:* `Striatum project/tcca/NOTES.md:15, 66, 126, 180` · *Reproducible:* yes (pytest)

### ✅ In the tcca epoch analysis, canonical correlation and significant-dimension counts vary by an order of magnitude across animals for the same area pair and epoch, with no cohort-level aggregation saved

**Value.** Animal 1 DMS-ACC: cc1 naive 0.3762 / intermediate 0.2795 / expert 0.4630, n_sig 1/1/1, mi_sig 0.0763/0.0407/0.1206. Animal 2 DMS-ACC: cc1 0.2652/0.2342/0.0975, n_sig 12/3/0, mi_sig 0.4785/0.0930/0.0. Animal 4 DMS-ACC: cc1 -0.1840/-0.2930/0.0743 (negative held-out cc1), n_sig 0/2/1. Animal 1 DLS-ACC intermediate cc1 = -0.0315, animal 2 DLS-ACC naive cc1 = -0.0088. k_eff = 20 (animals 1, 2) or 16 (animal 4). 72 of 125 rows have n_sig>0.

*Source:* `Striatum project/tcca/results/epoch_metrics.csv (2026-07-28 12:05)` · *Reproducible:* Yes on disk, but the producing code (tcca/scripts/run_epochs.py) is MODIFIED and UNCOMMITTED, and all four CSVs are still git-untracked despite .gitignore negation rules written specifically to protect them.

### ✅ tcca cross-epoch subspace rotation angles are close to, and often exceed, their own shuffle floor — i.e. measured rotation is largely indistinguishable from chance

**Value.** Animal 1 DMS-DLS naive->intermediate: rot_x_cc1=60.14 vs floor_x_cc1=68.87 (rotation SMALLER than floor), rot_y=62.45 vs floor_y=63.91. intermediate->expert: rot_x=68.87 vs floor=71.32, rot_y=63.06 vs floor=75.01. naive->expert: rot_x=73.27 vs floor=74.40, rot_y=81.41 vs floor=70.75. Animal 1 DLS-ACC naive->expert: rot_x=89.57 vs floor=86.46, rot_y=89.23 vs floor=89.57. Animal 2 DMS-ACC naive->intermediate: rot_x=88.87 vs floor=81.53. Jaccard overlap of top-unit sets is 0.12-0.43 throughout.

*Source:* `Striatum project/tcca/results/epoch_cross.csv (2026-07-28 12:05)` · *Reproducible:* Yes on disk; producing code uncommitted; file git-untracked.

### ✅ CCA/tcca cohort sizes are hard-limited by probe coverage recorded in the raw depth metadata

**Value.** Neuropixels_Depth_Data.csv: 17 task mice with ACC/Striatum/DMS/DLS boundaries; DLS BLANK for 731, 823, 1206, 703; ACC BLANK for 1206. Neuropixels_V1_Depth_Data.csv: only 5 mice have V1 (1212, 1206, 1201, 1106, 1105) and only 3 of those have CA1 AND DG (1212 CA1 200-650 / DG 0-200; 1206 CA1 1160-1760 / DG 400-1160; 1201 CA1 1800-2400 / DG 1400-1800). 1105 and 1106 have V1 only. tcca epoch_metrics.csv covers 11 animals with learning points lp = 22, 44, 53, 54, 36, 34, 44, 67, 84, 23, 39.

*Source:* `Striatum project/RawData/Neuropixels_Depth_Data.csv, RawData/Neuropixels_V1_Depth_Data.csv, RawData/V1_depth.txt, tcca/results/epoch_metrics.csv` · *Reproducible:* Yes — small metadata CSVs on disk (gitignored via *.csv/*.txt). These are the numbers that explain why every CA1/DG pair in the CCA tables has n_animals of 0-2.


## 3. popsim method validation

### ✅ popsim ground-truth recovery benchmark — all 10 scenarios recover from population activity alone

**Value.** 10/10 passed (ar1 dynamics, n_timesteps 6000, k=5). Per scenario (cca1 / partial_cca1 / drop_frac / peak_lag / n_strong / latent_cca1): no_coupling 0.150/0.157/-0.043/5/0/0.155; zero_lag 0.720/0.721/-0.001/0/4/0.738; lagged 0.463/0.459/+0.009/10/1/0.473; mediated 0.866/0.239/+0.724/0/1/0.884; epoch_varying 0.563/0.563/+0.001/1/1/0.577 (epoch_lags 2, -2, -8); bidirectional 0.566/0.564/+0.003/5/2/0.581; common_input 0.739/0.192/+0.740/0/1/0.751; rotated_subspace 0.841/0.841/-0.000/0/2/0.860; partial_mediation 0.820/0.812/+0.010/0/2/0.844; noise_correlation 0.999/0.999/+0.000/0/1/0.197

*Source:* `Striatum project/popsim/data/generated/recovery_benchmark.json; popsim/NOTES.md:49-51; popsim/README.md:88-90; commit ec60caa` · *Reproducible:* YES — the json is committed to git (the only committed result table in the repo) and regenerates deterministically from the seeded script

### ✅ popsim mediated scenario at the latent vs population level

**Value.** with gain=2.0 and single-link maps A→C→B: marginal CCA(A,B) ≈ 0.56 (latents) / 0.87 (populations); partial CCA(A,B|C) drops to ~0.24 at the population level; scalar partial_correlation(zA0, zB0 | zC) ≈ 0 on the single linked dimensions

*Source:* `Striatum project/popsim/NOTES.md:29-34` · *Reproducible:* yes — asserted in the test suite

### ✅ popsim → real striatum_cca bridge demo (validates the actual analysis code against known ground truth)

**Value.** recovers configured ground truth: zero_lag CC at lag 0; lagged peak lag = 8; mediated collapses under partial CCA 0.85 → 0.08; partial_mediation survives 0.76 → 0.78. Epoch-gated cross-correlograms: epoch 1 peaks +2, epoch 2 -2, epoch 3 flat

*Source:* `commit 31f4156 body; commit a857674 body; popsim/README.md:99-110` · *Reproducible:* yes — scripts/striatum_cca_demo.py; needs h5py and the cca src dir

### ✅ The popsim synthetic benchmark validates that partial CCA discriminates direct coupling from mediated/common-input coupling, and that population correlation can be near-total while latent coupling is negligible — all 10 scenarios passed

**Value.** dynamics=ar1, n_timesteps=6000, k=5. no_coupling: cca1=0.1505, partial=0.1569, drop_frac=-0.0428. zero_lag: cca1=0.7200, peak_lag=0, n_strong=4. lagged: cca1=0.4628, peak_lag=+10 (expected +10). mediated: cca1=0.8656 -> partial=0.2388, drop_frac=0.7241. common_input: cca1=0.7386 -> partial=0.1919, drop_frac=0.7402. partial_mediation: cca1=0.8200 -> partial=0.8121, drop_frac=0.0097 (direct path survives). rotated_subspace: cca1=0.8409, n_strong=2. noise_correlation: cca1=0.9988 but latent_cca1=0.1967 with pop_corr=0.9927. epoch_varying: epoch_lags=[2,-2,-8]. bidirectional: peak_lag=+5, peak_lag2=-12. 10/10 passed=true.

*Source:* `Striatum project/popsim/data/generated/recovery_benchmark.json (TRACKED in git)` · *Reproducible:* Yes — JSON is git-tracked, and the benchmark is deterministic from popsim/scripts/recovery_benchmark.py. The strongest-provenance numeric result in the repository.


## 4. LFP

### ❌ LFP export integrity — exact-zero fractions per mouse (full-file, every stored value read)

**Value.** 1212 0.004%, 614 5.34%, 727 3.16%, 731 4.62%; zero ≥99%-zero one-second windows during behaviour; terminal padding runs 448 s (614) / 265 s (727) / 387 s (731)

*Source:* `Striatum project/lfp/SANITY_AUDIT.md:21-31; lfp/NOTES.md:25-28; PREDICTIONS.md:87` · *Reproducible:* NO from artefact — results/sanity_summary.csv is absent (lfp/.gitignore excludes results/*.csv). Rerunnable: the four voltage_data_384ch*.mat (11.4-16.3 GB) are on disk

### ❌ LFP ordinary median RMS (stored units) and high-amplitude bin prevalence

**Value.** median RMS 2.30e-4 (1212), 2.62e-6 (614), 2.57e-6 (727), 2.07e-6 (731); high-amplitude one-second bins (robust z>5 on log RMS) 17.97% / 1.43% / 1.44% / 1.43%; prevalence unchanged at z>8 or z>10; cadence 5 s (1212) vs 60 s (614/727/731); 1212 ≈100x ordinary RMS of the others

*Source:* `Striatum project/lfp/SANITY_AUDIT.md:21-33; lfp/NOTES.md:46-47, 165-168` · *Reproducible:* no (CSV gone); npz figure-source files present

### ✅ The earlier '99% empty' diagnosis was a threshold artefact — quantified

**Value.** the SD > 0.02 threshold in undocumented units was 87x above ordinary 1212 voltage and 7,600-9,700x above ordinary voltage in the other mice

*Source:* `Striatum project/lfp/SANITY_AUDIT.md:8-12` · *Reproducible:* yes (arithmetic from the RMS table)

### ❌ RETRACTED LFP claim (kept as error record)

**Value.** 'real voltage signal in only 0.0-0.7% of windows'; 'corr(LFP amplitude, spike count) ≈ 0 (-0.05..-0.01)'; 'LFP is flat-zero in 469/500 windows where spikes ARE present'

*Source:* `Striatum project/lfp/NOTES.md:221-239 (marked SUPERSEDED — WRONG)` · *Reproducible:* n/a — explicitly withdrawn; do not use

### ❌ LFP signal identity — temporal smoothness and bandwidth

**Value.** lag-1 r (1 ms): 1212 -0.070, 614 -0.053, 727 -0.108, 731 +0.009; power 1-100 Hz / 1-499 Hz: 15.6% / 17.2% / 16.0% / 18.2%; PSD slope 2-40 Hz: +0.46 / -0.43 / -0.66 / -0.91. Computed over 40 non-overlapping session-spanning ordinary corridor windows x 24 depth-spanning channels

*Source:* `Striatum project/lfp/SANITY_AUDIT.md:35-43; lfp/NOTES.md:36-40; PREDICTIONS.md:87` · *Reproducible:* no from CSV (absent); signal_identity_figure_data.npz on disk

### ❌ LFP narrowband contamination peaks (40-180 Hz)

**Value.** 614: 153.8 and 74.2 Hz; 727: 150.9 and 74.2 Hz; 731: 150.9 and 73.7 Hz; 1212: none sharply resolved. The persistent ~74 Hz peak lies inside the planned 30-80 Hz low-gamma band → low-gamma analysis invalid as specified

*Source:* `Striatum project/lfp/SANITY_AUDIT.md:38-43, 55-59; lfp/NOTES.md:41-45` · *Reproducible:* no from CSV; npz present

### ❌ Common-median referencing sensitivity is session- and frequency-specific (falsified a blanket claim)

**Value.** reduces 614's 153.8 Hz peak by 4.54 dB; 614's 74.2 Hz peak and both peaks in 727/731 change by only -0.06 to +0.03 dB (recomputed from all 384 channels). Registered prediction had been '<0.2 dB' → falsified

*Source:* `Striatum project/lfp/SANITY_AUDIT.md:55-59; lfp/NOTES.md:11-13; PREDICTIONS.md:69-79; StriatumACC/NOTES.md:8-11; GOTCHAS.md:8-10` · *Reproducible:* no from CSV (signal_identity_summary.csv absent, which is where the per-peak dB changes were meant to live)

### ❌ LFP event timing vs VR sync (robust 1 ms event-peak medians)

**Value.** +1.983 s (614), +4.256 s (727), +1.740 s (731); 1212 has a ~30 s IQR and provides no alignment marker. Registered prediction (±0.25 s of +1.98/+4.26/+1.74) confirmed. 'A several-second uncertainty is comparable to a corridor traversal and blocks position/trial analyses'

*Source:* `Striatum project/lfp/SANITY_AUDIT.md:62-70; lfp/NOTES.md:8-10; PREDICTIONS.md:66-77` · *Reproducible:* no from CSV (sanity_timing_summary.csv absent)

### ❌ PROVISIONAL LFP position decoding (crosses the project's own provenance gate; used as an alignment test)

**Value.** ridge, group-k-fold on position-binned band power: R² 0.12 (614) / 0.14 (727) / 0.17 (731); MAE 53 / 46 / 20 cm vs ~72 cm uniform chance; shuffle-p95 ≈ 0; 1212 at chance (R² ≈ 0)

*Source:* `Striatum project/lfp/NOTES.md:101-111` · *Reproducible:* NO — results/decode_summary.csv absent from disk (gitignored); SANITY_AUDIT.md:88-94 lists position decoding under 'Claims not allowed now'

### ❌ PROVISIONAL LFP cross-area CCA is volume-conduction limited

**Value.** held-out cross-area CC1 0.6-0.9 (vs trial-shuffle ~0.02-0.14) but always below the within-area split-half VC ceiling of 0.88-0.99; cross-area CC falls with area separation → shared field, not communication. 'No area-specific coupling above the VC ceiling'

*Source:* `Striatum project/lfp/NOTES.md:112-117` · *Reproducible:* NO — results/cca_summary.csv absent

### ❌ PROVISIONAL LFP band power vs learning (quarantined)

**Value.** learning points 614 LP44, 727 LP53, 731 LP36, 1212 LP23 (all learned); expert/naive band-power ratio 0.92-1.26 (mostly up in 614, ~flat in 727/731); theta LOWER during running in 2/4 mice (727 0.46-0.57, 731 0.33-0.36; 614/1212 ≈1); cross-animal area-averaged theta running ratio 727 0.52, 731 0.35, 614 0.94, 1212 1.07; 1212 band power ~10^11x the others, theta ratio 0.36 → EXCLUDED; 'no robust group-level learning or engagement effect' — the naive→expert rise is largely 614-driven

*Source:* `Striatum project/lfp/NOTES.md:119-154` · *Reproducible:* NO — learning_evolution_summary.csv survives only under lfp/figures/_quarantined_unaligned_learning/ as an audit trail; results CSV absent. SANITY_AUDIT.md:88-94 forbids learning-phase claims. LFP mouse→cohort-index mapping: 614=animal 2 (LP44), 727=animal 4 (LP53), 731=animal 6 (LP36), 1212=animal 13 (LP23), cross-checked against tcca epoch_metrics.csv lp column

### ❌ SUPERSEDED LFP dead-channel counts (withdrawn)

**Value.** 614: DMS 50/50, DLS 46/46, ACC 0/52 dead; 727: DLS 107/110, DMS 2/52, ACC 0/52 dead; 731: ACC 28/72, DMS 0/32 dead; 1212 striatal areas all dead. LF/HF ≈ 0.23 vs 900-3400 on good channels, 2.7x std

*Source:* `Striatum project/lfp/NOTES.md:206-219, explicitly withdrawn at 249-264 and 221-223` · *Reproducible:* n/a — withdrawn (614 has 64 sorted units at 2000-2500 µm, i.e. exactly the 'dead' ACC channels → live tissue)

### ✅ LFP data contract facts

**Value.** 4 mice: 1212 16.33 GB / 11.4 M samples; 614 11.38 GB / 8.4 M; 727 11.65 GB / 8.4 M; 731 11.48 GB / 8.4 M. data_to_save (n_samples, 384) float32, chunks (42,384); channels_to_save 1..384; depth_to_save = [0,0] placeholder except 1212 which is [0,0,20,20,...,3820,3820] = (c//2)*20 µm; sampling ≈1000 Hz INFERRED; 731 blank DLS in the CSV; Neuropixels_V1_Depth_Data.csv covers only 1212

*Source:* `Striatum project/lfp/NOTES.md:67-90, 192-200; lfp/README.md:7-20` · *Reproducible:* yes — files on disk

### ✅ LFP test-suite size

**Value.** 27 tests at Stage 0; 57 pytest tests green at audit completion

*Source:* `Striatum project/lfp/NOTES.md:20-21, 186; lfp/SANITY_AUDIT.md:114` · *Reproducible:* yes

### ✅ Recorded LFP session lengths and sampling rate, from the integrity audit

**Value.** sampling_rate_hz = 1000. Analysis windows per mouse: 1212 = 11400, 614 = 8400, 727 = 8400, 731 = ~8400 (253.1 kB file). 384 channels per probe. PSD curves on 1025 frequency bins; raw voltage excerpts 4000 samples x 6 channels. Signal-identity analysis uses 24 selected channels and 3 nominal areas over 3 correlation bands, with a 3x3 nominal_depth_band_correlations matrix per mouse.

*Source:* `Striatum project/lfp/results/sanity_audit_{614,727,731,1212}.npz, sanity_figure_examples_and_psd.npz, signal_identity_figure_data.npz` · *Reproducible:* Yes — npz on disk, gitignored. But the CSV summary tables that the lfp/results/README.md declares as the defensible outputs (sanity_summary.csv, sanity_timing_summary.csv, signal_identity_summary.csv, sanity_windows_<mouse>.csv) are ABSENT from disk and gitignored, so the headline integrity numbers themselves are lost.


## 5. RL model

### ✅ RL model parameter recovery v4 — the identifiability failure and its fix

**Value.** eta_a non-identifiable (r = -0.17); it degenerates with c_lick, the product eta_a·c_lick recovering at r = 0.95; fix = pin c_lick to C_LICK = 0.15 → 16 free parameters. Latents recovered: value/lick_rate/v_mean r > 0.99, precision 0.96, RPE 0.90 (12 synthetic mice, 120-trial sessions)

*Source:* `Striatum project/rl_model/UNDERSTANDING.md:411-419` · *Reproducible:* yes — results/recovery_v4.npz on disk

### ✅ RL model parameter recovery v6

**Value.** eta_a 0.89, rho 0.76; perceptual params remain sloppy; kappa_v 'now unidentified — a drop candidate'

*Source:* `Striatum project/rl_model/UNDERSTANDING.md:355-359` · *Reproducible:* yes — recovery_v6.npz on disk

### ❌ RL model parameter recovery v7 (after the three correctness fixes; 12 synthetic mice, 120 trials, n_restarts=1)

**Value.** eta_w 0.99, gamma 0.98, beta 0.94, theta 0.95, lambda_max 0.97, v_base/v_slope/log_sigma_v 0.94-1.00, w_init 0.89, eta_a 0.89, rho 0.74, kappa_v 0.24 (still weak, drop candidate). Latents: value 0.996, RPE 0.994, lick_rate 0.999, v_mean 0.999, precision 0.938

*Source:* `Striatum project/rl_model/UNDERSTANDING.md:296-303; commit ebcaf36` · *Reproducible:* NO — results/recovery_v7.npz is absent (npz files stop at v6) although results/DONE_v7 exists. The v7 numbers have no backing array on disk

### ✅ RL model real fit v5 (graded reward + deterministic velocity actor, 16 mice)

**Value.** lick channel held-out CV gain +0.57 nats/bin, 16/16 mice positive (old model +0.12, 10/16); per-epoch lick change reproduced for 11/16 mice (mean Δr 0.70); velocity per-epoch reproduced 6/16 (mean Δr 0.37, up from 0.28); held-out velocity CV still marginally negative (-0.07)

*Source:* `Striatum project/rl_model/UNDERSTANDING.md:348-359` · *Reproducible:* yes — results/real_fits_v5/ on disk (16 mice); the input .mat is present

### ✅ RL model real fit v4 and the multi-restart trap

**Value.** per-epoch lick change reproduced for 10/16 mice (mean Δr = 0.66); M13 had landed in a bad basin at n_restarts=1, nll 26027 → 14240 after refit; velocity 5/16, Δr 0.28. Real fits now use N_RESTARTS = 4

*Source:* `Striatum project/rl_model/UNDERSTANDING.md:400-409, 317-318; commit 09f2646` · *Reproducible:* yes — real_fits_v4/ on disk

### ✅ RL model nested model-comparison ladder — SYNTHETIC validation only (3 full-model mice, forward CV)

**Value.** full is best for 3/3 mice on every comparison; total held-out LL/bin: full -1.260, no_actors -1.265 (Δ +0.005), no_value_learning -1.318 (Δ +0.058), fixed_agent -1.553 (Δ +0.293)

*Source:* `Striatum project/rl_model/UNDERSTANDING.md:336-341; commit 64a89e0` · *Reproducible:* yes for the synthetic demo (RLMODEL_LADDER_SYNTH=1; DONE_ladder_synth marker). The REAL ladder was never run — no artefact

### ✅ Velocity-actor gradient bug magnitude

**Value.** the omitted gamma·V' (kappa_v speed/accuracy) term was ~16% of the gradient in-RZ, ~0 elsewhere at the default kappa_v; closed-form vs jax.grad vs jvp all ~0.25 s/eval, ~115 s/mouse at 120 trials, maxiter 400, byte-identical values

*Source:* `Striatum project/rl_model/UNDERSTANDING.md:280-285, 305-315; commits 388720f, ec125db, e1d16b6` · *Reproducible:* yes — test_velocity_grad_matches_autodiff (rtol 1e-5)

### ✅ precision latent is structurally uninformative after spatial demeaning

**Value.** precision keeps only ~6% of its variance after per-bin demeaning (Kalman filter converges within ~1 trial), so its unique dR2 under the beh_spatial model is structurally ~0 regardless of biology

*Source:* `Striatum project/rl_model/UNDERSTANDING.md:286-291; commit 388720f` · *Reproducible:* yes — neural_encoding.spatial_demean_var_ratio quantifies it

### ✅ Forward/blocked CV scheme parameters

**Value.** FORWARD_FRAC = 0.25 (trailing quarter of trials held out, learning still teacher-forced through all)

*Source:* `Striatum project/rl_model/UNDERSTANDING.md:322-327; commit 09f2646` · *Reproducible:* yes — real_fits_v6_forward/ on disk

### ⚠️ STALE vault RL writeup (v3-era, 13-parameter model) — parameter recovery

**Value.** 9 of 13 parameters recover: v_base 1.00, log_sigma_v 1.00, v_slope 0.999, lambda_max 1.00, theta 0.99, gamma 0.99, w_init 0.99, eta_w 0.98, Q 0.85. Weakly identified: beta 0.59, R0 0.33, iti_inflation 0.26, R_slope -0.13. Latents: value r=0.999, RPE 0.997, precision 0.95, lick rate 0.99, expected velocity 0.9995 (12 synthetic mice, 160 trials)

*Source:* `ResearchVault/Methods/Belief-State RL Model.md:151, 155` · *Reproducible:* partially — recovery_v3.npz on disk. STALE: this predates the 2026-05-24 two-timescale redesign; the vault note has not been updated since 2026-05-22

### ⚠️ STALE vault RL writeup — real-data held-out likelihood (13-param model, 16 mice)

**Value.** lick channel beats the saturated per-bin null for 10 of 16 mice, mean gain +0.122 nats/bin; velocity fits worse than the null (mean -0.083 nats/bin, 2/16 positive). Per-mouse lick gains (nats/bin): M01 -0.02, M02 +0.03, M03 +0.44, M04 +0.13, M05 -0.03, M06 -0.09, M07 +0.25, M08 -0.04, M09 +0.07, M10 +0.00, M11 -0.06, M12 +0.76, M13 +0.15, M14 +0.05, M15 -0.09, M16 +0.41; engaged trials per mouse 125/225/71/100/109/76/90/92/131/79/182/100/155/112/27/51

*Source:* `ResearchVault/Methods/Belief-State RL Model.md:175, 181, 183-198` · *Reproducible:* partially — real_fits_v3/ on disk; superseded by v5 (+0.57 nats/bin, 16/16) and v6

### ✅ RL model held-out fit quality beats a matched null on licking but not on velocity, and an earlier generation shows a train/test inversion

**Value.** real_fits_v6_forward/M01 (n_trials=125, train 94 / test 31, nll=7227.008): model_lick_test=-0.5501 vs null_lick_test=-0.8626 (model better); model_vel_test=-0.6384 vs null_vel_test=-0.6097 (model WORSE); model_lick_train=-0.9143, model_vel_train=-0.6229. real_fits/M01 (n_trials=357, train 286 / test 71, nll=25836.807): model_test_ll=-1.7898 vs null_test_ll=-1.8281 (model better on test) but model_train_ll=-1.8097 vs null_train_ll=-1.7566 (model WORSE on train). fits_v6/mouse_00 nll=9927.243.

*Source:* `Striatum project/rl_model/results/real_fits_v6_forward/M01.npz, real_fits/M01.npz, fits_v6/mouse_00.npz` · *Reproducible:* Yes — npz on disk, gitignored. Note real_fits/ has only 3 of 16 mice and real_fits_v2/ only 2 of 16, so those generations cannot be summarised at cohort level.


## 6. CEBRA

### ❌ CEBRA embedding and decoding cohort dimensions

**Value.** 16 animals x 5 area subsets (all, DMS, DLS, ACC, V1) -> single_decoder_r2 (16x5) and single_decoder_rmse (16x5). Consistency: 240 scores over 240 animal pairs (16x15). Embeddings are 3-D, per-mouse sample counts 3461 (mouse3) to 14132 (mouse2). Training config: seed=42, output_dimension=3, max_iterations 15000 (single and multi), batch_size=512, learning_rate=0.0003, time_offsets=10, distance=cosine, conditional=time_delta, architecture=offset10-model, decoder_test_frac=0.25, ridge_alpha=1.0.

*Source:* `Striatum project/cebra_results/cebra_results.npz, cebra_results.mat, cebra_config.json (config + npz are git-tracked)` · *Reproducible:* Partly — config and npz are tracked, but the .mat holding the only copy of the 16 embeddings is gitignored, and cebra_data/*.mat inputs are gitignored and predate the producer's last commit.


## 7. TCA / ensembles

### ❌ Combined TCA supermouse tensor dimensions

**Value.** supermouse_tensor_raw = 30 trials x 50 spatial bins x 1693 units; supermouse_combined_valid = 30 x 50 x 1630 (63 units dropped by the validity filter). learning_points_task covers 14 animals. tensor_info records n_animals_task, n_animals_control, n_animals_total, mouse_units_starts/ends, trials_aligned. best_n_factors and best_mdl{lambda,u} are stored.

*Source:* `Striatum project/processed_data/tca_outputs.mat (h5py key inspection)` · *Reproducible:* File on disk but STALE — producer Run_TCA_pipeline.m committed 2026-06-01, artefact 2026-05-14; embedded task_data is the 33-field 5 cm schema, and the 1693 units still include DG despite DG being excluded from all figures/analyses from 2026-05-24. Gitignored.


## 8. MATLAB core & cohort

### ✅ popsim test-suite growth

**Value.** 68 → 75 → 86 tests (6 bridge tests), ruff clean

*Source:* `commits 5099f7a, ec60caa, 31f4156` · *Reproducible:* yes

### ✅ Cohort / anatomy counts (MATLAB side)

**Value.** 16 task mice; CA1 added for 3 mice (1212, 1206, 1201), DG the same 3; mice 1106 and 1105 have no CA1/DG annotation; 15 area pairs for n=6 areas; fr_threshold = 0.02 Hz everywhere (aligned 2026-05-07)

*Source:* `StriatumACC/NOTES.md:46-52, 34; vault Meetings/2026-04-24-Striatum-Meeting.md:10` · *Reproducible:* yes — RawData/Neuropixels_V1_Depth_Data.csv + project_cfg.m

### ✅ Repository audit counts (2026-05-07)

**Value.** 46 .m files moved up into 'Striatum project/' (5 from Preprocessing/, 41 load-bearing helpers from Legacy/ including three transitive deps); 26 orphan .m files remain in Legacy/; 24 confirmed-orphan files enumerated; SummaryProcessingPlotting.m is 122 KB; ProcessStriatumTask.m 77 KB / ProcessStriatumControl.m 79 KB; SpatioTemporalActivityEvolution.m 2100 lines / 104 KB

*Source:* `StriatumACC/NOTES.md:36, 209, 218, 227, 241, 357` · *Reproducible:* yes

### ❌ Known standardisation leak in loo_ridge_press.m (quantified, accepted)

**Value.** full-fit (not per-fold) feature standardisation introduces O(1/n) train/test leakage; at n = 200-300 trials the bias is ~0.3-0.5%, 'well below the noise floor'

*Source:* `StriatumACC/NOTES.md:119-122; AGENTS.md:124; CLAUDE.md project notes` · *Reproducible:* not independently verified — no artefact demonstrating the 0.3-0.5% figure

### ✅ Shuffle counts in the MATLAB decoders

**Value.** raised from 1 to cfg.n_shuffles = 25 (CrossSpatialBinDecoding, Nonlinear_Epoch_Decoding); one-sided empirical p via mean(shuffles >= real)

*Source:* `StriatumACC/NOTES.md:129-132; commit 098e195` · *Reproducible:* yes (code); NO recorded outcome — see gaps

### ✅ TCA rank selection is decorative (parameter, not an outcome)

**Value.** tca_with_bic_extended.m computes BIC in-sample with reconstruction error as the MINIMUM over 25 inits (an order statistic biased toward overfitting), and Run_TCA_pipeline then overrides BIC with a manual best_n_factors = 5

*Source:* `StriatumACC/NOTES.md:288, 366, 424` · *Reproducible:* yes (code); no recorded BIC curve or variance-explained value

### ✅ Modulation-class train/test split (de-circularised classifier)

**Value.** labels defined on naive trials 1:2 and expert 21:25; disjoint held-out tests naive trial 3 and expert 26:30; significance = p_FDR < 0.05 with BH across four panels (DMS/DLS/ACC/V1)

*Source:* `StriatumACC/NOTES.md:96-98; AGENTS.md:122-123` · *Reproducible:* yes (code); NO recorded class counts or p-values


---

## Result artefacts on disk

| Path | Kind | Produced by | Modified | Currency |
|---|---|---|---|---|
| `Striatum project/ (~140 loose .svg in the project root)` | figure (svg only) | IntegratedAll_v1.m, SpatioTemporalActivi | 2026-05-07 14:52 to 2026-05-24 23:47 (bulk 2026-05-24 22:55-23:47) | THIS IS THE CURRENT MATLAB FIGURE SET, and it lives loose in the repo root rather than in figures/. MIXED DG TREATMENT: the DG-exclusion commit landed |
| `Striatum project/Auto_Reports/` | report (.pptx) | UNKNOWN — a repo-wide grep for 'exportTo | 2026-05-15 09:36 and 2026-05-14 17:26 | PROVENANCE UNKNOWN / ORPHANED. No script in the repository writes to Auto_Reports/, so these decks cannot be regenerated. Gitignored (*.pptx). The two |
| `Striatum project/BehaviourOnly/` | .mat (behaviour-only cohort) | legacy/BehaviourOnlyAnalysis.m | 2025-01-29 (all) | STALE / LEGACY — only producer is in legacy/, files 18 months old. Note 1215_M2, 1217_M4 and 1219_M1 appear in NO other directory in the repo: they ar |
| `Striatum project/CCA_Results/ (24 .svg)` | figure (svg only, no png pair) | CCA_striatum_spatial_v2.m (Combined_CC_{ | 2026-03-04 23:35 (5 files), 2026-05-14 17:41-17:43 (12 files), 2026-05-15 11:53-11:56 (9 files) | Mar-04 set = ORPHANED (source .mat gone). May-14 set = current vs v2, superseded pipeline. May-15 set = STALE (see v3 .mat entry). No .csv/.mat compan |
| `Striatum project/CCA_Results/Striatum_CCA_Results_2026_05_14.mat` | .mat (v7.3) | CCA_striatum_spatial_v2.m:414 | 2026-05-14 17:41 | CURRENT relative to v2 code (CCA_striatum_spatial_v2.m last commit 2026-05-08) but v2 is SUPERSEDED by v3. Useful chiefly because it is the only .mat  |
| `Striatum project/CCA_Results/Striatum_CCA_v3_2026_05_15.mat` | .mat (v7.3) | CCA_striatum_spatial_v3.m:210 (cfg.save_ | 2026-05-15 11:52 | STALE. Both CCA_striatum_spatial_v3.m (2026-05-24 10:13) and project_cfg.m (2026-05-24 23:09) were committed 9 days AFTER this artefact, and that proj |
| `Striatum project/RawData/` | raw (immutable) .mat + metadata | upstream Kilosort/preprocessing, outside | 2024-10-02 to 2026-06-18 | IMMUTABLE raw. COHORT-DEFINING CONSTRAINTS visible here: (a) only 3 mice (1212, 1206, 1201) have BOTH CA1 and DG boundaries — 1105 and 1106 are V1-onl |
| `Striatum project/RawData/LFP/` | raw (immutable) voltage .mat | upstream acquisition | 2026-06-05 21:17 to 2026-06-07 16:43 | IMMUTABLE raw, but PROVENANCE IS FRAGILE: the filenames are Finder-style space-numbered duplicates carrying no mouse ID, and the mouse assignment rest |
| `Striatum project/RawDataControl/` | raw (immutable) .mat + metadata | upstream | 2024-10-04 to 2026-05-08 | IMMUTABLE raw. Note the control cohort HAS V1 recordings for 3 animals in raw form, yet preprocessed_data_control2p5cm.mat carries no is_v1 mask — the |
| `Striatum project/RawDataControl2/` | raw (immutable) .mat + metadata | upstream | 2024-10-04 to 2025-01-27 | IMMUTABLE raw, but the control2 cohort has NO spike-type classification at all — consistent with preprocessed_data_control2.mat holding only 5 fields  |
| `Striatum project/cca/figures/ (114 .png)` | figure (png only, no svg pair) | cca/scripts/plot_stage2.py, plot_stage3. | 2026-05-23 14:46 to 2026-06-02 10:37 | Mixed. The 2026-06-01 17:55-17:57 committed panels and 2026-06-02 10:37 directionality panels are CURRENT; the 2026-05-23 14:46-14:47 s5cm panels are  |
| `Striatum project/cca/figures/committed_ifi.csv + sweep_committed_results.csv` | .csv | cca/scripts/committed_ifi.py (commit 202 | 2026-05-24 10:49 / 2026-05-24 10:26 | committed_ifi.csv is STALE by 4 minutes — committed_ifi.py was committed 2026-05-24 10:53, the CSV written 10:49. Flag: committed_ifi.csv is 100 uncor |
| `Striatum project/cca/figures/directionality_partial.csv` | .csv | cca/scripts/directionality_table.py | 2026-06-02 10:37 | current with round-16 code; gitignored |
| `Striatum project/cca/figures/epoch_stats_partial.csv` | .csv | cca/scripts/epoch_anova.py --variant par | 2026-06-02 10:37 | current with round-16 code (2026-06-02); gitignored (root .gitignore *.csv) so never committed |
| `Striatum project/cca/figures/epoch_vs0_{plain,partial}.csv + epoch_stats_{plain,partial}.csv + directionality_{plain,partial}.csv` | .csv | cca/scripts/epoch_anova.py and cca/scrip | 2026-06-02 10:37 | CURRENT (written 1 min before the producing commit). CRITICAL STRUCTURAL FLAG: each file reports a dimension-level test and an animal-level test side  |
| `Striatum project/cca/figures/null_comparison.csv` | .csv | cca/scripts/compare_nulls.py | 2026-05-24 15:45 | 2026-05-24, pre-round-14 (200 surrogates era, plain not partial); gitignored |
| `Striatum project/cca/figures/null_comparison.csv` | .csv | cca/scripts/compare_nulls.py (commit 202 | 2026-05-24 15:45 | CURRENT. Scientifically load-bearing: the circshift null yields systematically MORE significant dimensions than the trial null (e.g. DMS-DLS naive 47  |
| `Striatum project/cca/figures/sweep_enrichment_p_naive_vs_expert.csv + sweep_enrichment_p_ifi_w3.csv + sweep_enrichment_focus_*.csv` | .csv | cca/scripts/summarise_sweep.py (enrichme | 2026-05-24 09:57 and 10:20 | CURRENT. These are the tables that quantify how much of the CCA result survives hyperparameter choice — see recorded_numbers. Gitignored. |
| `Striatum project/cca/figures/sweep_summary.csv + sweep_summary_temporal.csv` | .csv | cca/scripts/summarise_sweep.py | 2026-05-23 18:02 / 18:11 | 2026-05-23/24, round-8 sweep (2-epoch, pre-committed-config); gitignored |
| `Striatum project/cca/figures/sweep_summary_spatial.csv (+ .xlsx)` | .csv / .xlsx | cca/scripts/summarise_sweep.py:333-334 ( | 2026-05-24 09:49 | CURRENT (summarise_sweep.py last commit 2026-05-24 10:13, artefact 09:49 same day — 24 min before, so marginally pre-commit). Gitignored (*.csv, *.xls |
| `Striatum project/cca/results/ (3298 files)` | cache (.pkl) + sentinels (.done) + 1 .npz | cca/scripts/run_stage2.py, run_stage3.py | 2026-05-23 01:16 to 2026-05-24 15:20 | CURRENT vs the driver commits (artefacts written the same day as or after them). But 100% UNTRACKED (root .gitignore *.pkl, *.done) and 3 GB, so this  |
| `Striatum project/cca/results/stage1_validation.npz` | .npz | cca/scripts (stage 1 validation) | 2026-05-23 01:16 | CURRENT. Notable as the one artefact that stores held_out_cc1 AND in_sample_cc1 side by side, i.e. the in-sample optimism is directly measurable from  |
| `Striatum project/cebra_data/cebra_mouse{1..16}_data.mat` | .mat (v7.3) | save_for_cebra.m:126 | 2026-05-07 14:18 (all 16) | STALE — save_for_cebra.m's last commit is 2026-05-08 13:15, the day AFTER these were written. Gitignored (*.mat). |
| `Striatum project/cebra_results/ (12 fig_*.png) and cebra_results/embedding_panels/ (16 fig_emb_mouse*.png)` | figure (png only) | cebra_plot_results.py (commit 2026-05-08 | 2026-05-08 10:14 (all 28) | STALE — cebra_plot_results.py was committed 2026-05-08 13:15, ~3 h after these were written. NO .svg for any of the 28 panels — violates the svg+png r |
| `Striatum project/cebra_results/cebra_config.json` | .json (TRACKED IN GIT) | cebra_analysis.py | 2026-05-08 02:54 | STALE vs cebra_analysis.py (commit 2026-05-08 13:15, ~10 h after). Seeded (seed=42) — good. FLAG: decoder_test_frac=0.25 is a SINGLE held-out split, n |
| `Striatum project/cebra_results/cebra_results.mat` | .mat (MATLAB v7, NOT HDF5 — reads with scipy.io only) | cebra_analysis.py | 2026-05-08 02:54 | STALE vs producer commit. Carries the actual 3-D embeddings that the .npz omits — the only copy of them. Gitignored (*.mat). |
| `Striatum project/cebra_results/cebra_results.npz` | .npz (TRACKED IN GIT — force-added against root *.npz ignore) | cebra_analysis.py | 2026-05-08 02:54 | STALE vs producer commit. 240 = 16x15 ordered animal pairs for the consistency matrix. This is one of only two numeric result files tracked in git for |
| `Striatum project/figures/ (501 files)` | figure (272 png, 215 svg, 13 fig, 1 gif) | IntegratedAll_v1.m, SpatioTemporalActivi | 2025-01-21 to 2025-04-24 13:23 | ENTIRE DIRECTORY STALE — the newest file is 2025-04-24 while its three producing scripts were last committed 2026-05-24 (IntegratedAll_v1, SpatioTempo |
| `Striatum project/legacy/CCA_Results/Striatum_CCA_Results_2026_05_07.mat` | .mat (v7.3) | CCA_striatum_spatial_v2.m (earlier state | 2026-05-07 18:23 | SUPERSEDED — deliberately moved to legacy/. Dated the same day as the fr_threshold=0.02 Hz alignment, so it may straddle that change. |
| `Striatum project/legacy/Figures/ (121 files)` | figure (73 png, 45 svg, 3 fig) | legacy/ scripts (SummaryProcessingPlotti | 2024-10-02 to 2025-10-30 | SUPERSEDED audit trail. All producers are in legacy/. Gitignored. |
| `Striatum project/lfp/figures/ (current set)` | figure (.png + .svg pairs) | lfp/scripts/plot_sanity_audit.py, run_si | 2026-07-13 07:10-07:11 | CURRENT. These four are the only LFP figures the README endorses as defensible. Correctly paired png+svg with saved source npz. Gitignored. |
| `Striatum project/lfp/figures/_quarantined_unaligned_learning/` | figure (audit trail) | lfp/scripts/run_learning_evolution.py (c | 2026-07-13 07:05 | QUARANTINED — produced before the voltage-to-VR timing offset, source-band provenance and 1212 probe identity were established; the 30-80 Hz band is c |
| `Striatum project/lfp/figures/_superseded_absolute_threshold/` | figure (audit trail) | earlier state of plot_sanity_audit.py | 2026-07-12 22:14 | INVALID — explicitly quarantined by lfp/figures/README.md: 'must not be presented'. Retained as audit trail only. |
| `Striatum project/lfp/figures/cca_cross_area.{png,svg} + decode_position.{png,svg}` | figure | lfp/scripts/run_decode_cca.py | 2026-07-13 09:26 | STALE — run_decode_cca.py was committed 2026-07-28 10:16 (the 'lfp: harden voltage integrity audit' work), 15 days after these figures. No saved sourc |
| `Striatum project/lfp/results/` | csv (MISSING) + npz | lfp/scripts/run_sanity_audit.py, run_sig | 2026-07-12 to 2026-07-13 | npz from 2026-07-12/13, i.e. current with the hardened audit; the CSVs that hold the quoted per-peak dB and integrity numbers are gone (lfp/.gitignore |
| `Striatum project/lfp/results/sanity_audit_{614,727,731,1212}.npz` | .npz | lfp/scripts/run_sanity_audit.py (commit  | 2026-07-12 19:05-19:09 | CURRENT (written hours before the producing commit, no later commit touches the script). Gitignored (lfp/.gitignore results/**/*.npz). |
| `Striatum project/lfp/results/sanity_figure_examples_and_psd.npz` | .npz | lfp/scripts/plot_sanity_audit.py (commit | 2026-07-13 07:10 | CURRENT. Good practice — this is the exact plotted data saved alongside the figure, so sanity_audit_overview_v2 and sanity_audit_raw_examples_v2 are r |
| `Striatum project/lfp/results/signal_identity_figure_data.npz` | .npz | lfp/scripts/run_signal_identity.py (comm | 2026-07-13 07:10 | CURRENT. Gitignored. Note the area labels are 'nominal' by construction — the README states area labels are withheld for 1212 because its voltage-prob |
| `Striatum project/mat_results/run001..run010` | .mat (GPFA latent fits, v7.3) | legacy/GPFA_striatum.m — neuralTraj(runI | all 2025-02-27 (dirs restat 2025-11-26) | ORPHAN / LEGACY. The only producer is legacy/GPFA_striatum.m; no live script in the repo references GPFA. Worse, that script sets runIdx=10, xDim=8 an |
| `Striatum project/popsim/data/generated/recovery_benchmark.json` | .json | popsim/scripts/recovery_benchmark.py | 2026-05-30 | current; the ONE result artefact in the repo that is actually committed to git |
| `Striatum project/popsim/data/generated/recovery_benchmark.json` | .json (TRACKED IN GIT) | popsim/scripts/recovery_benchmark.py (co | 2026-07-28 10:17 (checkout timestamp) | CURRENT and one of only a handful of TRACKED result artefacts. This is the method-validation ground truth for the whole CCA approach: it demonstrates  |
| `Striatum project/popsim/data/generated/{zero_lag,lagged,epoch_varying,mediated,no_coupling}/metadata.json` | .json (TRACKED IN GIT) | popsim/scripts/generate_datasets.py (com | 2026-07-28 10:17 (checkout timestamp) | CURRENT, tracked. But only 5 scenario directories exist while recovery_benchmark.json reports 10 scenarios — bidirectional, common_input, rotated_subs |
| `Striatum project/presentations/` | report (.pptx) | manual | 2025-01-28 to 2026-05-25 11:53 | Historical record. Gitignored (*.pptx). Note two stale PowerPoint lock files left in the PROJECT ROOT (not in presentations/): ~$StriatumUpdate_202603 |
| `Striatum project/processed_data/all_data.mat` | .mat (v7.3/HDF5) | OrganiseStriatumDataIncV1.m (no explicit | 2026-05-14 12:24 | CURRENT vs producer (OrganiseStriatumDataIncV1.m last commit 2026-05-08, artefact 2026-05-14). Post-dates the fr_threshold=0.02 Hz alignment (2026-05- |
| `Striatum project/processed_data/all_data_control.mat` | .mat (v7.3) | OrganiseStriatumDataControl.m:143 | 2026-05-14 13:21 | CURRENT (producer last commit 2026-05-15 11:42 file mtime; git-tracked script older than artefact). Post-dates fr_threshold alignment. |
| `Striatum project/processed_data/all_data_control2.mat` | .mat (v7.3) | legacy/OrganiseStriatumDataControl2.m | 2025-01-27 | STALE — predates the 2026-05-07 fr_threshold=0.02 Hz alignment by 15 months, and the producer now lives in legacy/. CLAUDE.md flags _control2 caches e |
| `Striatum project/processed_data/cross_spatial_decoding_results.mat` | .mat (v7.3) | CrossSpatialBinDecoding.m:178 (cfg.save_ | 2026-05-14 14:25 | CURRENT vs code (CrossSpatialBinDecoding.m last commit 2026-05-08) but built on the 5 cm data generation, i.e. one bin-width regime behind the current |
| `Striatum project/processed_data/preprocessed_data2p5cm.mat` | .mat (v7.3) | ProcessStriatumTask.m:354 (save('preproc | 2026-05-23 12:19 | STALE by ~22 h — this is the CANONICAL task input (project_cfg.m:101 cfg.task_data_file) but ProcessStriatumTask.m was committed 2026-05-24 10:13, aft |
| `Striatum project/processed_data/preprocessed_data5cm.mat` | .mat (v7.3) | ProcessStriatumTask.m (5 cm-bin generati | 2026-05-14 16:49 | SUPERSEDED — project_cfg.m (committed 2026-05-24 23:09) points task_data_file at the 2p5cm file. Retained only as the input generation behind tca_outp |
| `Striatum project/processed_data/preprocessed_data_control2.mat` | .mat (v7.3) | legacy/PreprocessStriatumControl2.m | 2025-01-28 | STALE — 18 months old, producer in legacy/, predates the 2026-05-07 fr_threshold alignment. Still referenced live as project_cfg.m:103 cfg.control2_da |
| `Striatum project/processed_data/preprocessed_data_control2p5cm.mat` | .mat (v7.3) | ProcessStriatumControl.m:292 | 2026-05-24 16:01 | STALE by ~7 h — ProcessStriatumControl.m committed 2026-05-24 23:08, artefact 2026-05-24 16:01. Also note: control cohort carries NO V1/CA1/DG, so any |
| `Striatum project/processed_data/preprocessed_data_control5cm.mat` | .mat (v7.3) | ProcessStriatumControl.m (5 cm generatio | 2026-05-14 13:27 | SUPERSEDED by preprocessed_data_control2p5cm.mat per project_cfg.m:102, but it is the ONLY control cache holding the PCA-dimensionality fields and V1/ |
| `Striatum project/processed_data/tca_outputs.mat` | .mat (v7.3) | Run_TCA_pipeline.m:1179 (tca_outputs_fil | 2026-05-14 15:35 | STALE, two ways. (1) Run_TCA_pipeline.m was committed 2026-06-01 16:58, 18 days after the artefact. (2) The embedded task_data has 33 fields — the 5 c |
| `Striatum project/rl_model/data/generated/` | data directory | n/a | 2026-05-20 16:50 | NEVER POPULATED. The synthetic cohorts used for parameter recovery and the model ladder were never persisted, so recovery_*.npz and fits_v*/ cannot be |
| `Striatum project/rl_model/figures/ (20 .png)` | figure (png only) | rl_model/scripts/plot_real_data.py, plot | 2026-05-23 23:01 to 2026-06-02 15:39 | SPLIT: the seven 2026-06-02 15:39 panels are CURRENT; the encoding panels (05-25) are STALE vs plot_encoding_detail.py (commit 2026-06-02 09:59); the  |
| `Striatum project/rl_model/recovery.log` | log | rl_model/scripts/run_parameter_recovery. | 2026-07-28 10:17 (tracked; content from an earlier run) | Records an INTERRUPTED recovery run over a 14-mouse x 160-trial synthetic cohort. Every recovery_*.npz on disk is 12-mouse, so this 14-mouse run's out |
| `Striatum project/rl_model/results/` | npz / dirs | rl_model/scripts/run_parameter_recovery. | 2026-05-24 to 2026-06-02 | recovery_v7.npz is ABSENT although DONE_v7 exists and UNDERSTANDING.md records v7 numbers — the v7 recovery figures/correlations have no backing array |
| `Striatum project/rl_model/results/DONE*` | sentinel | rl_model/scripts drivers | 2026-07-28 10:17 (git checkout timestamp) | These 11 sentinels are the ONLY git-tracked files in rl_model/results — the actual numbers are all untracked. DONE_v7 records a completed v7 run but t |
| `Striatum project/rl_model/results/encoding_v5/ and encoding_v6/` | .npz (neural encoding) | rl_model/scripts/run_neural_encoding.py  | v5 2026-05-25 01:21-01:22; v6 2026-05-25 07:45-07:46 | BOTH STALE — run_neural_encoding.py committed 2026-06-02, 8 days after these. FLAG: unit-level p-values across 133 units x 3 latents x 2 model familie |
| `Striatum project/rl_model/results/fits/ fits_poisson/ fits_v3/ fits_v4/ fits_v5/ fits_v6/` | .npz (synthetic-cohort fits) | rl_model/scripts/run_model_ladder.py (co | 2026-05-24 10:10 (fits, fits_poisson, fits_v3), 10:54 (v4), 11:05 (v5), 14:17-14:22 (v6) | STALE vs run_model_ladder.py (2026-06-02). CORRUPT ARTEFACT: rl_model/results/fits/mouse_00.npz is 2 bytes and raises UnpicklingError — the v1 synthet |
| `Striatum project/rl_model/results/real_fits/ real_fits_v2/ real_fits_v3/ real_fits_v4/ real_fits_v5/` | .npz (real-data fits, superseded generations) | rl_model/scripts/fit_real_data.py (commi | 2026-05-24 10:10 (real_fits, v2, v3), 11:20 (v4), 14:33 (v5) | STALE / SUPERSEDED by v6. real_fits (3 of 16 mice) and real_fits_v2 (2 of 16) are INCOMPLETE partial runs. FLAG on real_fits/M01: the model is better  |
| `Striatum project/rl_model/results/real_fits_v6_forward/ and real_fits_v6_interleaved/` | .npz (real-data fits) | rl_model/scripts/fit_real_data.py (commi | forward 2026-06-02 15:51-16:55; interleaved 2026-06-02 14:26-15:31 | CURRENT (written after the 13:00 commit). Two split schemes retained side by side — 'forward' (held-out block at the end) vs 'interleaved' — with byte |
| `Striatum project/rl_model/results/recovery_{results,poisson,v3,v4,v5,v6}.npz` | .npz | rl_model/scripts/run_parameter_recovery. | 2026-05-24 10:10 (results, poisson, v3), 10:54 (v4), 11:05 (v5), 14:22 (v6) | ALL STALE — run_parameter_recovery.py was committed 2026-06-02, 9 days after the newest of these. The model gained 5 parameters between v1 (11) and v6 |
| `Striatum project/rl_model/results/rl_latents.npz + rl_latents.mat` | .npz + .mat (v7) | rl_model/scripts/fit_real_data.py / plot | 2026-06-02 15:39 | CURRENT. This is the hand-off artefact from the RL model into the MATLAB neural analyses — the only file bridging the two halves of the project. Untra |
| `Striatum project/tcca/figures/` | figure directory | n/a | 2026-07-28 10:17 | NEVER RUN — no tcca figure has ever been produced. The tcca epoch results exist as CSV only. |
| `Striatum project/tcca/results/epoch_cross.csv` | .csv | run_epochs.py | 2026-07-28 12:05 | current (2026-07-28); untracked, same gitignore issue |
| `Striatum project/tcca/results/epoch_cross.csv` | .csv | tcca/scripts/run_epochs.py:147 | 2026-07-28 12:05 | CURRENT vs working tree, code uncommitted. Good practice: reports rotation angles WITH a matched floor_* column, so rotation can be judged against cha |
| `Striatum project/tcca/results/epoch_dims.csv` | .csv | tcca/scripts/run_epochs.py:145 | 2026-07-28 12:05 | CURRENT vs working tree, code uncommitted. Same coverage imbalance as epoch_metrics. Any test run over these 1681 rows as independent units is pseudor |
| `Striatum project/tcca/results/epoch_metrics.csv` | .csv | Striatum project/tcca/scripts/run_epochs | 2026-07-28 12:05 | CURRENT with code (run 2026-07-28 12:05 against unchanged run_epochs.py and preprocessed_data2p5cm.mat of 23 May). BUT: NOTES.md:20 claims 'Results no |
| `Striatum project/tcca/results/epoch_metrics.csv` | .csv | tcca/scripts/run_epochs.py:144 | 2026-07-28 12:05 | CURRENT vs the WORKING TREE only. run_epochs.py mtime 2026-07-28 11:58 (7 min before the CSV) but its last commit is 2026-06-17 14:45 and git diff sho |
| `Striatum project/tcca/results/epoch_weights.csv` | .csv | tcca/scripts/run_epochs.py:146 | 2026-07-28 12:05 | CURRENT vs working tree, code uncommitted. |
| `cosyne2025/figures/` | figure (.eps only, 21 files) | unknown — no producer script in the repo | 2024-10-23 (dir), files Oct 2024 vintage | SUPERSEDED conference artefacts, DMS-only, ~21 months old. Provenance UNKNOWN — no producing script found. The whole /cosyne2025 path is gitignored at |

### Artefact detail

**`Striatum project/ (~140 loose .svg in the project root)`** — _Task__* and _Control__* families: Raw_FR / Z-Scored x Pooled / Hierarchical x Spatial / Temporal (+ _Subpops), Neuron_Types__* x same grid, {Spatial,Temporal}_Area_x_Type, MeanDist_{Epochs,Trials}_Hierarchical_{MSN,FSN,TAN,RS}, {Spatial,Temporal}_Skewness_Profile_Hierarchical, Scatter_Tr1_vs_Tr{4,10,21}, KDEDensity_Tr1_vs_Tr{4,10,21}. Plus Area_Activity_{raw,z}_{Pooled,Hierarchical}(_TrialEvo), Stability_AllGroups_{Pooled,Hierarchical}_{Raw,ZScored}(_ByArea,_WholePopulation), Behavioral_{Lick_Heatmaps_Group1-3, Epoch_ZError, ZError_AllTask, Stability_AllGroups_Epochs, Mouse3_VerticalAlign}, Behavioural_Evolution_3Groups_Yoked, Decoding_{Evolution_3Groups_Yoked, Lick_Prediction_3Groups, Spatial_Bin_Error_3Groups, Spatial_Entropy_Bin_Profile_3Groups}, Group_{1,2,3}_Cross_Modal_Correlations

  *Currency:* THIS IS THE CURRENT MATLAB FIGURE SET, and it lives loose in the repo root rather than in figures/. MIXED DG TREATMENT: the DG-exclusion commit landed 2026-05-24 23:08, but panels dated 22:55-23:05 (e.g. _Task__Raw_FR_-_Hierarchical_Spatial_Subpops, _Task__MeanDist_*, _Task__*_Skewness_Profile_*) PREDATE it while panels dated 23:20-23:47 postdate it — so one apparently-uniform figure set spans two different unit-inclusion rules. No .csv/.mat companions. Gitignored (*.svg).

**`Striatum project/Auto_Reports/`** — Auto-generated experiment data summary metrics decks for the 2026-05-14 data state

  *Currency:* PROVENANCE UNKNOWN / ORPHANED. No script in the repository writes to Auto_Reports/, so these decks cannot be regenerated. Gitignored (*.pptx). The two files differ 27x in size for the same date, suggesting the smaller 17:26 file is a truncated or partial run.

**`Striatum project/BehaviourOnly/`** — 1105_M2, 1106_M3, 1201_M1, 1206_M2, 1212_M3, 1215_M2, 1217_M4, 1219_M1

  *Currency:* STALE / LEGACY — only producer is in legacy/, files 18 months old. Note 1215_M2, 1217_M4 and 1219_M1 appear in NO other directory in the repo: they are behaviour-only animals with no Neuropixels counterpart, so any 'n animals' statement must distinguish the behaviour cohort from the recording cohort. Gitignored.

**`Striatum project/CCA_Results/ (24 .svg)`** — Three distinct generations in one folder. Oldest (Mar 4): Network_Connectivity_Group, Spatial_Corr_Group, Spatial_Precession_Group, Trial_Corr_Epochs, Trial_Precession_Epochs — no matching .mat on disk at that date.

  *Currency:* Mar-04 set = ORPHANED (source .mat gone). May-14 set = current vs v2, superseded pipeline. May-15 set = STALE (see v3 .mat entry). No .csv/.mat companion saved next to any panel, so none can be replotted without re-running.

**`Striatum project/CCA_Results/Striatum_CCA_Results_2026_05_14.mat`** — group_results, 33 fields — every metric in 4 flavours (raw / _held / _shuff / _held_shuff): all_bins_corr, all_bins_precession_curve, all_bins_precession_idx, trial_corr_{early,pre,post}, trial_precession_{early,...}_curve/idx. saved_config{max_shift_bins, min_units_per_region, n_components_reduced, n_shuffles, num_ccs_analyze, pca_selection_method, pca_variance_threshold}. analysis_lp / is_learner 14x1.

  *Currency:* CURRENT relative to v2 code (CCA_striatum_spatial_v2.m last commit 2026-05-08) but v2 is SUPERSEDED by v3. Useful chiefly because it is the only .mat that stores held-out AND shuffled variants side by side for the precession metrics.

**`Striatum project/CCA_Results/Striatum_CCA_v3_2026_05_15.mat`** — group_results{area_x, area_y, pair_name, per_epoch}; cfg (62 fields, incl. fr_threshold_hz, cca_fold_seed, cv_splits, area_pairs_v5, epoch_names, expert_starts_at, central_window_bins, behav_targets, au_to_cm, bin_size_cm, control_epoch_windows); analysis_lp (1x14); is_learner (1x14, uint8); learning_points (1x14)

  *Currency:* STALE. Both CCA_striatum_spatial_v3.m (2026-05-24 10:13) and project_cfg.m (2026-05-24 23:09) were committed 9 days AFTER this artefact, and that project_cfg commit is the one that repointed cfg.task_data_file from the 5 cm to the 2.5 cm file. So this .mat was fit on 5 cm data under a superseded config. All 9 StriatumCCAv3_*.svg panels (2026-05-15 11:53-11:56) inherit the staleness.

**`Striatum project/RawData/`** — 17 task mice with <id>_raw.mat: 409, 418, 507, 523, 614, 624, 703, 727, 730, 731, 822, 823, 1105, 1106, 1201, 1206, 1212. Only 16 have <id>_neurontype2025.mat — 507_raw.mat (109.0 MB) has NO neurontype file. 5 mice have V1 probes: <id>_V1_raw.mat + <id>_v1_neurontype2025.mat for 1105, 1106, 1201, 1206, 1212. Metadata: Neuropixels_Depth_Data.csv (697 B, 17 mice x ACC/Striatum/DMS/DLS boundaries in um); Neuropixels_V1_Depth_Data.csv (195 B, 5 mice x V1/CA1/DG); V1_depth.txt (430 B, free text).

  *Currency:* IMMUTABLE raw. COHORT-DEFINING CONSTRAINTS visible here: (a) only 3 mice (1212, 1206, 1201) have BOTH CA1 and DG boundaries — 1105 and 1106 are V1-only — which is exactly why every CA1/DG pair in the CCA and tcca tables has n_animals 0-2; (b) DLS Start/End are BLANK for 731, 823, 1206 and 703, and ACC is blank for 1206, so those animals contribute no DLS/ACC; (c) V1_depth.txt records structures absent from the CSV (CA3 1100-1400 and LG 600-1000 for 1201; LP 0-400 for 1206; RHP/SC for 1105/1106) and flags 'L1 might be damaged' for 1212, 1206, 1106 and 1105 — a V1 laminar caveat not encoded anywhere machine-readable. Gitignored (*.mat, *.csv, *.txt).

**`Striatum project/RawData/LFP/`** — 384-channel raw voltage per session. lfp_mapping.txt (57 B) is the ONLY provenance: '1212 - 16.33GB / 614 - 11.38GB / 727 - 11.65GB / 731 - 11.48GB'

  *Currency:* IMMUTABLE raw, but PROVENANCE IS FRAGILE: the filenames are Finder-style space-numbered duplicates carrying no mouse ID, and the mouse assignment rests entirely on a filesize-to-ID lookup in a 57-byte text file. This is consistent with (and probably the root of) the LFP README's statement that 1212's voltage-probe identity is unverified and that area labels are withheld for it. Any LFP claim inherits this. Gitignored.

**`Striatum project/RawDataControl/`** — 6 control mice with <id>_raw.mat + <id>_neurontype2025.mat: 407, 408, 513, 515, 817, 1205. Three (513, 515, 817) additionally have <id>_v1_raw.mat + <id>_v1_neurontype2025.mat (added 2026-05-07/08). Subdirectory '408_M2_all data_1ms(raw)/'. Metadata: Neuropixels_Depth_Data_control.csv (272 B), V1_depth.txt (255 B), mice_info2.xlsx (9.5 kB).

  *Currency:* IMMUTABLE raw. Note the control cohort HAS V1 recordings for 3 animals in raw form, yet preprocessed_data_control2p5cm.mat carries no is_v1 mask — the V1 control data has been acquired but not propagated into the current control cache. Gitignored.

**`Striatum project/RawDataControl2/`** — 7 sessions, <id>_raw.mat ONLY: 1011, 1103, 1107_M1, 1107_M2, 316, 317, 624. NO <id>_neurontype2025.mat for any of them. Metadata: Neuropixels_Depth_Data_control2.csv (284 B), mice_info3.xlsx (16.8 kB).

  *Currency:* IMMUTABLE raw, but the control2 cohort has NO spike-type classification at all — consistent with preprocessed_data_control2.mat holding only 5 fields and no final_neurontypes. Any MSN/FSN/TAN/RS breakdown is impossible for control2. Gitignored. Note 624 appears in BOTH RawData/ (99.1 MB, task) and RawDataControl2/ (17.6 MB, control) — the same animal ID in two cohorts, a collision risk for any ID-keyed join.

**`Striatum project/cca/figures/ (114 .png)`** — Families: stage2_comm_strength_* , stage2_ifi_win01..win10_* (x {committed, committed_partial, s5cm_res_fsexcl, s5cm_res_fsincl, s5cm_sig_fsexcl, s5cm_sig_fsincl}), ifi_fs_win10_{plain,partial}, partial_cca_committed, committed_ifi, null_comparison, directionality_{plain,partial}, sweep_pairs_spatial_{angle,p}, sweep_parcoords_focus_p_naive_vs_expert, subspace_similarity_summary_committed_partial

  *Currency:* Mixed. The 2026-06-01 17:55-17:57 committed panels and 2026-06-02 10:37 directionality panels are CURRENT; the 2026-05-23 14:46-14:47 s5cm panels are on the superseded 5 cm bin width. All gitignored (*.png). Violates the project rule that figures ship as .svg + .png pairs — there is not one .svg in this directory.

**`Striatum project/cca/figures/committed_ifi.csv + sweep_committed_results.csv`** — committed_ifi: pair, window (1-10), n_naive, ifi_naive, p_naive_vs0, n_expert, ifi_expert, p_expert_vs0, ifi_diff_expert_minus_naive, p_naive_vs_expert. sweep_committed_results: pair, fs, n_dims_{naive,expert}, cc_{naive,expert}, d_cc, p_naive_vs_expert, ifi_w3, p_ifi_w3, angle_minus_floor.

  *Currency:* committed_ifi.csv is STALE by 4 minutes — committed_ifi.py was committed 2026-05-24 10:53, the CSV written 10:49. Flag: committed_ifi.csv is 100 uncorrected tests over 10 windows x 10 pairs; sweep_committed_results.csv reports p_naive_vs_expert for 20 pair x fs cells with no correction and n as low as 0-2 dims.

**`Striatum project/cca/figures/directionality_partial.csv`** — per pair: n_animals, IFI per epoch + t-test vs 0, RM-ANOVA, Holm post-hoc, trend, peak_lag mean/median in bins and cm, n_sig_dims_pooled.

  *Currency:* current with round-16 code; gitignored

**`Striatum project/cca/figures/epoch_stats_partial.csv`** — per pair x metric(cc,ifi): n_dim per epoch, n_animals, one-way ANOVA F/p over dims, Tukey ni/ie/ne p, per-dim linear trend, per-animal RM-ANOVA F/p, Holm post-hoc, per-animal trend. The committed (partial) pipeline's epoch statistics.

  *Currency:* current with round-16 code (2026-06-02); gitignored (root .gitignore *.csv) so never committed

**`Striatum project/cca/figures/epoch_vs0_{plain,partial}.csv + epoch_stats_{plain,partial}.csv + directionality_{plain,partial}.csv`** — epoch_vs0: per (pair, metric in {cc,ifi}, epoch) n_dims, dim_mean, dim_vs0_p_wilcoxon, n_animals, animal_mean, animal_vs0_p_ttest. epoch_stats: n_dim_{naive,inter,expert}, n_animals, anova_dim_F/p, tukey_{ni,ie,ne}_p, trend_dim_slope/p, rm_anova_F/p, posthoc_*_p, trend_animal_slope/p. directionality: ifi per epoch + p_vs0_ttest, rm_anova_F/p, holm_{ni,ie,ne}_p, trend_slope/p, peak_lag_mean/median_bins, peak_lag_mean_cm, n_sig_dims_pooled. All 10 pairs (DMS-DLS, DMS-ACC, DLS-ACC, V1-DMS, V1-DLS, V1-ACC, CA1-DMS, CA1-DLS, CA1-ACC, CA1-V1) — no DG pairs.

  *Currency:* CURRENT (written 1 min before the producing commit). CRITICAL STRUCTURAL FLAG: each file reports a dimension-level test and an animal-level test side by side. The dimension-level columns (anova_dim_p, dim_vs0_p_wilcoxon, n_dim_* = 31-58) treat CCA dimensions as independent replicates although they are nested within 0-7 animals — pseudoreplication. Every dim-level p<0.05 in these tables has a non-significant animal-level counterpart (see recorded_numbers). n_animals is 0 for V1-DLS and CA1-DLS (no data at all) and 1-3 for four more pairs; several p-values are computed on n=2 (V1-DMS, CA1-V1) or n=1 (CA1-ACC, CA1-DMS) animals.

**`Striatum project/cca/figures/null_comparison.csv`** — per pair x epoch: n_learn, nsig_trials, nsig_circshift, cc_trials, cc_circshift, ifi_trials, ifi_circshift — the trial-perm vs circshift null dependence.

  *Currency:* 2026-05-24, pre-round-14 (200 surrogates era, plain not partial); gitignored

**`Striatum project/cca/figures/null_comparison.csv`** — pair, epoch, n_learn, nsig_trials, nsig_circshift, cc_trials, cc_circshift, ifi_trials, ifi_circshift — trial-shuffle null vs circular-shift null, head to head

  *Currency:* CURRENT. Scientifically load-bearing: the circshift null yields systematically MORE significant dimensions than the trial null (e.g. DMS-DLS naive 47 vs 15) at systematically LOWER cc (0.1429 vs 0.2413), i.e. null choice changes both the count and the effect size. Gitignored.

**`Striatum project/cca/figures/sweep_enrichment_p_naive_vs_expert.csv + sweep_enrichment_p_ifi_w3.csv + sweep_enrichment_focus_*.csv`** — Per hyperparameter level: n_cells, frac_p<0.05, enrichment_vs_chance (= frac / 0.05). Rows for pair, k_rule, min_units, lp_consec, z, bin, cca, fs. The focus_ variants add pair_baseline and enrichment_vs_pair.

  *Currency:* CURRENT. These are the tables that quantify how much of the CCA result survives hyperparameter choice — see recorded_numbers. Gitignored.

**`Striatum project/cca/figures/sweep_summary.csv + sweep_summary_temporal.csv`** — 270 rows = 27 spatial configs x 10 pairs (and 20 rows = 2 temporal configs x 10 pairs), each with cc_naive/cc_expert/d_cc/p_naive_vs_expert/n_sig/ifi_w3/p_ifi/angle_ne/sh_floor/angle_minus_floor/gini. Backs the round-8 'reorientation is the only robust effect' claim.

  *Currency:* 2026-05-23/24, round-8 sweep (2-epoch, pre-committed-config); gitignored

**`Striatum project/cca/figures/sweep_summary_spatial.csv (+ .xlsx)`** — One row per (sweep config x area pair). Cols: tag, bin, cca, fs, z, k_rule, min_units, lp_consec, pair, n_learn, cc_naive, cc_expert, d_cc, p_naive_vs_expert, n_sig_naive, n_sig_expert, ifi_w3, p_ifi_vs_0, angle_ne, sh_floor, angle_minus_floor, gini_naive, gini_expert. Companion sweep_summary.csv (270 rows, 2026-05-23 18:02) and sweep_summary_temporal.csv (20 rows, 2026-05-23 18:11).

  *Currency:* CURRENT (summarise_sweep.py last commit 2026-05-24 10:13, artefact 09:49 same day — 24 min before, so marginally pre-commit). Gitignored (*.csv, *.xlsx). This is the single most manuscript-relevant table on disk: it is the forking-paths audit for the whole CCA claim.

**`Striatum project/cca/results/ (3298 files)`** — Hyperparameter sweep, full factorial. Families: stage2_s2p5_* (547), stage3_s2p5_* (547), stage2_s5cm_* (540), stage3_s5cm_* (537). Grid axes recoverable from filenames and summarise_sweep.py PARAM_COLS = [tag, bin, cca, fs, z, k_rule, min_units, lp_consec]: bin = 2.5 cm | 5 cm; cca = signal (sig) | residual (res); fs = FS interneurons incl (fsI) | excl (fsX); z = z0 | z1; k_rule = fixed3/5/10/20/30, var75/85/95, samples15/25/40; min_units = 4/6/10; lp_consec = 7/8. Hand-run configs: stage2_main, stage2_committed_{circshift, circshift_fsincl, circshift_partial, circshift_fsincl_partial, trials, tpe15}, stage2_res_fs{incl,excl}, stage2_resid_fs{I,X}(_z0,_z1), stage2_fs_{included,excluded}, stage3_{main,z0,z1,t20,t40,committed}, partial{,_committed,_z0,_z1}.pkl

  *Currency:* CURRENT vs the driver commits (artefacts written the same day as or after them). But 100% UNTRACKED (root .gitignore *.pkl, *.done) and 3 GB, so this is a single-copy, single-machine artefact. The 1085 .done sentinels cover only the sweep pkls — the ~40 hand-run committed/main configs that actually underwrite the written claims have NO completion record.

**`Striatum project/cca/results/stage1_validation.npz`** — 126 rows x: animal, pair (<U7), epoch (<U12), role (<U10), lp, k, samples_per_pc, held_out_cc1, in_sample_cc1, var_x, var_y; scalar yoked_lp = 42

  *Currency:* CURRENT. Notable as the one artefact that stores held_out_cc1 AND in_sample_cc1 side by side, i.e. the in-sample optimism is directly measurable from it. yoked_lp=42 is a fixed yoked learning point for non-learner/control alignment.

**`Striatum project/cebra_data/cebra_mouse{1..16}_data.mat`** — Per mouse: neural_data (n_trials x 50 bins x n_units; mouse1 = 357 x 50 x 133), lick_rate (50 x n_trials), velocity (50 x n_trials), lick_errors (n_trials x 1), position (50 x 1), area_labels (n_units cell), neuron_types (1 x n_units), bin_size_cm, change_point, learning_point, group_id, mouse_id

  *Currency:* STALE — save_for_cebra.m's last commit is 2026-05-08 13:15, the day AFTER these were written. Gitignored (*.mat).

**`Striatum project/cebra_results/ (12 fig_*.png) and cebra_results/embedding_panels/ (16 fig_emb_mouse*.png)`** — fig_consistency_matrix, fig_consistency_vs_decoding, fig_decoder_confusion, fig_decoder_r2_{distribution,heatmap,paired,summary}, fig_decoder_rmse_summary, fig_decoding_error_per_bin, fig_embedding_density_by_epoch, fig_embedding_grid_by_position, fig_headline_composite; embedding_panels/fig_emb_mouse1..16

  *Currency:* STALE — cebra_plot_results.py was committed 2026-05-08 13:15, ~3 h after these were written. NO .svg for any of the 28 panels — violates the svg+png rule, so none is publication-ready as-is. All gitignored.

**`Striatum project/cebra_results/cebra_config.json`** — n_animals=16, seed=42, label_keys=[position, lick_rate, lick_errors], area_subsets=[all, DMS, DLS, ACC, V1], output_dimension=3, max_iterations_single=15000, max_iterations_multi=15000, batch_size=512, learning_rate=0.0003, temperature_mode=auto, time_offsets=10, distance=cosine, conditional=time_delta, architecture=offset10-model, hybrid=false, device=cuda_if_available, decode_target=position, decoder_test_frac=0.25, ridge_alpha=1.0

  *Currency:* STALE vs cebra_analysis.py (commit 2026-05-08 13:15, ~10 h after). Seeded (seed=42) — good. FLAG: decoder_test_frac=0.25 is a SINGLE held-out split, not k-fold, and ridge_alpha=1.0 is a fixed unfitted value, so the reported decoder R2 has no CV variance estimate and no tuned regulariser. area_subsets omits CA1 and DG entirely despite those areas existing in the dataset.

**`Striatum project/cebra_results/cebra_results.mat`** — single_decoder_r2 (16x5), single_decoder_rmse (16x5), consistency_scores (1x240), dataset_ids (1x16 cell), label_keys (1x3 cell), area_subsets (1x5 cell), plus embedding_mouse1..embedding_mouse16 (n_samples x 3; 3461 samples for mouse3 to 14132 for mouse2)

  *Currency:* STALE vs producer commit. Carries the actual 3-D embeddings that the .npz omits — the only copy of them. Gitignored (*.mat).

**`Striatum project/cebra_results/cebra_results.npz`** — single_decoder_r2 (16 animals x 5 area_subsets), single_decoder_rmse (16 x 5), consistency_scores (240), consistency_pairs (240 x 2, <U7), dataset_ids (16), label_keys (3), area_subsets (5)

  *Currency:* STALE vs producer commit. 240 = 16x15 ordered animal pairs for the consistency matrix. This is one of only two numeric result files tracked in git for the entire project.

**`Striatum project/figures/ (501 files)`** — Families: lickquant_animal* (16), cross_area_lickerror_animal* (16), cd_area_animal* (6), cd_area_diseng_animal* (6), codingdimension_animal* (4), interarea_corr_bins_animal* (2), decoder_{contour,average_error_trials,average_error_position,abs_error_heatmap}_animal* , pca_3d_animal{3,4}_{all,acc,dms,dls}, pca_animal3_{prepost,allareas}, popFF_animal3_prepost, zactivity_animal3_prepost, {TrialCorrelation,SpatialCorrelation,SpatialPrecession,Precession}_{DMS_DLS,DMS_ACC,DLS_ACC}, tca_supermouse_task_{trial,spatial,neuron}, tca_spatial_new, TCA5_spatial_task, population_stability_control, generalised_variance_task, stability_vs_behaviour_task, ensemble1_heat, single_ensemble_decoding_error, velocity_{average,allphases,allphases_heat}, firing_rate_animal*

  *Currency:* ENTIRE DIRECTORY STALE — the newest file is 2025-04-24 while its three producing scripts were last committed 2026-05-24 (IntegratedAll_v1, SpatioTemporalActivityEvolution) and 2026-06-01 (Run_TCA_pipeline), i.e. every panel predates its producer by 13+ months and predates both the 2026-05-07 fr_threshold alignment and the 2026-05-24 DG exclusion. NO .csv/.mat source data saved alongside any panel — violates 'save raw data alongside generated figures', so nothing here is replottable. All gitignored.

**`Striatum project/legacy/CCA_Results/Striatum_CCA_Results_2026_05_07.mat`** — identical 33-field group_results schema + saved_config (7 fields) + analysis_lp/is_learner 14x1 as the 2026-05-14 file

  *Currency:* SUPERSEDED — deliberately moved to legacy/. Dated the same day as the fr_threshold=0.02 Hz alignment, so it may straddle that change.

**`Striatum project/legacy/Figures/ (121 files)`** — Includes CCAVsBehaviour_DLS_ACC.png (newest, 2025-10-30), avg_fr_{523,614}.svg (oldest, 2024-10-02). Plus loose legacy/ png: TCA_nmu_{crossval_error,spatial_factors,trial_factors,unit_factors}, activity_correlation_control{1..5}, all_tca_patterns.svg

  *Currency:* SUPERSEDED audit trail. All producers are in legacy/. Gitignored.

**`Striatum project/lfp/figures/ (current set)`** — sanity_audit_overview_v2, sanity_audit_raw_examples_v2, sanity_audit_event_timing, signal_identity — each as .png AND .svg

  *Currency:* CURRENT. These four are the only LFP figures the README endorses as defensible. Correctly paired png+svg with saved source npz. Gitignored.

**`Striatum project/lfp/figures/_quarantined_unaligned_learning/`** — Per-mouse and cohort LFP band-power evolution across learning

  *Currency:* QUARANTINED — produced before the voltage-to-VR timing offset, source-band provenance and 1212 probe identity were established; the 30-80 Hz band is contaminated by a persistent ~74 Hz peak. The driver now refuses to run without --allow-unaligned, excludes 1212 and omits that band. The companion learning_evolution_summary.csv that the results README says is preserved as audit trail is NOT on disk. This is the only LFP learning result and it is unusable.

**`Striatum project/lfp/figures/_superseded_absolute_threshold/`** — Earlier sanity figures thresholded at SD > 0.02 in undocumented voltage units

  *Currency:* INVALID — explicitly quarantined by lfp/figures/README.md: 'must not be presented'. Retained as audit trail only.

**`Striatum project/lfp/figures/cca_cross_area.{png,svg} + decode_position.{png,svg}`** — LFP cross-area CCA and LFP position decoding

  *Currency:* STALE — run_decode_cca.py was committed 2026-07-28 10:16 (the 'lfp: harden voltage integrity audit' work), 15 days after these figures. No saved source .npz/.csv exists for either, so neither is replottable and neither is covered by the results README's list of defensible outputs. These are the only LFP *analysis* figures (as opposed to integrity figures) and they are the stalest.

**`Striatum project/lfp/results/`** — README.md lists sanity_summary.csv, sanity_timing_summary.csv, signal_identity_summary.csv, sanity_windows_<mouse>.csv as the defensible summaries — NONE are on disk. Only sanity_audit_{1212,614,727,731}.npz, sanity_figure_examples_and_psd.npz, signal_identity_figure_data.npz survive.

  *Currency:* npz from 2026-07-12/13, i.e. current with the hardened audit; the CSVs that hold the quoted per-peak dB and integrity numbers are gone (lfp/.gitignore excludes results/*.csv)

**`Striatum project/lfp/results/sanity_audit_{614,727,731,1212}.npz`** — Per mouse: window_start_sample, window_exact_zero_fraction, window_median_channel_rms, window_median_channel_abs, window_max_abs (8400 windows for 614/727, 11400 for 1212, matching session length), channel_exact_zero_fraction (384 channels), world, position_au, artefact_z

  *Currency:* CURRENT (written hours before the producing commit, no later commit touches the script). Gitignored (lfp/.gitignore results/**/*.npz).

**`Striatum project/lfp/results/sanity_figure_examples_and_psd.npz`** — sampling_rate_hz = 1000; selected_channels_zero_based (6); per mouse {614,727,731,1212}: ordinary/event frequency_hz + normalised_psd (1025 bins each), ordinary/event example_start_sample, ordinary/event example_voltage (4000 samples x 6 channels)

  *Currency:* CURRENT. Good practice — this is the exact plotted data saved alongside the figure, so sanity_audit_overview_v2 and sanity_audit_raw_examples_v2 are replottable without touching the 50 GB raw voltage files. Gitignored.

**`Striatum project/lfp/results/signal_identity_figure_data.npz`** — selected_channels_zero_based (24); nominal_areas (3, <U3); correlation_bands (3, <U20); per mouse: frequency_hz (1025), stored_psd (1025), common_median_referenced_psd (1025), identity_metrics (3), nominal_depth_band_correlations (3x3)

  *Currency:* CURRENT. Gitignored. Note the area labels are 'nominal' by construction — the README states area labels are withheld for 1212 because its voltage-probe identity is unverified.

**`Striatum project/mat_results/run001..run010`** — GPFA latent trajectories (seqTrain(iTrial).xsm, [xDim x numBins]) + fitted params. Runs are distinguished ONLY by the toolbox runIdx and the latent dimensionality in the filename: run001 = xDim 2; run002/003/004/005/007 = xDim 3 (differing 200x in size, so different unit/trial subsets); run008 = xDim 8. run006 and run010 are EMPTY DIRECTORIES; run009 does not exist.

  *Currency:* ORPHAN / LEGACY. The only producer is legacy/GPFA_striatum.m; no live script in the repo references GPFA. Worse, that script sets runIdx=10, xDim=8 and loads 'mat_results/run010/gpfa_xDim08.mat' (line 37) — run010 is empty, so the script as committed would fail; the xDim08 data actually sits in run008. Nothing here is reproducible without reconstructing the run-index bookkeeping.

**`Striatum project/popsim/data/generated/recovery_benchmark.json`** — dynamics ar1, n_timesteps 6000, k 5; 10 rows (scenario, cca1, partial_cca1, drop_frac, peak_lag, n_strong, latent_cca1, pop_corr, epoch_lags, expected, passed).

  *Currency:* current; the ONE result artefact in the repo that is actually committed to git

**`Striatum project/popsim/data/generated/recovery_benchmark.json`** — dynamics=ar1, n_timesteps=6000, k=5; 10 scenario rows each with cca1, partial_cca1, drop_frac, peak_lag, peak_lag2, n_strong, latent_cca1, pop_corr, epoch_lags, expected, passed. Scenarios: no_coupling, zero_lag, lagged, mediated, epoch_varying, bidirectional, common_input, rotated_subspace, partial_mediation, noise_correlation. ALL 10 passed=true.

  *Currency:* CURRENT and one of only a handful of TRACKED result artefacts. This is the method-validation ground truth for the whole CCA approach: it demonstrates on synthetic data that partial CCA collapses under mediation (drop_frac 0.724) and common input (0.740) but survives a direct path (0.0097), and that population correlation can be 0.993 while latent CCA is only 0.197 (noise_correlation scenario). Structurally the strongest artefact in the repo.

**`Striatum project/popsim/data/generated/{zero_lag,lagged,epoch_varying,mediated,no_coupling}/metadata.json`** — Ground-truth generative specs per scenario

  *Currency:* CURRENT, tracked. But only 5 scenario directories exist while recovery_benchmark.json reports 10 scenarios — bidirectional, common_input, rotated_subspace, partial_mediation and noise_correlation have no generated/ directory and no metadata.json. All *.npy arrays are gitignored by popsim/.gitignore and absent from disk (declared regenerable deterministically from generate_datasets.py).

**`Striatum project/presentations/`** — 20250128_LabMeeting, 20250205_Update, 20250218_Update, 20250305_Update, Ensemble_Analysis_Striatum_20250619, 'Lab Meeting 20250627', StriatumUpdate, StriatumUpdate_collab, StriatumUpdate_2026{0312,0319,0401,0525}. Newest = StriatumUpdate_20260525.pptx (29.0 MB).

  *Currency:* Historical record. Gitignored (*.pptx). Note two stale PowerPoint lock files left in the PROJECT ROOT (not in presentations/): ~$StriatumUpdate_20260312.pptx and ~$StriatumUpdate_20260319.pptx, 165 B each.

**`Striatum project/processed_data/all_data.mat`** — single struct all_data, 19 fields: average_{ACC,CA1,DG,DLS,DMS,V1}_fr, average_lick_rate, avg_fr_all, corrected_licks, corrected_vr_time, final_areas, final_neurontypes, final_spikes, mouseid, npx_time, vr_position, vr_reward, vr_trial, vr_world

  *Currency:* CURRENT vs producer (OrganiseStriatumDataIncV1.m last commit 2026-05-08, artefact 2026-05-14). Post-dates the fr_threshold=0.02 Hz alignment (2026-05-07).

**`Striatum project/processed_data/all_data_control.mat`** — all_data, same 19 fields as task version (incl. average_CA1_fr / average_DG_fr / average_V1_fr)

  *Currency:* CURRENT (producer last commit 2026-05-15 11:42 file mtime; git-tracked script older than artefact). Post-dates fr_threshold alignment.

**`Striatum project/processed_data/all_data_control2.mat`** — all_data, only 11 fields: avg_fr_all, corrected_licks, corrected_vr_time, final_areas, final_spikes, mouseid, npx_time, vr_position, vr_reward, vr_trial, vr_world. NO final_neurontypes, NO per-area average_*_fr.

  *Currency:* STALE — predates the 2026-05-07 fr_threshold=0.02 Hz alignment by 15 months, and the producer now lives in legacy/. CLAUDE.md flags _control2 caches explicitly as needing regeneration before any Task/Control parity claim.

**`Striatum project/processed_data/cross_spatial_decoding_results.mat`** — decoding_results struct with 6 area fields: ACC, CA1, DG, DLS, DMS, V1

  *Currency:* CURRENT vs code (CrossSpatialBinDecoding.m last commit 2026-05-08) but built on the 5 cm data generation, i.e. one bin-width regime behind the current project_cfg. Script also self-caches (line 56 exist-check + load), so re-running will silently reuse this file rather than recompute.

**`Striatum project/processed_data/preprocessed_data2p5cm.mat`** — preprocessed_data, 43 fields: binned_spikes_trials, npx_times_trials, spatial_binned_data, spatial_binned_fr_all, corridorData, darkData, trialData, trial_metrics, trial_lick_{errors,positions,fractions}, trial_average_fr_{dms,dls,acc,v1}, trial_sem_fr_*, is_{dms,dls,acc,v1,ca1,dg}, final_neurontypes, n_trials, change_point_mean, dark/stim_dimensionality_V1, temp_binned_dark_fr, shuffled_lick_error_{means,stds}, mean_cross_area_corr_{DMSACC,DMSDLS,V1ACC,V1DMS} and mean_abs_ variants

  *Currency:* STALE by ~22 h — this is the CANONICAL task input (project_cfg.m:101 cfg.task_data_file) but ProcessStriatumTask.m was committed 2026-05-24 10:13, after the 2026-05-23 12:19 artefact. Any CCA/TCA result loading it inherits the pre-commit generation.

**`Striatum project/processed_data/preprocessed_data5cm.mat`** — preprocessed_data, 33 fields. HAS z_spatial_binned_fr_all + zscored_lick_errors; LACKS binned_spikes_trials, npx_times_trials and all mean_cross_area_corr_* present in the 2p5cm file.

  *Currency:* SUPERSEDED — project_cfg.m (committed 2026-05-24 23:09) points task_data_file at the 2p5cm file. Retained only as the input generation behind tca_outputs.mat and the CCA v3 .mat.

**`Striatum project/processed_data/preprocessed_data_control2.mat`** — preprocessed_data, only 5 fields: firing_rates_per_bin, is_acc, is_dls, is_dms, trialData

  *Currency:* STALE — 18 months old, producer in legacy/, predates the 2026-05-07 fr_threshold alignment. Still referenced live as project_cfg.m:103 cfg.control2_data_file.

**`Striatum project/processed_data/preprocessed_data_control2p5cm.mat`** — preprocessed_data, 32 fields. Area masks are ONLY is_acc / is_dls / is_dms — no is_v1, is_ca1, is_dg. Has binned_spikes_trials, npx_times_trials, z_spatial_binned_fr_all, zscored_lick_errors, mean_(abs_)cross_area_corr_{DMSACC,DMSDLS}.

  *Currency:* STALE by ~7 h — ProcessStriatumControl.m committed 2026-05-24 23:08, artefact 2026-05-24 16:01. Also note: control cohort carries NO V1/CA1/DG, so any Task-vs-Control comparison is restricted to DMS/DLS/ACC.

**`Striatum project/processed_data/preprocessed_data_control5cm.mat`** — preprocessed_data, 41 fields. Unlike the 2p5cm control it DOES carry is_v1/is_ca1/is_dg plus pca_{dark,stim}_dimensionality_{acc,all,dls,dms} and stim/dark_dimensionality_V1 — fields absent from the current 2p5cm control file.

  *Currency:* SUPERSEDED by preprocessed_data_control2p5cm.mat per project_cfg.m:102, but it is the ONLY control cache holding the PCA-dimensionality fields and V1/CA1/DG masks. Dropping it loses those variables.

**`Striatum project/processed_data/tca_outputs.mat`** — supermouse_tensor_raw (30 trials x 50 bins x 1693 units), supermouse_combined_valid (30 x 50 x 1630), best_mdl{lambda,u}, best_n_factors, avg_learning_point, learning_points_task (14 animals), combined_labels{area,group,mouse,neurontype}_labels_all, labels_valid, tensor_info{bins, mouse_units_starts, mouse_units_ends, n_animals_task, n_animals_control, n_animals_total, trials_aligned}, cfg (15 fields), plus an EMBEDDED copy of task_data (33 fields)

  *Currency:* STALE, two ways. (1) Run_TCA_pipeline.m was committed 2026-06-01 16:58, 18 days after the artefact. (2) The embedded task_data has 33 fields — the 5 cm schema, not the 43-field 2p5cm schema — so the tensor was built on the superseded bin width. (3) Run_TCA_pipeline.m:19 now drops DG units from all figures/analyses (dated 2026-05-24); this tensor predates that exclusion and its 1693 units still include DG.

**`Striatum project/rl_model/data/generated/`** — EMPTY

  *Currency:* NEVER POPULATED. The synthetic cohorts used for parameter recovery and the model ladder were never persisted, so recovery_*.npz and fits_v*/ cannot be re-scored against their own ground-truth inputs — only the summary u_true/p_true arrays inside the npz survive.

**`Striatum project/rl_model/figures/ (20 .png)`** — fig_behaviour, fig_example_latents, fig_latent_recovery, fig_param_recovery (2026-05-24 14:22); fig_diag_epoch_lick_profiles, fig_diag_epoch_velocity_profiles, fig_diag_redesign_sanity, fig_diag_velocity_v2, fig_diag_velocity_v3 (2026-05-23/24); fig_neural_encoding (05-25 07:46), fig_encoding_examples, fig_encoding_stats (05-25 08:08); fig_epoch_validation_lick, fig_epoch_validation_vel, fig_real_example, fig_real_fit_quality, fig_real_latents_rpe, fig_real_latents_value, fig_real_lick_profiles (2026-06-02 15:39). Plus a stray empty _t.txt (0 B, 2026-05-20).

  *Currency:* SPLIT: the seven 2026-06-02 15:39 panels are CURRENT; the encoding panels (05-25) are STALE vs plot_encoding_detail.py (commit 2026-06-02 09:59); the recovery panels (05-24) are STALE vs run_parameter_recovery.py (2026-06-02). NO .svg anywhere in this directory — violates the svg+png rule. All gitignored.

**`Striatum project/rl_model/recovery.log`** — Truncated mid-run: MPLCONFIGDIR warnings, then '[20:29:37] Generating synthetic cohort: 14 mice x 160 trials' and a single line '[20:29:50] mouse 1/14 nll= 1105.7 (12.5s)'

  *Currency:* Records an INTERRUPTED recovery run over a 14-mouse x 160-trial synthetic cohort. Every recovery_*.npz on disk is 12-mouse, so this 14-mouse run's outputs never landed.

**`Striatum project/rl_model/results/`** — recovery_results/poisson/v3/v4/v5/v6.npz; fits_v3..v6; real_fits_v3..v5, real_fits_v6_{interleaved,forward}; encoding_v5, encoding_v6 (per-mouse .npz); rl_latents.mat/.npz (2026-06-02); DONE markers incl. DONE_v7 and DONE_ladder_synth.

  *Currency:* recovery_v7.npz is ABSENT although DONE_v7 exists and UNDERSTANDING.md records v7 numbers — the v7 recovery figures/correlations have no backing array on disk. No ladder result file for real mice (only DONE_ladder_synth).

**`Striatum project/rl_model/results/DONE*`** — DONE, DONE_ladder_synth, DONE_poisson, DONE_real, DONE_real_v6_forward, DONE_real_v6_interleaved, DONE_v3, DONE_v4, DONE_v5, DONE_v6, DONE_v7

  *Currency:* These 11 sentinels are the ONLY git-tracked files in rl_model/results — the actual numbers are all untracked. DONE_v7 records a completed v7 run but there is NO fits_v7/, real_fits_v7/ or encoding_v7/ directory on disk: a run is marked complete with zero surviving outputs.

**`Striatum project/rl_model/results/encoding_v5/ and encoding_v6/`** — Per mouse, per unit (133 units for M01): area (<U3), r2_beh, dR2_beh_{value,rpe,precision} + pval_beh_*, r2_beh_spatial, dR2_beh_spatial_{value,rpe,precision} + pval_beh_spatial_*. v6 ADDS pvalbin_beh_* and pvalbin_beh_spatial_* (bin-level p-values) absent from v5.

  *Currency:* BOTH STALE — run_neural_encoding.py committed 2026-06-02, 8 days after these. FLAG: unit-level p-values across 133 units x 3 latents x 2 model families per mouse x 16 mice with no FDR/BH column stored anywhere in the npz; units are nested in animals, so any pooled count of 'significant units' is pseudoreplicated. Ties directly to the CLAUDE.md rule that KS/test grids must use BH-FDR.

**`Striatum project/rl_model/results/fits/ fits_poisson/ fits_v3/ fits_v4/ fits_v5/ fits_v6/`** — mouse_00..mouse_11.npz, each holding u_fit (11 or 16 params) and scalar nll. fits_v6/mouse_00.npz: nll = 9927.24.

  *Currency:* STALE vs run_model_ladder.py (2026-06-02). CORRUPT ARTEFACT: rl_model/results/fits/mouse_00.npz is 2 bytes and raises UnpicklingError — the v1 synthetic ladder is missing its first mouse.

**`Striatum project/rl_model/results/real_fits/ real_fits_v2/ real_fits_v3/ real_fits_v4/ real_fits_v5/`** — real_fits/M01: n_trials=357 (vs 125 in v6 — the trial-selection criterion changed), u_fit (12 params), nll=25836.81, train 286 / test 71, model_test_ll=-1.7898 vs null_test_ll=-1.8281, model_train_ll=-1.8097 vs null_train_ll=-1.7566, latents 357x50

  *Currency:* STALE / SUPERSEDED by v6. real_fits (3 of 16 mice) and real_fits_v2 (2 of 16) are INCOMPLETE partial runs. FLAG on real_fits/M01: the model is better than the null on TEST (-1.7898 vs -1.8281) but WORSE than the null on TRAIN (-1.8097 vs -1.7566) — a train/test inversion that indicates the null and model are not being scored on comparable footing in that generation.

**`Striatum project/rl_model/results/real_fits_v6_forward/ and real_fits_v6_interleaved/`** — Per mouse M01..M16: mouse, n_trials, u_fit (16), params (16), nll, train_idx, test_idx, model_lick_{train,test}, model_vel_{train,test}, null_lick_test, null_vel_test, and latents lat_{value,rpe,precision,lick_rate,v_mean,belief_mean,sigma} each n_trials x 50 bins. M01: n_trials=125, train 94 / test 31, nll=7227.01.

  *Currency:* CURRENT (written after the 13:00 commit). Two split schemes retained side by side — 'forward' (held-out block at the end) vs 'interleaved' — with byte-identical file sizes per mouse, i.e. same data, different train/test partition. GOOD PRACTICE: both model_* and null_* held-out scores are stored, so the comparison is against a matched null rather than against zero. FLAG: for M01 forward the model beats the null on licking (-0.5501 vs -0.8626) but is WORSE than the null on velocity (-0.6384 vs -0.6097), so the fit quality claim is behaviour-channel dependent.

**`Striatum project/rl_model/results/recovery_{results,poisson,v3,v4,v5,v6}.npz`** — Synthetic parameter recovery: u_true, u_fit, p_true, p_fit (12 simulated mice x 11 params in v1, x 16 params in v6), nll (12), param_names, latent_keys (5), latent_r (5 x 12), param_r (per-param true-vs-fit correlation)

  *Currency:* ALL STALE — run_parameter_recovery.py was committed 2026-06-02, 9 days after the newest of these. The model gained 5 parameters between v1 (11) and v6 (16), so cross-version comparison is not apples-to-apples.

**`Striatum project/rl_model/results/rl_latents.npz + rl_latents.mat`** — value, rpe, precision, lick_rate, v_mean, belief_mean, sigma (16 object arrays, one per mouse) + mouse_id (16, <U3). The .mat holds a single 1x1 struct rl_latents.

  *Currency:* CURRENT. This is the hand-off artefact from the RL model into the MATLAB neural analyses — the only file bridging the two halves of the project. Untracked, single copy.

**`Striatum project/tcca/figures/`** — EMPTY

  *Currency:* NEVER RUN — no tcca figure has ever been produced. The tcca epoch results exist as CSV only.

**`Striatum project/tcca/results/epoch_cross.csv`** — 124 rows: animal, pair, transition, k_eff, rot_x_cc1, rot_y_cc1, floor_x_cc1, floor_y_cc1, rot_x_top3, rot_y_top3, jaccard_x, jaccard_y (degrees). Backs the rotation-vs-split-half-floor null.

  *Currency:* current (2026-07-28); untracked, same gitignore issue

**`Striatum project/tcca/results/epoch_cross.csv`** — animal, pair, transition (naive->intermediate | intermediate->expert | naive->expert), k_eff, rot_x_cc1, rot_y_cc1, floor_x_cc1, floor_y_cc1, rot_x_top3, rot_y_top3, jaccard_x, jaccard_y — subspace rotation across epochs against a shuffle floor

  *Currency:* CURRENT vs working tree, code uncommitted. Good practice: reports rotation angles WITH a matched floor_* column, so rotation can be judged against chance rather than in absolute terms.

**`Striatum project/tcca/results/epoch_dims.csv`** — animal, pair, epoch, dim, peak_cc, sig, ifi, lag — per-dimension detail behind epoch_metrics (dims 1..k_eff)

  *Currency:* CURRENT vs working tree, code uncommitted. Same coverage imbalance as epoch_metrics. Any test run over these 1681 rows as independent units is pseudoreplicated at two levels (dims within pair within animal).

**`Striatum project/tcca/results/epoch_metrics.csv`** — 125 rows x 19 cols: animal, role, lp, pair, epoch, n_bins, n_units_x/y/z, k_eff, cc1, n_sig, mi_sig, ifi, optimal_lag, gini_x/y, sh_x_cc1, sh_y_cc1. 11 learner animals (1,2,4,5,6,7,9,10,11,13,14), 15 pairs, 3 epochs. Backs every tcca number quoted in tcca/NOTES.md and PREDICTIONS.md 2026-07-28.

  *Currency:* CURRENT with code (run 2026-07-28 12:05 against unchanged run_epochs.py and preprocessed_data2p5cm.mat of 23 May). BUT: NOTES.md:20 claims 'Results now committed'; git status shows all four epoch_*.csv as untracked (??) — the root .gitignore ignores *.csv and only tcca/.gitignore (itself uncommitted, ' M') un-ignores them. Claim is false as of this session.

**`Striatum project/tcca/results/epoch_metrics.csv`** — animal, role, lp, pair, epoch, n_bins, n_units_x, n_units_y, n_units_z, k_eff, cc1, n_sig, mi_sig, ifi, optimal_lag, gini_x, gini_y, sh_x_cc1, sh_y_cc1. 11 animals (1,2,4,5,6,7,9,10,11,13,14); learning points lp = 22,44,53,54,36,34,44,67,84,23,39. EVERY row is role=learner — no non-learner or control rows exist. 72/125 rows have n_sig>0. k_eff is 16 or 20.

  *Currency:* CURRENT vs the WORKING TREE only. run_epochs.py mtime 2026-07-28 11:58 (7 min before the CSV) but its last commit is 2026-06-17 14:45 and git diff shows it MODIFIED (+13/-1) — so the code that produced these numbers is not in git. THREE STRUCTURAL FLAGS: (1) severely unbalanced coverage — animal 11 contributes 45 rows, animal 10 18 rows, animals 1/2/4/9/14 9 rows, animal 5 8 rows, animals 6/7/13 only 3 rows; pooling across rows silently weights animal 11 at 15x animal 6. (2) DMS-ACC has 10 expert and 10 intermediate rows but only 9 naive — one animal's naive cell is missing, so paired epoch contrasts are not on a common set. (3) It reports DG pairs (DG-ACC, DG-CA1, DG-DLS, DG-DMS, DG-V1) even though DG units were excluded from all figures/analyses across IntegratedAll_v1.m, SpatioTemporalActivityEvolution.m and Run_TCA_pipeline.m as of 2026-05-24 — the Python tcca pipeline is out of step with the MATLAB pipeline.

**`Striatum project/tcca/results/epoch_weights.csv`** — animal, pair, epoch, area, unit, contrib — per-unit loading magnitude on CC1

  *Currency:* CURRENT vs working tree, code uncommitted.

**`cosyne2025/figures/`** — con1/ (Lick_number.eps, DMS FR in corridor.eps, DMS FR.eps, 'Lick over trials in corridor' [NO EXTENSION]); con2/ (Lick_number.eps, DMS FR in corridor.eps, zscore1.eps, 'Lick over trials in corridor' [NO EXTENSION]); early/ (DMS FR.eps, DMS FR2.eps, DMS FR3.eps); visuallearningtask/ (Lick ratio in reward zone(smooth 5trial).eps, Lick_number.eps, DMS FR in corridor.eps, Cumulative reward.eps, Copy of lick_ratio_fig.eps, DMS FR.eps, zscore1.eps, Lick over trials in corridor(smooth 5trial).eps). Plus poster_zihao_93.pptx and cosyne_2025.docx at the cosyne2025/ root.

  *Currency:* SUPERSEDED conference artefacts, DMS-only, ~21 months old. Provenance UNKNOWN — no producing script found. The whole /cosyne2025 path is gitignored at the repo root. Two files carry no extension and cannot be opened by type. EPS only, no png/svg pair.


---

## Gaps: analyses with no recorded outcome, or outcomes that were lost

- Auto_Reports/ (2 pptx, 2026-05-14/15) has NO producer anywhere in the repository — a grep for 'exportToPPTX' and 'Auto_Reports' across all .m files returns nothing. The two decks cannot be regenerated and their provenance is unknown. The 27x size difference between them (988.4 kB vs 37.2 kB for the same date) suggests one is a partial run.
- BEHAVIOURAL LEARNING STATISTICS: no cohort-level record. Per-animal LPs are recoverable from tcca/results/epoch_metrics.csv (A1 22 … A11 84) and scattered LPs appear in cca/NOTES.md and lfp/NOTES.md, but there is no recorded cohort mean/SD/range, no learner/non-learner counts with criteria applied consistently (12 learners in cca/UNDERSTANDING, 13 in the tcca smoke, 11 in the tcca run), no lick-precision or z-scored-lick-error statistics, no velocity/reward-rate numbers, and nothing backing the paper's Fig 1C/1D optimality claims.
- BUZSÁKI TEMPORAL-CHUNKING RECIPE (M4) not written: the paper is still unidentified ('title/year TBC') and no recipe exists in ResearchVault/Methods/. The suspicion that a velocity threshold gates the chunking is unresolved ('(?)' in the original note).
- CEBRA: never run. cebra_README.md promises a per-mouse held-out decoder R² table and a multi-session consistency matrix; no cebra_results.npz, no cebra_data/ export, no consistency scores, no R². NOTES.md:457 still lists 'Resurrect the CEBRA pipeline' as P4.
- COHORT COVERAGE HOLES that limit what can be claimed: (a) RawData/507_raw.mat has no 507_neurontype2025.mat, so that animal cannot be spike-type classified. (b) RawDataControl2/ has NO neurontype file for any of its 7 sessions, so no MSN/FSN/TAN/RS breakdown is possible for the control2 cohort. (c) Only 3 animals (1212, 1206, 1201) have both CA1 and DG; V1-DLS and CA1-DLS have n_animals=0 in every CCA table and CA1-ACC/CA1-DMS have n_animals=1, so those pairs support no inferential statistics at all. (d) preprocessed_data_control2p5cm.mat carries only is_acc/is_dls/is_dms even though RawDataControl/ contains V1 recordings for 513, 515 and 817 — acquired V1 control data has not been propagated into the current control cache. (e) BehaviourOnly/ animals 1215_M2, 1217_M4 and 1219_M1 appear nowhere else, so behaviour-cohort and recording-cohort n differ and must be reported separately. (f) Animal ID 624 appears in both RawData/ (task, 99.1 MB) and RawDataControl2/ (control, 17.6 MB) — an ID collision across cohorts.
- DECREASER-ONLY COMMUNICATION SUBSPACE (M2) never run — no unit-subset filter in the striatum_cca loader, no Decreasers vs all-units vs Increasers contrast, no unit-count floor diagnostics.
- DG INCONSISTENCY ACROSS PIPELINES: IntegratedAll_v1.m, SpatioTemporalActivityEvolution.m and Run_TCA_pipeline.m all drop DG units from every figure and analysis as of 2026-05-24, and cca/figures/*.csv (2026-06-02) contain no DG pairs. But tcca/results/epoch_metrics.csv (2026-07-28) still reports five DG pairs (DG-ACC, DG-CA1, DG-DLS, DG-DMS, DG-V1), and tca_outputs.mat's 1693-unit tensor still includes DG. The Python tcca pipeline and the MATLAB pipeline disagree on the unit-inclusion rule.
- ENGAGED-vs-DISENGAGED (lab-meeting ask M1, due before 2026-06-04) never run: no held-out decoding accuracy or CC1 for the disengaged segment. The pipelines actively EXCLUDE that data (dataio truncation at change_point_mean; temporal_max_trial_ms drops over-long traversals).
- ENSEMBLE ABLATION (Fig 4C): the No-DMS / No-DLS / No-ACC / No-V1 (and formerly No-CA1/No-DG) knockout conditions are implemented in IntegratedAll_v1.m §7 and decode_ensemble_ablation.m, but no drop magnitudes were ever recorded. Also ensemble_analysis.m had a SEM copy-paste bug (bad_pre/bad_post error bars taken from good_post) fixed 2026-05-07 — any figure produced before that is wrong, and no post-fix numbers are recorded.
- GIT COVERAGE IS EFFECTIVELY ZERO FOR RESULTS. Only 310 files are tracked in the whole 'Striatum project' directory, and the complete set of tracked result artefacts is: cebra_results/cebra_config.json, cebra_results/cebra_results.npz, lfp/results/README.md, lfp/figures/README.md, popsim/data/generated/*.json (6 files), rl_model/results/DONE* (11 three-byte sentinels), tcca/results/.gitkeep, tcca/figures/.gitkeep. Everything else — all 10 processed_data .mat (5.9 GB), all 25 GB of mat_results, all 3 CCA_Results .mat, all 2210 cca/results .pkl (3.0 GB), all 16 cca/figures numeric CSV/XLSX tables, all 4 tcca CSVs, all 6 lfp .npz, all rl_model .npz (542 MB), all cebra_data and cebra_results.mat, and all ~900 figures — is untracked and exists in exactly one place on this machine. The root .gitignore blanket-excludes *.mat, *.csv, *.txt, *.xlsx, *.pkl, *.npz, *.png, *.svg, *.eps, *.fig, *.gif, *.pptx and /cosyne2025.
- GIT-COMMITTAL FAILURE MAKES MOST NUMBERS UNCITABLE: the root .gitignore ignores *.csv, *.pkl, *.npz, *.png, *.svg, *.mat, *.txt, so essentially no result artefact is under version control except popsim's recovery_benchmark.json and metadata.json. tcca/NOTES.md:19-20 asserts 'Results now committed (all four epoch_*.csv) so this cannot silently vanish again' — FALSE as of this session: git status shows all four as untracked (??) and the un-ignoring .gitignore edit is itself uncommitted. The 2026-06-17 tcca outputs were already lost once this way (PREDICTIONS.md:5-8).
- GPFA: mat_results/run006/ and mat_results/run010/ are EMPTY DIRECTORIES and run009 does not exist. legacy/GPFA_striatum.m sets runIdx=10 and loads 'mat_results/run010/gpfa_xDim08.mat' (line 37), which is absent — the script as committed cannot run. The 25 GB of GPFA output in run001-run008 is orphaned: no live script references GPFA, the run-index-to-configuration mapping is unrecorded, and five runs share the filename gpfa_xDim03.mat while differing 200-fold in size (86 MB to 17.4 GB) with no metadata distinguishing them.
- IntegratedAll_v1.m §11 CROSS-MODAL SCATTERS: 36 pooled Pearson tests, still uncorrected (BH-FDR was applied only to the SpatioTemporal KS grids; NOTES.md:99 flags §11 as pending) and pseudoreplicated (trials from the same mouse treated as independent). No r or p values recorded, so nothing can be re-corrected post hoc.
- LFP PROVENANCE UNRESOLVED, so the whole LFP arm is gated: source stream (LF/AP/wideband), physical gain/units, referencing, resampling/anti-alias, exact sample-zero convention and voltage↔VR offset are all unknown, and no producer script or .meta exists in-repo. SANITY_AUDIT.md:88-95 explicitly forbids: cross-mouse power comparisons, 'clean LFP' claims, sample-accurate alignment, learning-phase effects, position decoding, trial reliability, temporal CCA, and any 30-80 Hz low-gamma effect. The decoding and cross-area-CCA numbers above were produced in deliberate violation of that gate and are labelled PROVISIONAL.
- LFP integrity summary CSVs NEVER PRESENT ON DISK despite being declared as the defensible results: lfp/results/README.md lists sanity_summary.csv, sanity_timing_summary.csv, signal_identity_summary.csv and sanity_windows_<mouse>.csv as the 'defensible integrity summaries', and none exists. A recursive find over lfp/results returns only README.md and 6 .npz. lfp/.gitignore lines 'results/*.csv' and 'results/**/*.csv' mean they were never recoverable from git either. The headline LFP integrity numbers therefore do not exist in any form; only the figure-source npz survive.
- LFP raw provenance rests on a 57-byte text file: RawData/LFP holds four files named voltage_data_384ch.mat, 'voltage_data_384ch 2.mat', 'voltage_data_384ch 3.mat' and 'voltage_data_384ch 4.mat' (50.8 GB total) whose mouse identity is recorded ONLY as a filesize-to-ID mapping in lfp_mapping.txt ('1212 - 16.33GB' etc.). This is almost certainly the root cause of the unresolved 1212 probe identity that forces area labels to be withheld in signal_identity, and it is a single-point-of-failure for every LFP result.
- LFP: EVERY summary CSV that holds the audit's numbers is absent from disk (lfp/.gitignore excludes results/*.csv). sanity_summary.csv, sanity_timing_summary.csv, signal_identity_summary.csv (which was to hold the per-peak common-median dB changes — the exact quantity a registered prediction was falsified on), decode_summary.csv, cca_summary.csv and learning_evolution_summary.csv are all gone. Reproducing them means re-reading 50 GB of raw voltage.
- LICK DECODER: no recorded Pearson r vs trial-shuffle, per epoch or per group. Methods fully written in the manuscript; outcome never recorded.
- MANUSCRIPT ITSELF HAS NO NUMBERS. documents/Striatum-ACC paper.docx (70 paragraphs, 12.6 k chars) has a full Introduction and a detailed Methods section (Poisson naive-Bayes ML position decoder, log-link ridge lick decoder with lambda=1.0, LOTO CV, in-silico ablation, spatial-shuffle and trial-shuffle nulls, 3x10-trial epoch aggregation), but the entire Results section is figure placeholders: 'Figure 2A: Trial-to-trial neural stability increases early', 'Figure 2B: Spatial decoding accuracy improves rapidly', 'Figure 3A: Identification of neural ensembles using TCA', 'Figure 4B: Ensembles encode specific computational variables (??)'. Not one effect size, n, or p-value. Behavioural claim 'Within a few tens of trials, animals learned to restrict their licks' has no number.
- MIXED DG TREATMENT WITHIN ONE FIGURE SET: the DG-exclusion commit landed 2026-05-24 23:08, but the loose root .svg panels span 22:55 to 23:47 that same night. Panels written 22:55-23:05 (_Task__Raw_FR_-_Hierarchical_Spatial_Subpops, _Task__MeanDist_*, _Task__*_Skewness_Profile_Hierarchical, _Task__Neuron_Types__*_Subpops) predate the exclusion while those written 23:20-23:47 postdate it. A visually uniform figure set silently spans two different unit-inclusion rules.
- MODULATION CLASSIFIER OUTCOMES: no recorded Increaser/Decreaser/Maintainer counts per area or epoch, and no p_FDR values from the four BH-corrected KS grids in SpatioTemporalActivityEvolution.m. This blocks the standing lab-meeting ask (M2, Decreaser-only communication subspace), which needs modulation_class exported as a per-unit label.
- MUTUAL INFORMATION ANALYSES: none of MutualInformationStriatum_v2.m's four outputs exists on disk — processed_data/shannon_mi_results.mat (cfg.save_file, line 5), cross_spatial_mi_results.mat (line 280), cross_area_mi_results.mat (line 466), pid_shared_info_results.mat (line 662). A repo-wide find for '*mi_results*' and '*pid*results*' returns nothing. The entire MI / partial-information-decomposition strand of the project has either never been run to completion or its outputs have been deleted. All four are gitignored (*.mat).
- MUTUAL INFORMATION: MutualInformationStriatum_v2.m (Miller-Madow corrected, zero-aware bins, trial-shuffle null with 95th percentile, per-worker rng(cfg.seed+ianimal)) has NO recorded MI value anywhere — no bits, no per-pair numbers, no null margins. The MI cache is not on disk. Commit f50375b lists nine bug fixes to it and zero results.
- NEURON COUNTS AND CELL-TYPE COMPOSITION: summary_numbers.m prints per-area unit counts, firing rates and MSN/FSN/TAN breakdowns, but no numbers are recorded in any doc. The vault task 'Update Excel sheet with unit numbers' (from 2026-04-24) is still open. Auto_Reports/ holds only two .pptx exports (14 May) — unread, and .pptx is gitignored. So no cell-count table exists for the manuscript.
- NO CONTROL-GROUP (blank-corridor) NUMBERS ANYWHERE, despite Control being the learning-specificity control and being wired into IntegratedAll_v1.m as a three-group analysis. preprocessed_data_control.mat / _control2.mat predate the fr_threshold=0.02 alignment (NOTES.md:170) and NOTES says they should be regenerated for parity — no post-alignment control result is recorded.
- NONLINEAR EPOCH DECODING: Nonlinear_Epoch_Decoding.m writes processed_data/<decoder_type>_epoch_decoding_results.mat (line 16). No file matching *epoch_decoding* exists anywhere in the repo. Never run, or output lost.
- No .csv, .mat or .npz source data accompanies ANY of the 501 figures in Striatum project/figures/, the ~140 loose .svg in the project root, the 24 .svg in CCA_Results/, the 121 files in legacy/Figures/, or the 21 .eps in cosyne2025/figures/. The project rule 'save raw data alongside generated figures so plots can be regenerated without re-running the pipeline' is unmet for every MATLAB figure in the repository.
- No .svg exists for any Python-generated figure: cca/figures (114 png, 0 svg), rl_model/figures (20 png, 0 svg), cebra_results + embedding_panels (28 png, 0 svg). The 'figures ship as .svg + .png pair' rule is unmet for all 162 Python panels, so none is publication-ready.
- POSITION DECODING: no recorded outcome anywhere. The paper's headline Fig 2B/2C claims (RMSE improvement, Shannon entropy in bits, decoding-vs-learning-rate correlation) have no numeric record in any doc, log or CSV. processed_data/cross_spatial_decoding_results.mat exists (14 May, 424 MB) but NOTES.md:163-168 lists it as STALE (predates the V1/CA1/DG extension). processed_data/nonlinear_epoch_decoding.mat, ridge_epoch_decoding_results.mat, gpr_epoch_decoding_results.mat and striatal_cca_group_results.mat are all ABSENT from processed_data/.
- REAL-DATA RL MODEL LADDER never run. Only the synthetic demonstration exists (DONE_ladder_synth); rl_model/UNDERSTANDING.md:342-346 lists the real ladder, the real v6 refit, the per-epoch re-validation and the encoding re-run as 'Open … need the .mat (not in this container)'. So the claim that each learning channel 'earns its place' is untested on mice.
- RL NEURAL ENCODING (Fig 4B) has no recorded numbers: results/encoding_v5/ and encoding_v6/ hold per-mouse .npz (16 files each) but no dR2 per latent, per ensemble or per area is written down anywhere. The only recorded encoding number is the negative control (precision retains ~6% of variance after spatial demeaning).
- ROUND-17 'ARM A' IS A PHANTOM: cca/NOTES.md round 17 and ResearchVault/Methods/CCA_HH_Adapted.md §6.2 both describe a complete running-state temporal port with '169 tests', but segments.py, lagged_temporal.py and run_temporal_runstate.py do not exist on disk. Its smoke numbers (19-27 cm/s, 75-82% bin retention, DMS→ACC CC1 0.27-0.38) are unbacked. The vault methods contract has NOT been corrected and still presents Arm A as ported — this is live doc-drift into the shared cross-project contract.
- RawData/V1_depth.txt records structures and caveats that exist in NO machine-readable form: CA3 (1100-1400) and LG (600-1000) for 1201, LP (0-400) for 1206, RHP for 1105/1106, SC for 1105, and 'L1 might be damaged' for 1212, 1206, 1106 and 1105. Neuropixels_V1_Depth_Data.csv captures only V1/CA1/DG, so the V1 laminar-damage caveat cannot propagate into any analysis automatically.
- SPATIOTEMPORAL / STABILITY (Fig 2A): dozens of Stability_*/Area_Activity_* SVGs exist in 'Striatum project/' (14 May era) but no numeric record of the trial-to-trial correlation values, their epoch differences, or any test. The bin-wise skewness panels use an SEM patch as if it were a significance test (NOTES.md:298) — no cluster-based correction, no numbers.
- STALE-CACHE RISK TO SPECIFIC CLAIMS: (a) CCA_Results/Striatum_CCA_v3_2026_05_15.mat and its 9 StriatumCCAv3_*.svg panels were produced 9 days before both CCA_striatum_spatial_v3.m and project_cfg.m were committed, and that project_cfg commit is the one that repointed cfg.task_data_file from the 5 cm to the 2.5 cm data — so the v3 CCA result was fit on data the current config no longer uses. (b) processed_data/tca_outputs.mat embeds the 33-field 5 cm task_data and predates its producer by 18 days. (c) processed_data/preprocessed_data2p5cm.mat, the canonical task input, predates ProcessStriatumTask.m's last commit by 22 h; preprocessed_data_control2p5cm.mat predates ProcessStriatumControl.m's by 7 h. (d) all_data_control2.mat and preprocessed_data_control2.mat date from Jan 2025 and predate the 2026-05-07 fr_threshold=0.02 Hz alignment by 15 months, exactly as CLAUDE.md warns; any Task/Control parity claim resting on them is invalid. (e) lfp/figures/cca_cross_area.* and decode_position.* predate the 2026-07-28 'harden voltage integrity audit' commit to run_decode_cca.py by 15 days and have no saved source data.
- TCA: no recorded rank, no variance explained, no BIC values, no ensemble count, no per-ensemble spatial-tuning or emergence-timecourse statistics. best_n_factors = 5 is a hardcoded override of the (in-sample, min-over-25-inits) BIC machinery, i.e. a parameter not a result. tca_outputs.mat exists (14 May, 1.1 GB) with no summary.
- TEMPORAL BINNING RESTRICTED TO THE FIRST 3 TRIALS (M3) never run, despite being flagged as 'run-and-analyse, not build'.
- THE PLAIN preprocessed_data.mat DOES NOT EXIST. processed_data/ holds only preprocessed_data2p5cm.mat and preprocessed_data5cm.mat (plus control variants). Every MATLAB script and the RL model's documented entry point reference 'preprocessed_data.mat', so any doc claim about which file backs which figure is ambiguous, and the RL latents (rl_latents.mat, 2 June) cannot be traced to a specific binning without opening them.
- The 1085 .done sentinels in cca/results cover only the parameter-sweep pkls; the ~40 hand-run 'committed' and 'main' configurations that actually underwrite the written claims (stage2_committed_*, stage2_main, stage3_*, partial_*) have no completion record, so a partially-written pkl among them would be indistinguishable from a complete one.
- The four tcca CSVs that tcca/.gitignore explicitly un-ignores (!results/epoch_metrics.csv, !results/epoch_dims.csv, !results/epoch_weights.csv, !results/epoch_cross.csv) are STILL git-untracked — git status shows all four as '??'. The .gitignore negation was written precisely because these tables 'were silently lost once' (per the comment in the file and NOTES.md 2026-07-28), yet the protection has never been effected by an actual git add. They remain a single uncommitted copy.
- Two stale PowerPoint lock files (~$StriatumUpdate_20260312.pptx, ~$StriatumUpdate_20260319.pptx, 165 B each) and one empty stray file (rl_model/figures/_t.txt, 0 B) sit in tracked working directories, plus .DS_Store in 10+ result directories.
- cca commits for Stages 2-3 were blocked by a stale .git/index.lock in the sandbox (cca/NOTES.md:112-121); the notes repeatedly list stale pkls/figures the sandbox could not delete (rounds 6, 7, 8), so cca/results/ now mixes ~3300 entries across incompatible configurations with no manifest. Which pkl backs which published figure is not recorded.
- cca/RESULTS.md IS STALE AND SAYS SO: its own banner (lines 8-16) states it 'describes rounds 1-2 and is now out of date'; the pipeline has since changed significance test, epoch count, cohort gate, z-scoring location, null, and moved to partial CCA (round 14). No rewrite exists. The committed configuration's headline numbers therefore live ONLY in gitignored CSVs (epoch_vs0_partial.csv, epoch_stats_partial.csv, directionality_partial.csv) and in figure PNGs — nothing in prose.
- cca/scripts/summarise_sweep.py writes a sweep summary per named sweep, but only 'spatial' and 'temporal' (and one unnamed) summaries exist. sweep_summary_temporal.csv has just 20 rows against 10560 for spatial — the temporal-alignment arm of the CCA sweep is essentially unrun compared to the spatial arm.
- cosyne2025/figures/ has no producing script anywhere in the repo, and two of its files ('con1/Lick over trials in corridor', 'con2/Lick over trials in corridor') have NO file extension. Provenance unknown, format unidentifiable.
- figures/ (86 MB, 501 files) is WHOLLY STALE: newest file 2025-04-24, producers last committed 2026-05-24 and 2026-06-01. Every panel predates its producer by 13+ months and predates both the fr_threshold alignment and the DG exclusion. Meanwhile the current MATLAB figure generation writes ~140 loose .svg into the repository ROOT rather than into figures/, so the directory intended for figures holds only obsolete ones.
- lfp/figures/_quarantined_unaligned_learning/learning_evolution_summary.csv is declared preserved as an audit trail by lfp/results/README.md but is absent — only the 10 quarantined .png/.svg remain, so the contaminated 30-80 Hz numbers cannot even be inspected to judge the contamination.
- popsim: recovery_benchmark.json reports 10 scenarios but only 5 scenario directories exist (zero_lag, lagged, epoch_varying, mediated, no_coupling). The generated data and metadata.json for bidirectional, common_input, rotated_subspace, partial_mediation and noise_correlation were never written, so five of the ten validation scenarios — including the two that carry the strongest interpretive weight (common_input drop_frac 0.740, noise_correlation latent 0.197 vs pop 0.993) — have no reproducible ground-truth spec on disk.
- preprocessed_data_control5cm.mat is the ONLY cache holding pca_{dark,stim}_dimensionality_{acc,all,dls,dms} and is_v1/is_ca1/is_dg for the control cohort; the superseding 2p5cm control file has neither. If the 5 cm files are cleaned up as 'superseded', those variables are lost with no producer path to regenerate them at 2.5 cm.
- recovery_v7.npz is MISSING although DONE_v7 exists and UNDERSTANDING.md records the full v7 correlation table — the current best parameter-recovery result has no backing artefact.
- rl_model/data/generated/ is EMPTY. The synthetic cohorts behind recovery_*.npz and fits_v*/ were never persisted, so no recovery result can be re-scored or re-analysed against its own inputs.
- rl_model/recovery.log is truncated after 'mouse 1/14' of a '14 mice x 160 trials' synthetic cohort. Every recovery_*.npz on disk is 12-mouse, so this 14-mouse recovery run's outputs never landed.
- rl_model/results/DONE_v7 records a completed v7 run, but there is no fits_v7/, real_fits_v7/ or encoding_v7/ directory. A run is marked complete with zero surviving outputs.
- rl_model/results/fits/mouse_00.npz is 2 bytes and unreadable (UnpicklingError) — the v1 synthetic model-ladder cohort is missing its first mouse, so that ladder rung cannot be summarised over all 12 mice.
- rl_model/results/real_fits/ contains only M01, M03, M06 (3 of 16 mice) and real_fits_v2/ only M03, M06 (2 of 16). Both are incomplete partial runs; no cohort-level statistic can be computed from either.
- tcca Stage 3 and Stage 4 not run: run_trajectory (sliding window, 3 learning axes), run_ifi_windows (10 ms, ±250 ms held-out segment-aware IFI sweep), run_transition (Task vs Control, between-cohort), run_early_trials, run_kcca — all listed as next steps with no outputs. Note preprocessed_data_control2p5cm.mat exists (24 May) but tcca/NOTES.md:214-216 flags the Control cache as possibly stale (predates the fr_threshold alignment).
- tcca analyze_epochs.py WAS NEVER WRITTEN. tcca/NOTES.md:53-56 and 104-106 make it the immediate next step (per-animal Wilcoxon + LMM via paired_stats/mixed_effects, medians as headline, per-pair animal counts on every panel, rotation−floor as a distribution). All tcca cohort statistics quoted so far are in-driver previews with no stats table on disk.
- tcca coverage is severely unbalanced and no aggregation accounts for it: animal 11 contributes 45 of 125 rows, animal 10 contributes 18, animals 1/2/4/9/14 contribute 9 each, animal 5 contributes 8, and animals 6/7/13 contribute only 3 each. Any statistic pooled over rows weights animal 11 at 15x animal 6. Additionally DMS-ACC has 10 expert and 10 intermediate rows but only 9 naive, so paired epoch contrasts are not computed on a common animal set. All 125 rows are role=learner — no non-learner or control comparison exists in the tcca output at all.
- tcca/figures/ is EMPTY except .gitkeep — no tcca figure has ever been produced. The tcca epoch results (11 animals, 125 cells, 4 CSVs written 2026-07-28) exist as raw tables only, with no visual or statistical summary.
- tcca/scripts/run_epochs.py is MODIFIED and UNCOMMITTED (+13/-1 vs commit 2026-06-17), and its mtime (2026-07-28 11:58) is 7 minutes before the CSVs it produced. The code that generated the current tcca numbers does not exist in git history and cannot be recovered if the working tree is reset.