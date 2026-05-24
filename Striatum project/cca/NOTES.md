# CCA pipeline — development notes

Running log of progress and decisions. The *spec* lives in `UNDERSTANDING.md`
(decisions D1–D12 + edit log); this file is the chronological work log.

---

## 2026-05-23 — autonomous session (Theo asleep)

Theo asked me to continue through Stage 2, Stage 3 and the writeup using best
judgement, documenting decisions and updating notes frequently. Checkpoints
that would normally pause for his review are instead **documented here** for
him to read on waking — each is tagged **[REVIEW]**.

### Stage 1 — DONE (committed 8e05ec0)
Package `cca/` built: config, core (residualise/PCA/CCA/CV), dataio,
pipeline. 35 tests. Validated on the cohort. Three data-forced deviations
(per-epoch PCA, missing-bin imputation, rank-capped k) — see UNDERSTANDING.md
edit log v3. Stage 1 checkpoint: Theo chose `min_units` 5→4 and an FS-in/out
comparison (edit log v4).

### Stage 2 — IN PROGRESS
Built: `lagged.py` (lag slicing, lag curve, IFI), `surrogate.py` (trial
permutation + circular bin shift nulls, p-values), `analysis.py`
(`analyse_pair` — the parallelisable per-pair driver), refactored `pipeline.py`
into `prepare_pair` (data-heavy, main process) + `fit_pair`/`analyse_pair`
(pure compute). `scripts/run_stage2.py` runs the cohort in a process pool.
53 tests total, all passing; ruff clean.

**[REVIEW] Stage 2 design calls made without Theo:**
- `n_shuffles = 100` (not the 200 in D7). Both source papers used 100;
  100 gives p-value resolution 0.01, enough for a 95th-percentile threshold,
  and halves compute. Easy to re-run at 200 if wanted.
- Circular-shift null computes lag-0 CC1 only (not the full lag curve) — it
  is the robustness check on CC1 significance; the trial-permutation null
  carries the full lagged refit and supplies the IFI null. Halves the
  circshift cost.
- IFI uses held-out CC1 **clipped at 0** (a negative held-out CC = no
  generalising communication at that lag → contributes 0). Keeps IFI in
  [−1, 1] and well-defined.
- Parallelism: 3 GB RAM / 4 cores in the sandbox means workers cannot each
  hold the cohort. Instead the main process does the data-heavy `prepare_pair`
  and ships only the small (~0.5 MB) per-epoch PC-score tensors to the pool.

**[REVIEW] Significance test = in-sample permutation (compute-forced).**
A full 5-fold-CV surrogate loop measured at ~18 s per large pair → the whole
cohort would not fit the sandbox's 45 s-per-shell limit. Switched to: the
**significance test statistic is the in-sample CC1** — real and all 200/100
surrogates computed in-sample, identically. This is a textbook-valid
permutation test (real and shuffles carry the *same* overfitting bias, so
excess over the shuffle distribution is real structure; this is exactly how
H&H framed their shuffle). The **held-out CC1 (5-fold CV) is still computed
for the real data at lag 0** and reported as the unbiased *effect size*. So:
held-out CC = honest magnitude; in-sample-vs-shuffle = significance. Lag
curves / IFI use in-sample CC (in-sample IFI is mildly conservative — a near-
uniform additive bias shrinks the ratio toward 0; the peak lag is unaffected).
Easy to switch surrogates back to CV'd if ever wanted — just slower.
~6× speedup; full cohort now runs in well under a minute.
- The cohort runner is **resumable** (per-pair incremental save) so it
  survives the shell timeout.

**[REVIEW] Revised after the first Stage 2 run — directionality (IFI).**
The first run computed the lag curve *in-sample*; the in-sample bias is a
near-uniform additive offset that crushes the normalised IFI toward 0 (IFI
was ≈0 for every pair). Corrected:
- The **real lag curve is now held-out** (5-fold CV per lag) — the honest
  directionality readout. IFI and peak-lag come from it.
- The **CC1 significance test stays in-sample** (real vs trial-permutation and
  circular-shift nulls, 200 surrogates) — it only needs lag-0, so each
  surrogate is a single in-sample CCA fit (very cheap; dropped the per-
  surrogate lagged refit entirely).
- **IFI significance is assessed at the group level** — a test across animals
  of the per-animal held-out IFI (does the cohort IFI differ from 0 / shift
  across epochs). This is the scientifically appropriate inference for "does
  the direction of flow change with learning", and avoids an expensive
  per-pair held-out IFI surrogate. Done in Stage 3 / plotting.
- `n_shuffles` back to **200** (D7) now that surrogates are cheap.

**Other fix:** `cca_fit` is now SVD-based and rank-robust — bin-slicing for the
lagged CCA can leave a PC-score column with no variance over the slice;
the old QR version raised on that. The SVD version returns
min(rank_x, rank_y) canonical correlations gracefully.

### Stage 2 — FINDINGS (final, held-out lag curves; FS-excluded primary)

Strength (held-out CC1, learner-group mean, naive / intermediate / expert):
- DMS-DLS  0.27 / 0.34 / 0.26   (4-5 of 8 animals p<0.05)
- DMS-ACC  0.18 / 0.11 / 0.17   (4-5 of 10)
- DLS-ACC  0.13 / 0.10 / 0.14   (2-3 of 7)
- V1 pairs 0.03-0.15, mostly n.s.; CA1 pairs n=1-2 (anecdotal).

So **residual communication is real but modest** — about half the learner
animals show significant trial-to-trial coupling. DMS-DLS is the strongest
pair. Epoch trend: DMS-DLS bulges at the intermediate epoch; DMS-ACC/DLS-ACC
dip there. The trend is suggestive, not strong — needs a group epoch test
(Stage 3 / writeup).

Directionality: the held-out lag curves **peak sharply at 0 cm and fall off
roughly symmetrically** for every well-powered pair. Group IFI ≈ 0 with no
epoch showing a significant deviation. → **No detectable change in the
direction of information flow across learning** — residual communication is
spatially symmetric. (Theo's "DMS leads early → DLS leads later" hypothesis
is not supported by this analysis; honest null result.)

FS comparison: FS-excluded vs FS-included held-out CC1 correlate at r=0.89
across matched cells; conclusions are unchanged. FS-included has marginally
more power (more units, a few more animals clear the floor).

Stage 2 committed. Figures: stage2_cc1_significance_*, stage2_ifi_*,
stage2_lag_curves_*, stage2_fs_comparison (all in figures/).

**[REVIEW] git commits blocked after Stage 1.** A stale `.git/index.lock`
(0 bytes) cannot be removed — the sandbox's mounted filesystem allows
create/overwrite but not unlink/rename. Stage 1 committed cleanly (`8e05ec0`);
Stage 2 and Stage 3 are **saved to disk but not committed**. To commit them,
Theo runs from a normal terminal:
```
cd ~/Desktop/Experiments/StriatumACC && rm -f .git/index.lock
git add "Striatum project/cca" && git commit -m "Stage 2-3: lagged CCA, surrogates, membership, angles"
```
All code, results and figures are on disk regardless; nothing is lost.

### Stage 3 — FINDINGS (membership, reorientation, partial CCA)

Built `membership.py`, `subspace.py`, `stage3.py`, `partial.py`. 67 tests.

**[REVIEW] d_sub = 1 (dominant canonical direction only).** At d_sub=3 the
*within-epoch* split-half principal angle was already near-orthogonal — the
2nd/3rd canonical dimensions are not estimable from 10-trial epochs. Set the
"communication subspace" to the dominant direction (what H&H and the subspace
paper emphasise too).

Subspace reorientation (D10): cross-epoch principal angle of the dominant
direction ≈ 1.3–1.4 rad, **above** the within-epoch split-half floor ≈
1.0–1.2 rad. DMS-ACC and DLS-ACC show the naive→expert angle significantly
above the floor (paired t-test). So the communication direction **does
reorient across learning** — modestly, on a noisy baseline.

Membership (D9): cross-pair member-set Jaccard ≈ 0.22–0.31 (chance ≈ 0.25 for
top-quartile sets) → the units carrying communication are **largely
pair-specific, not a shared hub set**. Cross-epoch (naive vs expert) Jaccard ≈
0.22 → membership also **turns over across learning**. Weight sparsity (Gini)
≈ 0.6–0.8, with a mild increase naive→expert for the striatal pairs.

Partial CCA (D3): regressing out the third striatal area barely changes CC1
(e.g. DMS-DLS|ACC 0.30→0.30; DMS-ACC|DLS 0.22→0.25; DLS-ACC|DMS 0.14→0.12).
→ striatal-cingulate communication is **direct / pair-specific**, not an
artefact of a shared third-area input. Consistent with the pair-specific
membership result. n=7 animals with all three areas.

Figures: stage3_principal_angles, stage3_gini, stage3_membership_overlap,
partial_cca (all in figures/).

### Writeup — DONE
`RESULTS.md` — full Methods + Results, British English, destined for
ResearchVault `Methods/`. Covers all four sub-questions, caveats, next steps.

---

## 2026-05-24 — round 2 (Theo's 8-point feedback)

Eight refinements from Theo. Status:

- **Bin-agnostic refactor** — pipeline auto-detects n_bins from the data;
  consumes the forthcoming 100-bin (2.5 cm) file unchanged. Theo rebins in
  MATLAB (`ProcessStriatumTask.m` line 10, `bin_size = 4 -> 2`), saving a new
  `preprocessed_data_2p5cm.mat`. (Point 3.)
- **z-scoring** — clarified: we never used `z_spatial_binned_fr_all` (which IS
  z-scored across all trials incl. post-disengagement — leakage). We use the
  raw `spatial_binned_fr_all`. Added a `zscore_units` flag: per-unit z-score
  over the analysis trials only (leakage-free), default off. (Point 6.)
- **Per-dimension IFI + significance** — surrogate now tests every canonical
  dimension; n = number of significant subspace dimensions (Gonzalez/Buzsáki
  convention). Lag curve returns all dimensions. (Point 1.)
- **IFI lag-window sweep** — IFI computed over |lag| <= 1..5 separately. (Pt 4.)
- **Signal CCA** — run via the existing `subtract_trial_mean` flag. (Point 2.)
- **Factorial** — `run_stage2.py` now sweeps residual/signal x FS-excl/incl
  (4 configs); held-out vs in-sample is the holdout read-out. (Points 2, 8.)
- 74 tests; ruff clean.

**Round-2 findings.** Per-dimension IFI pooled over **357 significant
subspace dimensions** (striatal learner pairs): mean +0.023, one-sample
p = 0.18 — still no net directionality. Robust across lag-integration windows
(IFI ~0.005–0.03 for |lag| <= 1..5, all SEM bars touching 0). Subspace
dimensionality (significant canonical dims) is bimodal — mean 4.76, many
epochs with 0–1, a tail to ~20. Factorial: signal CCA gives slightly higher
CC1 than residual (≈0.33 vs 0.25 for DMS-DLS expert — it includes shared
position coding); FS-excl vs FS-incl barely differ; held-out vs in-sample gap
is large (≈0.25 vs ≈0.55). **The "no directional change" conclusion survives
every robustness check Theo asked for.**

**[REVIEW] still pending:** the 100-bin re-run (blocked on Theo's MATLAB
rebin); whether to make z-scoring the default. git commits still blocked.

---

## 2026-05-24 — round 3 (Theo's 3-point feedback)

- **Learning-point bug fixed (point 3).** `find_learning_point` reproduced the
  MATLAB `movsum` rule, which returns the *window start* — and that trial can
  be *above* threshold. Diagnostic confirmed it: under the old rule the
  returned LP trial had z = +1.28 (animal 3), −0.77 (animal 2), −0.14
  (animal 15) etc. Corrected: the LP is now the first trial that is **itself**
  below threshold *and* starts a sustained (≥7-of-10) sub-threshold run. New
  LPs are all sensible — animal 3 → 32 (was 29), every learner ≥ 22, none
  implausibly early. **The MATLAB pipeline had the same flaw — fixed too:**
  `processTaskData.m` and the shared helper `find_learning_points.m` now use
  `find((x<=thr) & (movsum(...)>=min_consec), 1, 'first')`. Verified — the
  fixed MATLAB rule, replicated exactly, gives identical LPs to the Python
  for all 16 animals.
- **Per-pair IFI (point 1).** The round-2 IFI-per-dimension plot wrongly
  *pooled* all 357 significant dimensions into one histogram. Replaced with
  `stage2_ifi_per_pair_*` — one panel per pair, IFI of that pair's significant
  subspace dimensions by epoch. (My error; corrected.)
- **z-scoring (point 2).** Added to the factorial — now 8 configs
  (resid/signal × FS-excl/incl × z-off/on). Stage 3 and partial CCA re-run for
  z-off and z-on too.
- 75 tests; ruff clean. Everything re-run with the corrected LP.

**Round-3 findings.** With the corrected LP the headline numbers barely move
(held-out CC1 expert: DMS-DLS 0.265, DMS-ACC 0.175, DLS-ACC 0.138 — the LP
shifts were only 1–3 trials). Per-pair directionality: IFI clouds centred on
0 for every pair, no epoch significant — the no-directionality result holds
pair-by-pair, not just pooled. z-scoring: residual CC1 essentially unchanged
(~0.26 DMS-DLS); signal CC1 drops a little under z-scoring. **All conclusions
stand.**

---

## 2026-05-24 — round 4 (regenerate on the 2.5 cm / 100-bin data)

Theo produced `preprocessed_data2p5cm.mat` (100 bins, 2.5 cm) and asked to
regenerate everything with the controls correct. Done.

**[REVIEW] Missing-data handling — switched from imputing to dropping.**
At 2.5 cm bins the file-wide NaN fraction is up to 33% (vs <3% at 5 cm) —
many tiny bins the animal skips on a given trial. *Within the analysis epochs*
(engaged trials only; post-disengagement trials are excluded) it is far lower:
0–0.2% for most animals, 23% for the worst (animal 10). Imputing 23% to zero
residual would corrupt the CCA, so the pipeline now **drops** missing
(trial, bin) samples at every fit (CCA, PCA, rank, structure coefficients,
partial regression) rather than imputing. The NaN is per-(trial, bin) across
all units, so the valid-sample mask is shared by both areas of a pair —
verified (identical %NaN across an animal's areas). `impute_missing_bins`
removed.

Other changes: `max_lag_bins` 5 → 10 (10 bins × 2.5 cm = ±25 cm, matching the
5 cm pipeline's ±25 cm); k now reaches the cap of 30 (1000 nominal samples/
epoch); `config.PREPROCESSED_DATA` points at the 2.5 cm file. 78 tests.

**Round-4 findings.** Held-out CC1 (residual, FS-excl, expert): DMS-DLS 0.24,
DMS-ACC 0.11, DLS-ACC 0.08 — slightly *lower* than at 5 cm (0.27 / 0.18 /
0.14). Finer bins did not raise the canonical correlations; the extra bins are
more autocorrelated and k rose to 30, so the held-out estimate is, if
anything, a touch more conservative. Lag curves still peak at 0 and are
symmetric; per-pair IFI still ~0; subspaces still reorient modestly. **All
qualitative conclusions are unchanged from the 5 cm analysis.**

---

## 2026-05-24 — round 5 (LP reproducibility, cohort split, licks check)

- **Learning-point reproducibility bug found & fixed.** `IntegratedAll_v1.m`
  calls the (already-fixed) `find_learning_points`, so the LP-rule fix is
  applied. Animal 8's LP is genuinely early — its z-scored lick error crosses
  −2 sustainedly from ~trial 14 (trial 9 = −3.2, 14–16 ≈ −2 to −3.2). Not a
  window-start artefact. BUT the LP is **non-deterministic across preprocessing
  runs**: `calculate_lick_precision.m` draws its 1000-shuffle baseline with
  un-seeded `rand()`, so `zscored_lick_errors` (hence the LP) wobbles by a
  trial or two each run — animal 8 came out 16 on the 5 cm file and 14 on the
  2.5 cm file, identical behaviour. Fixed: `rng(42,'twister')` added at the top
  of `ProcessStriatumTask.m`. (Theo must re-run preprocessing for this to
  take effect; animal 8 stays yoked either way — ~14–16 is too early for
  disjoint epochs.)
- **Licks/velocities bin count — checked.** In `preprocessed_data2p5cm.mat`,
  `spatial_binned_data.licks` and `.durations` are (100, 357) — correctly
  100 bins. The only 50-bin field is `temp_binned_dark_fr` (the dark-period
  *temporal* 100 ms binning — legitimately 50, unrelated to the spatial
  rebin). No separate velocity field exists in the preprocessed data (velocity
  is derived from `durations`). Could not reproduce a 50-bin lick/velocity —
  flagged back to Theo to point at the exact field. (The CCA uses
  `spatial_binned_fr_all` + per-trial `zscored_lick_errors`, so it is
  unaffected regardless.)
- **Cohort split.** Every key figure is now produced for two cohorts —
  `_learners` (clean dataset, learners only) and `_all` (learners + yoked
  non-learners). The pipeline always computed both (role-tagged results);
  the plot scripts now expose both. 15 figures per cohort.

---

## 2026-05-24 — round 6 (significance fix + collapse to one config)

**Significance bug found & fixed (point 1).** The per-dimension significance
test was over-calling badly — DMS-DLS had ~136 "significant" subspace
dimensions across the cohort. Cause: the test compared the real *in-sample*
CC of dimension j to the shuffle's dimension j. Genuine signal in the top
dimensions shifts the whole in-sample CC spectrum upward, so a noise
dimension j was effectively compared against a lower-index (larger) shuffle
value and spuriously passed. Diagnostic confirmed it (animal 2 DMS-DLS:
in-sample test n_sig = 13; held-out test n_sig = 3). **Fixed:** significance
now uses the **held-out** CC — a noise dimension's held-out CC is ~0
regardless of its index, so the test is properly calibrated. `surrogate.py`
reworked: `build_null` runs `n_shuffles` trial-permutations, each a full
cross-validated CCA, and tests real held-out CC_j vs the shuffle held-out CC_j
per dimension. Result: subspace dimensionality drops to ~1.5–2.4 per
(animal, epoch) — consistent with H&H's "1–2 significant dimensions". The
in-sample machinery (`cca_in_sample`, circshift null) is gone.

**Collapsed to one config (point 2).** Committed to a single configuration —
residual CCA, FS-excluded, z-scored units, held-out CC — which is now
`config.DEFAULT`. The 8-config factorial is retired (`plot_factorial.py` is a
deprecation stub). `run_stage2.py` runs the one config → `stage2_main.pkl`.
`plot_stage2.py` rewritten as a focused 4-figure set, each for two cohorts
(learners / all): communication strength (held-out CC1), subspace
dimensionality, IFI vs lag-integration window, and per-pair IFI of the
significant subspace dimensions. The IFI statistics use **n = significant
subspace dimensions** throughout. 77 tests; ruff clean.

(Stale figures/pickles from the old 8-config runs remain in figures/ and
results/ — the sandbox cannot delete them; `rm figures/*resid_fsX* results/
stage2_resid* ...` clears them.)

---

## Session end summary (2026-05-23)

All four planned stages complete. Package `cca/`: 12 source modules, 9 test
files (67 tests, all passing), 6 scripts, 3 design/log docs + RESULTS.md.
15 figures, 5 result files on disk.

**Headline findings.** Residual (trial-to-trial) communication in the
striatal–cingulate triangle is real but modest (held-out CC1 ≈ 0.1–0.34;
~half of learner mice significant), strongest for DMS–DLS. No directional
asymmetry (symmetric lag curves, IFI ≈ 0). Communication subspaces reorient
modestly across learning (above the split-half noise floor for DMS–ACC,
DLS–ACC). Membership is pair-specific and turns over across learning
(Jaccard ≈ chance). Coupling is direct — survives partialling the third
striatal area. All robust to FS-cell inclusion. V1/CA1 arms underpowered.

**Things for Theo to look at** — search this file for `[REVIEW]`:
the in-sample permutation test, the held-out-vs-in-sample split, d_sub=1,
n_shuffles=100→200, the multiprocessing/BLAS fix, and the blocked git commits.
Nothing here is irreversible; all are documented and easily changed.

**Not done / suggested next:** formal group epoch test of the CC1 profile;
temporal (within-bin) lag analysis for directionality; the non-subtracted
"signal" CCA variant (wired in, not run); more CA1/V1 co-recordings.

---

## 2026-05-23 — round 7 (two epochs, whole-period z-scoring, 4-config grid)

Four threads of feedback from Theo.

**Intermediate epoch dropped entirely.** `EPOCH_NAMES = ("naive", "expert")`;
the analysis contrasts naive vs expert directly. `epoch_windows` returns the
two windows only. With no intermediate window the cohort gate relaxes from
`lp >= 2*trials_per_epoch` (>=20) to `lp >= trials_per_epoch` (>=10) — the one
remaining constraint is that naive [0,10) and expert [lp,lp+10) not overlap.
This admits fast-learning animals (10 <= lp < 20); learner pairs rise from 39
to 42 (residual, FS-excluded). Stage 3 reduced to the single naive->expert
transition (principal angles, Gini, membership all 2-epoch now).

**Z-scoring moved to the front of the pipeline.** Previously each unit was
z-scored per epoch, on its 10-trial residual slice. Now z-scoring is the
*first* operation on the activity: each unit is divided by its std over the
*entire engaged period* (all disengagement-truncated trials), before epoch
slicing and residualisation (`prepare_pair`/`prepare_area`; `_residual` no
longer z-scores). The std is thus estimated from ~all-trials x n_bins samples
and is identical across epochs. Because z-scoring is a per-unit scalar it
commutes with residualisation, so the residual variant's numbers shift only
through the wider, raw-activity std estimate. Two new ground-truth tests pin
it: invariance to a global per-unit rescaling, and the distinguishing test
that an expert-only rescaling perturbs even the naive-epoch fit (which the old
per-epoch z-scoring would have left bit-identical).

**Four-config robustness re-instated.** `run_stage2.py` runs residual/signal
CCA x FS-excluded/included -> `stage2_{res,sig}_fs{excl,incl}.pkl`. All four
z-score over the whole engaged period and use held-out CC (those two axes are
now fixed, not swept — relaxes the round-6 single-config commitment). The
runner is resumable per config with a `--max-seconds` budget + atomic saves,
so it is driven in short chunks under the sandbox's 45 s shell limit.

**Naive-vs-expert statistics on every figure.** `plot_stage2.py` rewritten:
naive vs expert box plots (+ jittered points) over the significant subspace
dimensions, three tests per panel — (a) one-sample Wilcoxon vs 0 per epoch,
n = significant dims ('*' marker); (b) paired Wilcoxon naive vs expert over
per-pair means of the significant dims, n = pairs ('pair=' in title);
(c) Mann-Whitney naive vs expert over all significant dims pooled, n = dims
('unp=' in title). Produced for all four configs with config-tagged
filenames. IFI kept as box plots for all ten lag-integration windows (Theo's
choice) -> 13 figures per config, 52 in all. Lag curves are lines + shaded
SEM over the significant dims, naive vs expert.

The paired test reads `n/a` for nearly all comm-strength / IFI panels: it
needs the *same pair* to have >=1 significant dim in *both* epochs, and with
the strict held-out significance test (round 6) few pairs clear that. The
unpaired pooled-dims test, asked for alongside, carries the comparison. For
subspace dimensionality the per-pair value is a count (always defined), so
the paired test does compute there.

Cohort committed to **learners only** for Stage 2 and Stage 3 figures (the
"all animals" loop dropped). 81 tests; ruff clean.

**Round-7 run** — 4 configs, 200 surrogates. Learner held-out CC1
(naive / expert), residual FS-excluded:
DMS-DLS 0.25/0.29, DMS-ACC 0.17/0.13, DLS-ACC 0.14/0.08. First read: no clear
naive->expert change in communication strength (unpaired p n.s. for the
well-powered pairs); IFI mostly null; lag curves still peak at lag 0. The
4-config robustness and the per-window IFI panels are for Theo to review from
the figures.

**Stale files the sandbox cannot delete** (Theo: `rm` from a terminal):
`results/` — `stage2_main.pkl`, `stage2_fs_*`, `stage2_resid_*`,
`stage2_signal_*`, `stage2_*_z[01].pkl`, `partial*.pkl`, `stage3.pkl`,
`stage3_z*.pkl`; `figures/` — the round-6 no-config-suffix set
`stage2_{comm_strength,subspace_dim,lag_curves,ifi_win01..10}.png` and any
older factorial figures. The current valid result files are
`stage2_{res,sig}_fs{excl,incl}.pkl` and `stage3_main.pkl`.

**Still pending:** `RESULTS.md` is a rounds-1-2 writeup and is now out of date
(see its status banner) — a full rewrite against the round-7 figures is the
natural next step once Theo has reviewed them. git commits still blocked by
the stale `.git/index.lock`.

---

## 2026-05-23 — round 8 (parameter sweep + temporal arm)

**Spatial parameter sweep — 27 configs.** A 2x2x2x2 factorial (bin width
2.5/5 cm x residual/signal CCA x FS excluded/included x z-scoring on/off)
plus one-at-a-time sweeps around the baseline of every PC-count rule Theo
named — samples-per-PC (15/25/40), fixed PC count (5/10/20/30),
variance-explained (75/85/95 %) — and the cohort knobs (min units 4/6, LP
criterion 7/8). New: `striatum_cca/sweep.py` defines the grid; `core.py`
gained fixed-k and variance-explained PC modes (`k_mode`); `run_stage2.py` /
`run_stage3.py` became sweep drivers (resumable, `.done` markers give O(1)
resume); `summarise_sweep.py` collapses every config into a per-pair xlsx
(one sheet per metric, 10 pairs x configs) + per-pair grid figures.

**Key result — one robust, defensible finding.** Reported PER PAIR (never
pool across pairs). Across the 27 configs:
- communication-subspace REORIENTATION (naive->expert principal angle above
  the within-epoch split-half floor) holds in 19-27 of 27 configs for every
  pair — the only effect that survives the whole parameter space;
- communication STRENGTH change (naive vs expert) is significant in <=3/27
  configs for every pair — chance level; no robust effect;
- IFI / directionality: no consistent sign across configs.

So strength and direction are robust nulls; the communication channel
*rotates* across learning even though its strength does not.

**Temporal arm.** `corridorData.binned_spikes` (1 ms counts) re-binned to
fixed time bins from corridor onset; signal CCA only (Theo's call);
disengaged traversals (>60 s on a 250 cm corridor) excluded — this both
speeds the analysis and stops a single ~300 s "trial" dominating its epoch.
10 ms proved computationally intractable in the sandbox (one pair > 45 s at
200 surrogates), so the temporal arm runs 40 ms and 20 ms at 50 surrogates
(p-resolution 0.02 — adequate for an exploratory arm; a measured necessity,
not a preference). The temporal result MIRRORS the spatial one: no strength
change, subspace reorientation present (18 of 20 pair-config cells above the
floor). Two animals have one malformed corridor trial each (a stray (2,)
array) — guarded as empty.

87 tests; ruff clean.

**Stale files to delete** (`rm` from a terminal): the round-7 pkls
(`stage2_{res,sig}_fs*.pkl`, `stage2_s5cm_*.pkl`, `stage3_main.pkl`,
`stage3_s5cm.pkl`) — superseded by the sweep's `stage2_<tag>.pkl` /
`stage3_<tag>.pkl`; plus older non-sweep figures.

---

## 2026-05-24 — round 9 (commit the config + IFI deep-dive)

The sweep surfaced one defensible region; Theo committed to it and we made
it `config.DEFAULT`: **residual CCA, z-scoring on, 2.5 cm bins,
`samples_per_pc=15`, `min_units=6`, `lp_min_consecutive=7`,
`trials_per_epoch=10`, `n_shuffles=200`**. Parcoords plots
(`plot_parcoords.py`) were rebuilt as a per-pair grid after Theo's note that
**pair is never a hyperparameter — always show all pairs**.

Honest corrections logged this round:
- `min_units` 4→6 does **not** cost the V1 cohorts (the cohort tables show
  V1-DMS / V1-ACC = 2 learners at *every* min_units). The real cohort lever
  is the LP criterion: LP-7 → 3 learners on V1 pairs, LP-8 → 2.
- The min_units=6 / LP-8 / samp-15 config that sharpened V1-ACC strength
  (p ≈ 0.02–0.03) was a **2-animal artifact** — reverting to LP-7 (n=3)
  pushed it to p ≈ 0.26 and the V1-DMS IFI flip vanished. Both V1 "findings"
  were two-mouse noise; reported as such.
- `trials_per_epoch=15` gave no extra significant dims (134→129) and cost
  cohort (33→27 learner pairs) — stayed at 10.

## 2026-05-24 — round 10 (intermediate epoch back + circshift null)

Re-added the **intermediate epoch** — pipeline is 3-epoch again
(`EPOCH_NAMES = naive/intermediate/expert`; `dataio.epoch_windows` and
`stage3.EPOCH_TRANSITIONS` reverted to 3). LP stays at 7.

Added a second surrogate null. `config` gained `null_type`
(`"trials"` | `"circshift"`) and `circshift_min_bins=15`; `surrogate.py`
gained `circshift_bins` (per-trial `np.roll` of the bin axis by a random
shift in `[min_shift, n_bins-min_shift]`) and `build_null` now dispatches on
`null_type`. `run_committed.py` rewritten to run `config.DEFAULT` under
`--null-type {trials,circshift}`; `compare_nulls.py` builds the comparison
(`figures/null_comparison.{csv,png}`).

**Result — the circshift null is much more permissive.** Significant
subspace dims pooled over all pairs×epochs: **trial-perm = 201,
circshift = 533** (≈2.6×). Every pair×epoch gains dims. Why: circshift
keeps each trial paired with itself and only destroys *within-trial bin
alignment*, so any trial-level co-modulation (a shared per-trial gain — e.g.
arousal/running-speed drift) **survives** the null and is counted as real.
Trial-permutation destroys the trial pairing and is the stricter test.

What the null choice does and does not change:
- **Epoch shape is preserved** — which epoch peaks/declines is the same
  under both nulls for every pair (e.g. DLS-ACC peaks at intermediate and
  collapses at expert under both; V1-DMS / V1-ACC / CA1-V1 decline from
  naive under both). The learning-related *pattern* is robust to the null.
- **Magnitudes shift.** Mean held-out CC of the significant pool is
  uniformly *lower* under circshift (the extra dims it admits are weak
  ones — DMS-DLS naive 0.245→0.139). IFI (w3) is the least robust: signs
  flip between nulls on several cells (V1-ACC expert +0.18→−0.07).

**Decision — circshift committed as the primary null.** Theo's call: it is
the surrogate Gonzalez et al. use, it is the defensible one, and it recovers
many more significant subspace dimensions. `config.DEFAULT.null_type` is now
`"circshift"`; the two trial-permutation tests in `test_surrogate.py` are
pinned to `null_type="trials"` so that path stays covered. The caveat stands
and is worth keeping in mind when writing up: circshift leaves trial-level
co-modulation (e.g. arousal/running-speed drift) in the "real" signal, so it
is the more permissive test — `compare_nulls.py` remains the record of how
much the count depends on the null.

### Full figure set — committed config, circshift, three epochs
`run_committed.py` extended with `--stage {2,3}` (Stage 3 is null-independent
— one run, `stage3_committed.pkl`). All three plot scripts rewritten for the
3-epoch committed config:
- `plot_stage2.py` — single committed config (4-config grid dropped), reads
  `stage2_committed_circshift.pkl`: communication strength (held-out CC),
  subspace dimensionality, lag curves, IFI for every window 1–10. Each
  figure: 3-epoch boxes, vs-0 stars, naive→expert paired + unpaired p.
- `plot_stage3.py` — reads `stage3_committed.pkl`: principal angles (floor +
  the 3 transitions), Gini sparsity, membership overlap.
- `plot_common_units.py` (new) — re-derives the z-scored area tensors
  (`pipeline._zscore_area`) and plots the mean spatial activity of
  communication-subspace member vs non-member units. One figure per pair,
  a 2×3 grid (rows = X / Y area, columns = naive / intermediate / expert).
  Member and non-member profiles share spatial structure; member units sit
  modestly higher in amplitude in several pairs but are not a starkly
  distinct subpopulation.
`d_sub` stays 1 for Stage 3: the split-half stability floor sets it, and the
surrogate (a held-out-CC test, a different measurement) does not change that.

88 tests; ruff clean.
