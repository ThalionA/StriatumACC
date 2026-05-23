# CCA Communication Analysis — Shared Understanding

Status: **awaiting Theo's sign-off before Stage 1 coding**
Last updated: 2026-05-23

---

## 1. Scientific question

How does communication between striatal and cortical/hippocampal areas change
across task learning (naive / intermediate / expert epochs)?

Sub-questions:
1. Do inter-areal connections strengthen or weaken with learning?
2. Does the direction of information flow change (e.g. DMS leading early →
   DLS leading later)?
3. Which units take part in the communication subspaces? Are the same units
   shared across different area pairs?
4. How do the communication subspaces reorient across learning epochs?

Method: canonical correlation analysis (CCA) on spatially-binned Neuropixels
activity, fit per (animal × area-pair × epoch). Adapted — not copied — from
Han & Helmchen 2024 ("H&H") and Gonzalez et al. 2026 ("subspace paper").

---

## 2. Data reality (binding constraints — verified from `preprocessed_data.mat`)

16 animals. `spatial_binned_fr_all` = (n_units × 50 bins × n_trials), 5 cm bins,
200 cm corridor. Areas via logical masks `is_dms/dls/acc/v1/ca1/dg`.
Cell type in `final_neurontypes(:,5)`: 1=MSN, 2=FS, 3=TAN, 4=UIN (striatum);
2=FS, 5=RS (V1/CA1). **Type 2 = fast-spiking in every area.**

Usable animals per pair (learning point detectable, ≥30 usable trials for the
3 epochs, ≥5 units in *both* areas, disjoint epoch windows):

| Pair group | Pairs | Usable n |
|---|---|---|
| Striato-cingulate | DMS–ACC | 13 |
| Striato-cingulate | DMS–DLS, DLS–ACC | 9 each |
| Visual | V1–DMS, V1–ACC | 4 each |
| Visual | V1–DLS | 3 |
| Hippocampal | CA1–DMS, CA1–DLS, CA1–ACC, CA1–V1 | 2 each |

CA1 co-occurs with other areas in animals 11, 13 (learners) and 12
(non-learner — richest CA1/V1 yield, recovered via yoking, see D1).
Partial-CCA triplet (DMS+DLS+ACC all ≥5 units): n = 9.

Per-epoch sample budget: 10 trials × 50 bins = **500 samples**.

---

## 3. Resolved design decisions

**D1 — Pair scope: tiered; non-learners yoked.** Compute all 10 pairs. The
cohort splits into *learners* (real learning point — 12 animals) and
*non-learners* (animals 8, 12, 16 — given the cohort-mean "yoked" learning
point so their early/mid/late trial windows are still analysable as a
learning-specificity control). Learning-effect claims use learners only;
non-learners are the yoked control group. Group inference restricted to the 3
striato-cingulate pairs (learner n=9–12); V1 pairs (n=3–5) and CA1 pairs
(n=2–3) are per-animal exploratory case studies — no group inference.

**D2 — "Communication" definition: residuals primary.** Subtract each unit's
per-bin trial-mean (computed within each epoch) → CCA on the trial-to-trial
residual fluctuations (noise correlations / H&H "interaction structure").
Also run the non-subtracted (signal) version as a supplementary check to see
how much shared position coding contributes.

**D3 — Estimator: PCA→CCA primary; partial CCA add-on.** Plain canonical
correlation on PCA-reduced residuals (H&H style) for all 10 pairs. Partial CCA
as a targeted secondary analysis for the DMS/DLS/ACC triplet only (n=9):
does each striato-cingulate pair's communication survive removing the third
area's activity?

**D4 — Dimensionality: k ≈ 20 PCs.** PCA per area, k = floor(500 / 25) = 20,
capped at the smaller area's unit count, by `k_cap` = 30, **and by the
smallest per-epoch numerical rank** of the residual data; **symmetric within a
pair**, **fixed across the 3 epochs**. PCA is fit **per epoch** per area
(revised — see edit log v3: a shared 30-trial basis let components carry
near-zero variance inside one epoch, making that epoch's CCA ill-conditioned).
Cross-epoch comparison is done in neuron space instead — see D10. Achieved
samples/PC ratio logged per fit.

**D5 — Spatial bins: keep 50.** Existing 5 cm bins, full corridor, 500
samples/epoch. Trial-blocked CV handles within-trial bin autocorrelation
honestly. No re-binning.

**D6 — Directionality: lagged refit, ±5-bin scan.** Refit CCA at each spatial
lag from −5 to +5 bins (±25 cm). Readouts: (a) Information Flow Index
IFI = (X-leads − Y-leads)/(X-leads + Y-leads), bounded [−1,1], averaged over
the lag window; (b) lag at which CC1 peaks. Both tracked across the 3 epochs.
X = first-listed area in each pair (fixed sign convention).

**D7 — Surrogate null: trial-permutation primary + circshift check.** Primary:
permute trial correspondence between the two areas (H&H), 200 surrogates,
recomputed through the full pipeline including the lagged refits; significance
by non-parametric percentile p-value. Secondary robustness null: circular
position-bin shift, to confirm conclusions don't hinge on the shuffle choice.

**D8 — Cross-validation: 5-fold over whole trials.** Hold out 2 trials at a
time (train 8 / test 2); fit CCA projections on train, project test, correlate.
Whole-trial folds prevent autocorrelation leakage. Held-out CC reported as
primary; in-sample alongside as a bias check. Surrogates run through the same
CV. (PCA, being unsupervised, is fit on the full epoch set per D4.)

**D9 — Unit membership: both scores, cross-checked.** Score each neuron two
ways — structure coefficient (correlation of its residual activity with the
canonical variate) and raw CCA weight (back-projected through PCA loadings).
Members = top quartile of |score| across significant dims; robustness sweep
over the threshold (10th–90th pct). Gini coefficient for weight sparsity.
Cross-pair commonality via member-set overlap (Jaccard) + score-vector
correlation for shared areas; cross-epoch membership stability tracked.

**D10 — Subspace rotation: principal angles, all transitions.** Principal
angles between canonical subspaces for every epoch pair (naive↔inter,
inter↔expert, naive↔expert), X-side and Y-side separately, calibrated against
a within-epoch split-half angle (noise floor). Compared at matched dimension.
Subspaces are expressed in **neuron space** — the frame common across epochs —
by back-projecting the canonical coefficients through each epoch's PCA
loadings (this replaces the shared-PCA-basis idea; see edit log v3).

**D11 — Compute: NumPy + CPU multiprocessing.** numpy/scipy CCA reference
implementation; surrogate/lag loops parallelised across CPU cores. No GPU
(matrices too small to benefit).

**D12 — Delivery: planning doc → full pipeline (TDD) → staged check-ins →
writeup.** This document first. Then build the full pipeline test-first, in
3 checkpointed stages with a pause for review between each. Methods + Results
writeup at the end.

### Self-made decisions (not grilled — flag if you disagree)

- **FS exclusion:** drop all `final_neurontypes(:,5) == 2` units in every
  area. Verified type-2 exists for V1/CA1 too.
- **Epochs:** naive = trials 1–10; intermediate = (lp−9):lp; expert =
  (lp+1):(lp+10). Disengagement truncation at `change_point_mean` applied
  before epoch assignment.
- **Learning point `lp`:** project rule — zscored lick error ≤ −2, window 10,
  ≥ `min_consecutive` within window. `min_consecutive` is a config knob;
  **default 7** (project default — 14 raw learners, stable LPs). Sensitivity
  check (edit log v2): raising it to 8 cleans animal 8 (lp 16 → 20) but
  reclassifies animals 5 & 6 as non-learners and destabilises animal 11
  (lp 84 → 152) — so a global switch to 8 is rejected.
- **Animal 8:** lp=16 at the default criterion is implausibly early (learning
  a hidden reward zone within 16 trials) — likely the "tricky bug" Theo
  flagged. Treated as a **non-learner and yoked**, which removes the
  naive/intermediate overlap without destabilising the rest of the cohort.
- **Yoked non-learners:** animals 8, 12, 16 receive the cohort-mean lp
  (≈40–43, computed from the trusted learners). Animal 15 dropped entirely
  (27 usable trials — even yoked epochs don't fit).
- **Subspace dimensionality:** d = number of canonical dims passing the
  surrogate significance test, per (animal, pair, epoch). CC1 used for the
  strength/direction questions; the d-dim subspace for membership and angles.
- **Code location:** this self-contained Python subfolder
  `Striatum project/cca/` (mirrors the `rl_model/` precedent), with
  `src/`, `tests/`, `scripts/`, `figures/`, `results/`; `uv` venv.

---

## 4. Pipeline architecture and stages

**Stage 1 — data IO + core residual CCA + CV.** [DONE — see edit log v3]
h5py loader for `preprocessed_data.mat`; epoch/LP/FS/unit-selection logic;
missing-bin imputation; per-epoch residualisation; per-epoch PCA; plain CCA
with 5-fold whole-trial CV. 35 synthetic-ground-truth tests. Validated on the
full cohort × 10 pairs. **Checkpoint: review the held-out CCs before Stage 2.**

**Stage 2 — surrogates + lagged/IFI + full-cohort run.** Trial-permutation
and circshift nulls; ±5-bin lagged refit; IFI and peak-lag; run all 10 pairs
× 3 epochs × cohort with multiprocessing. **Checkpoint: review significance
and directionality.**

**Stage 3 — unit membership + principal angles + figures + partial-CCA
add-on.** Structure coefficients + weights + Gini; cross-pair / cross-epoch
comparisons; principal angles with split-half control; partial CCA on the
DMS/DLS/ACC triplet; all figures (axis labels + units + titles per CLAUDE.md);
results saved alongside figures. **Checkpoint: review, then writeup.**

---

## 5. Won't-Do (explicitly out of scope)

- **DG** — excluded per Theo's instruction (too few units / animals).
- **Fast-spiking cells** — excluded for this first pass (may revisit later).
- **Temporal-bin CCA** — analysis is on spatial bins only; no time-resolved
  within-bin CCA.
- **Reduced-rank regression** — not used (neither paper uses it; CCA only).
- **GPU / batched-torch backend** — not used (D11).
- **Re-binning to finer spatial resolution** — not done (D5).
- **Dark / ITI periods** — corridor traversal only.
- **Reusing the old MATLAB v2/v3 CCA code** — starting fresh in Python; the
  v3 driver depends on `v5_*` primitive files that no longer exist anyway.
- **Exact replication of either paper** — deliberately a hybrid adaptation.
- **The downstream RL-latent regression** — separate initiative, not this.

---

## 6. Open items

1. ~~Epoch-disjointness / animal 8~~ — RESOLVED (v2): animal 8 yoked as a
   non-learner; `min_consecutive` kept at 7. See §3 and edit log v2.
2. Code subfolder `Striatum project/cca/` — confirmed by Theo ("Let's go").
3. ResearchVault not mounted; the Methods/Results writeup will be drafted in
   `cca/` and moved once that folder is shared.

---

## 7. Edit log

- **2026-05-23 — v3 (Stage 1 built).** Python package `cca/` created
  (config, core, dataio, pipeline; 35 passing tests; `uv` venv). Three
  deviations from the v2 spec, all forced by the real data:
  1. **Per-epoch PCA, not a shared 30-trial basis (D4/D10).** A shared basis
     made a global component carry near-zero variance inside one epoch (units
     silent in naive but active later) → that epoch's CCA was rank-/condition-
     deficient. Fixed: PCA per epoch; cross-epoch subspaces compared in neuron
     space by back-projecting through each epoch's loadings.
  2. **Missing-bin imputation (new).** `spatial_binned_fr_all` has 0.01–3.3%
     NaN — spatial bins the animal did not occupy on a trial. Filled with the
     per-(bin,unit) trial mean → zero residual fluctuation after D2.
  3. **k also capped by per-epoch numerical rank**, so every epoch's CCA gets
     full-rank input.
  *Realised cohort (FS excluded):* fits are sparser than the v2 projection —
  FS exclusion drops more area×animal cells below the 5-unit floor. Learner
  fits per pair: DMS-DLS 8, DMS-ACC 10, DLS-ACC 7, V1-DMS 3, V1-DLS 2,
  V1-ACC 3, **CA1-DMS/DLS/ACC 1 each, CA1-V1 2**. Animal 13 (a CA1 learner)
  loses all striatal areas — >50% of its DMS/DLS/ACC units classify as FS.
  The CA1 arm is effectively a single-animal anecdote (animal 11). Flagged
  for Theo at the Stage 1 checkpoint.
- **2026-05-23 — v2.** Theo signed off ("Let's go"). Non-learners now
  *included* via the yoked (cohort-mean) learning point rather than dropped —
  recovers animal 12 (rich CA1/V1). Cohort = 12 learners + 3 yoked
  non-learners (8, 12, 16); animal 15 dropped (too few trials). Ran an LP
  criterion sensitivity check: `min_consecutive` 7 vs 8 vs 9 — 8 cleans
  animal 8 but destabilises animals 5/6/11, so default stays 7 and animal 8
  is yoked instead. D1 and the epoch decisions updated accordingly; open
  item 1 resolved.
  *Prior (v1) D1:* "Pair scope tiered — compute all 10 pairs, group stats
  only for striato-cingulate pairs, V1/CA1 exploratory, non-learners dropped."
  *Prior (v1) epoch rule:* "require lp ≥ 20; drop animal 8 (lp=16)."
- **2026-05-23 — v1.** Initial document. Twelve design decisions resolved via
  structured interview (D1–D12). Data reality verified directly from
  `preprocessed_data.mat`. Awaiting sign-off before Stage 1.
