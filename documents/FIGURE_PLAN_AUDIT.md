# Figure-Plan Audit — 2026-08-10

**Scope.** Every panel of the handwritten manuscript plan (Figures 1–4, notes dated 10/08/26) verified
against current code and on-disk artefacts, superseding stale verdicts in
[`MANUSCRIPT_REPORT.md`](MANUSCRIPT_REPORT.md) (2026-07-30). Method: 8 parallel verification agents,
each re-reading code at file:line and recomputing numbers from caches/CSVs (read-only; no MATLAB runs).
Plus: the tcca ↔ TomLearning alignment check (§6).

> ✅ **Adversarial skeptic pass completed 2026-08-11** (8 refuters, one per cluster):
> **0 verdicts overturned, 15 modified, the rest upheld.** All modifications are folded into
> the text below; the two that changed recommendations are the RL lick-channel headline
> (§1 — the saturated null is exposure-confounded) and the sweep "chance rate" reading (§4).

**Verdict key.** ✅ SUPPORTED = code sound AND citable artefact on disk · 🔧 FIXABLE = defined fix,
effort stated · ❌ REFUTED = claim contradicted by data/code · ⬜ UNBUILT = never run / no asset ·
⚠️ UNTESTABLE = artefact lost or provenance broken.

**The one-line summary.** Figures 1–2 are *runnable but unrecorded* (sound cores, zero citable
numbers, every figure artefact stale by two data revisions); Figure 3 is *blocked* (TCA cache deleted;
rank/seed/circularity defects unfixed; the RL bridge is half a day from its first honest run);
Figure 4's *learning framing is refuted by the project's own well-powered nulls* and must pivot to a
static-architecture + local-refinement story.

> ### ⚠️ 2026-08-11 UPDATE — the "local refinement" half is now refuted too.
> Everything was regenerated on the clean 5 cm caches and, for the first time, **quantified**
> (per-animal CSVs now tracked in git). At the animal level: **trial-to-trial stability does not
> increase** in any area (all p_BH > 0.14; DMS 6 up / 7 down), and **position decoding does not
> improve more than in control mice** (Task Δ −0.032 p=0.168; Control 1 Δ −0.044 p=0.062;
> between-group p=0.924), while decoder *certainty falls* (entropy p=0.021). Combined with the
> TCA control decomposition — where control mice reproduce the same trial-factor "learning" steps,
> because the tensor's slot axis has a ~20-trial gap at the Naive|Pre-LP boundary — **the paper's
> entire two-stage narrative (early map stabilisation → later ensemble emergence) is unsupported
> by the analyses as they stand.** What survives is architecture, not change: reliable spatial
> coding far above shuffle, decodable position, pair-specific communication subspaces, and
> task-specific *spatial* structure in the TCA components that controls lack. The manuscript needs
> reframing around that, or needs the LP-shuffled null + within-epoch decoder to rescue the
> learning claims — see §7.

---

## 1. Figure 1 — behaviour

| Panel (handwritten plan) | Verdict | Core finding |
|---|---|---|
| Behaviour schematic + control groups | ⬜ / 🔧 | Schematic = illustration (no asset). Controls are **no-task baselines, not controls** for learning/drift claims; caption them as such. |
| Learning heatmap + performance quant → epochs | 🔧 | Code sound (animal-level SEM, learners-only quant) but **no behavioural number is recorded anywhere on disk**; all SVGs stale (May 24 / Mar 4). Heatmaps must be regenerated at 5 cm. |
| Increasing stability of licks and velocity | 🔧 | Computation exists (IntegratedAll S4.4) but **no null, no test of any kind** — currently an uncaptioned visual claim. |
| Optimality vs RL model | ⬜ optimality / ✅ RL | Optimality: orphaned, zone geometry unrecorded, circular expert window — **drop it; let the RL model carry the panel.** RL lick channel is the strongest quantitative asset in the project. |

Key facts, re-derived today from `preprocessed_data5cm.mat` (Aug 10 13:54):

- **LPs are now deterministic and reproduce July's numbers**: project rule → 14/16 learners, LPs
  `[22 44 32 53 54 36 33 14 44 67 84 NaN 23 39 26 NaN]`, mean 40.8. But **three LP conventions still
  coexist** (`Nonlinear_Epoch_Decoding.m:43-45` / `CrossSpatialBinDecoding.m:41-43` → 8 learners mean
  51.5, animal 1 moves 22→85; MI v2 adds +9), plus a fourth wrinkle: IntegratedAll's Expert = `lp:lp+9`
  vs processTaskData's `lp+1:lp+10`. Same mouse is a learner in one figure and not in another.
- **Mouse 823 (LP 14): Naive and Intermediate share 7 of 10 trials** — no overlap guard anywhere.
- **Occupancy denominator STILL unfixed** (`spatial_binning.m:45`, last-minus-first = (k−1)·dt). In the
  live path of *both* new caches. The 5 cm repoint roughly **halves** the inflation (~8–20 % now) but
  the speed-dependence — the property that matters, since mice speed up with learning — is intact.
  **No speed control exists anywhere in the repo.**
- The July "velocity wrong by 2×" verdict is **obsolete**: `(4*1.25)` now equals the true 5 cm bin
  width against the caches all three consumer scripts load. Correct-by-coincidence — tie the literals
  to `cfg.velocity_factor` (exists, unused) before anyone re-bins.
- **RL model (Fig 1's model panel)** — skeptic-corrected 2026-08-11. **Citable today:** the
  Naive→Expert lick-profile reproduction, **11/13 evaluable learners, mean Δr +0.70** (artefact-backed,
  untouched by the null critique), plus the honest velocity caveat (6/16, Δr +0.36; CV ≈ 0,
  CV-scheme-dependent). **NOT citable as-is:** the held-out CV gain (+0.60 nats/bin v6 interleaved,
  16/16; +0.45 forward). The saturated per-bin null has **no dwell-time exposure term** while the
  model scores Poisson(rate·dt) with dt from observed same-bin velocity — MANUSCRIPT_INVENTORY.md:4993
  rates this "FATAL for the headline claim"; exposure-matched re-scoring collapses the gain to
  ~+0.11 (inventory spec, 11/16) or ~+0.36 (skeptic's independent spec, 12/16) — direction confirmed
  twice, magnitude spec-dependent, and neither collapse number has a backing script yet. **Fix is
  cheap**: null re-scoring only, no refits, hours — run it before any caption quotes a CV gain.
  (The v6 aggregates ARE recorded — inventory:4930 — contrary to the first-pass claim.)
  Real-data model-comparison ladder: never run (runnable today, hours).
- Control caches (regenerated today) still carry the July §5.9 defect: **446 unlabelled
  V1/CA1/DG units inside control-1 "all units"**. Control 2 is now 6 sessions — but the animal
  count is **unresolved**: `mice_info3.xlsx` suffixes suggest 1107_M1/M2 are two *different* mice
  recorded the same day (6 animals), not one mouse twice (5) — **confirm identity with Zihao before
  any per-mouse n is written**, and note 1107_M2's depths are flagged "??" (provisional). 624's
  exclusion is acknowledged in NOTES:37 ("row unused") but unrationalised. fr_threshold moved
  0.1→0.02 (old control-2 numbers incomparable).

---

## 2. Figure 2 — recordings and learning effects

| Panel | Verdict | Core finding |
|---|---|---|
| Neuropixels track + Allen Atlas | ⬜ | **Nothing exists** — no histology, no registration, no coordinates. Localisation is depth-interval CSVs alone (edited twice this week). Honest caption or import histology. |
| Recording examples | 🔧 | Raster material exists (16/16 `*_raw.mat`, 1 ms floor) but no plotting code; `raw_data_bin.m`'s four deps still absent from the repo (Methods-reproducibility hole, not a panel blocker). |
| Activity evolution across epochs/areas/types | ⬜ | SpatioTemporal machinery is the best MATLAB arm (BH-FDR ×4 sites, de-circularised classifier) but **blocked: `tca_outputs.mat` was deleted** — and zero outcomes were ever recorded. Cell-type panels **still apply striatal criteria to ACC** (the "RS" patch is hollow — no code produces type 5; section :1550 still ungated). |
| Increasing stability of neural activity | ❌ **REFUTED 2026-08-11** | Regenerated and **quantified for the first time** (`figures/stability_by_animal.csv`). Animal-level Wilcoxon naive→expert, Task: DMS median Δ **−0.0002** (n=13, p=0.735), DLS **+0.0027** (n=10, p=0.557), ACC **−0.0259** (n=14, p=0.049, **p_BH=0.148**). Nothing survives BH across the three areas. DMS splits **6 animals up / 7 down** — no consistent direction. The apparent rise in cross-animal medians (0.221→0.259) is a composition effect, not a within-animal change: the paired delta is zero. Reliability *is* far above shuffle in every epoch — the code is reliably structured, it just doesn't become more so. Prior code defects (single unseeded shuffle, edge windows) now fixed/seeded. |
| Improved decoding of position | ❌ **REFUTED as a learning effect 2026-08-11** | Quantified (`figures/decoding_by_animal.csv`). Task error 0.317→0.287→0.266, median Δ **−0.032, p=0.168 (n.s., n=13)**. **Control 1 improves as much or more**: 0.349→0.310, Δ −0.044, p=0.062 (n=5) — mice with no task, no cues, nothing to learn, over LP-yoked windows. Task-vs-Control-1 difference in improvement: **Mann-Whitney p=0.924**. Control 2 flat. Secondary metric contradicts sharpening outright: normalised decoder entropy **increases** naive→expert (Task 0.073→0.097, p=0.021; Control 2 p=0.031) — the posterior gets *flatter*, not sharper. Caveat that cuts both ways: the template is session-global, inflating the naive epoch equally in both groups. |

**Cohort correction (Methods-critical).** Current counts, verified from the caches: 16 task mice,
**DMS 394 / DLS 522 / ACC 699** (striatal+ACC 1615; +V1 264, CA1 133, DG 18 = 2030 total).
The July report's "1540 (367/490/683)" matches **no artefact, old or new** — re-derive, never copy.
Depth fix (mouse 731: DMS band 0–300→500–800 µm, 12 wrong units out, 5 correct in, 17→10 total) is
**confirmed propagated** into both `all_data.mat` and the new cache (content-verified, not just mtimes).
Note 731 also silently discards its 17 deepest units (outside every band) — the depth gate's
no-accounting problem in miniature.

---

## 3. Figure 3 — TCA, ensembles, RL bridge

| Panel | Verdict | Core finding |
|---|---|---|
| TCA components + ensemble evolution | ❌ rank / ⚠️ artefact | `Run_TCA_pipeline.m:1019` still hardcodes `best_n_factors = 5` *after* BIC and relabels the diagnostics figure "Selected = 5" — **a false methods statement**. BIC invalid anyway (2.45 M entries as independent n; RSS = min over 25 unseeded inits). `tca_outputs.mat` **deleted** → the whole arm is unrunnable until a rerun. `tca_with_cv.m` (held-out rank selection) exists and is live in the *control* pipeline — wire it in or declare rank a priori. |
| Ensemble emergence around LP | ⬜ null | Tensor slots (1:10 / lp−10:lp−1 / lp+1:lp+10) make pre/post differ **by construction**; **no LP-shuffled null exists anywhere**. Tractable: ~1 day code + parallel compute. |
| Ensembles track RL variables | ⬜ | **Never run on current latents.** Stale `encoding_v5/v6` npz predate the corrected latents; worse, a naive re-run **silently aggregates the stale results** (done-marker skip logic) — bump ENCDIR first. New blocker found: **M13 off-by-one in today's cache** (156 behaviour vs 155 neural trials, cp=NaN) crashes the run. RPE is near-collinear with licking (r up to −0.995) — add a dwell-time (1/v) nuisance before attributing dR² to RPE. Precision caveat sharpened: retained variance **median 0.000** (14/16 mice ≈0; M16 = 0.73) — and value retains only 0.19, which nothing currently states. ~Half a day to the first honest run. |
| Ablations/manipulations | 🔧 / language | `decode_ensemble_ablation.m:60-70` leakage confirmed byte-for-byte (trains on 21:30; `trn` computed, unused; LOO line commented). **5-minute fix**, rides on the TCA rerun. **No physiological manipulation exists in the repo** — language must stay at decoder sufficiency, never "necessary". |
| (July 3B/3D/3E carry-overs) | 🔧/❌ | Anatomical ANOVA still pools units across mice (`plotNeuronFactorsByArea.m:65`) while the correct per-mouse means sit untested in `plotFactorHeatmapsByMouseArea.m:51`. t-SNE panel circular (colours = argmax of embedded coords). Cell-type composition unmasked (ACC "MSNs" contaminate). Ensemble SEM fix is in code but no artefact postdates it. |

**Latent-recovery numbers** (Fig 3/4 caption material): the cited tuple (value 0.996 / RPE 0.994 /
lick 0.999 / precision 0.938) is **v7 prose-only** — `recovery_v7.npz` doesn't exist locally and the
tracked `DONE_v7` flag is a false completion marker. Citable from disk (`recovery_v6.npz`): value
0.996, RPE 0.994, lick 0.999, **precision 0.915 mean / 0.937 median**. Cite v6 or re-run v7 (~1 h).

---

## 4. Figure 4 — communication subspaces: THE PIVOT

Planned: CCA schematic · canoncorr across learning · information-flow direction · Gini across
learning · network schematic. **Three of the five planned claims are refuted by the project's own
artefacts — all values re-read from the CSVs today.**

**Refuted as learning effects (well-powered, convergent):**

1. **CC change across epochs**: every animal-level rm-ANOVA n.s. — actual p range **0.155–0.947**
   (note: `MANUSCRIPT_RESULTS.md:340`'s "0.33–0.69" is wrong as a range; conclusion unchanged). Even
   dimension-level (pseudoreplicated) CC ANOVAs are all n.s.; the three dim-level hits are IFI-only and
   die at animal level. Sweep reading **corrected by the skeptic pass**: the enrichment CSV's "chance"
   baseline is the observed grand rate, not nominal 5% — the sweep-wide naive-vs-expert p<0.05 rate is
   **8.5% (1.7× nominal)**, concentrated in DMS-ACC (14.0%, 2.8× nominal) and high-k configs, with
   DMS-DLS (4.5%) and DLS-ACC (5.4%) at nominal. Sweep cells are massively non-independent (same
   animals, 1056 re-analyses), so this is weak evidence either way — the committed-config animal-level
   tests, all n.s., remain the verdict. tcca replicates the null temporally.
   **[2026-08-11 grid update:]** the temporal strength null is now shown robust across the full
   factorial — bin {25, 10 ms} × FS {excl, incl} × {partial, plain}: 24 pair×config tests, one
   nominal hit, none BH-surviving (`tcca/results/grid_summary.csv`, registered priors scored in
   PREDICTIONS.md 2026-08-11). Bonus finding: plain CCA sits *below* partial FS-excluded — the
   coupling is pair-specific, not shared drive from the other recorded areas.
2. **Directional flow**: triangle IFI vs 0 — all p ≥ 0.052, all epochs, both variants; the two
   near-misses are *negative* and unsustained; **peak-lag median = 0 bins for every pair**; IFI signs
   flip between null models. The honest claim is *symmetric, zero-lag coupling*.
3. **Gini across learning**: spatial rise (0.66→0.76 DMS-DLS) was **never tested** (PNG-only,
   eyeballed); temporal Gini recomputed today from `tcca/results/epoch_metrics.csv`: **flat**
   (per-animal triangle medians 0.440/0.441/0.430; paired naive→expert Wilcoxon **p = 1.0**, n=10).
   And see §6: the exported temporal Gini is partner-invariant — it isn't a communication metric at all.
   **[2026-08-11 grid update: closed.]** The partner-*dependent* `gini_pearson` (the §6 fix, now
   exported) is **also flat** across epochs (x p=0.73, y p=0.30, committed config) — the null stands
   under the corrected metric.
4. (From tcca 2026-07-28, **demoted to inconclusive-leaning-null by the skeptic pass**:) temporal
   reorientation does not exceed its floor (n=10, rot−floor median +1.03°, Wilcoxon p=0.38; A10 alone
   carries the positive mean) — but the *measurement* is ceiling-compressed: rotation and floor both
   sit within ~15° of the 90° ceiling, and the floor comes from half-data single-draw fits
   (inventory ~4188c says report as INCONCLUSIVE). No evidence for temporal reorientation, but a
   modest real rotation could hide. The *spatial* reorientation result is separate and survived its
   residual/signal factorial. Also: none of the tcca p-values is produced by a committed script —
   they were ad hoc (arithmetically verified); the grid analysis scripts this.

**Supported (the static architecture):**

- **Held-out CC1 > 0 in every sampled pair, every epoch, at animal level**: DMS-DLS 0.154/0.140/0.180
  (p=.0019/.031/.027, n=5); DMS-ACC 0.120/0.096/0.115 (p≤7.7e-4, n=7); DLS-ACC 0.107/0.094/0.093
  (p≤.0064, n=4); V1-ACC similar (n=3). Caption caveats: dims are significance-selected (magnitudes
  conditioned upward; existence rests on the per-dim circshift nulls, 141–160 pooled sig dims per
  triangle pair) and epoch-incomplete animals are dropped.
- **Spatial subspace reorientation** across epochs survives its sweep (19–27/27 configs per pair) —
  the one *change*-with-learning result still standing, spatial arm only.
- Reporting rules that bind every panel (tcca lessons): **lead with medians** (DLS-ACC intermediate
  mean 0.134 vs median **0.017**); **never pool above-floor proportions across pairs** (eight pairs
  rest on one animal each); annotate per-pair backing-animal counts (DMS-ACC 7–10, DMS-DLS 5–7,
  DLS-ACC 4–7, V1 pairs 2–3, CA1/DG 1).

**Recast Figure 4:** (a) residual+partial CCA schematic (must depict the *committed* pipeline, not
textbook CCA); (b) CC>0 per pair × epoch, per-animal points + medians; (c) IFI/lag panel *as a null*
(symmetric, zero-lag); (d) drop Gini or show the flat connection-specific version after the §6 fix;
(e) **static** coupling network annotated with backing-animal counts. Combined with Figures 1–3, the
defensible headline becomes: **learning refines local representations without reorganising
inter-areal communication strength or direction** (spatial reorientation being the nuance).

**Provenance (gates any Fig-4 number):** every committed cca/tcca artefact is 2.5 cm-era, single-copy,
gitignored; the input (`preprocessed_data2p5cm.mat`) was **deleted today**; the committed configs were
repointed to 5 cm and the IFI window halved (uncommitted), so **current code no longer reproduces the
on-disk CSVs even from the surviving pkls**. The depth fix changes the substrate (731's DMS is a
disjoint unit set), so even a rebuilt 2.5 cm file would not reproduce the recorded numbers.
**tcca/NOTES.md's "results now committed" is FALSE** — the four epoch CSVs are untracked (`??`) and
the .gitignore negations uncommitted; and the CSVs contain columns (`n_units_*`, `k_eff`) only the
*uncommitted* `run_epochs.py` produces. A committed-config 5 cm rerun (prediction registered first) is
the only route back to reproducibility; expect A7's rows to shift (LP 34→33 under the seeded baseline).

---

## 5. Cross-cutting state (from the provenance agent)

**Resolved since July:** all three caches seeded & regenerated today (task `rng(42)` committed;
control-1 seeded today; control-2 deterministic); fr_threshold 0.02 Hz aligned in all three scripts
(control-2 was 0.1 until today); depth fix content-verified end-to-end; LPs deterministic; cca **143**
and tcca **165** tests pass (run today; cca grew +8 vs the report); bin-size decision is
registered-and-scored with on-disk artefacts (5 cm beats 2.5 cm in **50/50** animal-areas, median Δr
+0.16–0.18, no sub-5 cm structure anywhere, P3 honestly scored as a miss).

**Open hazards, ranked:**

1. **[Resolved 2026-08-10/11.]** The main bundle was committed the same evening (`ecbd34a`: depth
   CSVs + helper + test, 5 cm repoint, seeds, PREDICTIONS incl. the bin-decision numbers,
   compare_bin_sizes code); the tcca epoch CSVs + producing code landed 2026-08-11 (`249ae8c`,
   `428d2bb`). Remaining unprotected: the `figures/compare_bin_sizes_*` artefact files themselves
   (gitignored; the P1–P3 numbers are in git via PREDICTIONS.md) with their 2.5 cm input deleted —
   the decision is recorded but the raw comparison table is single-copy.
2. **Code mtimes postdate the caches they produced** (16:31 vs 13:54/16:25) and PREDICTIONS describes a
   dual-bin producing variant that no longer exists — benign-looking (the diffs are load-guards) but
   formally unprovable. Commit now and note it, or re-stamp with one rerun.
3. `NOTES.md` top entry is still 2026-07-13 — today's whole-pipeline repoint is unlogged in the
   project's canonical memory.
4. Task organiser still uses inline depth logic, not the tested `assign_areas_by_depth.m` (only the
   control organisers call it); its new `save` writes to cwd, not `processed_data/`.
5. `cfg.plot.zone_params` is read by live scripts (SpatioTemporal, ensemble_analysis, Run_TCA) but
   **defined nowhere** — hand-built in lost interactive sessions.

---

## 6. tcca ↔ TomLearning alignment (spatial + temporal CCA question)

Both arms exist and are coherent: **spatial** (`cca/`, committed config: partial CCA, residual,
FS-excl, held-out CV, circshift null) and **temporal** (`tcca/`, port of tom_cca). Alignment of the
temporal arm with TomLearning's **latest** iteration (temporal running-state arm, PRIMARY):

**Aligned (verified at code level today):**

- Gaussian σ=2.5 ms smoothing; 2 cm/s running gate; z-score over the engaged running reference;
  partial CCA vs **all** other recorded areas concatenated; samples-rule k with cap; leak-free
  held-out whole-trial CV (`config.py:136-166`, `runner.py:25-63`).
- **Per-area split-half floors** (`runner.py:105-106`: floor_x from X's halves, floor_y from Y's) —
  Tom's 2026-08-03 shared-floor bug is **absent**.
- **CC1-level rotation floors recorded** (`subspace_window.py:251`, `_max_angle(...,1)`) — Tom's
  "top-3 floors are unmeasurable (~78–82°), test at CC1" lesson is respected in the exported columns.
- Significance attached to the same fit's held-out cc within one function
  (`subspace_window.py:217-222`) — Tom's bug 1 (sig from a different fit) absent.
- Animals-as-n verdicts; the medians and no-pooling reporting rules are already adopted in tcca NOTES.

**Deliberate, documented divergences (fine, but state them in Methods):**

- **25 ms epoch bins vs Tom's 10 ms primary** — config default is 10 ms (Tom-aligned); the epoch
  driver's 25 ms was chosen on magnitude-reference evidence (tcca NOTES / commit 84270ea).
- IFI integrated over ±10 bins (±250 ms at 25 ms) vs Tom's ±50 ms headline window with curves to ±250.

**Misaligned — action needed:**

1. **The exported Gini is Tom's retracted metric.** `epoch_metrics.csv`'s `gini_x/y` =
   `membership.gini(subspace_contribution(canonical_weight_scores(..., d)))` with **d = min(kx,ky)**
   (`subspace_window.py:215,242-245`) — the L2 row-norm over the full retained set cancels the
   partner-determined orthogonal factor, i.e. **partner-invariant, area-intrinsic loading geometry**,
   exactly what TomLearning proved (analytically + r=0.981 across partners) and retracted on
   2026-07-28. **[Fixed 2026-08-11:]** `run_epochs.py` now exports the partner-dependent
   `gini_pearson_x/y` in every config, and the epoch result under the corrected metric is **also
   flat** (x p=0.73, y p=0.30) — the Fig 4d null survives, now as a genuine communication claim.
2. **No frozen-axes arm.** Tom's current iteration freezes canonical axes for every epoch contrast
   (only CC1 has a stable identity; refit-per-epoch contrasts confound axis drift with change). tcca
   refits per epoch. For the scalar cc1-strength null this is conventional and probably harmless; for
   any future membership/direction-tracking claim it is the bug Tom hit twice. Adopt frozen axes
   before extending the temporal arm.
3. **Per-fold per-dim CV stats are averaged by rank** across folds — the mild cousin of Tom's bug 2;
   safe for cc1, fragile beyond dim 1. Keep per-dim claims to CC1.
4. **FS-included co-primary**: Tom reports both conditions; tcca ran FS-excluded only
   (`exclude_fast_spiking: bool = True` flip exists, unused for the epoch run).

**Contract note.** `ResearchVault/Methods/CCA_HH_Adapted.md` §3.1 is stale against BOTH projects —
the striatum spatial arm departs on cv_folds/n_shuffles/sig_threshold/k-rule (recorded in
MANUSCRIPT_RESULTS), and Tom's primary is now 10 ms/frozen-axes. Update the contract or mark it
historical.

---

## 7. Do these first (ranked)

1. **Commit today's work** — depth CSVs + `assign_areas_by_depth.m` + test, 5 cm repoint, seeds,
   PREDICTIONS/tcca entries, the four tcca CSVs, and un-ignore `figures/compare_bin_sizes_*`. Minutes;
   removes the biggest loss risk. Log the session to `NOTES.md` top.
2. **Fix `spatial_binning.m:45`** (occupancy k·dt) + regenerate both caches — the last pipeline-wide
   estimator bug; then every Fig 1–3 rerun happens once, on clean rates. Add a speed-matched or
   velocity-covaried control for the stability/decoding panels.
3. **TCA rerun** (after: seed it, fix the ablation LOO line — 5 min, settle rank via `tca_with_cv` or
   a-priori declaration) — unblocks Fig 2c and all of Fig 3; export numeric artefacts this time.
4. **Run the RL bridge honestly** (ENCDIR bump, M13 guard, 1/v nuisance) — half a day to the paper's
   mechanistic claim; and write down the v6 behavioural aggregates + rerun `recovery_v7`.
5. **Unify LP conventions + add the LP-shuffled null**; then the Fig-4 recast (medians, per-animal
   points, static network) and the tcca Gini export fix (§6).

*Deferred to next session: the adversarial skeptic pass (resume `wf_4dff1f89-a46` after the usage
reset); committed-config 5 cm cca/tcca reruns with fresh registered predictions.*
