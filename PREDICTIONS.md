# Predictions (newest first)

## 2026-08-11 — tcca epoch grid: bin {25,10 ms} × FS {excl,incl} × {partial,plain} + IFI integration-window sweep

Eight run_epochs configs on the seeded 5 cm cache (A7's LP is 33 vs the recorded
34; the other 10 learner LPs are identical — smoke run reproduced A1's nine cc1
values to 4 d.p.). New per-cell exports: CC1 lag curves to ±250 ms
(epoch_lagcurves*, IFI recomputable at any window via `lagged.ifi_by_window`)
and the partner-dependent `gini_pearson_x/y`. Registered before any cohort run.

- **P1 (strength null is config-robust):** paired per-animal Wilcoxon on cc1
  naive→expert, striatal triangle, all 8 configs (24 tests): 0–2 nominal
  p<0.05, none surviving BH within its config. Confidence ~80%. Basis: the
  spatial sweep sat at chance, the 25 ms temporal null, and Tom's threefold
  strength null.
- **P2 (FS inclusion raises coupling, changes no verdict):** cc1(FS-incl) >
  cc1(FS-excl) in ≥70% of matched (animal, pair, epoch) cells at 25 ms partial.
  Confidence ~75%. Basis: Tom's hierarchy uniformly higher FS-incl; spatial
  FS-incl/excl agreement r=0.89.
- **P3 (plain − partial isolates shared drive):** plain cc1 > partial cc1 in
  ≥80% of matched cells, median uplift ≥ +0.05 (popsim: common-input 0.74→0.19
  under partialling). The epoch-strength null persists even in plain.
  Confidence: uplift ~85%; null-persists ~65% — this is the arm most likely to
  produce a (spurious-looking) epoch effect, since shared drive tracks
  behavioural state.
- **P4 (IFI ≈ 0 at every integration window):** per-animal IFI(w), w up to
  ±250 ms: (a) no striatal-triangle (pair × window) cell survives BH across
  windows within pair in any config (~70%); (b) session-pooled existence-level
  IFI vs 0 at ±50 ms also null (~75% — the 25 ms run already gave IFI 0.000 ±
  0.09), unlike Tom's CA1→RSC where a flow exists.
- **P5 (10 ms is noisier, verdicts unchanged):** per-cell cc1 magnitudes lower
  at 10 ms in the majority of matched cells; no cohort verdict flips. ~70%.
- **Falsifiers:** (i) any pair with a BH-surviving, same-sign IFI window band
  (≥2 contiguous windows) in BOTH FS conditions → a genuine directional flow —
  Fig 4c revives; (ii) any pair with same-sign p<0.05 naive→expert cc1 change
  in ≥3/8 configs → the strength-change story revives.

**Outcome (2026-08-11, same day) — ✓ P1; ✗ P2; ✗ P3 REVERSED (the registered
surprise); ✓ P4 with one single-config footnote; ✓ P5. Neither falsifier
triggered.** Full tables: `tcca/results/grid_summary.csv`,
`grid_ifi_windows.csv` (script `scripts/analyze_epoch_grid.py`; animals-as-n,
BH within family).

P1: 24 pair×config tests, exactly 1 nominal hit (b25/fsincl/plain DLS-ACC
p=0.039), 0 survive BH. The strength null is robust to bin size, FS condition
and partialling.

P2: FS inclusion raises cc1 in only **54%** of matched cells in the partial
frame (p_animal 0.43 b25 / 0.11 b10) — prediction wrong. The predicted uplift
exists only in the PLAIN frame (65% p=0.039 b25; 78% p=0.004 b10). Reading: FS
units contribute mostly **shared** variance, which partialling removes — Tom's
"uniformly higher FS-incl" does not transfer to the partial striatal pipeline.

P3: **REVERSED.** plain − partial is *negative* FS-excluded (median −0.012
b25, p_animal=0.016; −0.031 b10, p=0.039; only ~⅓ of cells positive) and ≈0
FS-included. I anchored 85% confidence on popsim's common-input scenario;
these data are not that scenario. Lesson: the triangle's coupling is **not
inherited from shared drive off the other recorded areas** — partialling acts
as *denoising* (without residualisation, PCA-k spends components on
high-variance global directions and crowds out the coupling-carrying ones).
This is a positive, citable statement: the coupling is pair-specific.
Null-persists sub-prediction ✓ (all plain configs n.s.).

P4: learning-change: **0/420** (pair × window × config) cells BH-survive.
Existence: null everywhere except one config-island — b10/fsincl/plain
DLS-ACC, contiguous ±200–240 ms band, median IFI +0.02, p=0.0078. The
falsifier required the band in BOTH FS conditions; it is absent FS-excluded
and absent under partialling → not triggered. Parsimonious read: a small
shared-drive asymmetry visible only in the least-controlled config.

P5: b10 − b25 cc1 negative in all four frames (medians −0.034…−0.040, 20–31%
of cells positive, p_animal ≤ 0.02); no verdict flips.

Bonus (the audit's §6 fix, first use): the partner-DEPENDENT
`gini_pearson_x/y` is **also flat across learning** (b25 committed config:
x p=0.73, y p=0.30; cohort medians 0.34–0.38) — the Fig 4d Gini null survives
the corrected metric, so the panel stays a null (or is dropped) either way.

## 2026-08-10 — spatial bin-size soundness comparison (2.5 cm vs 5 cm, task cohort)

First run of `compare_bin_sizes.m` on the dual-bin outputs of the reworked
`ProcessStriatumTask.m` (post depth-fix `all_data.mat`, 16 mice). Question:
does 2.5 cm binning cost reliability as sparse-spike theory predicts, or is
it an unmitigated win for n_bins?

- **P1 (reliability):** 5 cm beats 2.5 cm on split-half tuning reliability in
  the striatal areas — per-animal median Δr = r(5cm) − r(2.5cm) > 0 in ≥12/16
  animals for DMS, DLS and ACC. Confidence: ~75%. Basis: ~0.125 expected
  spikes/bin/trial at 2.5 cm for a 1 Hz MSN; CV ∝ 1/√count.
- **P2 (no sub-5cm structure):** the interpolated-coarse test shows no genuine
  fine structure in striatum — per-animal median d_str = r_fine − r_cross ≤
  0.02 for DMS/DLS/ACC. V1 is the plausible exception (finer spatial coding);
  if any area shows d_str > 0, it will be V1. Confidence: ~65%.
- **P3 (sparsity):** zero-spike fraction at 2.5 cm exceeds 5 cm by >10
  percentage points (per-animal median, striatal areas). Confidence: ~80%.
- **Falsifier:** d_str > 0.05 in a majority of animals for any striatal area
  → real 2.5 cm-scale structure → 2.5 cm becomes the justified primary and
  the 5 cm default recommendation is wrong.

**Outcome (2026-08-10, same day) — ✓ P1 exceeded; ✓ P2 confirmed and stronger
than predicted; ✗ P3 wrong in magnitude; falsifier not triggered.**

P1: 5 cm beat 2.5 cm in **every animal in every area** — 15/15 DMS, 10/10 DLS,
15/15 ACC (predicted ≥12/16), plus 5/5 V1, 3/3 CA1, 2/2 DG. Median Δr
+0.16 to +0.18, signrank p ≤ 0.002 (striatal areas).

P2: no detectable sub-5cm structure anywhere — d_str −0.08 to −0.12, and the
predicted V1 exception did NOT materialise (V1 d_str −0.10, 0/5 animals with
d_str > 0.05). Not a single animal-area crossed the falsifier line.

P3 (the registered surprise): the zero-fraction gap was 3–9 pp, not >10 pp —
because BOTH bin sizes are already >90% zeros in striatum (94.8% vs 91.3% DMS).
Ceiling effect I failed to anticipate: at ~0.1 expected spikes/bin, halving
the bin width cannot add much to an already-saturated zero fraction. Lesson:
the sparsity cost of small bins shows up in tuning-curve reliability (P1's
+0.17), not in the zero-fraction — the naive sparsity metric saturates.

Decision: **5 cm is the primary bin size for population/trial-resolved
analyses.** 2.5 cm stays only for pre-declared fine-spatial questions; no
current analysis qualifies.

## 2026-07-28 — tcca epoch run (reproduction of the lost 2026-06-17 cohort)

The 25 ms `run_epochs` outputs were gitignored and are gone from disk; only the
summary in `tcca/NOTES.md` survives. This rerun is therefore a reproduction test
of a recorded-but-unverifiable result, on unchanged code (`165` tests green) and
unchanged data (`preprocessed_data2p5cm.mat`, 23 May).

- **Prediction (determinism):** the run reproduces 125 cells across 11 learners,
  animals 3 and 15 skipped for too few run-trials. Confidence: high (~85%).
  Basis: no RNG seed is set for the circshift null, so `n_sig` may drift, but
  cell count and held-out cc1 should not depend on it.
- **Prediction (magnitudes):** held-out cc1 for the striatal triangle lands
  within ±0.03 of DMS-DLS 0.25/0.30/0.19, DMS-ACC 0.17/0.19/0.19, DLS-ACC
  0.22/0.13/0.30 (naive/int/expert). Confidence: medium (~65%).
- **Prediction (the open question):** cross-epoch rotation will **not** clear the
  within-window split-half floor for most pairs — i.e. round 8's temporal
  reorientation (18/20 cells above floor) will *fail* to survive residual+partial
  CCA with held-out CV, because that arm was signal CCA and shared position/time
  tuning plausibly carried the rotation. Confidence: low-medium (~45%). This is
  the prediction I would most like to be wrong about, and the one the run exists
  to settle.
- **Falsifier:** cell count differs by >5, or any triangle cc1 misses by >0.05,
  or rotation-minus-floor is positive for ≥7/9 striatal-triangle transitions.

**Outcome — ✓ P1 exact; ✓ P2 exact; ✓ P3 substantively confirmed but its
falsifier was mis-specified.**

P1: 125 cells, 11 learners, animals 3 and 15 skipped — reproduced exactly.
P2: all nine striatal-triangle cells matched the recorded values to within
0.005 (worst |Δ| = 0.004), confirming the recorded numbers were *means* and
that held-out cc1 is deterministic; the unseeded circshift null touches only
`n_sig` (median 1, max exactly 12 — the recorded per-cell outlier reproduced).
IFI ≈ 0 (mean +0.000, sd 0.09) and Gini_x median 0.42 also reproduced.

P3: cross-epoch rotation does **not** exceed the within-window split-half floor
above chance. With animal as the inferential unit (n=10): mean +1.92°, median
+1.03°, Wilcoxon p=0.38, t p=0.26 — and the positive mean is carried by one
animal (A10, +22.4°; without it the mean is −0.35°). At n=10 the Wilcoxon
p-floor is 0.001, so this is a **powered null, not a power floor** — unlike the
epoch-strength result. Round 8's signal-CCA arm reported 90% of cells above
floor; this arm gives 56% of sides (p=0.10 even before correcting for
non-independence). Aggregation units differ between the two arms, so treat the
90%→56% gap as indicative rather than a formal comparison — but the animal-level
null stands on its own.

**Lesson 1 (the important one): I set the falsifier at the chance rate.** The
"either of two sides clears the floor" criterion has P=0.75 under an
exchangeable null; my ≥7/9 (78%) threshold therefore could not distinguish
signal from noise in either direction. Pick thresholds against the null's
expectation, not against intuition about what "most" means.

**Lesson 2: pooled proportions across pairs are an artefact here.** The
all-pairs figure (61% above floor, p<0.001) is manufactured by eight pairs that
rest on a *single animal each* (CA1-*, DG-*), which score 83–100% by noise.
Report per pair with the backing animal count attached, never pooled.

**Lesson 3: the recorded cc1 headline used means on a right-skewed n=7.**
Mean and median diverge badly (DMS-DLS naive 0.248 vs 0.148; DLS-ACC
intermediate 0.134 vs **0.017**). The typical animal in DLS-ACC intermediate
shows essentially no communication. Lead with medians, or show the per-animal
points.

## 2026-07-13 — LFP audit reproducibility checks

- **Prediction:** Raw within-event voltage peaks will reproduce the documented
  signed offsets from the nearest VR sync edge (within 0.25 s of +1.98/+4.26/+1.74 s
  for 614/727/731), and common-median referencing will alter the ~75/~151 Hz peaks
  by less than 0.2 dB. Confidence: medium-high (~75%).
- **Falsifier:** Any offset misses by >0.25 s, or any peak changes by >=0.2 dB,
  when recomputed by a documented script from the cached event bins and raw voltage.

**Outcome — ✓ timing confirmed; ⚠ referencing claim invalidated.** Robust raw
voltage peaks reproduced +1.983/+4.256/+1.740 s for 614/727/731. However,
common-median referencing reduced 614's 153.8 Hz peak by 4.54 dB. Its 74.2 Hz
peak and both peaks in 727/731 changed by only -0.06 to +0.03 dB. Lesson: do
not generalise a referencing check across frequencies or sessions; persist the
per-peak change in the machine-readable output.

## 2026-07-12 — LFP sanity reanalysis

- **Prediction:** The earlier claim that the four LFP exports are ~99% empty will be invalidated once exact zeros and signal occupancy are measured without the absolute `SD > 0.02` threshold. Confidence: high (~85%). Basis: voltage units are unknown and the files occupy 11–16 GB, close to dense float32 storage.
- **Prediction:** Scale-free diagnostics will still identify intermittent, synchronous broadband artefacts, but ordinary low-amplitude LFP will be present through most corridor epochs. Confidence: medium (~65%).
- **Falsifier:** If sample-level exact-zero fractions are near 99%, or robust within-session amplitude/PSD diagnostics remain absent across channels and corridor epochs independently of threshold and scaling, the empty-export diagnosis stands.

**Outcome — ✓ first prediction confirmed; ↔ second prediction partly confirmed.** Full-file exact-zero fractions were 0.004% (1212), 5.34% (614), 3.16% (727), and 4.62% (731), with zero fully-zero windows during behaviour. The latter three zeros are single terminal padding blocks. Ordinary task voltage is continuous, but not clean conventional LF-band data: lag-1 correlations are near zero, only 16–18% of 1–499 Hz power lies below 100 Hz, and 614/727/731 contain strong ~74–75 and ~151 Hz peaks. Periodic high-amplitude events recur every 60 s or 5 s. Lesson: never threshold undocumented voltage in absolute units; verify exact zeros and signal bandwidth separately.
