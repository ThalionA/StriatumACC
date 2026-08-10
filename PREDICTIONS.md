# Predictions (newest first)

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
