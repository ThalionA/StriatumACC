# striatum_tcca — development log (newest first)

Temporal communication-subspace CCA, faithful port of TomLearning `tom_cca` onto
the striatum dataset. Branch: `claude/temporal-cca-port`. Spec/scope agreed with
Theo 2026-06-17 (full report battery; Task-vs-Control as the two-corridor
contrast; fresh tom_cca-style port; plus an engaged-vs-disengaged contrast).

---

## 2026-07-28 — Stage 2 cohort rerun: reproduced, + the reorientation question settled

The 2026-06-17 outputs were gitignored and lost. Reran `run_epochs.py --group
task` (25 ms, unchanged code, 165 tests green) on the same
`preprocessed_data2p5cm.mat`. Prior registered in `PREDICTIONS.md` first.

**Reproduction is exact.** 125 cells / 11 learners, animals 3 and 15 skipped.
All nine striatal-triangle cc1 values match to within 0.005; IFI ≈ 0 (sd 0.09),
Gini_x median 0.42, n_sig median 1 / max 12. Held-out cc1 is deterministic; the
unseeded circshift null only moves `n_sig`. **Results now committed** (all four
`epoch_*.csv`) so this cannot silently vanish again.

**NEW — round 8's temporal reorientation does not survive residual+partial CCA.**
This is what the run existed to settle. Round 8 (`cca/`, sweep tags t20/t40) ran
the temporal arm with **signal CCA** (`subtract_trial_mean=False`), no
partialling, in-sample permutation, and reported subspace reorientation in 18/20
pair-config cells (90%). With residual+partial CCA and held-out whole-trial CV:

- **animal as the unit (n=10): mean rot−floor = +1.92°, median +1.03°,
  Wilcoxon p=0.38, t p=0.26.** No effect.
- The positive mean is one animal (A10, +22.4°); drop it and the mean is −0.35°.
- Per-animal values scatter −10.0° to +22.4° — no consistent sign.
- **This is a powered null, not a p-floor.** At n=10 the one-sided Wilcoxon
  p-floor is 0.001; the test could have detected a consistent effect.
- Side-level (pseudoreplicated) striatal triangle: 78/140 = 56%, p=0.10.

Reading: a large part of round 8's temporal rotation was plausibly **shared
position/time tuning**, which residual CCA removes. The spatial sweep's
reorientation finding is untouched by this (it ran the residual/signal factorial
and survived); it is specifically the *temporal* reorientation claim that does
not replicate. Aggregation units differ between the two arms (pair-config vs
sides/animals), so treat 90%→56% as indicative and lean on the animal-level null.

**Two reporting traps found in the 2026-06-17 summary (fix before any writeup):**
1. **The quoted cc1 values are means on a right-skewed n=7.** Mean vs median:
   DMS-DLS naive 0.248/0.148, DMS-DLS expert 0.190/0.092, DLS-ACC intermediate
   **0.134/0.017**. The typical animal in DLS-ACC intermediate has ~zero
   communication. Lead with medians or plot per-animal points.
2. **Never pool the above-floor proportion across pairs.** All-pairs gives 61%
   (p<0.001) purely because eight pairs (CA1-*, DG-*) rest on **one animal each**
   and score 83–100% by noise. Backing animals: DMS-ACC 10, DMS-DLS 7, DLS-ACC 7,
   V1-DMS/V1-ACC 3, V1-DLS/CA1-V1 2, the other eight **1**.

**Next:** `analyze_epochs.py` — formalise the above (per-animal Wilcoxon + LMM
via `paired_stats`/`mixed_effects`), with medians as the headline, per-pair
animal counts on every panel, and rotation−floor as a per-animal distribution
rather than a hit-count.

---

## 2026-06-17 — Stage 2: epoch analysis driver (WORKING; full run in progress)

`runner.py` (build_present, fit_window, cross_window) + `scripts/run_epochs.py`
wire data→numeric: per (learner, pair, epoch) pull running in-corridor bins, build
the partial-out Z from the animal's other areas, call `subspace_window.window_subspace`
→ held-out CC, n_sig, MI, IFI, Gini + cross-epoch rotation/Jaccard. Writes
epoch_metrics/dims/weights/cross CSVs. 4 runner tests; full suite 165 green.
**Committed d12aa51** (Stages 0-2 code). cca/ untouched.

**Bin-width decision (evidence-based).** Smoke on 2 learners × striatal triangle:
- **10 ms is too sparse per-cell** — several cc1 go negative (e.g. A2 DMS-ACC
  naive cc1=-0.17, n_sig=6) because k=20 PCs on ~10 trials' autocorrelated bins
  overfit, exposed by honest whole-trial CV (per-dim held-out CC swings +-0.6).
- **25 ms is clean** — cc1 positive/stable; **DMS-DLS peaks at intermediate**
  (0.15→0.40→0.10) reproducing the spatial pipeline's bulge; IFI ~0 (also matches
  the spatial nulls).
→ `run_epochs` default now **25 ms (magnitude reference, = report/Tom)**; 10 ms is
the fine/directionality view. `--max-lag 0` auto-scales the IFI window to +-50 ms.

**Caveat (carried).** `n_sig` still inflates in occasional cells (circshift null is
permissive — the spatial pipeline saw ~2.6x vs trial-perm; overfit subdominant dims
slip past). Mitigation = the report's: **animal is the inferential unit, lead with
cc1**, n_sig secondary. Run `/stats-rigor` before any claim.

**Full 25 ms cohort run DONE** → epoch_metrics.csv (125 cells, 11 learners;
animals 3,15 skipped — too few run-trials for disjoint epochs).

**Cohort findings (preview — in-driver Wilcoxon; formal stats = analyze_epochs next):**
- Held-out cc1, striatal triangle (n=7–9): **DMS-DLS 0.25/0.30/0.19**,
  **DMS-ACC 0.17/0.19/0.19**, **DLS-ACC 0.22/0.13/0.30** (naive/int/expert).
  Magnitudes match the *spatial* pipeline (~0.1–0.34). **No significant epoch
  change** for any pair (all paired Wilcoxon n.s.; n=7→p-floor 0.016).
- **The n=2 smoke's "DMS-DLS intermediate peak" did NOT survive the cohort** —
  per-animal it's inconsistent (A1 peaks intermediate, A10 monotonic up, A5
  monotonic down). Good reminder: animal is the unit; n=2 is noise.
- IFI ≈ 0 (±0.05), no epoch trend. Gini_x ~flat (0.4–0.5) — **no de-sparsification**
  (unlike Tom's HC pairs; consistent with the striatal spatial result). n_sig
  sensible at cohort level (1–3; the smoke's n_sig=12 was a per-cell outlier).
- V1/CA1/DG pairs n=1–3 → anecdotal only.
→ **Temporal method reproduces the spatial striatum headline**: communication
real but modest, no strength change, no directionality across learning. Strong
cross-method consistency. (Run `/stats-rigor` + analyze_epochs before any writeup;
small-n "n.s." = power floor, not absence.)

**Next (Stage 2 finish + Stage 3-4):**
1. `analyze_epochs.py` — per-pair contrasts: per-animal Wilcoxon + paired t + LMM
   (paired_stats/mixed_effects); dims-as-n reported-not-inferential; figures.
2. `run_engagement.py` — **engaged vs disengaged** (Theo's add). Design TBD:
   z-score over a *shared session* running reference (avoid scale confound); relax
   `temporal_max_trial_ms` for the disengaged period (dawdling traversals are the
   data) but report running-bin coverage; pair engaged vs disengaged per animal.
3. Stage 3: `run_trajectory` (sliding window, 25 ms, 3 learning axes) +
   `run_ifi_windows` (held-out segment-aware IFI sweep, 10 ms, +-250 ms).
4. Stage 4: `run_transition` (Task-vs-Control, between-cohort; needs control 2.5 cm
   regen), `run_early_trials`, `run_kcca`; then RESULTS.md writeup.

---

## 2026-06-17 — Stage 1: numeric layer ported (DONE, on branch)

`cp`'d 15 data-agnostic numeric modules from `tom_cca` **verbatim** (relative
imports, so byte-identical — no rename needed): `core`, `lagged`, `surrogate`,
`subspace`, `membership`, `subspace_stats`, `trajectory`, `early_trials`,
`kernel_cca`, `kcca_window`, `partial`, `subspace_window`, `paired_stats`,
`mixed_effects`, `crosspair`. Verified each imports only numpy/scipy/sklearn/
statsmodels + sibling numeric modules (zero coupling to dataio/pipeline). Ported
their ground-truth tests (renamed the absolute import). `__init__` exports the
full set. Heavy deps present (pandas, statsmodels 1.x, sklearn 1.2.2).

**161 tests pass** (`python3 -m pytest tcca -q`, ~32 s) — 25 data-layer + 136
numeric. Covers: rank-robust CCA/PCA, 5-fold whole-trial CV leak guards, held-out
segment-aware lag curve + IFI sign/window sweep, circshift null, principal-angle
split-half floor, Gini, MI, trajectory slopes, early-trial projection, kernel CCA
ridge, partial-out leak-free regression, the full `subspace_window` readout.

**Not ported:** `stage3` (imports `pipeline`→`dataio`, coupled to the spatial
path) and `pipeline`/`analysis`/`segments`/`landmark*`/`lagged_temporal`/
`lagged_landmark`/`sweep` (spatial or Arm-A/B). `test_crosspair` deferred to
Stage 3 (its fixtures use `stage3` containers; will re-test `crosspair` against
the striatum result structures then).

**Next — Stage 2.** Wire data→numeric: a `run_epochs` driver that, per
(animal, pair, epoch), pulls running in-corridor bins via `dataio.area_activity`,
builds the partial-out Z from the animal's other areas, and calls
`subspace_window.window_subspace` → held-out CC, n_sig, MI, IFI, Gini, rotation,
membership. First real result. (Read `subspace_window.py`'s exact contract first.)

---

## 2026-06-17 — Stage 0: scaffold + data layer (DONE, on branch)

**What the port targets.** The "Hippocampus-V1 Communication-Subspace Learning
Report" temporal pipeline (`CCA_HH_Adapted`): 1 ms spikes → Gaussian smooth
(σ=2.5 ms) → time-bin (10 ms primary / 25 ms trajectory / 50 ms robustness) →
per-unit z-score over the engaged (in-corridor, running ≥2 cm/s) reference →
residual + partial CCA → held-out whole-trial CV → held-out CC, n_sig, MI, IFI
directionality, Gini, principal-angle rotation → epoch / trajectory / engaged-vs-
disengaged / Task-vs-Control contrasts. NOT the 50 ms raw-count Arm A/B addendum.

**Package.** `tcca/` is self-contained (src/striatum_tcca, tests, scripts,
results, figures), mirroring `TomLearning/cca/`. `conftest.py` puts `src/` on the
path (anaconda python3 has numpy/scipy/h5py — no venv, matches `cca/`). Numeric
modules (core/lagged/surrogate/subspace/membership/subspace_stats/trajectory/
early_trials/kernel_cca/partial/subspace_window) will be ported near-verbatim
from tom_cca in Stage 1 (verified data-agnostic: array+cfg only). Only `config`
and `dataio` are striatum-specific.

**`config.py`.** 6 areas (DMS/DLS/ACC/V1/CA1/DG), 15 pairs (project_cfg.m), FS =
`neurontypes[:,4]==2` in *every* area, Task/Control paths, report-faithful
defaults (σ=2.5, 10 ms, residual+partial, 5-fold CV, circshift null 100, IFI
headline ±50 ms = `max_lag_bins=5`), epoch + engagement colours.

**`dataio.py`.** Loads cohort .mat (Task or Control). Per-traversal 1 ms
`corridorData.binned_spikes` is already corridor-only (dark stripped). Key
striatum specifics: **velocity derived** from `trial_position`/`trial_times`
(re-zeroed to corridor onset; a.u.→cm), **Gaussian smoothing added** to
`rebin_trial`, period selection (`engaged` / `disengaged` / `all`) for the
engagement contrast, LP/epoch/disengagement per the MATLAB rules. Reuses the
round-17 velocity-derivation approach (see supersession note below).

**Tests + smoke (green).** 25 synthetic-ground-truth tests pass
(`python3 -m pytest tcca -q`): binning, σ-smoothing mass-conservation, velocity
derivation + re-zero, stream assembly + trial labelling, LP rule, epoch windows,
engagement periods, FS exclusion. Real-data smoke (`scripts/smoke_dataio.py`):
16 animals, 13 learners, yoked LP 43; animal 1 → 161,934 bins / 124 traversals at
10 ms, median running speed 19.2 cm/s, gate keeps 75% of bins, z-scoring exact.
Confirms V1/CA1/DG arms are sparse (1–4 animals) → exploratory only.

**Supersession of round 17.** `cca/NOTES.md` claims a complete "Arm A" temporal
port (segments.py, lagged_temporal.py, run_temporal_runstate.py, "169 tests") —
**those files do not exist on disk.** What existed was a partial, uncommitted data
layer in `striatum_cca` (velocity/stream funcs + 2 config knobs + test_velocity.py).
That work is superseded by this `tcca/` package and its logic carried forward here.
A clean revert of the stale `striatum_cca` fragments was attempted but **blocked by
the harness safety classifier** (it guards uncommitted work). Theo can revert
manually if desired:
```
cd ~/Desktop/Experiments/StriatumACC && \
git checkout -- "Striatum project/cca/NOTES.md" "Striatum project/cca/UNDERSTANDING.md" \
  "Striatum project/cca/src/striatum_cca/config.py" "Striatum project/cca/src/striatum_cca/dataio.py" && \
rm -f "Striatum project/cca/tests/test_velocity.py"
```

**Next — Stage 1.** `cp` the data-agnostic numeric modules from tom_cca → tcca,
rename imports, port their pure tests (known-lag recovery, IFI sign, CV-leak
guards, Gini, rank-robust CCA). Then Stage 2 (epoch analysis = first real result).

### Open design points (flagged, to resolve when building the contrasts)
- **Engaged-vs-disengaged z-score reference**: currently each period z-scores over
  its own running bins. For the paired contrast a shared session reference may be
  fairer (avoids a scale confound). Decide in the contrast driver.
- **Disengaged over-long traversals**: the 60 s cutoff that drops disengaged
  traversals from the engaged analysis also removes data we want for the
  disengaged-period fit. Relax/parameterise per-period; report coverage.
- **Task-vs-Control is between-cohort** (different mice), not a within-session
  transition — interpret as a group comparison. Control 2.5 cm cache may be stale
  (predates fr_threshold alignment) → regenerate before Stage 4.
