# striatum_tcca — development log (newest first)

Temporal communication-subspace CCA, faithful port of TomLearning `tom_cca` onto
the striatum dataset. Branch: `claude/temporal-cca-port`. Spec/scope agreed with
Theo 2026-06-17 (full report battery; Task-vs-Control as the two-corridor
contrast; fresh tom_cca-style port; plus an engaged-vs-disengaged contrast).

---

## 2026-09-18 — tom_cca's intercept fix ported; one conclusion dies, the rest hold

Theo flagged that Tom's temporal CCA had been fixed again. `tom_cca` 9e03883
(2026-09-15) added an INTERCEPT to `partial.partial_out_cv`; this port's copy was
byte-identical to the PRE-fix version. Our windows are exactly the trap:
`dataio.zscore_over_reference` scores over the engaged reference and every epoch
window is a slice of it, so both areas' residuals kept the same `(I - P_Z)*1`
image and CCA read it as a shared channel. Call sites: `subspace_window.py:96`,
`early_trials.py:51`, `kcca_window.py:49`.

**Impact, measured like-for-like** (same code both ways, canonical b25/FS-excl/
partial, 107 cells). The pre-fix re-run reproduces the committed CSVs exactly
(1571 dim rows, 106 cross), so the committed grid WAS the buggy one:

| | |
|---|---|
| dim-1 cc1 moves > 0.05 | 52/107 cells (49%) — tom_cca saw 31% |
| all dim rows > 0.05 | 672/1561 (43%); > 0.20 in 12%; max 1.19 |
| n_sig total | 156 -> 82 (-47%); median per cell 1 -> 0 |
| optimal lag changed | 103/1561 rows |
| IFI | Spearman(pre, post) = 0.966 |

### One conclusion is DEAD

**"plain < partial FS-excluded; partialling denoises" (2026-08-11 item 3) was the
bug.** Pre-fix medians -0.012 (b25) / -0.031 (b10), p ~ 0.02 / 0.04. Post-fix:
-0.0006, p = 0.25 (b25) and 0.000, p = 0.81 (b10). The intercept was missing
inside the partialling path, so partial fits were inflated by exactly the channel
the fix removes. There is now no evidence that partialling denoises.

### What holds

1. **Strength null** — still 24 tests. Two nominal hits now (was one), and one of
   them survives BH: b10/fsincl/partial DLS-ACC, dNE -0.033, p = 0.0078. It does
   NOT replicate: the same pair at b25/fsincl/partial is **+0.028** (opposite
   sign, p = 0.38). Same character as the pre-fix non-replicating IFI band. Read
   as null with a config-specific artefact, not as a result.
2. **IFI null** — 0 of 84 BH-surviving (pair x window x config). The Spearman
   0.966 above says why: the directionality arm barely moves under the fix. This
   is the most robust verdict in the subproject.
3. **10 ms uniformly weaker** — b10 - b25 median -0.035 to -0.039, p <= 0.0098 in
   all four frames. 25 ms stays the magnitude reference.
4. **Gini** — area-intrinsic `gini_y` falls naive -> expert (-0.108, p = 0.002,
   animals as n) both PRE and POST fix, while the CCA-independent `gini_pearson`
   control stays flat (x p = 0.92, y p = 0.63). De-sparsification remains a
   property of the weight metric, not of the data. Unchanged by the fix.
5. FS-incl uplift broadens slightly: nominal in 3 of 4 frames now (b25/plain
   p = 0.039, b10/partial p = 0.0195, b10/plain p = 0.0039), was plain-only.

Also killed by the fix: a nominal cc1 decline with learning in the canonical
config (animals as n, p = 0.027 -> p = 0.432).

### Still diverged from tom_cca

Not ported: the connection-specific Gini variants (`gini_x_conn`, `gini_y_conn`,
`gini_x_sig`, `gini_y_sig`) -- tom_cca e099eca's newest result turns on exactly
the area-intrinsic vs connection-specific distinction that item 4 above cannot
currently test; `core.pca_fit_flat` (this port still duplicates the four-line PCA
in `subspace_window` and `early_trials`); `early_trials.variates`' `dims`
argument; and 24 newer tom_cca modules including `perdim_ifi`, `single_trial`,
`population_geometry`, `unit_timescales`, `preprocess`. tom_cca has also collapsed
its 8-config grid to ONE canonical configuration (7e2dd1c).

---

## 2026-08-28 (later) — CORRECTION to the entry below, and the whole grid re-run on corrected cell types

**The entry below claims the `cca_fit` back-port makes "no committed CSV move". That is wrong.**
It was measured on ONE cell (animal 1, DMS-DLS, 3.1e-13) and generalised to 1158. Corrected
statement, measured on the full grid with the cell-type variable held fixed (the FS-**included**
arms, where unit selection cannot change, so only the CCA route differs):

| frame | cells moved | worst cc1 | worst n_sig |
|---|---|---|---|
| **plain CCA** (no partialling) | **0 / 153** | 0 | 0 |
| **partial CCA** (Z = the other areas) | **7 / 152** | 0.40 | 6 |

**The two routes differ only when partialling is on**, because partialling out 13–177 units
destroys rank: the directions the covariance route cuts (relative singular value < 1e-5) and the
SVD route kept (down to 1e-9) turn out to have relative singular values of **1e-9 to 4e-9** —
double-precision debris. Measured on the affected cells: a14 ACC has 21 units and Z = 13, and only
**14** independent directions survive partialling, but the SVD route fitted 17. Its extra canonical
dimensions were noise, which is why its cc1 came out *lower* (0.0964 vs 0.1846) — the leading
direction was polluted. **The covariance route is the correct rank rule here and is kept.**

### The grid was stale for a bigger reason, and is now re-run

The committed 2026-08-11 grid predates the **2026-08-12 cell-type fix** (root `NOTES.md`). At grid
time FS-exclusion was a **no-op in V1, CA1 and DG** (they carried no cell type at all) and ACC was
labelled with the striatal four-way rule, not the FS/RS split. Re-running the same code on the same
`.mat` today:

* FS-**excluded** b25: **122 → 107 cells**; animal 11's DG drops from 6 units to 3 (< `min_units`),
  taking all five DG pairs with it. 83 of the 107 shared cells move, worst cc1 **0.73**
  (a11 CA1-ACC expert), `optimal_lag` moves in 21 cells by up to 13 bins.
* FS-**included**: cell counts unchanged (inclusion uses every unit regardless of label).
* Unit sets changed for animals 5, 9, 10, 11, 13 (V1 67→54, CA1 28→22, ACC 62→61, …).

All 8 configs re-run (b25 ≈ 15 min, b10 ≈ 22 min: 305/428/250/323 s per config), plus
`analyze_epoch_grid.py` and `figs_epoch_grid.py`. Results, logs and summary tables committed.

### Every headline finding of the 2026-08-11 entry survives

| finding | committed | re-run |
|---|---|---|
| 1. strength null | 24 tests, 1 nominal (b25/fsincl/plain DLS-ACC p=0.0391), 0 BH | **identical** |
| 2. IFI null | 0/420 change; 5 exist, all b10/fsincl/plain DLS-ACC ±200–240 ms | **identical** |
| 3. plain < partial, FS-excl | b25 −0.012 p=0.016 / b10 −0.031 p=0.039 | −0.0173 p=0.016 / −0.0267 p=0.039 |
| 4. 10 ms weaker | median ≈ −0.04, p ≤ 0.02 all frames | −0.038…−0.042, p ≤ 0.0039 |
| 5. `gini_pearson` flat | x p=0.73 / y p=0.30 | x p=0.43 / y p=0.25 (still null; medians flat) |
| FS-incl uplift only in the plain frame | 78 % at b10, p=0.004 | 79.7 %, p=0.0039 |

⚠ **Do not read the identical p-values as identical data.** At n = 9–11 animals the two-sided
Wilcoxon p sits on a coarse discrete lattice (0.0039, 0.0156, 0.0391, …), so a p can be unchanged
while its median moves by 40 %. Two p-values did move: `b25/partial fsincl-fsexcl` 0.43 → 0.57 and
`b25/fsincl plain-partial` 0.16 → 0.074. The conclusions are robust; the arithmetic is not a
reproduction.

### New methodological trap (also in root `GOTCHAS.md`)

`k_eff` in `epoch_metrics*.csv` is `min(K, n_units_x, n_units_y)` — the *pre*-partial dimensionality.
Nothing records the rank that survives partialling, so a cell where Z has eaten most of the subspace
is indistinguishable in the CSV from a well-conditioned one. Those are exactly the cells whose cc1
swings by up to 0.4 on the rank rule alone, i.e. cells whose communication estimate was never
trustworthy under *either* implementation. A post-partial rank column would make them visible.

---

## 2026-08-28 — Back-port from TomLearning `tom_cca` (numerics + power); zero result change

The port was frozen on 2026-07-28; `tom_cca` moved 37 commits since. A module-by-module
diff (both packages imported side by side) put 8 of 19 numeric modules byte-identical and
the shared numerics in agreement to ~3e-15. Four things worth having had drifted in; three
are back-ported here, one went the other way.

**1. `core.cca_fit` — covariance route (tom 2026-08-17).** Each population is whitened
through the eigen-decomposition of its own k×k covariance instead of a thin SVD of the n×k
data matrix. The old implementation is kept verbatim as `core._cca_fit_svd`, tests only.
Rank rule documented (`_COV_EIG_FLOOR = 1e-10` relative eigenvalue ≈ 1e-5 relative singular
value). **Measured, not assumed:** on animal 1 / DMS-DLS / all three epochs through
`runner.fit_window` at the committed b25 config, every field of `WindowSubspace` agrees with
the frozen SVD route to **3.1e-13** (cc1, IFI, n_sig, Gini, weights, split-half, lag curve),
and the fit is **2.7× faster** (2.7 s vs 7.2 s for the three cells). 4 equivalence tests
ported from `tom_cca` (rank-deficient, n < p, zero-variance column, weights up to a per-pair
sign flip). No committed CSV moves.

**2. `lagged.py` — now byte-identical to `tom_cca/lagged.py`.** Adds `ifi_sides` (tells the
degenerate IFI = 0 "no coupling either way" apart from a genuinely balanced curve),
`heldout_lag_curve_flat_perdim` (per-dimension held-out lag curves; `heldout_lag_curve_flat`
is now its d=0 slice), and `perdim_significance` / `PerDimSignificance` — the **held-out
per-dim circular-shift null with BH correction**, the like-for-like alternative to the
in-sample dominant-dim null in `subspace_window._significance` (which is unchanged and still
what every committed epoch result used). New module `lagpairs.py` (ported verbatim) is now
the single within-group lag pairer; `lagged._segment_lagged_pairs` delegates to it and
`tests/test_lagpairs.py` pins the delegate against the frozen inline loop for lags −12…+12.

⚠ **`perdim_significance` is not usable at the current shuffle count.** `config.fdr_dims = 10`
is added (inert — no driver reads it) but a permutation p cannot go below 1/(n_shuffles+1),
and BH at 10 dims needs 0.005. At `SURROGATE_SHUFFLES = 100` the floor is 0.0099, so **no
dimension can pass**; a driver adopting this null must raise shuffles to ≥ 200 (floor
0.00498). Left as a flagged decision — changing the shuffle default would move committed
numbers.

**3. `paired_stats.paired_t` / `welch_t`.** The parametric siblings of `wilcoxon_signed` /
Mann-Whitney. At the cohort n (11–13 animals) the exact signed-rank p sits ON its floor
(2/2^10 = 0.00195 with all deltas one-signed); the t-test is unbounded below. 9 new tests,
mirrored into `tom_cca` — both functions had shipped there untested.

**4. Sent the other way (`tom_cca` gained it from here):** `subspace_window.WindowSubspace`
now exports `lags` + `lag_cc1`, so IFI can be recomputed at any integration window offline.
Its test came with it.

**Still divergent (deliberately).** `config.py` / `dataio.py` / `runner.py` vs `pipeline.py`
are the dataset boundary and stay separate. Not taken: `core.pca_fit_flat` & friends (would
be dead code until the three inline copies are rewired), and `membership
.subspace_contribution_connection` + `gini_*_conn` / `gini_*_sig` — note that this project's
"corrected Gini" (`gini_pearson_x/y`, 2026-08-11 entry below) is the CCA-independent Pearson
control, which `tom_cca` also has; its two *connection-specific* corrected definitions have
never been run on this dataset. Parameter defaults still differ: `max_lag_bins` 5 (±50 ms)
here vs 25 (±250 ms) in tom's `TemporalDefaults`, and `n_shuffles` 100 vs 200.

220 tests (was 167).

---

## 2026-08-11 — Epoch grid (8 configs): every verdict robust; partialling is denoising; corrected Gini also flat

Full factorial on the seeded 5 cm cache: bin {25, 10 ms} × FS {excl, incl} ×
{partial, plain}, max_lag extended to ±250 ms, per-cell CC1 lag curves +
`gini_pearson_x/y` exported. Priors registered first (PREDICTIONS.md
2026-08-11; scored same day — P3 reversed, everything else as predicted).
Driver additions: `--no-partial`, `epoch_lagcurves*.csv`; analysis in
`scripts/analyze_epoch_grid.py` (animals-as-n Wilcoxon + BH — the first
scripted tcca statistics, replacing the ad hoc 07-28 arithmetic). 167 tests.

1. **Strength null in all 8 configs** (24 tests, 1 nominal hit, 0 BH).
2. **IFI null at every integration window to ±250 ms** — 0/420 learning-change
   cells; existence null except a non-replicating b10/fsincl/plain DLS-ACC
   band (±200–240 ms, median +0.02) absent FS-excluded and absent under
   partialling. Registered falsifier not triggered.
3. **plain < partial FS-excluded** (median −0.012 b25 / −0.031 b10, p≈0.02/0.04):
   the coupling is NOT shared drive from the other recorded areas; partialling
   denoises (PCA-k crowding). FS-incl uplift exists only in the plain frame
   (78% at b10, p=0.004) → FS units carry mostly shared variance.
4. **10 ms is uniformly weaker** (b10−b25 median ≈ −0.04, p≤0.02, all frames);
   25 ms stays the magnitude reference.
5. **Partner-dependent Gini (gini_pearson) is flat across epochs**
   (x p=0.73 / y p=0.30, b25 committed config) — the de-sparsification null
   survives the metric correction flagged in FIGURE_PLAN_AUDIT.md §6.

A7 runs at LP 33 (seeded baseline; recorded runs used 34) → its epoch windows
shift one trial vs the frozen 07-28 CSVs; cohort counts 122 (b25 fsexcl
partial) vs 125 recorded. All grid CSVs + analysis outputs are tracked
(gitignore negations widened).

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
