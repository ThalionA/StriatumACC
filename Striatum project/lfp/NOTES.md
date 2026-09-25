# striatum_lfp — running log (newest first)

## 2026-09-25 — Soundness review before simplifying: most headline claims are at risk

Agent-written review (four independent reviewers on 2d1fc3f, key items re-measured
by the main session). Nothing was re-run on real data; every number is from the
committed tables or synthetic known-answer scripts. **Treat the claims below as
UNDER REVIEW until the fixed pipeline re-runs.** Branch `lfp-simplify`.

Note: `main` had been left 28 commits behind `meeting-2026-08-28-figures`; it was
fast-forwarded to it (local only) on 2026-09-25.

**Claims at risk**
- *DLS theta task ≠ control* — passes only with the trials 4-10 baseline and no
  DP rule. Dropping the two animals whose Expert window is past DP (418; control
  1205, DP = 18) → p_FDR 0.18-0.40. NOTES 08-28 "also survives the speed control"
  was false even then (speed-resid p_FDR 0.0508, `differs=False`).
- *Decoding beats its null in every striatal/ACC cell (BH-FDR)* — no committed
  code runs that test; `plot_lfp_arms.py:175` prints it anyway.
- *Reliability gap is behavioural* — never tested; control 407 runs stereotypically
  (speed split-half 0.935) yet DMS beta reliability −0.03; task 624/731/822 run at
  control speed and reach 0.53-0.74.
- *No direction survives removing the far field (PSI)* — the "bipolar" pairs are
  same-row (depth = (c//2)*20, so depth-sorted neighbours share a row: 0 µm
  vertical, ~32 µm lateral), 13-15 % sign-flipped by unstable argsort, and
  `area_signals` never notches mains (55-96 % of low-gamma power in 1105).
- *Coupling is distance and nothing else* — theta CI ≈ [−0.21, +0.09] cannot
  exclude an area term of ~0.1; matched separations only probe DMS/DLS.
- *PAC is real, 7 % null vs 71 %* — the 7 % is a hard-coded constant measured
  before DP clipping (`presentations/make_meeting_figures.py:217`).
- *infotheory: band power carries information beyond speed* — the label shuffle
  is global, so shared slow drift reads as information (synthetic: 95-101 %
  "survives speed" from drift alone), and speed is conditioned on a 2-bin split.
- Survives review: PAC does not change with learning (robust to count and
  reference); decoding CV and circular-shift null; LP/epoch ports vs MATLAB;
  filters, overlap-save, notch in the band-power cubes; animal as unit.

**Data-path defects (confirmed)**
1. Control probe-2 clock: V1-bundle `VR_times_synched` is 6-48 ms (513) and
   116-159 ms (817) off probe 1's; task bundles identical. Which side is right is
   untested (needs a 1 ms lag scan of probe-2 LFP vs V1 MUA).
2. `validate_lfp_bandpower.py` compares Python with Python — the "0.0 ms" is circular.
3. Exports are already CMR'd (channel median exactly 0 in 42 % of samples, 822);
   ch191 (probe reference, SD 13.7× median) is binned as tissue in 11 area signals.
4. `truncated_trials` fires on the last trial of every session by rounding
   (compare against `n_lfp − crop_start0`, not the crop).
5. 1212 raw trial 102 is non-good; infotheory pairs 52-53 trials with the NEXT
   trial's behaviour (raw vs good-filtered index).
6. LFP depths 0-3820 µm vs unit depths 20-3840 µm — area boundaries 2 channels off.
7. DP is applied only in coupling and infotheory; arms, group contrast, PSI and
   distance ignore it. MATLAB treats NaN DP as "no clip" (`min([cp, n])`).
8. Three epoch schemes coexist (4-window 3/7/10/10 in arms + PSI; 4-10 vs Expert
   in the contrast; 3-window in coupling/MI) — "one epoch definition everywhere"
   is not yet true.

**Decisions (Theo, 2026-09-25):** delete the July code (git tag); split NOTES by
date; band power on a *vertical* bipolar derivation as primary with the
as-exported signal (ch191 dropped) as sensitivity; group contrast primary test =
Δlog10 + exact permutation, pre-registered before the re-run.

## 2026-09-17 (b) — Direction of communication: nothing survives removing the far field

The third question from 2026-09-09, asked with the phase-slope index because it
is the one measure blind to instantaneous mixing by construction -- and the
distance control that morning had just shown cross-area coupling here IS a shared
field. Two references computed per pair from the raw voltage: **monopolar** (the
area's mean channel) and **bipolar** (non-overlapping adjacent-channel
differences, which cancel the far field to first order -- the re-referencing this
file has been asking for since August).

**Answer: no.** Task cohort, animal as the unit, Wilcoxon against zero, BH-FDR
over the 12-cell family:

| reference | cells surviving BH-FDR | the one survivor |
|---|---|---|
| monopolar | **1 / 12** | DLS→DMS beta, z = +4.35 ± 0.89, N = 12, p = 0.0005 |
| bipolar | **0 / 12** | — every cell within \|z\| ≤ 1.73, all p > 0.09 |

**The decisive number is not either of those — it is that the two references
agree at chance.** Over 268 cells, monopolar and bipolar z correlate at
**r = −0.05** and their signs agree in **46%** (chance 50%). If bipolar were
simply a noisier view of a true direction, the signs would agree well above
chance. They do not, so the monopolar effect is not a stronger version of the
bipolar one; they are unrelated measurements, and the monopolar one is a property
of the far field.

PSI is blind to INSTANTANEOUS mixing, not to a shared source carrying a delay.
A field that propagates or is filtered differently with depth has a real phase
slope, and that is what the monopolar numbers are reading. Bipolar removes it and
leaves nothing.

**Change with learning: not answerable on a non-result.** The epoch panel is drawn
but rests on cells that are null overall.

**Task versus control: not answerable at all.** No control area pair reaches
N = 6 animals, the floor at which a two-sided signed-rank test can return p < 0.05.
The control cohort has five striatal animals split across pairs.

### The statistic nearly went out uncalibrated
The jackknife z divides by a spread over segments that overlap by half, so I
distrusted it and built a mismatched-trial surrogate as the "proper" null. On real
data the two disagreed fivefold, which forced a calibration: 40 synthetic cells
with NO interaction and a strong shared field gave **jackknife sd(z) = 1.12, 10%
false positives** against **surrogate sd(z) = 2.48, 38%**. The jackknife is the
calibrated one; the surrogate is anti-conservative because every trial of a
session shares the same slow field, so a mismatched pair stays coupled. It had
briefly produced mean \|z\| = 14. The surrogate is kept with that measurement in
its docstring and a test pinning it, so it is not re-adopted. Logged in
`~/.claude/MISTAKES.md` (2026-09-17, no-verification).

Two other defects caught before any number was reported: spectral windows were
straddling trial boundaries (segmentation is now per trial), and the bipolar
derivation was `mean(np.diff(...))`, which telescopes exactly to
`(last − first)/(n−1)` — one wide pair across the area rather than local ones.

**Code.** `src/striatum_lfp/psi.py` (+ 17 ground-truth tests, including that pure
instantaneous mixing gives nothing and that a real lag survives heavy mixing),
`scripts/run_lfp_psi.py`, `scripts/plot_lfp_psi.py`, a `psi` step in the pipeline.
269 tests pass.

## 2026-09-17 — The cross-area coupling is distance, and nothing else

**The control asked for on 2026-09-09.** DMS, DLS and ACC sit on one shank, so
"different area" and "further apart" have been the same axis in every cross-area
number this package reports. The test: compare channel pairs the SAME distance
apart, within one area against across an area boundary.

**Answer: at matched separation the boundary makes no difference.**

| cohort | band | within − across | N mice | test |
|---|---|---|---|---|
| Task | theta | −0.057 ± 0.069 | 14 | p = 0.43 |
| Task | beta | +0.005 ± 0.052 | 14 | p = 0.95 |
| Task | low gamma | +0.011 ± 0.052 | 14 | p = 0.90 |
| Task | high gamma | +0.018 ± 0.034 | 14 | p = 0.58 |
| Task | total | −0.012 ± 0.049 | 14 | p = 0.72 |
| Control 1 | all five | −0.073 to +0.051 | 5 | underpowered |

Animal is the unit of analysis, Wilcoxon signed-rank against zero, BH-FDR across
bands. Nothing is significant anywhere. **Control 1 is marked underpowered, not
"n.s.": at n = 5 the smallest attainable two-sided signed-rank p is 0.0625, so
that test cannot reject at 0.05 however large the effect is.** The task cohort at
n = 14 is the one carrying the negative result.

The decay curves say the same thing more directly: coupling falls from r ≈ 0.8 at
adjacent channels to ≈ 0.17 at 1.6 mm, and the within-area and across-area curves
lie on top of each other the whole way.

**This settles the standing caveat.** The held-out CC1 ordering that looked like a
result — CA1–DG 0.93 at 510 µm down to ACC–DLS 0.55 at 2044 µm — is the distance
axis and not an area axis. No cross-area LFP coupling claim should be made from
these data, and the answer does not change with re-referencing: re-referencing
would change the decay constant, not the fact that the two classes share a curve.

**A trap I walked into first, kept in the code and pinned by a test.** Restricting
both classes to a shared separation RANGE is not the same as matching their
separation DISTRIBUTIONS. An area is only a few hundred µm thick, so inside the
shared range within-area pairs sit **216 µm closer** than across-area pairs, and
that imbalance correlates with the contrast at **r = −0.63**. The range-restricted
comparison therefore reported **within − across = +0.073** on data whose exact-matched
answer is **−0.008**. Channel depths lie on an exact 20 µm grid, so the classes can
be matched at IDENTICAL separations rather than merely overlapping ranges;
`distance.exact_matched_contrast` does that and both numbers stay in
`lfp_distance_matched_<cohort>.csv` so the confound is visible rather than
quietly corrected away.

A ground-truth fixture bug in the same area, also fixed: the "distance-only" null
field was built by summing Gaussian-weighted sources over a finite depth range, so
channels near the ends drew on fewer sources, were more correlated with their
neighbours, and produced a spurious within > across of 0.08. The field is now
sampled from an exactly stationary kernel, and the null test passes for the right
reason.

**Code.** `src/striatum_lfp/distance.py` (+ 11 ground-truth tests),
`scripts/run_lfp_distance_control.py`, `scripts/plot_lfp_distance_control.py`,
a `distance` step in `run_lfp_pipeline.sh`. Reads the band-power caches only — no
re-extraction. 252 tests pass.

**Still open from that meeting: direction of communication.** The cached products
are binned to 50 spatial bins per trial (~120–250 ms each), which cannot resolve a
communication lag; a directional measure needs a fresh extraction from the voltage
exports. Scoped for Theo, not started.

## 2026-09-09 (c) — Task non-learners take the cohort average LP; CA1 and DG are whole again

**Requested by Theo.** The two task animals that never reach criterion, **703 and
1206**, were left at `learning_point = None`, so the learning-point-relative
Intermediate and Expert windows did not exist for them. Since 1206 is one of only
three task animals with a probe in CA1 and DG, those two areas fell to n = 2 in half
the epochs and dropped out of the figures. They now inherit the task cohort's average
learning point (**41**) exactly as the yoked controls do — `IntegratedAll_v1.m`'s own
rule, applied to the same kind of animal.

**Coverage, measured before → after** (task animals contributing, per area):

| area | Trials 1-3 | Trials 4-10 | Intermediate | Expert |
|---|---|---|---|---|
| DMS | 16 | 16 | 14 → **16** | 14 → **16** |
| DLS | 12 | 12 | 12 | 12 |
| ACC | 15 | 15 | 14 → **15** | 14 → **15** |
| V1  | 5 | 5 | 4 → **5** | 4 → **5** |
| CA1 | 3 | 3 | 2 → **3** | 2 → **3** |
| DG  | 3 | 3 | 2 → **3** | 2 → **3** |

Identical gains in the reliability and decoding arms. CA1 and DG now draw across all
four epochs in `lfp_evolution_z_task_vs_control` instead of stopping after trials 4-10.

**What it costs, stated plainly.** For 703 and 1206 "Expert" is now a matched TIME
window, not a matched level of performance — the caveat every control already carried.
The arm tables gained an `lp_source` column (`measured` | `cohort_average`) so any
result can say which animals are on a borrowed number, and the combined figure's
caption says it.

**What moved in the results, checked like-for-like against the pre-change tables:**
- **No cell gained or lost significance in the group contrast.** Decoding stays
  9/30, reliability 13/30, moving reliability 0, CCA 0, behaviour 2/3.
- **DLS theta is unchanged as an effect and slightly weaker as a q-value**: task
  −0.1860 vs control +0.0758, p = 0.0011 in both runs; **p_FDR 0.0215 → 0.0323**,
  purely because CA1 and DG cells are now testable so the BH family grew from 56 to
  84 comparisons. It remains the only evolution cell where the groups differ.
- Within-task evolution stats went 7/48 → 13/72 surviving BH-FDR, again because the
  CA1/DG cells entered the family.

**Code.** `analysis.measured_learning_points` is the raw measurement and is what
MATLAB parity is owed on; `analysis.cohort_learning_points` is the analysis-facing map
that fills the gaps; `analysis.task_average_learning_point` averages over LEARNERS
only, so the definition cannot go circular; `analysis.learning_point_sources` labels
each animal. The MATLAB-parity test was repointed to the measurement and a test added
that the fill touches the two non-learners and nothing else. 241 tests pass.

## 2026-09-09 (b) — Why the CA1/DG task line stops after trials 4-10

Not missing trials. Every animal has all of its trials; the **epochs are the
problem, not the data**.

`Intermediate` and `Expert` are defined RELATIVE TO THE LEARNING POINT
(`lp-10 .. lp-1` and `lp .. lp+9`). Two task animals never reach criterion and
have no learning point at all — **703 and 1206** (`lp=None` in the arms log) —
so neither window exists for them and they contribute to `Trials 1-3` and
`Trials 4-10` only. Measured from `lfp_arms_evolution_task.csv`:

| area | animals, Trials 1-3 → Expert | who drops |
|---|---|---|
| DMS | 16 → 14 | 703, 1206 |
| ACC | 15 → 14 | 703 |
| DLS | 12 → 12 | — |
| V1  | 5 → 4 | 1206 |
| CA1 | 3 → **2** | 1206 |
| DG  | 3 → **2** | 1206 |

CA1 and DG have only three task probes (1201, 1206, 1212), and 1206 is one of
them, so those two areas fall to n = 2 — below the three-animal floor the
combined figures require before drawing a mean ± SEM. Hence the line stops.

The figure now prints `task N=3/3/2/2` in each panel and says this in its
caption, so the gap explains itself rather than reading as truncated data.

**The session-aligned figure does not have this problem at all** — trial 1 is
trial 1 for every animal, learner or not, so CA1 and DG are drawn across the
whole 20-trial window with n = 3 throughout. That is a second reason to prefer
the session axis for anything involving the small hippocampal cohort.

## 2026-09-09 — Corrected: the reliability figure Theo asked for is the MOVING one

**I built the wrong figure yesterday.** "Trial-to-trial reliability in the first
20 trials" was read as a static split-half over trials 1-20 (a bar per area x
band). What was wanted is the **evolution of the moving-window reliability** —
the learning-point-aligned figure, re-aligned to the start of the session. The
data for it was already on disk (`lfp_arms_moving_reliability_*.csv`, per trial),
so the "First 20" window added to the arms and the two re-runs it cost were both
unnecessary. `plot_lfp_reliability_first20.py` and its figure are deleted; the
`First 20` rows stay in the reliability table as a summary statistic but nothing
plots them. Logged in `~/.claude/MISTAKES.md` (6th `misread-spec`, and the first
to recur through a project rule that already carries the promoted line).

**New: `scripts/plot_lfp_combined.py`** — task and Control 1 on the same axes,
one figure per result instead of two, all on a shared y-scale. Splitting each
result across two per-cohort files meant the comparison that matters had to be
made by holding two images side by side and trusting their axes matched.

1. `lfp_reliability_moving_session_task_vs_control` — **the requested figure.**
   Moving-window reliability against trial number from session start, first 20
   trials, area x band grid, both cohorts, no shuffle series. The session axis is
   the honest one for a cohort comparison: yoked controls have no learning point
   and inherit the task cohort's average (41), so an aligned control curve is an
   artefact of that borrowed number. Task sits above control in nearly every
   panel from the first trials — with the same speed-stereotypy caveat as every
   other reliability result.
2. `lfp_reliability_moving_lp_task_vs_control` — the same statistic on the
   learning-point axis, kept as the direct counterpart so the two alignments can
   be compared.
3. `lfp_evolution_z_task_vs_control` — band power across the four epochs, both
   cohorts, previously split across two 6x4 grids.

`run_lfp_pipeline.sh` now calls this in its `plots` step; exercised end to end
with `--from plots`.

### Answering a question that came up: the single-unit curve in
### `lfp_reliability_moving_vs_units_*` does not show the naive → expert rise

Both that curve and `integrated_09_stability_allgroups_hierarchical_zscored`
come from the SAME numbers — `figures/stability_by_animal.csv`, written by
`IntegratedAll_v1` from `hier_z` / `hier_z_shuff`. The difference is entirely in
the rendering, and the underlying naive → expert change is small:

| area | observed reliability, Naive → Expert | shuffle | obs − shuffle |
|---|---|---|---|
| DMS | 0.269 → 0.278 (+0.008) | 0.145 → 0.142 | +0.011 |
| DLS | 0.204 → 0.232 (+0.028) | 0.114 → 0.134 | +0.008 |
| ACC | 0.249 → 0.211 (**−0.038**) | 0.099 → 0.115 | −0.055 |
| V1  | 0.161 → 0.164 (+0.003) | 0.095 → 0.073 | +0.025 |
| CA1 | 0.131 → 0.102 (−0.029) | 0.081 → 0.089 | −0.036 |

Two things produce the apparent disagreement. **(a) The MATLAB figure draws each
epoch as its own 10-trial trajectory**, so what reads as "increasing towards
expert" is largely the rise WITHIN the Naive block (DMS climbs ~0.24 → 0.30
across trials 1-10, DLS similarly) — the first trials of a session are the least
reliable. The LFP figure collapses each epoch to one mean, which removes that.
**(b) The LFP figure plots observed MINUS its own trial-shuffled control**, and
in ACC the shuffle level RISES with learning (0.099 → 0.115) while the observed
falls, so the subtraction steepens ACC's decline from −0.038 to −0.055.

Neither figure is wrong, but the epoch-mean of the single-unit statistic is flat
to slightly falling in this cohort, and any claim that single-unit reliability
increases with learning should be made on the within-epoch trajectory, with the
epoch means quoted alongside.

## 2026-09-08 — First-20-trials reliability, one pipeline entry point, project_cfg parity

**New figure: `lfp_reliability_first20_task_vs_control`** (requested). Split-half
reliability of the spatial band-power profile over the **first 20 trials of the
session**, one panel per area, all five bands, task and Control 1 side by side on
a **shared y-scale**, no shuffle series. The window is deliberately NOT
learning-point aligned: controls have no learning point of their own (they inherit
the task cohort's average, 41), so an aligned control window is a matched *time*
window, not a matched level of performance. The first 20 trials are the same
stretch of exposure in both cohorts and need no alignment assumption.

Task exceeds control in every area and nearly every band (DMS theta +0.41 vs
+0.08; DLS beta +0.35 vs -0.03; ACC high gamma +0.42 vs -0.04; V1 and the
hippocampal pair follow the same direction at n = 3-5). **This is the same
behavioural confound as the epoch-aligned reliability arm, seen earlier in the
session:** task animals already run the corridor more stereotypically than
controls (speed-profile split-half r 0.98 vs 0.68), and the LFP spatial profile
largely tracks speed. Read it as a behavioural difference read out through the
LFP until a speed-matched control says otherwise.

**Arms: a new "First 20" window.** `run_lfp_arms.py` gained an unaligned
20-trial window alongside "All" and the four epochs, and the loop's index
convention was unified — windows are 0-based at construction now, instead of
"All" being 0-based and every other window silently decremented inside the body.
**Verified value-preserving:** 1020 shared rows against the committed table,
maximum absolute difference 0.00e+00 on every column, plus 270 new rows. The
group contrast filters on `window == "All"` and is untouched: 2/56, 9/30, 13/30,
0/20, 2/3, 0/30, identical to 09-07.

### Interoperability
- **`scripts/run_lfp_pipeline.sh` is now the single entry point** for the chain
  the project convention requires. It replaces `results/rerun_chain_2026-09-07.sh`,
  a dated one-off retyped from the NOTES recipe — the arrangement that lets a
  step be skipped when recipe and script drift. `--cohort`, `--from`, `--only`,
  `--list`; a failed step aborts; the cross-cohort contrast skips itself rather
  than run against a stale partner table.
- **`tests/test_project_cfg_parity.py`** parses `project_cfg.m` and asserts the
  Python mirrors match it: `AU_TO_CM`, `max_bin`, the four learning-point
  constants, the derived 5 cm grid and bin count, and the area list. The mirrors
  were documented as "follows project_cfg.m" in comments, and a comment cannot
  fail — when the grid was re-cut 2.5 cm → 5 cm on 2026-08-10 a stale mirror
  would have gone on producing plausible numbers on the wrong grid. All 9 pass
  today, so nothing has drifted.
- **`preprocessed_data_control2.mat` now carries `mouseid`** and
  `PreprocessStriatumControl2.m` takes its paths from `project_cfg` (its
  `exist()` check pointed at `Striatum project/...` while load/save used a bare
  filename, so it neither found nor wrote the file every consumer reads).
  Regenerated and verified identical to the previous product on every array,
  plus the new field.

237 tests pass, 18 skipped (228 + the 9 new parity checks).

## 2026-09-07 (b) — Full task cohort re-run: 16/16 striatum, 1212 at full length

The recipe in the previous entry was run end to end (`results/rerun_chain_2026-09-07.sh`,
log alongside; 21 min). Every `*_task.csv`, `lfp_group_contrast.csv` and the task /
task-vs-control figures are regenerated; the per-step `lfp_*_run_task.log` files are
the chain log split by step (the unsuffixed task logs are removed). 228 tests pass.

**Data.** 21 named exports, none skipped, no duplicate fingerprints. Identity:
**18/21 confirmed in all three windows**; 409, 703 and both 1212 probes are 3/3, and
418 joins 823/1105 as MAJORITY(2/3) — at t = 1200 s 822's MUA edges it (3.48× vs
3.24×), the other two windows are 14.7× and 8.3×, so the filename stands. 1212 is now
11.4 M samples on both probes with VR ending at 10,879 s inside the grid — the
truncation is gone and its expert trials are in (154 trials, LP 23). Band power for
409 / 418 / 703 / 1212: 99.7–99.9 % of cells filled; **21/21 files reproduce the MATLAB
bin map to 0.0 ms**. 409's `[TRUNCATED to 200 trials]` is the driver's 200-trial cap,
not a short export (VR ends at 7806 s of 8400). Per-area N is now DMS 16, ACC 15,
DLS 12, V1 5, CA1 3, DG 3.

**What moved, like-for-like (same estimator, same families, 3 more animals + 1212's
expert end):**

1. **DLS theta strengthened.** Task −0.186 vs control +0.076, p = 0.0011,
   **p_FDR = 0.022** (was 0.049 on n = 10); speed-residualised p_FDR = 0.019 (was
   0.051). It is still the only evolution cell where the groups differ — the second
   "cell" in the contrast log is the same cell's speed-residualised metric. The
   provisional flag on this result is lifted.
2. **Decoding now differs task > control in 9/30 cells** (ACC θ/γL/γH/total, DMS
   γL/γH/total, DLS γL, DG γH; q 0.02–0.04). This is a threshold crossing, not a new
   effect: the same cells had p_raw 0.003–0.03 and q 0.052–0.064 on the 08-28 tables,
   and the effect sizes are unchanged to the third decimal (e.g. ACC θ +0.062 vs
   +0.012 both runs). The 08-28 sentence "position decoding does not depend on
   reward" is therefore withdrawn; but the behavioural caveat that governs the
   reliability arm applies here just as much — task animals' speed profile is more
   stereotyped (0.98 vs 0.68) and the LFP profile tracks speed, so a decoding
   advantage is expected from behaviour alone. Control n is 4–5. Not a neural claim.
3. **Reliability 13/30, moving reliability 0/20, CCA 0/30, behaviour 2/3** — unchanged
   in count and reading. Behaviour: speed-profile split-half r 0.98 vs 0.68
   (p = 0.013), mean speed 36 vs 21 cm/s (p = 0.0007).
4. Within-task evolution stats: 7/48 cells survive BH-FDR (was 2/48) — DLS theta
   (both metrics) plus the gamma `frac_of_total` rises in DMS/DLS/ACC. The gamma rise
   is present and larger in yoked controls (08-28 entry), so it stays "time in
   apparatus", not learning.

**Figure hygiene.** `plot_lfp_task_vs_control.py` had the 08-28 numbers typed into
its panel titles ("no decoding cell differs", "p_FDR = 0.049"); the first re-plot
carried them over unchanged on top of the new bars. Titles are now computed from the
contrast table (counts, cell lists, p_FDR, behaviour numbers). Rule: never type a
result into a title.

**Still open.** 1212's expert end is in every table but its two "MAJORITY" peers
(823, 1105) and 418 are worth an eyeball in the identity figure. CA1/DG remain n = 3.


## 2026-09-07 — STALE RESULTS WARNING: the data moved under the committed tables

Housekeeping session (consolidate branches onto `main`). Checking every export on
disk against `lfp_inventory_*.csv` turned up two changes since 2026-08-28, and
**both invalidate part of the committed result set**. Nothing was re-run here.

**1. 1212 was re-exported at full length** (striatum 2026-08-30 21:05, visual
2026-08-31 11:01). It is now 11,400,000 samples on both probes and
grid-compatible — the 41 min truncation that removed its expert end is gone.
Every cached 1212 product predates this: `lfp_band_trials_task/1212_*.npz` were
built from the 8.4 M export, so 1212's rows in the evolution, decoding,
reliability, moving-reliability and CCA tables are all from the short session.
`tests/test_alignment.py` asserted the truncation and now asserts the full-length
export instead; the `truncated_trials` guard is kept as a regression test because
the defect was silent (index clipping made a short export look like a session
ending exactly on its last sample).

**2. The three missing task animals arrived**: 409 (08-28 13:22), 418 (08-28
14:24), 703 (08-28 17:45), all striatum, all 8,400,000 samples. **The task
striatum cohort is now complete at 16/16.** None of them is in any committed
table, so every per-area N in the 2026-08-27/28 entries is short by up to three
animals — DMS 13 → 16, ACC 12 → 15, DLS 10 → 12 (703 has DMS only; 409/418 have
all three).

**What to re-run, in order** (roughly 30 min on this machine):

```
python scripts/run_lfp_inventory.py  --jobs 5 --cohort task
python scripts/run_lfp_identity.py   --jobs 5 --cohort task
python scripts/run_lfp_bandpower.py  --jobs 5 --cohort task --only 409,418,703,1212
python scripts/validate_lfp_bandpower.py --cohort task
python scripts/run_lfp_arms.py       --jobs 6 --cohort task
python scripts/run_lfp_group_contrast.py
python scripts/plot_lfp_arms.py --cohort task && python scripts/plot_lfp_task_vs_control.py
```

Control is unaffected — all 8 control exports are byte-for-byte as inventoried.

**Which conclusions are at risk.** The group contrast is the exposed one: DLS
theta was the single cell distinguishing task from yoked controls at p_FDR =
0.049, on n = 10 task animals. Adding 409/418 to DLS and restoring 1212's expert
trials will move it either way, and 0.049 has no margin. Treat that result as
provisional until the re-run. The decoding result (13/24 cells, large margins)
and the behavioural finding (speed-profile reliability 0.98 vs 0.68) are unlikely
to turn on three animals, but they are not confirmed either.

**Repo housekeeping.** `lfp-cohort-exploration` and `tcca-tom-sync` were both
strict ancestors of the tip, so `main` was fast-forwarded onto it and both were
deleted, local and remote. The two MATLAB regeneration chains
(`processed_data/regen_chain*.sh`) are now tracked. One pre-existing stash from
the deleted `cca-python-pipeline` branch was left alone — it is the only thing
keeping commits 8e05ec0 ("Stage 1: residual CCA pipeline") and 738dc3a alive.


## 2026-08-28 (b) — Control LFP arrives; the cohort abstraction; what survives a yoked control

**The task-only caveat is retired.** `RawDataControl/LFP/` holds 9 exports; 8 are
usable — 5 striatum probes (407, 513, 515, 817, 1205) and 3 visual (513, 515,
817). **408 has an export but is not in `OrganiseStriatumDataControlIncV1.m:20`**,
exactly as 507 is absent on the task side, and `parse_lfp_filename`'s mouse-list
guard drops it without being asked. Control 2 is dark-only and ships no voltage.

**Cohorts are now a first-class object** (`config.Cohort`), not a fork of the
scripts. Every driver takes `--cohort task|control` and every output is suffixed.
What actually differs, and is now encoded once: the control probe-2 spike bundle
is lowercase (`513_v1_raw.mat` vs the task's `1105_V1_raw.mat`); depth boundaries
live in separate CSVs whose mouse-number ranges overlap the task ones; and
**yoked controls have no learning point**, so `IntegratedAll_v1`'s rule applies —
every control animal inherits the task cohort's average LP (41), making control
"epochs" matched time windows rather than learning windows.

**Control data quality.** All 8 files are (8,400,000 × 384), 1/f slopes −1.16 to
−2.05, adjacent r 0.78–0.99, distant r ≈ 0, mains mostly 1.1–3.8× (407 is 29.6×).
Gain sits in the task August batch's regime. **8/8 reproduce the MATLAB bin map
to 0.0 ms.** Identity: 7/8 confirmed in all three windows; **817/striatum is
2/3 and weak (0.99×, 1.29×, 2.93×)** — its raw bundle has only 48 sorted units,
so the MUA reference is sparse and the test is underpowered there, not the file
suspect. Grid mismatches are benign except one: the three visual probes have
11.4 M spike bins against 8.4 M of LFP, but the organiser already crops probe 2
to probe 1's window and all their behaviour fits inside the export (validation
recovers the same trial counts as probe 1). **407 is the real loss** — 10.08 M
bins, VR to 9972 s, so the export stops 26 min early and 169 of 209 trials survive.

### What the control group changes

**1. The gamma rise is not learning.** Naive→expert gamma increases in DMS and
DLS are present in yoked controls too and are *larger* there (DMS low gamma
+0.236 control vs +0.085 task; DLS +0.192 vs +0.058). Four of the task's
"near-miss" cells are in this category. Time in the apparatus, not task learning.

**2. DLS theta survives, and it is the only thing that does.** A proper group ×
epoch contrast (Welch on the per-animal Δ, BH-FDR within arm,
`run_lfp_group_contrast.py`) finds **1 of 56 evolution cells differing between
groups: DLS theta, task −0.157 vs control +0.076, p = 0.0024, p_FDR = 0.049.**
It also survives the speed control. That is now a dissociation, not a lone
significant cell.

**3. Position decoding does not depend on reward.** 0 of 30 decoding cells differ.
Control 1 runs the same corridor, and the LFP's (small) position information is
there just the same.

**4. The reliability gap is behavioural, and I nearly reported it as neural.**
13 of 30 split-half cells differ, task ≫ control everywhere in striatum and ACC
(e.g. DMS beta 0.60 vs 0.02). Two checks before believing it:
- **Trial count does not explain it.** Matching both groups to exactly 100 trials
  leaves 11/24 cells differing with near-identical effect sizes.
- **Behaviour does.** The split-half reliability of the **speed profile itself**
  is 0.98 in task animals and 0.68 in controls (p = 0.013), and controls run at
  21 vs 36 cm/s (p = 0.005). Task animals traverse the corridor the same way every
  trial; controls do not. Since the spatial LFP profile largely tracks speed
  (population-profile r ≈ −0.9 for beta), this is a behavioural difference read
  out through the LFP, **not** a neural one. V1 is the one area with no group
  difference, and V1 theta is also the one profile uncorrelated with speed
  (r ≈ −0.01 against +0.65/−0.58 elsewhere) — consistent with a position-locked
  visual drive, though not established.

`behaviour` is now a first-class arm (`lfp_arms_behaviour_<cohort>.csv`:
speed-profile split-half r, mean speed, within-bin CV) so this control is run
every time rather than remembered.

**5. Cross-area CCA and moving reliability: 0 cells differ.** Consistent with the
shared-field reading — volume conduction should not care which group an animal
is in.

### Correction to the 2026-08-27 entry
That entry said beta's spatial profile is largely a speed profile "and theta's is
not". That was the **per-channel median** r (theta: DMS +0.09, DLS +0.15, ACC
−0.33). At the **population-profile** level theta is strongly speed-related too,
with opposite signs by area: DMS +0.65, DLS +0.63, ACC −0.58, V1 −0.01. The gap
between the two says the speed-locked component is shared across an area's
channels while each channel adds enough idiosyncratic variance to dilute its own
correlation.

### Corridor structure (measured, task cohort)
Speed minimum 12.8 cm/s at 132 cm — inside the 125–169 cm reward zone — and
maximum 45 cm/s at 208 cm. **Beta peaks at 128 cm in DMS, DLS and ACC together**,
at the reward-zone onset. For a "beta codes position" claim the −0.9 speed
correlation is fatal; for basal-ganglia beta, high beta exactly where the animal
stops to collect reward is the expected movement-suppression signal. Same
measurement, two readings, decided by the question.

### Code
`config.Cohort` + `TASK`/`CONTROL` + `get_cohort`; `results_io.py` (shared table
loader and hierarchical aggregator, hoisted out of both plotting drivers);
`run_lfp_group_contrast.py`; `plot_lfp_task_vs_control.py`. `cohort`, `geometry`
and `analysis` all take a cohort. Results renamed to `<name>_<cohort>` throughout
and band-power caches to `lfp_band_trials_<cohort>/`. 228 tests pass.

### Standing caveats (updated)
Control n is 5 striatal / 3–4 elsewhere, so "no group difference" is weak evidence
of absence everywhere except the reliability arm, where the effect is large.
1212 (task) and 407 (control) are truncated. CA1/DG are n = 3 per group.
Absolute power is never compared across animals.


## 2026-08-28 — 1201_v1 added; moving-window reliability ported from the unit code; dedup pass

**Cohort is now 18 files.** `1201_v1_voltage_data_384ch.mat` landed and is in:
13 striatum probes + **5 visual** (1105, 1106, 1201, 1206, 1212). Only 409, 418
and 703 are still missing, all probe 1. 1201/visual is grid-compatible, clean
(adjacent r 0.97, distant −0.15, 1/f slope −1.34) with a moderate 50 Hz excess
(67×, notched like everything else), and it contributes **V1 102 / CA1 60 / DG 42**
channels — which lifts CA1 and DG from n = 2 to n = 3. Identity: CONFIRMED in all
three windows. Whole cohort re-validated: **18/18 reproduce the MATLAB bin map to
0.0 ms** on median, p99 and max.

**Moving-window reliability — the project's own metric, not a new one.**
`arms.batch_triu_corr_mean` is a faithful port of `batch_triu_corr_mean.m`
(z-score each trial profile across bins, SD 0 → 1, NaN → 0, `Z'Z/(bins-1)`, mean
of the strict upper triangle, all-NaN cell → NaN), checked against a naive
pairwise loop. `arms.moving_window_reliability` applies it on
`max(1, t-2):min(n, t+2)` — the 5-trial centred, edge-clipped window from
`IntegratedAll_v1.m:565-630` — with the same `randperm` trial-shuffle control.
One deliberate deviation: MATLAB substitutes 0 for a missing bin before
z-scoring because 0 Hz is a real firing rate; log power has no zero, so the NaN
is left to reach the z-score step where it becomes that trial's own mean. It
affects 0.2–3% of cells.

Rolled up over the **three-window** epoch convention the neural analyses use
(not the four-window corridor-vs-dark one) into
`results/lfp_arms_moving_reliability_epochs.csv`, whose columns mirror
`figures/stability_by_animal.csv` so the two tables are directly comparable.

**The like-for-like result.** Same statistic, same window, same epochs, same
animals, each minus its own trial-shuffled control:

| area | single units | LFP theta | LFP low gamma |
|---|---|---|---|
| DMS | 0.125 / 0.152 / 0.135 | 0.007 / 0.030 / 0.025 | 0.018 / 0.010 / 0.007 |
| DLS | 0.090 / 0.082 / 0.098 | 0.003 / 0.021 / 0.035 | 0.006 / 0.017 / 0.016 |
| ACC | 0.151 / 0.128 / 0.096 | 0.018 / 0.019 / 0.015 | 0.014 / 0.018 / 0.009 |
| V1  | 0.066 / 0.090 / 0.091 | 0.030 / 0.056 / 0.026 | 0.020 / 0.042 / 0.023 |
| CA1 | 0.050 / 0.051 / 0.013 | 0.034 / 0.085 / 0.070 | 0.045 / 0.029 / 0.026 |

(Naive / Intermediate / Expert.) **In striatum and ACC the LFP's single-trial
spatial structure is 4–10× weaker than the spiking recorded on the same probe**;
in V1 the gap narrows to ~1.5–2×, and in CA1 theta LFP exceeds the units — though
CA1 is n = 2–3 and carries no claim. This reconciles the two reliability numbers
that looked contradictory: the split-half figure (0.48–0.92) averages ~100 trials
per half, so it measures the reproducibility of the *mean* profile; the moving
metric measures whether any *single* trial resembles its neighbours, and there
the LFP is weak. It also fits the small decoding effect (~1–2.5 cm of a 62 cm
chance error).

CA1 and DG are not drawn on the LP-aligned trace: only 2 of their animals have a
learning point, below the 3-animal floor for a mean ± SEM.

**Dedup pass** (no behaviour change intended, and none observed — the identity
verdicts are byte-for-byte the same 16/18 CONFIRMED, only third-decimal shifts):
- `figstyle.py` now owns `save_pair`, `AREA_COLOUR`, `AREA_ORDER`, `BAND_LABEL`,
  `PLOT_BANDS`, `MAX_PNG_PX`; both plotting drivers had their own copies.
- `analysis.log_power` and `analysis.read_behaviour` hoisted out of the drivers;
  `run_lfp_inventory.behaviour_bounds` and the depth-heatmap panel now call them
  instead of re-implementing.
- `bandpower.band_power_series` uses the existing `features.design_band_sos`.
- `inventory.coupling_envelope` now composes `band_power_series` + `bin_mean`
  instead of repeating filter → square → smooth → root; the separate smoothing
  pass was redundant before binning over the same width.
- Deleted `arms.mean_pairwise_trial_r` (duplicated `batch_triu_corr_mean`, which
  is the project's own version of the same statistic) and the unused
  `zscore_channels`.

208 tests pass.

**New figures.** `lfp_reliability_moving` (LP-aligned trace, observed vs
shuffled), `lfp_reliability_moving_session` (absolute trial, keeps the two
non-learners), `lfp_reliability_moving_depth` (channel × trial image, the LFP
analogue of `ProcessStriatumTask.m:997`'s neurons × trials panel),
`lfp_reliability_moving_vs_units` (the table above).


## 2026-08-27 (b) — Band power as the firing-rate analogue: extraction, then four arms

**The product.** `results/lfp_band_trials/<mouse>_<probe>.npz`, one per export:
mean power in 5 bands (theta 4–8, beta 15–30, low gamma 30–80, high gamma
80–150, total 1–150) per channel, per 5 cm corridor bin and per 100 ms dark bin,
per trial, capped at 200 trials (~136 MB each, 2 GB total). 50 Hz/100/150 notched
unconditionally on every file. Extraction is one streaming pass, ~4 min/file.

**The bin map is MATLAB's, verified cell by cell.** `validate_lfp_bandpower.py`
compares every (trial, bin) span against `spatial_binned_data.durations`:
**17/17 files, median / p99 / max absolute difference all 0.0 ms**, no bin present
in one pipeline and missing from the other. Two wrinkles found on the way, both
now reproduced deliberately: (1) MATLAB's `durations` field is the *unclipped* VR
span while the spike sum it feeds uses npx indices clipped to the trial length, so
each trial's last bin is genuinely shorter in the data than in `durations`;
(2) `npx_index` clips to the recording, so 1212's one trial that runs off the end
of its truncated export looked like a trial finishing exactly at the last sample —
`bandpower.truncated_trials` catches it from the unclipped VR time instead, and
1212 now yields 107 complete trials rather than 107 + a half.

**One deliberate difference from the unit pipeline:** band power is the plain mean
over a bin's samples, with no occupancy denominator. That avoids the `(k-1)*dt`
speed bias the FR corridor arm carries — but it makes the two a **different
estimand**, so an LFP corridor panel must never be captioned "the same analysis".

**Learning points ported and checked against MATLAB for all 16 animals**
(`analysis.learning_point` vs the values CorridorVsDarkActivity logged): exact match.

### Results

**1. Evolution.** Consistent spectral tilt across learning — theta down, gamma up —
in DMS, DLS and directionally ACC. Declared family = area × band, animals as n,
paired trials 4–10 → Expert, BH-FDR q=0.05: **2/48 cells survive, both DLS theta**
(raw Δ = −0.157, p_FDR = 0.020; speed-residualised Δ = −0.145, p_FDR = 0.029).
Everything else is a near-miss (DMS theta/low/high gamma p = 0.02–0.04 raw).
**Running speed rises 24.9 → 33.4 cm/s (+34%) over the same epochs**, so every
effect is reported twice, raw and after removing the linear log-speed component
per channel. The one surviving effect survives the control; that is the whole
point of running it.

**2. Spatial decoding — reliable but small, and the first null was broken.**
The original null permuted trials, which does nothing: every trial carries the
same 0–49 bin sequence, so `y` came back bit-identical and the "null" silently
re-ran the real decoder. Replaced with a within-trial circular rotation of the
position labels (`arms.circular_shift_targets`). With the correct null, position
IS decodable: **12/12 striatal and ACC area × band cells survive BH-FDR**
(R² − null = +0.010 to +0.062). V1 has the largest effect (+0.10) but n = 4 and
does not reach significance. Practically the effect is slight: median error
59.5–61.9 cm against a 62 cm chance level, i.e. ~1–2.5 cm on a 250 cm corridor.
Per-epoch decoding is **not estimable** — 10 trials give the ridge ~500 samples
for 30–140 channels and every animal returns a negative R².

**3. Reliability — high, with a large qualifier.** Split-half r (interleaved
halves, Spearman-Brown) of the spatial profile: theta 0.72–0.92, high gamma
0.65–0.79, beta 0.50–0.85, low gamma 0.48–0.75; V1/CA1/DG above striatum.
But the profile's correlation with the **speed** profile is r ≈ −0.40 to −0.59
for beta in *every* area, and −0.15 to −0.37 for the gammas, against −0.33 to
+0.23 for theta. **Beta's spatial profile is substantially a speed profile;
theta's is not.**

**4. Cross-area CCA — not separable from a shared field with this design.**
Held-out CC1 is far above the trial-permutation null (0.35–0.95 vs 0.03–0.13)
and far below the within-area split-half ceiling (0.97–0.999). It falls with
electrode separation along the shank (Spearman ρ = −0.24 to −0.40; p = 0.012 for
theta, 0.12 for high gamma, n.s. for beta and low gamma), and the adjacent pairs
score highest (CA1–DG 0.90–0.95, DLS–DMS 0.47–0.73) versus the distant ones
(ACC–DMS/DLS 0.36–0.54). That is the shape of a volume-conducted field, but the
scatter is large enough that distance alone does not explain it. **No cross-area
LFP coupling claim should be made until this is re-derived under bipolar or CSD
referencing.**

### Code
New: `bandpower.py` (trial/spatial/dark bin geometry, notches, band power,
streaming segment accumulator), `analysis.py` (learning point, epoch windows,
bin speed, joint z-score — all ports checked against MATLAB), `arms.py`
(design matrices, split-half + pairwise reliability, trial-grouped held-out CCA,
trial-permutation null, volume-conduction ceiling, circular-shift decoding null,
speed residualisation, BH-FDR verified against statsmodels).
Drivers: `run_lfp_bandpower.py`, `validate_lfp_bandpower.py`, `run_lfp_arms.py`,
`plot_lfp_arms.py`. 190 tests pass (15 skipped: the July cached-result checks, whose CSVs are gitignored).

### Figures
`lfp_evolution_z`, `lfp_evolution_speed_residual`, `lfp_evolution_fraction`,
`lfp_evolution_speed`, `lfp_decoding`, `lfp_reliability`, `lfp_cca`,
`lfp_cca_vs_distance` (svg + png).

### Standing caveats
Task-only (no control-group LFP exists). 1212 excluded from nothing here but
carries a 41 min truncation. CA1 and DG are n = 2 — plotted, never claimed.
Absolute power is never compared across animals (two gain regimes, ~30×).


## 2026-08-27 — Full-cohort download: 17 files inventoried, every filename verified

The LFP set is no longer four size-keyed files. `RawData/LFP/` now holds **17
named exports** — 13 striatum probes (523, 614, 624, 727, 730, 731, 822, 823,
1105, 1106, 1201, 1206, 1212) and **4 visual probes** (1105, 1106, 1206, 1212,
as `<mouse>_v1_voltage_data_384ch.mat`). Missing: 409, 418, 703 (probe 1) and
1201 (probe 2); one further download was in flight during this run. `lfp_mapping.txt`
is dead — the map now comes from the filename (`cohort.parse_lfp_filename`;
note 727's file has **no** underscore before `voltage`).

**Every file is the animal its name claims — verified, not assumed.**
`scripts/run_lfp_identity.py` correlates each file's 30–90 Hz envelope against
*every* task animal's MUA over three independent 10 min windows. Raw |r| is not
comparable across candidates, so each candidate column is divided by its median
over all files (`cohort.column_normalise`): the question becomes "does this file
couple to that animal more than the other files do". 15/17 win their own row in
all three windows; 823 and 1105/striatum win 2/3, losing only the t = 4800 s
window where a synchronous artefact makes several animals' MUA correlate with
everything — in both cases the claimed animal still scores 2.5–4.9× its column
median, so this is a contaminated window, not a mislabelling. No duplicate
content fingerprints: **the 614/731 duplicate is gone.**

**1212 is truncated, not wrong.** Both its probes export 8,400,000 samples while
`binned_spikes` runs 11,400,000 (VR to 10,879 s), so ~41 min of behaviour — the
late, expert end — has no LFP. The offset scan settles what kind of defect it is:
coupling to its own MUA is 0.263 at offset 0 and 0.011–0.018 at +600/+1500/+3000 s,
so the export is the **truncated head of the same session**, sample-for-sample
aligned from t = 0. Usable for the first 140 min; excluded from any trial-indexed
or learning analysis until the missing tail is re-exported.

**Two export batches with different gain.** Median channel RMS is
6.3e-6–1.3e-5 for the eight July files and 1.6e-4–3.2e-4 for the nine August
ones — ~30× in amplitude, ~1000× in power. **Absolute power is not comparable
across animals.** The batches differ in two more ways: terminal zero padding is
present in all eight July files (starting 7863–8135 s, 4.4–7.5% exact zeros) and
**absent in all nine August files** (0.5–1.2% zeros); and `lfp/NOTES.md`'s
"padding runs from ~130–135 min to the 140 min end" is a July statement, not a
cohort one.

**Mains is the big new problem, and it is per-session.** 50 Hz power over its
±5 Hz shoulders: 1105 visual **1348×**, 1105 striatum **1097×**, 1106 visual
**450×**, 1212 striatum **161×** — then 7 files at 3–17× and 6 files below 3×.
1105 additionally carries odd harmonics (150 Hz 27×, 250 Hz 9–12×, 350 Hz 5–8×),
the signature of a clipped rather than sinusoidal mains pickup. **Notch 50/100/150 Hz
unconditionally on every file**: conditional notching would make the estimator a
function of the mouse. The July 75/151 Hz instrument peaks are gone everywhere
(ratios 0.94–1.06) except where they coincide with a mains harmonic.

**Everything else passes.** All 17: (8,400,000 × 384) float32, gzip chunks
(42, 384), zero non-finite values, zero dead channels, `channels_to_save` = 1..384,
and `depth_to_save` reproducing `geometry.channel_depths` **exactly** (max error
0.0 µm) — the 2-channels-per-20 µm assumption is now measured, not inherited.
1/f slope −0.78 to −2.89 and LF/HF 55–21,343, all far from the scrambled June
signature (0.6–1.5); adjacent-channel r 0.52–0.96 with distant r ≈ 0, i.e. the
layout is right. Flag not exclude: 823 adjacent r = 0.52 (all others 0.83–0.96),
822 LF/HF = 21,343 with slope −2.89 and a common-**mean** residual of 2.42.

**Referencing does not generalise.** Common-median residual is 0.030–0.084
everywhere (consistent with common-median referencing), but the common-**mean**
residual runs 0.12–0.18 in the July files and up to 1.18–2.42 in 822/1105/1106·v1.
Treat referencing as a per-session covariate, not a cohort property.

**Area coverage is ragged and decides the real n.** DMS 13 animals (32–140 ch),
ACC 12, DLS 10 (blank CSV cell for 731, 823, 1206), V1 4, **CA1 2, DG 2**. A CA1
or DG LFP claim is not available from this cohort.

**Code.** New: `cohort.py` (filename→(mouse, probe) discovery, binning, per-column
Pearson, coupling score, column normalisation), `inventory.py` (structure,
integrity, spectra, fingerprint, coupling envelope), `scripts/run_lfp_inventory.py`,
`scripts/run_lfp_identity.py`, `scripts/plot_lfp_inventory.py`. `config.py`'s
`FILE_BY_MOUSE`/`LFP_MICE` now resolve lazily from the directory (old drivers keep
working); `RAW_MAT` covers all 16 task mice and gains a probe-2 twin. Fixed:
`geometry.channel_area_masks` applied no precedence, so 1206's probe-2 channel at
1160 µm was labelled **both** CA1 and DG — MATLAB assigns in CSV column order and
lets the last write win (DG), and the Python now matches. Two stale tests
corrected: 731's DMS band (0–300 → 500–800, per the 2026-08-10 CSV fix) and the
1212 length assertion (11.4 M → the measured 8.4 M truncation). 110 tests pass.

**Artefacts.** `results/lfp_inventory.{csv,json}`, `results/lfp_psd.npz`,
`results/lfp_identity_matrix.csv`, `results/lfp_identity.json`;
`figures/lfp_cohort_overview`, `lfp_spectra`, `lfp_depth_by_frequency`,
`lfp_session_integrity`, `lfp_identity_matrix` (svg + png).

**Next**: band power as the analogue of firing rate — see the Stage B/C plan.
Not started; nothing in this entry position-bins, decodes, or couples areas.


## 2026-08-11 — Zihao's re-export: the gate is CLEARED (with two data gaps)

Zihao re-exported the LFP (CAR only, no filter; the previous files were scrambled
by the save step). New files live on the lab share, **not** in this repo:
`/Volumes/INCR-RochefortLab/Striatum_ACC/Archive/All mice_task_1ms_binned/<mouse>/voltage_data_384ch.mat`.
**22 task-mouse files, regenerated 2026-07-29 → 08-07.** Read-only verification from
disk (h5py over the mount); no files copied.

**New format.** `data_to_save` (8,400,000 × 384) float32, gzip-chunked (42,384);
`channels_to_save` 1–384; **`depth_to_save` 0–3820 µm** (2 channels per 20 µm — NP 1.0
geometry, and the same depth convention as `goodcluster2`, so the existing depth-band
CSVs assign LFP channels to DMS/DLS/ACC directly).

**The alignment blocker is solved.** Every mouse's LFP has *exactly* the same sample
count as its `binned_spikes` (8,400,000 = 140 min at 1 kHz), i.e. the LFP is on the
project's 1 ms grid. Verified physiologically, not just by shape: 727's 30–90 Hz
envelope × MUA cross-correlation **peaks at lag 0 ms** (r=0.061, falling to 0.031 by
±50 ms). LFP sample *i* ↔ `binned_spikes` sample *i* ↔ `VR_times_synched`.

**Before → after** (mid-session block, per-channel z, fs=1 kHz):

| | old (local Jun files) | new (lab share) |
|---|---|---|
| adjacent-depth r | 0.985–0.989 | 0.83–0.96 |
| distant r (+100 ch) | **+0.42 to +0.46** | **0.00 to +0.09** |
| LF/HF (1–10 vs 100–200 Hz) | **0.6–1.5** (broadband) | **97–1400** (1/f LFP) |
| 75 / 151 Hz line ratio | contaminated (audit 2026-07-12) | **1.0 / 0.9–1.1 — gone** |
| 60 s periodic events | exact 60 s cadence | none in 727/730 (peak/median 1.5–2.1) |

So the three July blockers — scrambled layout, broadband dominance, and the ~75/151 Hz
contamination that invalidated the 30–80 Hz band — are all resolved. **Low gamma is
usable again.** Mild 50 Hz mains in some sessions (2.2–2.6× neighbours in 727/730,
~1.0 in 523/731) — notch or avoid.

**Two data gaps — do not analyse until resolved:**

1. **614 and 731 are byte-identical.** Same size (5,142,176,092), same mtime
   (Jul 29 16:46:44), same SHA1 on every slice tested. Identity resolved by
   LFP↔spike coupling: the shared file correlates with **731's** spiking
   (mean|r| = 0.322 across 384 ch) and **not** with 614's (0.008). The test is
   validated — matched pairs always win (727×727 0.032, 523×523 0.033, 730×730 0.010
   vs off-diagonal 0.004–0.008), and 731's rate matches *only* this file. **Conclusion:
   the file is 731's; 614 has no genuine LFP and must be re-exported.**
2. **1212 was not regenerated** — still the June file (11,400,000 samples, std ≈ 1.64,
   LF/HF 1.3, peaks at ~417 Hz). Consistent with the standing rule that 1212 is
   qualitatively different and stays separate.
3. **No control-group LFP exists** (`All_mice_control1/control2_*` contain no
   `voltage_*` files) — so any LFP claim is task-only.

**Open questions for Zihao** (none blocking the fix above): units (values are ~3–5e-6,
i.e. µV-scale if volts — he flags default gain, fine for correlation/coherence, matters
for absolute power); referencing (the across-channel **mean** is not zeroed —
residual 0.12–0.20 of a channel SD — while the **median** residual is 0.05–0.07, which
looks like common-*median* referencing rather than CAR); and whether "no filter" still
implies an anti-alias filter in the 30 kHz → 1 kHz decimation.

**Housekeeping.** Terminal zero padding runs from ~130–135 min to the 140 min end —
mask it. The four June files in `RawData/LFP/` are superseded; `lfp_mapping.txt`
(1212/614/727/731) no longer describes the current set.

## 2026-07-13 — Validation hardening and figure/code reconciliation

- Corrected the state histograms to exclude periodic high-amplitude bins and
  replaced an unexplained half-scaled temporal metric with its raw value plus
  explicit white-noise expectations.
- Added robust 1 ms event-peak timing to the reproducible timing CSV. Peak↔sync
  medians are +1.983/+4.256/+1.740 s for 614/727/731; 1212 has a ~30 s IQR and
  does not provide an alignment marker.
- Recomputed common-median-reference sensitivity from all 384 channels. It
  reduces 614's 153.8 Hz peak by 4.54 dB, invalidating the earlier blanket
  <0.2 dB claim; the ~74 Hz peak and both peaks in 727/731 remain within 0.06 dB.
- Withheld area labels for 1212, whose voltage-probe identity is unverified.
  For other mice, area panels are explicitly nominal depth-band diagnostics.
- Fixed filtered-traversal indexing in the quarantined learning driver, removed
  low gamma and 1212 from diagnostic reruns, and moved old outputs under
  `_quarantined_unaligned_learning/`.
- Full-scan processing no longer regenerates rejected v1 figures. Exact figure
  source arrays are saved beside summaries. Validation: 57 pytest tests pass;
  all four deliverable PNG/SVG pairs were visually inspected and are ≤1500 px.

## 2026-07-12 — Deep sanity audit: continuous voltage, signal identity unresolved

- Full-file out-of-core audit read every stored value once. Exact zeros: 1212
  0.004%, 614 5.34%, 727 3.16%, 731 4.62%. There are **no ≥99%-zero one-second
  windows during corridor or dark/ITI behaviour**. In 614/727/731 all zero windows
  form one terminal padding run after VR ends (448/265/387 s respectively).
- Earlier “99% empty” plots were invalid: `SD > 0.02` in undocumented units was
  ~four orders above ordinary voltage and selected only extreme periodic events.
- Periodic high-amplitude mode: exact 60 s cadence in 614/727/731; exact 5 s cadence
  in 1212. Events are synchronous across depths and instrument-like. Their phase is not
  sample-locked to VR sync edges (median peak offsets +1.98/+4.26/+1.74 s for
  614/727/731; 1212 unrelated), so they do not prove exact alignment.
- Ordinary-voltage identity (median over 40 corridor windows, 24 depths): lag-1
  correlation −0.07/−0.05/−0.11/+0.01 and only 15.6/17.2/16.0/18.2% of total
  1–499 Hz power below 100 Hz (1212/614/727/731). This is broadband-dominated,
  not a clean low-pass LFP export. 614/727/731 retain a declining 2–40 Hz component
  and some correlation within nominal depth bands, so physiological LF structure may be
  embedded in the broadband voltage.
- Strong narrow peaks at ~74–75 Hz and ~151–154 Hz in 614/727/731. Reference
  sensitivity is not uniform: common-median referencing reduces 614's 153.8 Hz
  peak by 4.54 dB, while its 74.2 Hz peak and both peaks in 727/731 change by only
  −0.06 to +0.03 dB. The persistent ~75 Hz peak contaminates the planned 30–80 Hz
  low-gamma band.
- 1212 is qualitatively different (about 100× ordinary RMS, rising 2–40 Hz spectrum,
  5 s events) and its LFP probe identity is unconfirmed. Keep separate; do not pool
  or attach striatal area labels.
- No producer script or source `.meta` exists in-repo. Input band, gain/units,
  referencing, resampling/anti-alias method and exact voltage↔VR offset remain unknown.
  **Gate remains:** no position decoding, temporal CCA or learning claims until these
  are established. Theta/beta exploration may become possible; low gamma is currently
  confounded and should not be analysed.
- Concurrent learning outputs are quarantined. Their mouse matching happened to be
  correct but used a brittle nearest-trial-count rule; phase indices were off by one
  relative to `epoch_indices.m`; and traversal extraction assumes the unresolved timing.
  Mouse mapping and phase indexing are fixed, and the script now refuses to run unless
  explicitly passed `--allow-unaligned`.
- Supersession rule for the historical entries below: any claim of "clean 1/f",
  biologically good/dead channels, or confirmed timing alignment is obsolete.
  The current data contract and gate above control interpretation.
- New reproducible outputs: `results/sanity_summary.csv`,
  `results/sanity_timing_summary.csv`, `results/signal_identity_summary.csv` and
  figures `sanity_audit_*_v2`, `sanity_audit_event_timing`, `signal_identity`.
  Processing and cached-result contracts are tested (57 pytest tests green).

## Data contract (verified 2026-07-09, before any code)

- **Files:** 4 mice, `RawData/LFP/voltage_data_384ch*.mat` (v7.3/HDF5), mapped by
  size in `lfp_mapping.txt`: 1212 (16.33 GB, 11.4 M samp), 614 (11.38 GB, 8.4 M),
  727 (11.65 GB, 8.4 M), 731 (11.48 GB, 8.4 M).
- **Layout:** `data_to_save` h5py-view `(n_samples, 384)` float32, chunks `(42,384)`;
  `channels_to_save` = 1..384; `depth_to_save` = `[0,0]` placeholder (no depth in file).
- **Sampling ≈ 1000 Hz (inferred, not documented).** LFP `n_samples` == `binned_spikes`
  bin-count for every mouse and VR max ms-index < n_samples — this is **length/grid
  compatibility only**. It does NOT prove the exact sample↔ms offset or that
  `data_to_save[t]` is behavioural millisecond `t` (see `align.py`, corrected). Offset
  provenance unresolved.
- **Behaviour** in `RawData/<ID>_raw.mat`: `VR_data` (MATLAB 10×N: row2 position,
  row4 velocity, row7 trial#, row8 lick) + `VR_times_synched` (N×1 **seconds**;
  ×1000 → ms). Trial windows also in `processed_data/preprocessed_data2p5cm.mat`.
- **Areas:** `Neuropixels_Depth_Data.csv` (µm from tip; DMS/DLS/ACC) covers all 4 —
  **731 blank DLS**. `Neuropixels_V1_Depth_Data.csv` (V1/CA1/DG) only 1212.
  Channel→depth: NPx 1.0 `depth[c]=(c//2)*20 µm` (0..3820). **Probe caveat:** LFP
  is 384ch = one probe; 614/727/731 are probe-1 (striatal). 1212 has two probes —
  its LFP probe identity is unconfirmed; default probe-1, revisit V1/CA1 at Stage 3.
- **Spectrum:** broadband-dominated with a declining 2–40 Hz component in 614/727/731
  and strong ~75/~151 Hz narrow peaks. Anti-aliasing and source band are unverified.
- **Units:** float32 stored voltage units; physical calibration unknown. Do not use
  absolute thresholds or compare absolute power across mice.

## Decisions (with Theo, 2026-07-09)
1. Features = θ (4–8) / β (15–30) / low-γ (30–80) band power **+ broadband (1–100)**.
2. Granularity = all channels per area (drop-in for per-unit) **+** per-area top-PC diagnostic.
3. Home = hybrid: Python `lfp/` front-end → MATLAB basics + Python `tcca/` temporal CCA.

Plan: `~/.claude/plans/magical-tumbling-owl.md` (approved).

## Progress

**2026-07-12 — Decoding + cross-area CCA (PROVISIONAL; crosses the README gate deliberately).**
New `decode.py` (+ `test_decode.py`, 6 tests): `bin_by_position`, group-k-fold `ridge_cv_decode`,
`residualise_by_group`, split-half `heldout_cca`. Driver `scripts/run_decode_cca.py` (position-binned
band power per traversal, artefact-masked). Outputs `results/decode_summary.csv`, `cca_summary.csv`,
`figures/decode_position`, `figures/cca_cross_area`.
- **Position decoding = ALIGNMENT TEST, and it PASSES.** LFP band power (area×band, ridge, group-CV)
  decodes corridor position **above chance in 614/727/731** (R² 0.12 / 0.14 / 0.17; MAE 53/46/**20** cm
  vs ~72 cm uniform-chance; shuffle-p95 ≈ 0). **1212 at chance** (R²≈0; artefact-contaminated). You
  cannot decode position from mis-timed data ⇒ **the export offset is roughly correct** — this
  materially eases the timing worry behind the gate (offset ≲ position-bin scale). LFP carries modest
  position info; all areas contribute similarly (volume conduction).
- **Cross-area CCA is volume-conduction limited (as predicted).** Held-out cross-area CC1 is high
  (0.6–0.9, ≫ trial-shuffle ~0.02–0.14) BUT always **below the within-area split-half VC ceiling**
  (0.88–0.99), and cross-area CC falls with area separation — the signature of shared field, not
  communication. **No area-specific coupling above the VC ceiling** on this single probe. To assess
  genuine inter-areal LFP communication you'd need VC-robust referencing (bipolar/CSD) or LFP→spike
  CCA (which needs the offset). tcca temporal CCA not wired in — VC makes raw LFP–LFP CCA uninformative.

**2026-07-12 — LFP band-power vs learning (PROVISIONAL, provenance-gated).**
New `learning.py` (+ `test_learning.py`, 8 tests) and `scripts/run_learning_evolution.py`.
Per corridor traversal: per-area (DMS/DLS/ACC) band power (θ/β/low-γ) via Welch, median over
area channels; periodic artefact excluded by robust log-outlier on traversal peak; log-power
z-scored over CLEAN traversals only; learning phases from the REAL learning point
(`tcca.find_learning_point`, mice matched to cohort by trial count — `animal_id` is a
positional index 1–16, NOT the mouse number). Outputs `results/learning_evolution_summary.csv`,
`figures/learning_evolution_{mouse}.{png,svg}`.
- **Validation:** all areas incl. **ACC show clean 1/f LFP + a theta shoulder** (my earlier
  "ACC dead" was wrong — volume conduction makes DMS/DLS/ACC PSDs nearly identical). A ~75 Hz
  narrowband peak (line-noise harmonic?) sits in low-γ — flag as a possible confound.
- **Learning points found:** 614 LP44, 727 LP53, 731 LP36, 1212 LP23 (all learned).
- **Refined result (all 4; lines = 21-trial moving avg ± SEM; LP + disengagement marked;
  running/stationary by SPEED not corridor/dark; naive/inter/expert = 1:10 / lp-10:lp-1 / lp+1:lp+10):**
  band power is **NOT stationary** — it tracks **engagement**: β/low-γ (and θ) rise through the
  engaged period and **drop at the disengagement point** (614 clearest: peak ~trial 150–200,
  fall at cp=225; 727 peaks ~cp=100). The narrow LP-aligned naive→expert contrast is still
  modest (expert/naive 0.92–1.26, mostly ↑ in 614, ~flat in 727/731). **θ is LOWER during
  running** in 2/4 (727 0.46–0.57, 731 0.33–0.36; 614/1212 ≈1) — a real behavioural modulation
  (opposite sign to hippocampal running-θ, plausible for striatum/cortex).
  **Confound (flag):** the slow rise/fall co-varies with engagement/arousal/running state, so
  it can't be cleanly attributed to *learning* per se; slow electrode drift is a further
  candidate for the slow component. Areas track together (volume conduction). 1212 artefact-
  contaminated (5 s period) → its numbers unreliable, plotted with a caveat only.
- **Cross-animal summary (`figures/learning_evolution_summary_animals`) TEMPERS this:** the
  naive→expert rise is **largely 614-driven** and NOT consistent — 614 rises across all bands
  (β dips at disengaged); 727/731 are flat/variable. So **no robust group-level learning or
  engagement effect**. The one semi-consistent effect is **θ lower during running in 2/4**
  (727 0.52, 731 0.35 area-avg; 614 0.94, 1212 1.07). Disengaged epoch mixed (θ ↑, β/γ ↓ in 614).
  Refined driver: 10-trial MA, disengaged as a 4th epoch, all bands × epochs, cross-animal panel.
- **1212 EXCLUDED:** 5 s artefact period ≈ traversal length → pervasive contamination; its band
  power is ~10^11× the others and its numbers (0.5× change, θ ratio 0.36) are artefact-driven.
- **Caveats:** provenance still unresolved (scale/band/referencing/offset) → nulls are
  provisional; volume conduction → areas not separable; alignment offset could wash out
  corridor-vs-dark contrasts. Next options: position-resolved power profiles + trial-to-trial
  reliability across phases; theta/γ or cross-frequency measures.

**2026-07-12 — CORRECTION: my "99% empty / broken / misaligned" diagnosis was WRONG.**
A full-file, scale-aware audit (`audit.py` / `sanity.py` + `scripts/run_sanity_audit.py`;
outputs `results/sanity_summary.csv`, `results/sanity_timing_summary.csv`,
`figures/sanity_audit_overview_v2`, `figures/sanity_audit_raw_examples_v2`) overturns it:
- **Continuous during behaviour.** Exact-zero fraction is 0.004 % (1212) to ~5 %
  (614/727/731), and `corridor_zero_window_fraction == 0` for all four. The zeros are
  purely **terminal padding after behaviour ends**, not gaps. Ordinary corridor windows
  show LFP-like oscillatory morphology across depth + a low-freq 1/f + theta spectral
  shape (614/727/731; 1212 more noise-like) — **plausibly real LFP**.
- Real signal is **low amplitude in undocumented "stored units"** (median RMS ~2.6e-6 for
  614/727/731, ~2.3e-4 for 1212).
- A **periodic high-amplitude artefact** recurs every 60 s (614/727/731) or 5 s (1212),
  ±55–131; for 614/731 100 % of events are within 2 s of a VR sync transition
  (sync-locked instrument artefact) → **mask it, it is not the signal**.
- **My errors:** (1) used an absolute amplitude threshold (`std>0.02`) on undocumented-scale
  data — it sat ~4 orders of magnitude above the real LFP, so I discarded the signal and
  kept only the artefact; (2) inverted signal vs artefact; (3) computed
  `corr(LFP amplitude, spikes)` on artefact-dominated amplitude → meaningless; (4) sampled
  windows instead of auditing every value; (5) chained confident wrong claims
  (spike-grid-lock → ACC-dead → 99%-empty → broken). Scale-free metrics (exact zeros for
  continuity; spectral shape for signal) are the right tools — see `sanity.py`.
- **Genuinely unresolved:** units, input band (LF/AP/wideband), gain, referencing,
  anti-alias, exact sample↔ms **offset**. No producer script / source `.meta`. **Gate:
  no position-binning, decoding, or CCA until provenance is established** (per README).
- Cleanup owed: my flawed threshold figures (`figures/sanity_{1212,614,727,731}.png`,
  `figures/stage0_qc_614.*`) and the `figures/stage0_qc` QC verdict are WRONG — supersede/remove.

**2026-07-09 — Stage 0 built; BLOCKED at checkpoint on channel→area mapping.**
- Scaffolded `lfp/` mirroring `tcca/` (conftest + src-layout + system anaconda python;
  no uv/pyproject — matches the siblings; ruff not installed).
- `config.py`, `geometry.py`, `reader.py` (out-of-core overlap-save streamer),
  `features.py` (θ/β/low-γ/broadband Hilbert envelopes), `qc.py` (flat-spectrum +
  high-std rejection), `align.py`. **27 tests green.** Real-file reads work on 614.
- **Alignment confirmed for all 4 mice** (LFP n_samples == spike bins; VR max < n):
  the 1 kHz spike-grid lock holds — the temporal enabler is solid.

- **Stage 0 probe (`scripts/run_stage0_probe.py`, fig `figures/stage0_qc_614`).**

  **Channel→area mapping CONFIRMED CORRECT** (I first suspected it was scrambled —
  it isn't). **1212's `depth_to_save` IS populated** = `[0,0,20,20,...,3820,3820]`,
  exactly `(c//2)*20` and depth-sorted → columns are in depth order and the geometry
  is right. (614/727/731 have `depth_to_save=[0,0]`, an empty placeholder from the
  same export — that's why I first saw only a placeholder. Reuse 1212's formula.)

  **Two real data-quality issues remain (NOT mapping):**
  1. **Zero-filled time gaps** (~6% of the recording, in multi-second chunks) — almost
     certainly non-corridor/ITI periods zeroed at export (cf. the spike pipeline's
     dark-stripping). Harmless: bin only corridor traversals, but **mask zero samples**
     in the binner as a guard.
  2. **Dead channels**: contiguous per-mouse bands of flat-spectrum, high-amplitude
     noise (LF/HF≈0.23 vs 900–3400 on good channels; 2.7× std; near-identical stats
     across the band → systematic, a probe/headstage section or referencing, not
     scattered electrodes). They land on real area depths and kill:
     - 614: DMS 50/50 ✓, DLS 46/46 ✓, **ACC 0/52 dead**.
     - 727: DLS 107/110 ✓, DMS 2/52 thin, **ACC 0/52 dead**.
     - 731: ACC 28/72 ✓, **DMS 0/32 dead** (DLS blank in CSV).
     - 1212: striatal areas **all dead** (its low channels are dead too, unlike 614) → 1212 out for striatal LFP.
     Net: **no mouse has DMS+ACC both good** → the headline DMS–ACC LFP CCA is not
     feasible as-is; **DMS–DLS is feasible in 614** (both good).

  Open question for Theo: are the dead bands a **fixable export/referencing artifact**
  (re-export could recover ACC) or genuinely bad channels on these recordings? The
  systematic contiguous near-identical pattern hints at the former.
  → Held before Stage 1 per Theo ("pause until resolved"). Front-end (27 tests) ready.

- **[SUPERSEDED — WRONG. See the 2026-07-12 correction at the top of Progress.]** The
  block below is an **absolute-threshold artefact and is false**: the files are continuous
  during behaviour (audit: `corridor_zero_window_fraction==0`). Kept only as an error record.
- ~~**DEFINITIVE DIAGNOSIS (same day): the LFP files are broken — ~99% empty and NOT
  time-aligned to the spike/behaviour grid. Unusable as delivered.**~~
  Corridor-epoch check + alignment test (scratchpad `corridor_lfp.py`, `align_test.py`,
  `align` multi-mouse) show, for all 4 mice:
  - real voltage signal in only **0.0–0.7 %** of windows (rest is exact-zero or a ~1e-5
    noise floor); real fragments are sparse and scattered (~1 per 10 min).
  - **`corr(LFP amplitude, spike count) ≈ 0`** (−0.05..−0.01); LFP is flat-zero in
    469/500 windows where spikes ARE present. So LFP sample t does NOT correspond to
    spike bin t — the matching 8.4M/11.4M length is a red herring (length matched, content
    not). During corridor traversals the LFP is essentially all zero.
  - This — not tissue, not channels, not depth mapping — is why "LFP looked dead but units
    fine": the spikes are a separate, complete, correctly-synced file; the LFP file itself
    is mostly empty and mis-timed. My earlier "1 kHz spike-grid lock" enabler was inferred
    from matching LENGTH only; I never checked content alignment (corr vs spikes) — that was
    the core error. Everything built on it (ACC-dead, per-mouse feasibility, DMS–DLS-in-614)
    is **withdrawn**.
  **Needed:** re-export the LFP at source as a continuous, time-aligned LFP-band signal —
  the raw `.lf.bin` (2500 Hz) synced via `Synch_NP_VR` exactly like the spikes, placed on
  the same grid so `LFP[:,t]` matches `binned_spikes[:,t]`. Whatever produced
  `voltage_data_384ch` allocated the right-length array but did not fill it correctly.
  **Pipeline code (reader/features/qc/geometry/align, 27 tests) is sound and ready** — it
  will run once a correct LFP file exists. TODO: harden `align.check_alignment` to also
  assert CONTENT alignment (corr of LFP amplitude with spike rate over the session), not
  just equal length.

- **RETRACTION (same day): "ACC LFP is dead" was premature — do NOT trust it.**
  Theo pushed back (LFP and spikes share electrodes; how can LFP be dead but units fine?).
  Checked units as ground truth: **614 has 64 sorted units at 2000–2500 µm = exactly the
  "dead" ACC channels** (units span the whole probe; unit density matches anatomy incl.
  a real 0-unit white-matter gap at 1500–2000 µm). Kilosort can't find neurons on dead
  channels ⇒ the channels are **live tissue**, so flat-LFP-there is not a tissue fact.
  Re-examined and the classification is **window-dependent and unstable**: the recording
  is a mix of (a) near-zero gaps, (b) **high-amplitude broadband artifact bursts** that
  dominate many windows (all channels go white/flat, theta_pow≈beta_pow≈2.4e-4), and
  (c) a minority of clean windows where striatal channels show proper 1/f LFP and ACC
  looks flatter. Single-window QC (my `qc.py` hf_frac criterion) over-concluded from one
  60 s window. **So: the "ACC dead / no DMS–ACC CCA" conclusion is withdrawn.** Cannot
  reliably classify channels or trust band power on this file until (i) Theo explains how
  `voltage_data_384ch` was produced (raw wideband? LFP band? referencing? why gaps + bursts)
  and (ii) the pipeline gains artifact/gap-aware, clean-period selection (not naive
  per-window QC). Diagnostics in scratchpad (unit-depth counts, robust PSD, lowband slope).
