# striatum_lfp — running log (newest first)

## 2026-09-28 — Band power across the corridor and across trials (descriptive)

Agent-written. New: `arms.area_position_trial_map`, which gives log10 power
z-scored per channel over corridor + dark and then the channel mean, as a
bin × trial map, optionally with the within-trial speed component removed via
`residualise_on`. Three tests. `scripts/run_lfp_position_trial.py` writes
`results/lfp_position_trial_<cohort>/`, and `scripts/plot_lfp_position_trial.py`
draws per-animal heatmaps, epoch position profiles and trial evolution aligned
to LP, raw and speed-residualised. It reads tables only. No statistics; read by
eye from the figures:
- **Beta peaks at ~100-125 cm** in every area. In the striatum and ACC it
  shrinks by roughly half after speed residualisation, so it is largely the
  deceleration before the reward-zone stop. In **V1, beta (~+0.6 z) and high
  gamma peak at the visual cue zone** (starts 80 a.u. = 100 cm) and survive
  speed residualisation. That looks like a sensory response. N = 5; untested.
- **Theta is higher in Naive** than in later epochs (DLS, CA1, DG) and falls
  early in the task session. That is consistent with the tested DLS-theta
  within-task fall. Yoked controls show no early fall.
- **Low and high gamma climb steadily across the session in task AND
  control** (DMS, DLS, ACC). This is session time or drift, not learning. Do
  not read the task-only gamma trend as a learning effect.
- Caveat: `analysis.bin_speed_cm_s` quantises. A 5 cm bin crossed in one VR
  frame reads ~150 cm/s, so the top of the speed scale is coarse.

## 2026-09-26 — The re-run on the fixed pipeline: what survives

Agent-written. MATLAB products regenerated (`regen_chain.sh`, all steps exit 0;
old products in `processed_data/_archive_2026-09-25/`), then the whole LFP
pipeline, both cohorts. Predictions registered beforehand: `PREDICTIONS.md`
2026-09-25 (b). Everything below is DP-clipped, Naive = good trials 1-10, exact
permutation tests with floors, BH within the declared family.

**Validation (the falsifier) passed.** 29/29 files: no LFP-good trial MATLAB
dropped, >99 % of bin spans within 1 ms of MATLAB's `durations`; the only
trial-count gap is 407's short export (31 trials). Identity with the notched
envelope: 20/21 task files confirmed in all three windows (was 18/21; 418 and 1105
striatum now pass), 823 striatum 2/3; control 7/8, 817 striatum 2/3.

**Task vs yoked control (`lfp_group_contrast.csv`).**
- *Pre-registered primary* (Δlog10 power, Naive→Expert): **0/30 cells differ**
  (20 reachable; smallest p_FDR 0.58). DLS theta: task −0.060 vs control +0.012,
  p = 0.051, p_FDR = 0.58 (n 11 vs 4 after DP). **The 2026-08-28 DLS-theta
  dissociation does not survive.** All evolution metrics: 0/120.
- Decoding 0/30, moving reliability 0/30, CCA 0/30.
- Split-half reliability of the spatial profile: task > control in **15/30** cells,
  and **13/30 after removing each channel's within-trial speed slope** — the gap is
  not explained by the linear speed component. Behaviour: speed-profile split-half
  r 0.97 vs 0.54 (p_FDR < 0.001); mean speed 33 vs 22 cm/s is no longer
  significant after DP clipping (p = 0.086).

**Within the task cohort.**
- Evolution (primary): **DLS theta −0.060 (p_FDR 0.029, n 11)** and DMS beta +0.029
  (p_FDR 0.018, n 15). In the figure the DLS theta fall is present in the dark ITI
  as well as the corridor, so it is not corridor-specific; whether it is learning
  cannot be said — the control cohort (n ≤ 5) cannot reach 0.05 on any
  within-group test (0/30 reachable), and the group contrast is null.
- Decoding beats its rotated-label null in **15/15 striatal and ACC cells**; V1/CA1/DG
  (n ≤ 5) are unreachable. Effect sizes are small (R² above null 0.01-0.08).
- CC1 falls with separation **within animals** (theta −0.17/mm p_FDR 0.012; high
  gamma −0.16, 0.023; total −0.18, 0.010; beta/low gamma p_FDR 0.07).

**Coupling and direction.**
- PAC vs its calibrated null (same trials, amplitude re-paired): task bipolar
  within-area 77 % vs 6 %, between 69 % vs 4 %; control 78 % vs 2 %, 66 % vs 2 %.
  Cells = animal × area × band: descriptive, not an across-animal test.
- PSI with vertical bipolar pairs: 33 % of task animal-cells |z| > 2, but **0/24**
  pair × band cells with a consistent direction across animals (12 reachable);
  monopolar 1/24. Monopolar and bipolar now disagree (r = −0.25, 45 % sign
  agreement). Controls: nothing reachable.
- Distance control, DLS-DMS boundary (n 12): within − across +0.04 to +0.08, all
  CIs include 0 (e.g. low gamma [−0.04, +0.19]) — an area term of ~0.1-0.2 cannot
  be excluded. ACC-DMS: n 4, unreachable.

**Information (../infotheory).** LFP: shuffle-subtracted I(power; feature) is
0.0004-0.0026 bits and, conditioned on speed, sits at or below the speed floor
for 8/9 features; time-to-reward-zone clears it (p = 0.042 uncorrected, an ad hoc
per-animal check, 1 of 9) — **"band power carries behavioural information beyond
speed" does not survive.** Spike MI, Expert − Naive: nothing survives correction
(smallest p 0.018 uncorrected, first-lick position).

**LFP vs units, moving reliability (same mice, areas and epochs;
`lfp_reliability_moving_vs_units_task`):** in DMS/DLS/ACC the LFP's raw
reliability is 0.01-0.07 against units 0.19-0.29 -- below even the units' own
trial shuffle (0.10-0.15). In V1 the LFP (0.03-0.08) sits near the units' shuffle,
and in CA1 LFP theta (0.08-0.12) matches the units (0.10-0.13); CA1 is n = 3.

Housekeeping: 48 figures no current script produces → `figures/_archive_pre_2026-09-25/`;
July npz → `results/_archive_july/`. `ProcessStriatumTask.m` had an undefined
`task_data` in its tail (hidden while the chain ignored exit codes) — fixed, and
the fixed script then ran end to end (exit 0, 2026-09-26 10:45; `processed_data/regen_2026-09-25.log`).

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

**Progress (same day, branch `lfp-simplify`):**
1. July code deleted (tag `lfp-july-archive`). `features.py` → `filtering.py`.
2. One trial layer, `trials.SessionTrials`: MATLAB's own good mask (read back
   from `corridorData.trial_reward`), LP on the good numbering, DP as a raw
   trial number (NaN = no clip, as MATLAB), three epochs, windows dropped not
   shortened. Every lfp/ and infotheory/ driver uses it; a guard test forbids
   drivers building windows. infotheory's cache is now raw-indexed (1212 fixed).
3. Signal layer. **Probe-2 clock settled by measurement**
   (`scripts/audit_probe2_clock.py`, `results/lfp_probe2_clock_audit.csv`): the
   probe-2 LFP is on probe 2's clock (LFP→MUA lag 0 ± 3 ms while the bundle
   offset drifts 116→158 ms in 817) and the V1 bundle's VR times give all three
   controls the same onset transient; **MATLAB's crop of control probe-2 UNITS
   with probe 1's times is the misalignment** (not fixed here; flagged as a
   separate task). Depths now in the unit convention (+20 µm); ch191 never in an
   area; bipolar = vertical pairs (c, c+2) on raw voltage; coupling/PSI reads
   mains-notched; `truncated_trials` against the export length (cube good flags
   now equal MATLAB's mask in 28/29 sessions; 407 genuinely truncated).
4. Statistics. `striatum_lfp.stats` (exact sign-flip / two-sample permutation,
   floors, BH). Group contrast primary = `delta_log_corridor` (log of mean
   linear power). Decoding now actually tested. CCA null permutes cube trials;
   the within-area ceiling is gone. CC1-vs-distance is a within-animal slope.
   PSI/distance tests moved into src. infotheory: 5-trial within-block null,
   tie-safe splits in both arms, shared-feature contrasts, speed-confound floor.
   **Open decision:** the spike-MI per-animal value is the median over units and
   is exactly 0 in 13-15/16 animals; old Wilcoxon silently dropped the zeros.

5. Every other flagged issue fixed in code (Theo: "fix all the issues"):
   MATLAB `probe2_on_probe1_clock.m` (control probe-2 units onto probe 1's clock
   through the shared VR frames; tested) used by both organisers;
   `ProcessStriatum{Task,Control}.m` now filter corridorData/darkData too and
   save `good_trials` -- before, lick errors (so the LP) and the spatial unit
   arrays were the first n_trials RAW trials (1212, 409 only); Python readers
   accept both product generations. Spike MI averages ACTIVE units (the median
   was 0 in 13-15/16 animals). Identity envelope notched. PAC calibration null
   computed per cell (`p_trial_repaired_null`) instead of a typed-in 7/96.
   Distance contrast per boundary (only DMS-DLS is testable by exact matching;
   ACC is too far). Speed slope from within-trial variation. Depth panel and
   the units join computed/joined properly. Last dark bin closed as histcounts.
   Per-trial PAC on a fixed 4 s. De-duplication (CSV writer, --cohort, area
   floor = 5 everywhere, save_pair/colours). `regen_chain.sh` stops on failure
   and runs IntegratedAll_v1.
   Left as is, deliberately: the one-VR-frame lag (shared with the units, so
   like-for-like) and controls taking the task-average LP (MATLAB's convention;
   no yoked-partner map exists).

Not yet done: MATLAB regeneration, the LFP re-run (step 5) and docs (step 6).
Every committed table and figure still predates all of the above.

## 2026-09-17 (b) — Direction of communication: nothing survives removing the far field *[UNDER REVIEW 2026-09-25 — see the top entry]*

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

## 2026-09-17 — The cross-area coupling is distance, and nothing else *[UNDER REVIEW 2026-09-25 — see the top entry]*

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

1. **DLS theta strengthened.** *[UNDER REVIEW 2026-09-25 — see the top entry]* Task −0.186 vs control +0.076, p = 0.0011,
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
It also survives the speed control *[FALSE even then: speed-residualised p_FDR 0.0508, `differs=False` — 2026-09-25]*. That is now a dissociation, not a lone
significant cell.

**3. Position decoding does not depend on reward.** 0 of 30 decoding cells differ.
Control 1 runs the same corridor, and the LFP's (small) position information is
there just the same.

**4. The reliability gap is behavioural, and I nearly reported it as neural.** *[UNDER REVIEW 2026-09-25 — see the top entry]*
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
spatial structure is 4–10× weaker than the spiking recorded on the same probe** *[UNDER REVIEW 2026-09-25 — see the top entry]*;
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
IS decodable: **12/12 striatal and ACC area × band cells survive BH-FDR** *[no committed code ran this test — 2026-09-25]*
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

---
Entries before 2026-08-27 (the July audit of the superseded export, the
2026-08-11 re-export, the original data contract and decisions) are in
`NOTES_archive.md`, verbatim.
