# striatum_lfp — running log (newest first)

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
