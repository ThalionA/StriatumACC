# striatum_lfp — LFP band power as the analogue of unit firing rate

Band power per trial × 5 cm spatial bin from the 384-channel Neuropixels voltage
exports, on the unit pipeline's 1 ms grid, for the task cohort and yoked Control 1,
and the analyses built on it. The running log is `NOTES.md` (newest first); the
July audit of the superseded export is in `NOTES_archive.md`.

> **Status (2026-09-26).** Reviewed, rebuilt and re-run end to end on branch
> `lfp-simplify` (MATLAB products regenerated first). Current results are the
> `NOTES.md` top entry; results from before 2026-09-25 are superseded.

## Where things stand (2026-09-26)

| question | answer on the fixed pipeline |
|---|---|
| Does learning change band power differently in task animals than in yoked controls? (pre-registered) | No cell differs (0/30; DLS theta p_FDR 0.58). Controls are n = 4-5. |
| Does band power change Naive → Expert within task animals? | DLS theta falls (−0.060 log10, p_FDR 0.029) — also in the dark ITI; DMS beta rises (+0.029, 0.018). |
| Can position be decoded from band power? | Yes, in all 15 striatal/ACC cells, but R² above null is 0.01-0.08. |
| Is the spatial profile more reliable in task animals? | Yes, 15/30 cells, 13/30 after removing speed; task animals also run far more stereotypically. |
| Is there theta-gamma coupling? | Yes: 66-78 % of cells vs a 2-6 % calibrated null (descriptive). |
| Is there a consistent direction between areas (PSI)? | No: 0/24 bipolar cells consistent across animals. |
| Does band power carry behavioural information beyond speed? | No: 8/9 features at or below the speed-confound floor. |

## One command

```
cd "Striatum project/lfp" && ./scripts/run_lfp_pipeline.sh
```

Steps, in dependency order (`--list` prints them, `--from <step>` resumes,
`--cohort task|control` limits to one cohort, `--only 409,418` limits band power):

| step | what | reads |
|---|---|---|
| inventory | per-file audit: size, 1/f slope, mains, adjacency | voltage |
| identity | each file's notched 30-90 Hz envelope vs every animal's MUA | voltage |
| bandpower | band power per trial × bin (theta, beta, low/high gamma, total), mains-notched | voltage |
| validate | cube trials vs MATLAB's good mask; bin spans vs MATLAB `durations` | caches + MATLAB |
| arms | evolution, decoding, reliability, moving reliability, CCA, behaviour, and their across-animal tests | caches |
| distance | within- vs across-area coupling at identical separation, per boundary | caches |
| psi | phase-slope index (direction), monopolar and vertical bipolar | voltage |
| coupling | theta-gamma PAC and envelope coupling, with a re-paired-trial null | voltage |
| mi | `../infotheory`: trial caches, spike MI, LFP MI | MATLAB + caches |
| contrast | task vs control per cell, exact permutation, BH within arm × metric | both cohorts |
| plots | every figure, reading tables only | tables |

A failing step stops the chain (`pipefail`); every run logs to
`results/pipeline_<timestamp>.log`. The MATLAB products it reads are rebuilt by
`../processed_data/regen_chain.sh`.

## Three shared layers — use them, do not re-derive

- **Trials — `trials.SessionTrials`.** MATLAB's own good-trial mask, the learning
  point on the good numbering, the disengagement point as a raw trial number
  (NaN = no clip, as MATLAB does), and one three-epoch scheme: Naive = good trials
  1-10, Intermediate = the ten before LP, Expert = the ten from LP, all before DP.
  A window that crosses DP or leaves the recording is dropped, never shortened.
  Every driver in `lfp/` and `infotheory/` gets its trials here; a test fails if
  one builds its own windows.
- **Signal — `geometry`, `bandpower`, `area_signals`.** Depths in the unit
  (Kilosort) convention, 20-3840 µm; channel 191 (the probe reference) is never
  tissue; mains notched on every read; bipolar = vertical pairs one row apart
  (c, c+2), deep minus shallow, on raw voltage.
- **Statistics — `stats`.** The animal is the unit. Exact sign-flip (paired) and
  two-sample permutation tests with their floors; a cell that cannot reach 0.05 at
  its n is marked, not reported as null. BH-FDR within a declared family. Tests
  live in run scripts and `src/`; plotting scripts only read tables.

Pre-registered primary test for "does learning change band power differently in
task animals than in yoked controls": `delta_log_corridor` (Expert − Naive log10
of mean linear power), exact permutation, BH over area × band.

## Data facts (measured)

- **Cohorts.** Task: 16 striatum probes (409, 418, 523, 614, 624, 703, 727, 730,
  731, 822, 823, 1105, 1106, 1201, 1206, 1212) and 5 visual (1105, 1106, 1201,
  1206, 1212). Control 1: 5 striatum (407, 513, 515, 817, 1205) and 3 visual (513,
  515, 817); 408's export is outside the organiser's list; Control 2 has no voltage.
  Files are found by name (`cohort.parse_lfp_filename`).
- **Grid.** Every export is on its probe's own 1 ms spike-bin grid (live test).
  Probe-2 LFP sits at lag 0 ± 3 ms against probe-2 MUA while the control bundles'
  VR clocks drift 6-159 ms apart (`scripts/audit_probe2_clock.py`).
- **Truncations.** 407's export stops 26 min early (169 of 209 trials covered);
  1212 was re-exported at full length.
- **Referencing.** The exports are already common-median referenced at source.
- **Units.** Two gain regimes (~1000×) and undocumented physical units: every
  outcome is within-session relative. Never compare absolute power across animals.
- **Mains.** 50 Hz up to 1348× the shoulder (1105); notched everywhere.
- **Coverage.** DMS 16, ACC 15, DLS 12, V1 5, CA1 3, DG 3 animals. CA1/DG cannot
  support a cohort claim (n = 3 per cohort; a sign-flip floor of 0.25).

## What this probe cannot answer

- Area-specific communication from band-power co-fluctuation: DMS/DLS/ACC share a
  shank, and only the DMS-DLS boundary has pairs at identical separation.
- A cortico-striatal boundary effect by exact distance matching (ACC is too far).
- Position coding separate from running speed, unless it survives the
  within-trial speed slope.

## Running tests

```
cd "Striatum project/lfp" && /opt/anaconda3/bin/python -m pytest -q
```
`conftest.py` puts `src/` on the path; the explicit interpreter avoids a Homebrew
`python3` without pytest. Lint: `ruff check src scripts tests`.
