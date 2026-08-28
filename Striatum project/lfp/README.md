# striatum_lfp — voltage-export audit and provisional LFP pipeline

Audits the 384-channel Neuropixels voltage exports for both the task and
Control 1 cohorts, and analyses band power as the analogue of unit firing rate. Provenance and 1 ms grid alignment were resolved on
2026-08-11; the full 17-file cohort was inventoried and every filename verified
against spiking on 2026-08-27. What is still gated is listed under Current gate.

## Data: verified facts (measured 2026-08-27 over the full cohort)
- **Two cohorts.** Every driver takes `--cohort task|control`; outputs are
  suffixed. `config.Cohort` holds what differs: mouse list, depth CSVs, the
  probe-2 raw suffix (task `_V1_raw.mat`, control `_v1_raw.mat`), and the fact
  that yoked controls inherit the task cohort's average learning point (41), so
  their epochs are matched time windows rather than learning windows.
- **Control 1: 8 usable exports** in `RawDataControl/LFP/` — 5 striatum (407,
  513, 515, 817, 1205) and 3 visual (513, 515, 817). 408 has an export but is not
  in the organiser's list and is dropped automatically. Control 2 has no voltage.
  8/8 reproduce the MATLAB bin map to 0.0 ms; 7/8 identity-confirmed in all three
  windows (817/striatum is 2/3 and weak — only 48 sorted units make its MUA
  reference sparse). 407's export stops 26 min before its session does.
- **18 named exports** in `RawData/LFP/`: 13 striatum probes (523, 614, 624, 727,
  730, 731, 822, 823, 1105, 1106, 1201, 1206, 1212) and 5 visual probes (1105,
  1106, 1201, 1206, 1212, suffixed `_v1`). Absent: 409, 418, 703 (probe 1 only). The file→mouse map comes from the filename now, not from file size —
  `lfp_mapping.txt` is superseded. Note 727's file has no underscore before
  `voltage`; use `cohort.parse_lfp_filename`, do not re-derive the pattern.
- Every file is `data_to_save` = (8,400,000 × 384) float32, gzip, chunks (42, 384),
  no non-finite values, no dead channels. `channels_to_save` = 1..384 and
  `depth_to_save` = 0–3820 µm reproduces `geometry.channel_depths` exactly, so the
  2-channels-per-20 µm geometry behind the area mapping is measured, not assumed.
- **Every filename has been verified against spiking** (`scripts/run_lfp_identity.py`):
  each file's 30–90 Hz envelope beats every other animal's MUA in 3 (16 files) or
  2 of 3 (823, 1105 striatum) independent windows. No duplicate content fingerprints.
- 1212 is the one exception to grid compatibility: 8.4 M LFP samples against
  11.4 M spike bins. The offset scan places it at offset 0, so it is the truncated
  head of the same session — the last ~41 min of behaviour simply has no LFP.
- **Two gain regimes.** Median channel RMS 6.3e-6–1.3e-5 (July batch) vs
  1.6e-4–3.2e-4 (August batch). Never compare absolute power across animals; every
  outcome must be within-session relative. Values remain "stored voltage units":
  gain, physical units and the anti-alias filter are still undocumented.
- **Mains must be notched on every file.** 50 Hz / shoulder ratio reaches 1348×
  (1105 visual), 1097× (1105 striatum), 450× (1106 visual), 161× (1212 striatum),
  with odd harmonics to 350 Hz in 1105. Six files are below 3×. Notch unconditionally.
- Terminal zero padding exists only in the July batch (onset 7863–8135 s). No
  mid-session dropouts anywhere.
- Referencing is per-session: common-*median* residual is 0.030–0.084 throughout,
  but the common-*mean* residual runs 0.12–0.18 (July) up to 2.42 (822).
- Area coverage: DMS 13 animals, ACC 12, DLS 10, V1 5, CA1 3, DG 3. CA1/DG still
  cannot support a cohort claim.

## Current gate

The July gate (no position binning, decoding or CCA until provenance is resolved)
was **lifted on 2026-08-11**: the re-export is real LFP on the project's 1 ms grid,
verified physiologically. What remains gated is narrower and specific:

- Absolute-power and cross-animal amplitude comparisons — blocked by the two gain
  regimes and the undocumented units.
- 1212 in any trial-indexed or learning analysis — blocked by the 41 min truncation.
- Any band overlapping 50 Hz or its harmonics before notching.
- Cross-area coupling claims without both a trial-permutation null *and* the
  within-area split-half volume-conduction ceiling: DMS/DLS/ACC sit on one shank.
- CA1 and DG cohort claims — n = 3 per cohort.
- Any "task animals differ from controls" claim on the LFP *spatial profile*
  without the behavioural check: task animals run the corridor far more
  stereotypically (speed-profile split-half r 0.98 vs 0.68) and faster (36 vs
  21 cm/s), and the LFP profile tracks speed. Run the `behaviour` arm first.

## Layout
`src/striatum_lfp/` — configuration, cohort discovery and file-identity
statistics (`cohort.py`), geometry, out-of-core reading, per-file inventory
(`inventory.py`), trial/spatial/dark binning of band power (`bandpower.py`),
the shared epoch and learning-point layer (`analysis.py`), the decoding /
reliability / CCA primitives (`arms.py`), shared figure conventions
(`figstyle.py`), integrity/sanity helpers, provisional feature extraction, and
quarantined learning helpers. `scripts/` contains reproducible audit drivers; `tests/` contains
synthetic-ground-truth pytest checks. The old single-window `qc.py` thresholds
are retained only as tested numerical primitives and are not an analysis gate.

## Running tests
```
cd "Striatum project/lfp" && /opt/anaconda3/bin/python -m pytest -q
```
`conftest.py` puts `src/` on the path. The interpreter is explicit because the
current shell may resolve `python3` to a Homebrew installation without pytest.

See `NOTES.md` for the running log and the full data contract.
