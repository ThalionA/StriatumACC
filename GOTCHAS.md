# StriatumACC gotchas

- LFP `data_to_save` units and source band are undocumented: never use an
  absolute amplitude threshold to decide whether signal exists; audit exact
  zeros, finite values and scale-free temporal/spectral structure separately.
- Equal LFP/spike array lengths and VR timestamps fitting inside the nominal
  grid are necessary compatibility checks, not proof of sample-zero alignment.
- Referencing effects are session- and frequency-specific: common-median
  referencing suppresses 614's ~154 Hz peak but not its ~74 Hz peak, and does
  not materially change either peak in 727/731.
- MATLAB `run('/abs/path/script.m')` cds into the script's folder for the
  duration: relative `load()` inside resolves there first and can silently
  shadow-load a same-named file (bit us with a synthetic test fixture).
- Never hold two `preprocessed_data*.mat` structs in RAM at once — each
  expands to tens of GB (binned_spikes_trials/darkData/corridorData payloads);
  load sequentially and extract slim fields. Same class: never snapshot
  `all_data` while the per-animal loop mutates it (copy-on-write doubles peak
  RAM); both patterns OOM-killed MATLAB with no error dialog (2026-08-10).
- `clearvars -except all_data` + load-only-if-absent lets a leftover cohort's
  all_data be silently processed under another cohort's filename; the Process
  scripts now assert cohort identity by mouseid — keep that guard.
- Velocity in MutualInformationStriatum_v2 / Nonlinear_Epoch_Decoding /
  CrossSpatialBinDecoding is hardcoded `(4*1.25)./durations` (assumes 5 cm
  bins): every 2.5 cm-era velocity-dependent result was 2× too high.
- Control probe-2 raw files are lowercase `<id>_v1_raw.mat`; task ones are
  uppercase `<id>_V1_raw.mat`. Case-insensitive macOS hides mismatches that
  break on the Linux cluster.
- Task depth-CSV id 507 = recording `0705_M1_Vishal` (MMDD swap), deliberately
  excluded from `all_mouse_ids`; control2 CSV row 624 likewise unused.
- LFP file `727voltage_data_384ch.mat` has **no underscore** before `voltage` while
  every other export does. An f-string pattern `f"{mouse}_voltage_data_384ch.mat"`
  drops mouse 727 silently, with no error. Use `cohort.parse_lfp_filename`. (2026-08-27)
- The probe-2 LFP file is lowercase `<m>_v1_voltage_data_384ch.mat` but the probe-2
  spike bundle is uppercase `<m>_V1_raw.mat`. Deriving one path from the other by
  string substitution works by accident on this case-insensitive volume and fails
  on a case-sensitive one. Use `config.raw_mat(mouse, probe)`. (2026-08-27)
- Depth bands in the CSVs can touch: 1206's probe 2 has DG ending and CA1 starting
  at 1160 µm. MATLAB assigns areas in CSV column order and lets the **last** write
  win (DG), so independent boolean masks double-label that channel.
  `geometry.channel_area_masks` now applies the same precedence. (2026-08-27)
- LFP file identity: raw |r| between an LFP envelope and a candidate's MUA is not
  comparable across candidates. An animal whose MUA carries a synchronous artefact
  correlates with *every* file and steals rows. Divide each candidate column by its
  median across files (`cohort.column_normalise`) before taking the argmax. (2026-08-27)
- A decoding null that permutes TRIALS is no null at all when every trial carries
  the same target sequence: for corridor position, `y` = 0..49 repeated per trial,
  so a trial permutation leaves it bit-identical and the "null" re-runs the real
  decoder. It looked like a real null because it returned a plausible non-zero R².
  Rotate the labels *within* each trial instead (`arms.circular_shift_targets`).
  With the broken null, LFP decoding looked like chance; with the correct one, 12/12
  striatal/ACC cells survive BH-FDR. (2026-08-27)
- MATLAB's `spatial_binned_data.durations` is the UNCLIPPED VR span, while the spike
  sum it feeds uses npx indices clipped to the trial length. Validating a new binning
  against `durations` therefore flags each trial's LAST bin as a mismatch even when
  the binning is correct. Compare against the clipped range. (2026-08-27)
- `npx_index`-style clipping hides a truncated trial: a trial running off the end of
  a short export comes back looking like one that finishes exactly at the last
  sample. Only the UNCLIPPED VR time distinguishes them (`bandpower.truncated_trials`);
  this bites 1212, whose export stops 41 min before its session does. (2026-08-27)
- Two reliability numbers on the same data can both be right and look contradictory:
  split-half over ~100-trial halves measures the reproducibility of the MEAN spatial
  profile (LFP: 0.48–0.92), while the project's 5-trial moving window measures whether
  any SINGLE trial resembles its neighbours (LFP: 0.00–0.09 above shuffle). Say which
  one a number is. The unit pipeline's `stability_by_animal.csv` is the second kind.
  (2026-08-28)
- Any `tcca` result written before 2026-08-12 predates the cell-type fix: FS-exclusion
  was a NO-OP in V1/CA1/DG (no cell type assigned at all) and ACC used the striatal
  four-way rule. Re-running the 2026-08-11 epoch grid on corrected types drops
  122 -> 107 FS-excluded cells at b25 (animal 11's DG falls to 3 units) and moves 83
  of the 107 survivors, worst cc1 0.73. FS-INCLUDED arms are immune (inclusion ignores
  the label) — use them to separate a labelling change from a code change.
- Partial CCA can annihilate rank, and nothing in `epoch_metrics*.csv` shows it:
  `k_eff` is `min(K, n_units_x, n_units_y)`, the PRE-partial dimensionality. Measured:
  animal 14's ACC has 21 units and Z = 13, but only 14 directions survive partialling —
  the rest sit at relative singular values of 1e-9. Such cells' cc1 swings by up to 0.4
  purely on the CCA rank tolerance, so their communication estimate is not trustworthy
  under any implementation. Check the post-partial spectrum before believing a
  partial-CCA cell; a plain-CCA run is unaffected (0/153 cells move on the rank rule).
- At n = 9-11 animals the two-sided Wilcoxon p sits on a coarse discrete lattice
  (0.0020, 0.0039, 0.0156, 0.0391, ...). An unchanged p across two runs is NOT evidence
  of unchanged data — medians moved by up to 40% under identical p-values in the
  2026-08-28 grid re-run. Compare medians and effect sizes, not p-values.
