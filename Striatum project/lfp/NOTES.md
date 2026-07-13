# striatum_lfp — running log (newest first)

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
