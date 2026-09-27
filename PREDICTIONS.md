# Predictions (newest first)

## 2026-09-26 (d) — committed partial CCA re-run with fold-wise partialling

The fix: confounds (other areas' PC scores) are now partialled INSIDE each CV
fold. Held-out CC, the lag curves and the surrogate null all use unpartialled
scores + the confound; Stage 3 is unchanged. On synthetic independent areas the
old in-sample step added +0.013 at k=20 with 16 confounds, and nothing at
strong coupling.
- **F1:** partial held-out CC1 falls vs the pre-fix pickle in most
  animal-pair-epochs. The median fall is 0.003–0.02, and largest where CC is
  low. ~70%.
- **F2:** the per-dimension significant count (`n_significant`) falls in a
  minority of cells, and none rises by more than 1. ~65%.
- **F3:** the committed conclusions hold: partial CC > 0 at the animal level,
  and no BH-surviving naive → expert change. ~80%.
- **Falsifier:** a committed conclusion flips (a pair's partial CC is no
  longer > 0, or an epoch effect appears).
- **Outcome (same day; details in cca/NOTES.md round 18):**
  - **F1 confirmed.** Like-for-like, the median fall is 0.0096 (FS-excl, 64% of
    epochs), −0.012 in the low-CC half and −0.003 in the high-CC half.
    FS-incl: 0.0043.
  - **F2 falsified.** `n_significant` fell in 45% of cells but ROSE in 20%, by
    up to 4 dims. The total is 686 → 616.
  - **F3 mostly held. The falsifier fired once:** V1–DMS expert partial CC > 0
    is lost at the animal level (p 0.031 → 0.080; V1 pairs are few-animal).
    The striatal triangle and V1–ACC CC > 0 hold, and no rm-ANOVA epoch
    effect appears.
  - **IFI (directionality) is fragile:** one vs-0 result is lost, one is new,
    and several means change sign.
  - **Lesson:** a leak with a small mean effect on CC (−0.01) can still move
    near-threshold, sign-based summaries such as the IFI.

## 2026-09-26 (b) — movement encoding at 20 ms (temporal arm), 4 task mice

Same unit set, drift terms and circular-shift null as the 5 cm analysis, but
per 20 ms bin (the cca temporal_bin_ms) of the corridor period. Spike column 0 =
the corridor's 2nd VR row (verified: n_1ms = last − 2nd row time + 1 in every
trial checked). Covariates are interpolated to bin centres and entered at lags
{−200, −100, 0, +100, +200} ms. Base = position (5 cm one-hot) + drift +
lagged VR speed and lick. Tested: lagged face SVD (10), or lagged ROI ME (4).

- **T1:** face SVD beyond VR flags more units at 20 ms than at 5 cm in ≥ 3/4
  animals, pooled over areas. ~60%. Fast whisking and licking are no longer
  averaged away.
- **T2:** the median ΔR² stays < 0.01 in every area × animal cell. ~80%.
  Single 20 ms spike counts are Poisson-noisy, so R² is small for everything.
- **Falsifier for "the 5 cm null was a resolution artefact":** T1 fails.
- **Outcome (same day, `run_temporal_encoding.py`):**
  - **T1 confirmed.** Units modulated by face SVD beyond VR, 5 cm → 20 ms:
    1105 26→56%, 1106 52→79%, 1201 12→52%, 1206 41→70%. The null's empirical
    false-positive rate is 6.5–6.8%. The ROI ME give the same picture
    (44–95% per area).
  - **T2 falsified in one cell.** The median ΔR² is < 0.01 everywhere except
    1106 DMS (+0.021), the same outlier as at 5 cm.
  - No consistent Expert − Intermediate change.
  - **Reading:** fast face movement is tracked by most units, but it explains
    very little of any one unit's 20 ms spike count (itself Poisson-dominated,
    so R² is small for everything). The 5 cm analysis under-counted modulated
    units by averaging over ≥ 250 ms.

## 2026-09-26 (c) — does shared movement drive the cross-area CCA? (spatial + temporal arms)

Four video animals (18 animal-pairs, cca cohort entries; 1206 is a non-learner).
Variants of the held-out CC1 per epoch:
- plain `prepare_pair`;
- VR confound (|speed|, lick);
- VR + video confound (4 ROI ME + 10 face-SVD components);
- control: the VR + video block circularly shifted by half the session.
The confounds are regressed out of the residual neuron tensors before PCA
(new `pipeline.prepare_pair_confounded`, tested on synthetic data). Temporal
arm: 20 ms bins, confounds at lags ±200 ms. Spatial arm: 5 cm bins, no lags.
The effect is (control CC1 − confounded CC1), so the drop from regressing out
unrelated signals of the same dimensionality is subtracted.

- **C1 (temporal):** VR + video lowers CC1 beyond the control by ≥ 0.05 in
  ≥ half of animal-pairs. ~50%. The 20 ms encoding flags 70% of units in 1206,
  so movement is a common input at this timescale.
- **C2 (spatial):** the median drop beyond the control is < 0.03. ~65%. Per-unit
  movement ΔR² is < 0.01 at 5 cm.
- **C3:** no naive → expert CC1 change appears or disappears after movement
  removal. That is judged by sign consistency only: 3 learners, and no test
  can be run. ~80%.
- **Falsifier for "movement is a negligible confound for the CCA":** C1 true.
- **Outcome (same day, `video/scripts/run_cca_movement.py`, `figures/cca_movement.png`).**
  The first run's single shifted control showed the problem: regressing out 16
  UNRELATED signals raised spatial CC1 by a median 0.027. That is a leak, now
  reproduced on synthetic data: `partial_out_tensor` fits on ALL epoch samples,
  test folds included, so the projection mixes training rows into test rows.
  With no true coupling, 16 random confounds lift held-out CC1 from 0.024 to
  0.046. The bias fades with real coupling (+0.005 at CC ~0.27, ~0 at 0.8).
  The final run therefore uses 10 shifted controls of the SAME confound per
  animal-pair-epoch.
  - Real CC1 falls below all 10 controls in 0.31 / 0.20 of cases in the
    spatial arm (VR / VR + video) and 0.33 / 0.37 in the temporal arm. Chance
    is 0.09.
  - The median drop vs the control median is +0.005 / +0.011 (spatial) and
    +0.001 / +0.006 (temporal). A drop ≥ 0.05 happens in 4–20% of cases.
  - **C1 falsified** (temporal ≥ 0.05 in only 7%). **C2 confirmed** (spatial
    median 0.011). **C3 confirmed**: expert − naive CC1 over 15 learner
    animal-pairs has a median of +0.040 → +0.029 in the spatial arm with 3/15
    sign flips, and +0.006 → +0.008 in the temporal arm with 0 flips.
  - **Reading:** movement is a real but pair-specific confound. Most coupling
    survives it, and a few pairs lose a lot: 1106 DMS–DLS spatial intermediate
    0.82 → 0.52 with VR alone; 1206 CA1–V1 temporal expert 0.24 → 0.09.
  - n = 4 animals, 3 of them learners: no test.

## 2026-09-26 — how much single-unit activity does movement explain beyond position? (video + VR, 4 task mice)

Per unit, cross-validated (5 folds by trial) ridge on FR per (trial, 5 cm bin).
Position model = bin one-hot. Movement = |VR speed|, and ME in the wheel,
whisker, mouth and spout ROIs, plus lick fraction. ΔR² = R²(pos+mov) −
R²(pos). Null: the movement block is permuted across trials within bin (50
shuffles); a unit counts as modulated if ΔR² > the 95th percentile of its null.
Areas DMS, DLS, ACC, V1 and CA1, FS excluded (cca config). n = 4 animals:
results are per animal × area, and no pooled p.

- **M1 (striatum is movement-modulated):** in DMS and DLS, ≥ 20% of units are
  modulated in ≥ 3/4 animals, and the median ΔR² is > 0. ~70%. This rests on the
  striatal locomotion literature.
- **M2 (small unique share):** the median ΔR² is below 0.05 in every area. The
  position one-hot already carries each trial's average speed profile, so
  movement adds only trial-to-trial deviations. ~70%.
- **M3 (no learning change):** the held-out ΔR², Expert − Intermediate, is not
  sign-consistent across animals in any area. ~65%.
- **Falsifier for "movement is a nuisance worth regressing":** < 10% of units
  are modulated in every area. Then movement regression can be skipped
  downstream.
- **Outcome (same day), after a correction.** The first run used an
  exchangeable within-bin trial shuffle with no drift terms. It flagged 50–94%
  of units with median ΔR² ~0.003, which is shared slow drift. A synthetic test
  now reproduces that false positive. The final design puts drift terms in both
  models and uses a whole-trial circular-shift null. Its empirical
  false-positive rate on the real data, with each shift treated as if it were
  the real alignment, is 6.7–7.0% against a nominal 5%.
  - **M1 half-right.** DMS is ≥ 20% modulated in 4/4 animals (24 / 84 / 30 /
    60%), but its median ΔR² is > 0 in only 2/4. DLS is untestable
    (2 animals).
  - **M2 falsified narrowly.** The median ΔR² is < 0.01 in 14 of 15
    area × animal cells. 1106 DMS is 0.060, and 1106 is also the animal with
    the fewest usable trials (79).
  - **M3 confirmed.** The held-out Expert − Intermediate ΔR² is ~0 and
    sign-inconsistent in every area.
  - Video beyond VR speed + licks: 0–20% modulated in most cells (null ~7%).
    The exceptions are 1106 DMS 79%, 1206 CA1 50%, 1106 DLS 47% and V1 ~40% in
    1105/1206. The median ΔR² is ≤ 0.0035 everywhere.
  - **Follow-up prior, face motion SVD** (top 10 components over the face ROI,
    numpy implementation of Stringer et al. 2019), same design, contrast =
    beyond VR speed + licks. **F1:** it flags more units than the 4 ME ROIs
    (`video_beyond_vr`) in ≥ 3/4 animals, pooled over areas. ~60%. **F2:** the
    median ΔR² is still < 0.01 in every area × animal cell. ~70%.
    **Outcome: F1 falsified, F2 confirmed.** Units flagged, pooled over areas,
    4 ROIs vs SVD: 1105 19 vs 26%, 1106 52 vs 52%, 1201 12 vs 12%, 1206 40 vs
    41%. The richer representation adds nothing over the 4 boxes. The median
    ΔR² is ≤ 0.008 everywhere.
    - What the components are (`figures/motion_svd_masks.png`): SVD 1 (50–68%
      of variance) is global motion including the wheel flanks. The next ones
      are left whisker pad / snout / paw patterns, plus one mouth–spout
      component per animal.
    - What was NOT tested: frame-rate (33 ms) encoding. 5 cm bins average over
      ≥ 250 ms and wash out fast whisking and licking.
  - **Reading:** movement modulation is widespread, reliable and tiny per unit.
    The video adds little beyond VR speed and licks. Not checked: SHARED
    movement drive at the population level. Small per-unit effects that are
    common across units could still inflate inter-area correlation or CCA.

## 2026-09-25 (c) — face motion vs learning, speed-controlled (video, 4 task mice)

Data: `video/results/<id>_binned.npz` for 1105, 1106, 1201 and 1206, as raw
trial × 5 cm bins. Speed model: ME ~ smooth function of |VR speed| per animal,
fit on usable trials OUTSIDE the compared epochs. Contrast: Expert − Naive in
the speed residual, per zone (pre-reward 75–125 cm; reward zone 125–169 cm).
n = 4 animals, so the floor is p = 0.125 and I report direction per animal,
not p-values.

- **R1 (the ROI is valid):** the speed-residual mouth ME correlates positively
  with licks per bin within animal, with partial r > 0.1 in ≥ 3/4 animals. ~55%.
  The raw frame-level r with licks was only 0.08–0.23.
- **R2 (anticipation):** the Expert − Naive speed-residual mouth ME in the
  pre-reward zone is > 0 in ≥ 3/4 animals. ~50%. The only hint is 1201's
  profile.
- **R3 (the drift confound is real):** the still-frame ME floor differs between
  the first and last thirds of a session by more than 10% in ≥ 2/4 animals.
  ~50%. If so, Naive vs Expert is uninterpretable, and Intermediate vs Expert
  (adjacent in time) is the contrast to read.
- **Falsifier for "face ME tracks learning":** the residual Expert − Naive
  difference is inconsistent in sign across animals, or vanishes in
  Intermediate vs Expert.
- **Outcome (same day, `scripts/run_face_learning.py`):**
  - **R1 falsified.** Partial r(mouth ME, lick fraction | speed) =
    −0.01 / +0.04 / +0.08 / +0.13 (1105/1106/1201/1206); 1/4 > 0.1. The
    hand-drawn mouth box does not measure licking.
  - **R2 falsified, and the falsifier fired.** Pre-reward speed-residual mouth
    ME, Expert − Naive, is −0.54 / −0.32 / +2.04 / +1.61 fit-set SD, 2/4
    positive. Expert − Intermediate is −0.45 / +0.41 / +0.60 / +0.37.
  - **R3 untestable as designed.** These mice are almost never still: 2–7% of
    frames have VR velocity 0. So 3/4 animals have no per-trial still floor in
    early trials. In 1206, where it exists, the whisker floor falls 31% from
    the first to the last third of the session, so drift is real there.
  - **Lesson:** validate the measurement (does the ROI see the behaviour?)
    before contrasting it across epochs. Next: a data-driven lick map
    (lick vs lick-free frame differences, speed-stratified).
- **Lick map (`figures/lick_maps.png`, 1206 + 1201):** lick-specific motion is
  1.5–3 grey levels, against ~20 for running. It sits on the SPOUT, meaning the
  spout block and tube edges, where the running map is ≈ 0. It does not sit in
  the mouth box.
- **Prior for a spout ROI** (x 250–370, y 295–525), chosen on 1201/1206 only:
  **S1:** partial r(spout ME, lick fraction | speed) per bin > 0.3 in all four
  animals, including held-out 1105/1106. ~65%. **S2:** raw r(spout ME, VR
  speed) per bin is below 0.2 in magnitude in all four. ~60%.
- **Outcome (2026-09-26): S1 and S2 are both falsified.**
  - Partial r(spout ME, lick fraction | speed) = +0.07 / +0.12 / +0.38 / +0.21
    (1105/1106/1201/1206). It is weakest in the two held-out animals. Raw r with
    licks is 0.16–0.46.
  - r(spout ME, speed) = −0.31 / −0.40 / −0.29 / +0.14. Licking happens when
    slow, so a lick signal inherits speed.
  - The pre-reward spout contrast Expert − Naive is positive in 4/4 animals
    (+0.12 / +0.29 / +1.16 / +2.76 SD), but Expert − Intermediate is not
    (−0.39 / −0.16 / +0.18 / −0.39). So Naive → Expert does not survive the
    time-adjacent contrast, and the floor at n = 4 is p = 0.125. **No claim.**
  - The ROI was placed on 2 animals and fails on the 2 held out. Its apparent
    fit on 1201 is partly selection.
  - **Lesson:** a box drawn on a pixel map from two animals does not transfer.
    And the VR lick sensor already measures licking, so a video lick readout
    adds nothing. The video's value is what the sensor can't see.

## 2026-09-25 (b) — the full LFP re-run on the fixed pipeline (branch lfp-simplify)

Registered before `regen_chain.sh` + `run_lfp_pipeline.sh`. Everything is now
DP-clipped, Naive = good trials 1-10, exact permutation tests with floors.

- **P1 (pre-registered primary):** DLS theta, task vs control, Δlog10 power
  Naive→Expert: p_FDR > 0.05. ~80% (it needed the 4-10 baseline and no DP rule).
- **P2:** decoding beats its null (sign-flip, BH) in most striatal/ACC area×band
  cells of the task cohort: ≥ 9/15. ~70%.
- **P3:** task > control split-half reliability persists after the within-trial
  speed slope is removed, in fewer cells than raw. ~55% (single-animal evidence
  argued against "purely behavioural").
- **P4:** PSI with vertical bipolar pairs shows no consistent direction across
  animals in any bipolar pair×band cell (BH). ~60%.
- **P5:** PAC observed significant-cell rate exceeds its re-paired-trial null
  rate in every measure×reference group. ~80%.
- **P6:** no Naive→Expert change survives BH in the primary log-power evolution
  stats for the task cohort. ~65%.
- **Falsifier for the whole fix:** validation shows an LFP-good trial MATLAB
  dropped, or <99 % of bin spans within 1 ms, in any file.
- **Outcome (2026-09-26): 5 of 6 held; P6 FALSIFIED; falsifier did not fire.**
  Validation 29/29. P1 held: DLS theta Δlog task −0.060 vs control +0.012,
  p = 0.051, p_FDR = 0.58; 0/30 primary cells. P2 held: 15/15 striatal/ACC
  decoding cells beat the null. P3 held: reliability task > control 15/30 raw,
  13/30 after the within-trial speed slope. P4 held: 0/24 bipolar direction cells
  (monopolar 1/24). P5 held: PAC 66-78 % observed vs 2-6 % calibrated null in every
  group. **P6 wrong:** two within-task cells survive (DLS theta −0.060, p_FDR 0.029;
  DMS beta +0.029, p_FDR 0.018). What I missed: I anchored on the group contrast
  failing and carried that to the within-task test, which has 11-16 animals and a
  real floor; the DLS theta decline is there within task animals -- it is the
  CONTROL comparison that is unpowered (n ≤ 5, 0/30 reachable). Lesson: "doesn't
  differ from control" and "doesn't change" are different predictions; register
  them separately.

## 2026-09-25 — which clock the control probe-2 (visual) bundles are on

The control V1 bundles' `VR_times_synched` differ from probe 1's by 6-48 ms (513)
and 116-159 ms (817), drifting over the session; task bundles are identical.
Python bins probe-2 LFP with the V1 bundle's times; MATLAB crops probe-2 units
with probe 1's. Test: V1-probe MUA (and LFP) locked to corridor onset, aligned
once with each bundle's VR times; task 1105 (bundles identical) is the reference.

- **P1 (each bundle is synced to its own probe):** with the V1 bundle's times the
  onset-response latency in 513/515/817 matches task 1105's within ±10 ms; with
  probe 1's times 817 is off by roughly its 116-159 ms offset. ~70%.
- **P2 (the LFP shares probe 2's spike clock):** probe-2 LFP high-gamma envelope
  vs probe-2 MUA peaks at lag 0 ± 2 ms. ~85%.
- **Falsifier:** latency matches 1105 with probe 1's times and not with the V1
  bundle's, or no onset response clear enough to time (then use the lag scan
  against a probe-1 signal instead).
- **Outcome (same day): P2 CONFIRMED, P1 HALF-RIGHT.** `scripts/audit_probe2_clock.py`.
  LFP→MUA lag is 0 ± 3 ms in every window of every animal while the bundle
  offset drifts (817: 0,0,1,0,−1,−1 ms against offsets 116→158 ms), so the probe-2
  LFP is on probe 2's clock. With the V1 bundle's times all three controls show
  the same sharp onset transient (half-rise 106-116 ms, peak 113-128 ms); with
  probe 1's times 817's response is late (227/246 ms) and smeared, since the
  drifting offset blurs it. So the V1 bundle's times are right for probe-2 data
  and **MATLAB's crop of control probe-2 units with probe 1's times is the
  misalignment.** The part that failed: 1105 is no latency reference (half-rise
  16 ms, a weak response from 53 units) — the controls agree with each other,
  not with it. Lesson: pick a reference with a response as sharp as the thing
  being timed; the across-control consistency was the diagnostic, not 1105.

## 2026-09-25 — top-camera video ↔ VR alignment, task 1201 and 1206

There is no camera sync line. The video clock comes from filename timestamps and
a frame rate estimated as part-1 frames / (part-2 start − part-1 start). The
clock link is behavioural: wheel-ROI motion energy against VR |velocity|.

- **P1 (the wheel carries the clock):** the whole-session peak Pearson r between
  wheel motion energy and |VR velocity| is ≥ 0.5, at a lag within ±90 s of the
  filename prior (video start → VR start: 1201 ≈ 645 s, 1206 ≈ 642 s; the VR
  filename has minute precision only). ~75%.
- **P2 (the nominal rate is close):** 5-min windowed lags fit a line with
  |drift| < 0.5% and residuals < 0.2 s. There is no step at the part-1/part-2
  boundary larger than 0.5 s. ~60%. The per-session rates of 25–33 fps make
  me unsure whether the camera drops frames.
- **P3 (independent check):** the mouth ROI's motion energy against VR licks
  peaks at the same lag as the wheel's, within 0.2 s. ~65%. The mouth ROI also
  sees paw and head motion, which is correlated with running.
- **Falsifier:** the wheel peak r < 0.3, or the mouth and wheel lags disagreeing
  by more than 1 s, means the clock is not trustworthy. Stop and look for a
  visible sync event before going further.
- **Outcome (same day): FALSIFIED.** Wheel peak r = 0.20 (1201) and 0.31 (1206).
  The mouth lag disagrees with the wheel lag by 11 s (1201) and 97 s (1206). In
  1206, 12 of 23 windows pass r ≥ 0.3, and their lags scatter with SD 9 s.
  Diagnostics: the wheel motion energy saturates, so it is a moving/stopped
  signal, not speed (1201: median 12 of a maximum of ~19). VR velocity does
  track the wheel in both worlds: corr(dx, v·dt) = 0.88–0.89, and it is not
  zeroed in world 12. On the filename clock, 1206 shows the wheel moving in the
  video about 90 s before VR registers movement, and the two bouts differ in
  length (5.7 vs 4.7 min). That is not a single shift. **Lesson:** frame-difference
  energy on a textured wheel cannot carry a sub-second clock. What remains open
  is whether the video clock itself is non-uniform (dropped frames).
- **Follow-up prior (registered before the speed run):** signed wheel speed
  from phase correlation. A 6-min 1206 snippet gave clean signed bouts, with
  video running starting ~89 s before VR on the filename clock.
  **Q1:** the whole-session continuous r ≥ 0.5 in both sessions. ~55%.
  **Q2:** event matching pairs ≥ 70% of VR onsets, with residual SD < 0.3 s,
  and agrees with the continuous fit within 0.5 s. ~50%.
  **Q3:** the windowed and event residuals show a step or curvature larger
  than 1 s, i.e. the video clock is non-uniform. ~40%. If Q3 is true, a
  piecewise clock is needed.
- **Outcome: Q1 and Q2 were falsified, and the premise behind all three was
  wrong.** Continuous r = 0.26 (1206) and 0.15 (1201). Events matched 9/67 and
  13/101, and onset patterns matched no better than a shuffled-interval null at
  any lag. **Why:** video frame count = VR row count exactly (1201: 246,826;
  1206: 236,074). The camera is triggered once per VR frame, so frame i = VR row i,
  and the filename start times and "fps" are meaningless. At frame shift 0, video
  wheel speed vs VR velocity gives r = 0.98 (1206) and 0.95–0.97 on frames with
  phase-correlation response > 0.5 (1201). The best shift stays within 0–1 frame
  in all 12 blocks of 1206. Every behavioural lag I estimated was fitting a clock
  that doesn't exist. **Lesson:** check structural invariants (sample counts)
  before building a statistical clock.

## 2026-08-12 — spatial CCA rerun on the corrected 5 cm cache

First spatial-arm run since (a) the 5 cm standardisation, (b) the depth fix,
(c) LP seeding, and (d) today's cell-type fix. (d) matters most here: the
pipeline excludes fast-spiking units by `FS_TYPE_CODE = 2`, and before today
ACC carried striatal-criteria FS labels while V1/CA1/DG had **no type at
all**, so no unit was excluded in those areas. The FS-excluded cohort
therefore changes for every non-striatal area.

- **P1 (striatal triangle is stable):** DMS-DLS and DLS-ACC per-animal
  held-out CC1 land within ±0.03 of the recorded committed-config values
  (DMS-DLS ≈0.14–0.18, DLS-ACC ≈0.09–0.11). DMS/DLS unit sets are untouched
  by the fix; only ACC's changes. Confidence ~70%.
- **P2 (ACC pairs shift most):** DMS-ACC changes more than DMS-DLS does,
  because ACC now excludes 76 FS units instead of the ~26 the striatal rule
  picked. Direction unsigned — excluding more units cuts n but removes
  broad-waveform contamination. Confidence ~65%.
- **P3 (the null survives):** no pair shows a BH-surviving naive→expert
  change in held-out CC at the animal level. This is the load-bearing one —
  it has held across 1056 sweep configs, the tcca 8-config grid and the
  committed-config ANOVAs. Confidence ~85%.
- **P4 (CC>0 survives):** every striatal-triangle pair stays significantly
  above zero at the animal level in all three epochs. Confidence ~85%.
- **P5 (V1/CA1 stay anecdotal):** V1 pairs keep n≤3 learners and CA1 pairs
  n≤2 after the new FS exclusion, so they remain uninferable. ~90%.
- **Falsifier:** any striatal pair with a BH-surviving epoch effect on
  held-out CC → the learning-change claim revives and Fig 4B is back.

**Outcome (2026-08-12, same day) — ✗ P1 wrong (CC rose systematically);
~P2 directionally right; ✓ P3; ✓ P4; ✓ P5. Falsifier not triggered.**

P1: **wrong, and in an informative way.** Every pair's per-animal held-out CC1
came out ABOVE the recorded committed-config values, by more than the ±0.03 I
allowed: DMS-DLS 0.200/0.184/0.191 (was 0.154/0.140/0.180), DMS-ACC
0.172/0.143/0.164 (was 0.120/0.096/0.115), DLS-ACC 0.132/0.132/0.138 (was
0.107/0.094/0.093) [naive/intermediate/expert]. Uplift +0.01 to +0.05, most
consistent for DMS-ACC (+0.05 in all three epochs). I predicted stability
because I was thinking of the FS-exclusion change as the only difference and
treated the binning change as neutral — but the recorded values came from the
**2.5 cm** committed config, and 5 cm has better per-bin SNR (the same effect
that won the bin-size comparison: +0.17 split-half reliability). Lesson: when
several things changed at once, predict against the change with the largest
known effect size, not the one most recently on my mind.

P2: DMS-ACC did shift most (+0.049 mean), consistent with ACC's FS set
changing from ~26 striatal-criteria units to 76 — but since every pair rose,
the binning is the better explanation for most of the movement, and the ACC
contribution is not separable in this run.

P3 ✓ **the load-bearing null holds**: animal-level rm-ANOVA on held-out CC is
n.s. for every striatal pair — DMS-DLS p=0.854, DMS-ACC p=0.450, DLS-ACC
p=0.904 (dimension-level likewise 0.92/0.51/0.58). Nothing to correct.

P4 ✓ CC>0 at the animal level in all three epochs for all three pairs
(p=0.016 worst, 4e-05 best).

P5 ✓ V1-ACC n=3, V1-DMS n=2, CA1-* n=1, V1-DLS/CA1-DLS n=0 — still
uninferable. DLS-ACC gained one animal (4→5).

**Net:** the spatial arm now agrees with the temporal arm on both headline
claims — communication is reliably present and does not change with learning —
and the magnitudes are ~30% higher than previously recorded because the arm
finally runs on the bin size the project chose on evidence.

## 2026-08-11 — tcca epoch grid: bin {25,10 ms} × FS {excl,incl} × {partial,plain} + IFI integration-window sweep

Eight run_epochs configs on the seeded 5 cm cache (A7's LP is 33 vs the recorded
34; the other 10 learner LPs are identical — smoke run reproduced A1's nine cc1
values to 4 d.p.). New per-cell exports: CC1 lag curves to ±250 ms
(epoch_lagcurves*, IFI recomputable at any window via `lagged.ifi_by_window`)
and the partner-dependent `gini_pearson_x/y`. Registered before any cohort run.

- **P1 (strength null is config-robust):** paired per-animal Wilcoxon on cc1
  naive→expert, striatal triangle, all 8 configs (24 tests): 0–2 nominal
  p<0.05, none surviving BH within its config. Confidence ~80%. Basis: the
  spatial sweep sat at chance, the 25 ms temporal null, and Tom's threefold
  strength null.
- **P2 (FS inclusion raises coupling, changes no verdict):** cc1(FS-incl) >
  cc1(FS-excl) in ≥70% of matched (animal, pair, epoch) cells at 25 ms partial.
  Confidence ~75%. Basis: Tom's hierarchy uniformly higher FS-incl; spatial
  FS-incl/excl agreement r=0.89.
- **P3 (plain − partial isolates shared drive):** plain cc1 > partial cc1 in
  ≥80% of matched cells, median uplift ≥ +0.05 (popsim: common-input 0.74→0.19
  under partialling). The epoch-strength null persists even in plain.
  Confidence: uplift ~85%; null-persists ~65% — this is the arm most likely to
  produce a (spurious-looking) epoch effect, since shared drive tracks
  behavioural state.
- **P4 (IFI ≈ 0 at every integration window):** per-animal IFI(w), w up to
  ±250 ms: (a) no striatal-triangle (pair × window) cell survives BH across
  windows within pair in any config (~70%); (b) session-pooled existence-level
  IFI vs 0 at ±50 ms also null (~75% — the 25 ms run already gave IFI 0.000 ±
  0.09), unlike Tom's CA1→RSC where a flow exists.
- **P5 (10 ms is noisier, verdicts unchanged):** per-cell cc1 magnitudes lower
  at 10 ms in the majority of matched cells; no cohort verdict flips. ~70%.
- **Falsifiers:** (i) any pair with a BH-surviving, same-sign IFI window band
  (≥2 contiguous windows) in BOTH FS conditions → a genuine directional flow —
  Fig 4c revives; (ii) any pair with same-sign p<0.05 naive→expert cc1 change
  in ≥3/8 configs → the strength-change story revives.

**Outcome (2026-08-11, same day) — ✓ P1; ✗ P2; ✗ P3 REVERSED (the registered
surprise); ✓ P4 with one single-config footnote; ✓ P5. Neither falsifier
triggered.** Full tables: `tcca/results/grid_summary.csv`,
`grid_ifi_windows.csv` (script `scripts/analyze_epoch_grid.py`; animals-as-n,
BH within family).

P1: 24 pair×config tests, exactly 1 nominal hit (b25/fsincl/plain DLS-ACC
p=0.039), 0 survive BH. The strength null is robust to bin size, FS condition
and partialling.

P2: FS inclusion raises cc1 in only **54%** of matched cells in the partial
frame (p_animal 0.43 b25 / 0.11 b10) — prediction wrong. The predicted uplift
exists only in the PLAIN frame (65% p=0.039 b25; 78% p=0.004 b10). Reading: FS
units contribute mostly **shared** variance, which partialling removes — Tom's
"uniformly higher FS-incl" does not transfer to the partial striatal pipeline.

P3: **REVERSED.** plain − partial is *negative* FS-excluded (median −0.012
b25, p_animal=0.016; −0.031 b10, p=0.039; only ~⅓ of cells positive) and ≈0
FS-included. I anchored 85% confidence on popsim's common-input scenario;
these data are not that scenario. Lesson: the triangle's coupling is **not
inherited from shared drive off the other recorded areas** — partialling acts
as *denoising* (without residualisation, PCA-k spends components on
high-variance global directions and crowds out the coupling-carrying ones).
This is a positive, citable statement: the coupling is pair-specific.
Null-persists sub-prediction ✓ (all plain configs n.s.).

P4: learning-change: **0/420** (pair × window × config) cells BH-survive.
Existence: null everywhere except one config-island — b10/fsincl/plain
DLS-ACC, contiguous ±200–240 ms band, median IFI +0.02, p=0.0078. The
falsifier required the band in BOTH FS conditions; it is absent FS-excluded
and absent under partialling → not triggered. Parsimonious read: a small
shared-drive asymmetry visible only in the least-controlled config.

P5: b10 − b25 cc1 negative in all four frames (medians −0.034…−0.040, 20–31%
of cells positive, p_animal ≤ 0.02); no verdict flips.

Bonus (the audit's §6 fix, first use): the partner-DEPENDENT
`gini_pearson_x/y` is **also flat across learning** (b25 committed config:
x p=0.73, y p=0.30; cohort medians 0.34–0.38) — the Fig 4d Gini null survives
the corrected metric, so the panel stays a null (or is dropped) either way.

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
