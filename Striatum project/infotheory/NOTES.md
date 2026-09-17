# striatum_info — running log (newest first)

## 2026-09-17 — MI arm of the Lemke/Panzeri mirror: no learning change survives correction

Design settled with Theo: **align to reward-zone entry**, and compute **every**
behavioural feature rather than choosing one (in the paper, which feature is best
encoded was itself a result).

**Result: nothing survives correction, and the yoked control moves the same way.**

| feature | Expert − Naive (bits) | N mice | p |
|---|---|---|---|
| lick_error_z | **+0.0042 ± 0.0014** | 15 | 0.007 |
| first_lick_position_au | +0.0022 ± 0.0013 | 16 | 0.105 |
| corridor_duration_ms | +0.0009 ± 0.0009 | 16 | 0.298 |
| mean_velocity_cm_s | +0.0007 ± 0.0008 | 16 | 0.252 |
| the other five | −0.0008 to +0.0004 | 16 | 0.40–0.98 |

**0/9 features survive BH-FDR at q = 0.05.** The lick-error result is the only
one below 0.05 uncorrected and it needs p ≤ 0.0056 to survive a family of nine.
And the yoked control shows **+0.0025** on the same feature, plus increases on
velocity at the reward zone, lick count and path length — so what change there is
looks like time in the apparatus rather than learning, the same reading the
gamma-rise and reliability results already forced.

**A second negative worth as much as the first: the information is not
time-locked to reward-zone entry.** Panel (a) is flat from −1000 to +500 ms in
every area. Lemke's kinematic information peaked 50–500 ms before their alignment
event; ours does not peak at all. Either these features are encoded tonically
across the traversal, or reward-zone entry is not the right event. That is a
result about the alignment choice and should be settled before the PID/TE arm is
built on top of it.

Ordering of information by area (epoch=All, pooled over features):
DMS ≈ 0.004 > DLS ≈ V1 ≈ 0.003 > ACC ≈ CA1 ≈ 0.0015 > DG ≈ 0.0008 bits.

### Three errors of mine, caught and fixed
1. **The peak test was circular.** I took each unit's peak MI over 30 time windows
   and tested it against a SINGLE window's null — the maximum of 30 draws beats a
   one-draw threshold far more than 5% of the time. It called 63–78% of cells
   significant. Replaced with a max-statistic permutation (each shuffle's own peak
   over windows, leave-one-out corrected), which gives 31%.
2. **One NaN killed a whole feature.** `zscored_lick_errors` is NaN on trial 1 for
   every animal, and an `isfinite().all()` guard dropped `lick_error_z` entirely
   from the first run — the feature that turned out to matter most.
3. **The project's 10-trial epochs cannot support an information estimate.** Every
   learning epoch (3, 7, 10, 10 trials) fell below the minimum, so the first run
   silently computed nothing but "All". Epochs are now count-matched thirds of the
   session — which is also closer to the paper, whose naive/skilled are the first
   and last DAYS, a time-based split rather than a performance-based one.

### Data traps fenced (also in the repo GOTCHAS)
- VR position rises during the dark inter-trial period, so a bare position
  threshold finds reward-zone entry **in the dark**: 523 trial 1 gives 3185 ms
  naive against 10478 ms corridor-restricted, corridor starting at 5001 ms.
- The per-trial cell arrays disagree in length: 1212 and 409 carry one more spike
  trial than behavioural trial. `n_trials` is authoritative.

### Code
`src/striatum_info/{estimators,trials}.py`, `scripts/{extract_trials,run_mi,plot_mi}.py`,
29 tests. Estimators cover MI (plug-in, shuffle-subtracted, Miller-Madow), conditional
MI, transfer entropy and an I_min PID — pinned on XOR, COPY and COMMON. `pid_imin`
is **not** BROJA, which the paper uses for two-source shared information.

**Not yet built:** the PID, TE and FIT arms between areas. Worth settling the
alignment question first. Note also that the cross-PROBE pairs (DMS↔V1, ACC↔CA1)
are the clean ones for any directed claim — within-shank LFP pairs carry the
shared field established on 2026-09-17.
