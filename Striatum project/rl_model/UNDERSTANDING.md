# UNDERSTANDING — Belief-state RL model of corridor behaviour

_Rolling design doc. Current understanding at top; edit log below._
_Created 2026-05-20. (Intended location `.claude/session/` is protected — kept with the model code.)_

## Goal

Fit a generative behavioural model to each task mouse so its internal latent
variables (value, RPE, perceptual precision, policy outputs) become per-trial
regressors against the simultaneously recorded striato-cingulate neural data.
This is the model behind paper Fig 1E (RL agent reproduces behaviour) and
Fig 4A/B (ensembles encode computational variables).

## Redesign — two-timescale actor-critic (agreed 2026-05-24, not yet implemented)

_Resolved via a grill-me design interview. This section is the current design
intent; the `Model framing` / `The agent` / `Fitting` sections below describe the
model **as built before this redesign** and are kept for reference until the
redesign lands._

### Problem

The fitted model collapses the within-session learning curve. The `value` and
`precision` latents are saturated within the first few trials, while the mice
take tens of trials to improve (see `figures/fig_diag_epoch_lick_profiles.png`
and `fig_diag_epoch_velocity_profiles.png`). Root cause: the model has exactly
one learning channel — the TD critic — driven by teacher-forced reward that is
available from trial 1 (mice lick in the RZ from the start, only diffusely). The
policy-gradient actor specified in the original `Model framing` was never
implemented: `agent.py` collapsed it to a static `z_lick = β(V − θ)` readout of
the critic. Perceptual precision has no across-trial dynamic at all (the Kalman
filter converges within trial 1). So nothing can carry slow behavioural
improvement.

### Behavioural target (from the data)

Per-epoch (Naive / Intermediate / Expert) profiles show that the RZ-anticipatory
lick peak and the RZ value are already present in the Naive epoch — reward
*location* is learned fast. What changes slowly is **action control**: (1)
misplaced licks collapse — pre-RZ, post-RZ overshoot, corridor-start; (2) the
corridor speeds up while the RZ slowdown sharpens. The slow phenomenon is
inhibitory / efficiency control of action, not appraisal.

### Architecture — fast appraisal, slow control

1. **Perception (fast).** Kalman belief over position — unchanged. `precision`
   (`1/σ`) is a fast latent.
2. **Critic `V(b)` (fast).** Reward-only TD: `δ_critic = r + γV′ − V`,
   `w_critic += η_w·δ_critic·b`. Fast `η_w` → a sharp RZ value bump within a few
   trials. The critic never sees the costs. `value` locks on fast.
3. **Lick actor (slow, NEW).** Spatial weight `w_lick`; the policy sets a lick
   *rate* `λ_rate = λ_max·σ(β·V + w_lick·b − θ)` and the per-bin count is
   `Poisson(λ_rate·dt)` — the value drive gives the early RZ peak, the actor
   term gives slow spatial gating, and `dt` couples licking to velocity (run
   fast → no time to lick). Policy-gradient updated by the actor RPE. Starts
   permissive (broad early licking) → learns to suppress licks where they go
   unrewarded.
4. **Velocity actor (slow, NEW).** Spatial weight `w_vel`;
   `log_v_mean = v_base + v_slope·V + w_vel·b`. Policy-gradient updated by the
   actor RPE. Made viable by *consequential speed*: path-integration noise
   scales with velocity (`Q_eff = Q·(1 + kappa_v·v)`), so fast running widens
   the belief and blurs the value read-out — RZ licking and reward become less
   reliable. The actor learns to slow approaching the RZ and run fast in the far
   corridor; `precision` consequently evolves across trials.
5. **Costs → actor only.** Per-lick cost `c_lick`; time cost `−ρ·dt` (`ρ` a fixed
   fitted parameter; `dt` is already computed in `bin_step`). Actor RPE:
   `δ_actor = (r − c_lick·lick − ρ·dt) + γV′ − V`. Because the critic is
   reward-only, the costs are never predicted away, so misplaced-action negative
   RPEs persist and fade only on the behavioural-learning timescale.
6. **Timescale asymmetry.** `η_a ≪ η_w`: appraisal fast, control slow. Both
   learning rates are fitted.

### Latents exported

`value` (fast), `RPE` = `δ_actor` (emergent — negative on misplaced licks / slow
corridor running, fading over learning), `precision` (fast), `lick_rate`,
`v_mean`, `belief_mean`, `sigma`, plus the new control fields `w_lick·b` and
`w_vel·b`.

### Resolved design decisions (the tree)

- The slow channel lives in the **actor**, not in value or perception —
  perception and reward-location are learned fast (the position decoder also
  improves rapidly).
- The `value` latent is **fast**; **`RPE` carries the within-session emergence**.
- The negative RPE on an unrewarded lick comes from a **per-lick cost kept
  outside the critic** — two learning signals: reward-only for `V`,
  reward-minus-cost for the actors.
- **Velocity is in scope** — it gets its own slow actor.
- The velocity cost is a **time / reward-rate cost** (`−ρ·dt`), not an effort
  cost — the data show the corridor *speeds up* over learning.

### Fitting

Per-mouse, teacher-forced, MAP, JAX autodiff + L-BFGS-B — as now. ~6 new
parameters (`c_lick`, `ρ`, `η_a`, `w_lick` init, `w_vel` init, actor gain) → ~19
total. The `η_a ≪ η_w` asymmetry and the permissive actor initialisations are
**fitted, not hard-coded**: teacher-forcing plus keeping the early epoch in the
objective gives MLE a reason to recover them. **Gate (CLAUDE.md TDD):** the
parameter-recovery test must generate synthetic mice with `η_a` and actor-init
spanning the slow learning-timescale range and confirm recovery *before* any
real-data fit. If recovery shows the slow timescale swallowed by the
post-learning bulk, epoch-weighting the likelihood is the registered remedy.

### Validation

The headline fit-quality check becomes **per-epoch** (Naive / Intermediate /
Expert, via `find_learning_points` + `epoch_indices`) observed-vs-predicted lick
**and** velocity profiles, per mouse, plus per-epoch held-out log-likelihood. The
model must reproduce the epoch-to-epoch *change*, not just the session average.
The session-averaged `_profile()` figures in `plot_real_data.py` are demoted — a
session average hides exactly the failure this redesign targets.

### Won't-Do (redesign scope)

- **Across-trial precision sharpening** — `precision` stays the fast Kalman
  latent. A fast across-trial sharpening is a model-comparison-ladder variant,
  pursued only if the flat `precision` latent proves inadequate as a regressor.
- **Learned reward-rate `ρ`** (`ρ += η_ρ(r − ρ·dt)`) — `ρ` kept fixed-fitted;
  learned-`ρ` is a ladder variant.
- **Effort cost `c_eff·v²`** — omitted (wrong-signed against the velocity data);
  ladder variant.
- **Separate `η_a` per actor** — start with one shared actor learning rate; split
  only if recovery demands it.
- **Hierarchical / partial-pooling fit** — still deferred; per-mouse first.
- **The neural regression itself** — downstream.

## Model framing

**Belief-state RL agent, fit per mouse by maximum likelihood** — the workhorse.
A rational agent with *subjective* parameters (sensory precision, costs, learning
rates) inferred from behaviour. "IRC in spirit", but with a *learning* value
function rather than an optimal one, so RPE and value genuinely emerge over
trials — matching the paper's within-session emergence narrative.

**Optimal-agent IRC benchmark** — a later, separate deliverable: an
optimal reward-rate-maximising agent that defines the normative optimum for
Fig 1D ("perfect knowledge produces the optimum"). Not the workhorse.

Rejected: classical Inverse RL (reward is known — water in the RZ — so there is
no reward function to infer); forward-only hand-tuned agent (no per-mouse fit,
cannot generate mouse-specific regressors).

## The agent: two processes, two timescales

1. **"Where am I?" — perception (fast).** A POMDP belief over corridor position.
   Sensory precision governs belief width; the belief sharpens fast across early
   trials. This is the paper's "rapid spatial-map stabilisation".
2. **"Given where I am, what's best?" — value/policy (slow).** An actor-critic.
   The critic `V(belief)` is TD-learned; the actor learns to withhold licking
   until the belief says the RZ is near and to trade running speed for accuracy,
   i.e. maximise rewards per unit time. This is the "slow emergence of RL
   variables".

No separate explicit reward-zone-location belief: `V` developing a bump near the
RZ *is* the agent learning where reward is. Dropping it removes a degeneracy.
The two processes load on distinct behavioural signatures — precision → spatial
**spread** of licking; value/policy → lick **timing & vigor** — so they are
identifiable.

### Environment
- Corridor 0–200 a.u.; spatial grid, `dx = 4 a.u.` → 50 bins (matches the neural
  `spatial_binned_fr` binning, so latents align 1:1 with `cells × bins × trials`).
- Visual landmark (VZ) at 80 a.u. (bin 20). Reward zone (RZ) 100–135 a.u.
  (bins 25–33.75). Reward delivered on the first lick inside the RZ per trial.
- Per trial the agent steps bin-by-bin; time in a bin `dt = dx / v`.
- Models the corridor period only (the dark pre-corridor period is excluded).

### Perceptual process (Kalman belief over position)
- Belief held as a grid distribution `b_t` over the 50 bins (Gaussian-shaped,
  summarised by mean `μ` and variance `Σ`).
- Predict: `μ_pred = μ + dx`, `Σ_pred = Σ + Q·dx`  (process / path-integration noise `Q`).
- Observe: visual position observation with noise `R(x) = R_slope·|x − VZ| + R_min`
  (precise near the landmark, vague far from it).
- Update: Kalman gain `K = Σ_pred/(Σ_pred+R)`; `μ, Σ` updated accordingly.
- Across trials `Σ` is carried over and inflated by `ITI_inflation`; the filter
  converges across early trials → the fast perceptual sharpening, parameter-light.
- Latent **precision** = `1/Σ` per trial × bin.

### Value / policy process (linear-function-approx actor-critic)
- Belief-state features = the belief vector `b_t`.
- Critic: `V(b) = w_val · b`; average-reward TD:
  `δ = r − ρ·dt + V(b') − V(b)`, `w_val += η_w·δ·b`, `ρ += η_ρ·(r − ρ·dt)`.
- Actor, two heads, policy-gradient updated by `δ`:
  - **Lick** — Bernoulli, `P_lick = σ(β · w_lick·b)`.
  - **Velocity** — Gaussian (fit in log-space, i.e. log-normal emission since
    `v > 0`), mean a function of `b`, std `σ_v`.
- Reward `r = +1` on first RZ lick; subjective costs: per-lick cost `c_lick`,
  effort cost `c_eff·v²`. Costs make the reward-rate tradeoff real.

### Free (subjective) parameters per mouse (~13)
Perceptual: `Q, R_slope, R_min, ITI_inflation, Σ0`.
Value/policy: `η_w, η_ρ, η_a, β, σ_v, ρ0, c_lick, c_eff`.
Which of these are actually recoverable is decided empirically by the
parameter-recovery test, not assumed.

## Latent variables exported (per trial × spatial bin)
`V(belief)`, `RPE / TD error δ`, `belief uncertainty / precision (1/Σ)`,
plus policy outputs (`P_lick`, expected velocity, belief mean, advantage,
expected reward-rate `ρ`). Written aligned to `spatial_binned_fr`.

## Fitting
- **Per mouse, maximum likelihood.** Teacher-forced forward pass: the agent is
  run through all trials, learning is driven by the mouse's *observed* actions;
  the parameters are the subjective constants.
- Per-timestep likelihood: `Bernoulli(lick | P_lick) · logNormal(velocity)`,
  summed over trials × bins.
- JAX autodiff → gradient-based optimiser.
- Validation: one-step-ahead held-out predictive log-likelihood (trial `t`
  predicted from history up to `t−1`).
- Model-comparison ladder (nested simpler models): drop precision sharpening,
  drop value learning, fully fixed agent — each component must earn its place
  by cross-validated likelihood.

## Animals / data
- Fit per-mouse on the **Neuropixels task mice** in `processed_data/
  preprocessed_data.mat` — the animals whose neurons are regressed in Fig 4.
- Later (not now): behaviour-only mice (`BehaviourOnly/`, 8 mice) as pipeline
  validation and for Fig 1/2 behavioural panels; blank-corridor controls as a
  negative control (expect no value emergence).

## Tooling / layout
- New self-contained Python package `Striatum project/rl_model/` (JAX for
  autodiff). Reads `.mat` via `mat73`/`h5py`; writes latents to `.mat` + `.npz`
  so existing MATLAB plotting / neural-regression code consumes them unchanged.
  Mirrors the existing CEBRA precedent (Python compute, MATLAB-friendly I/O).
- Synthetic / model-generated data lives under `rl_model/data/generated/`
  (segregated from real data, per CLAUDE.md).
- A repo-wide `src/ tests/ scripts/` restructure is **not** done — CLAUDE.md
  says flag structural changes, not perform them silently.

## This session's deliverable
1. This UNDERSTANDING.md.
2. The forward generative model in JAX (corridor env + two-process actor-critic).
3. A synthetic-data generator.
4. A **parameter-recovery test** — generate a synthetic cohort with known
   ground-truth parameters, fit, verify recovery — the TDD gate from CLAUDE.md.
   No real mouse data is touched until recovery passes.

## Open assumptions to confirm at plan review
- Velocity emission modelled as log-normal (since `v > 0`); alternative:
  truncated normal / Gamma.
- Belief held as a Gaussian-summarised grid; alternative: full grid filter.
- Teacher-forced fitting (learning driven by observed actions) — standard for
  RL behavioural fitting, but worth a conscious nod.
- Perceptual sharpening is mechanistic (Kalman convergence). An explicit
  *learned-precision* term is registered as a variant in the model-comparison
  ladder rather than in the base model.
- Corridor geometry (VZ 80, RZ 100–135) taken from `project_cfg.m` /
  `ProcessStriatumTask.m`; assumed fixed across trials and mice.

## Won't-Do (explicitly out of scope)
- Classical Inverse RL / reward-function inference — reward is known.
- Full optimal-POMDP IRC as the workhorse — only as a later normative benchmark.
- Hierarchical / partial-pooling fit across mice — later extension; per-mouse first.
- Fitting behaviour-only mice and controls — later; task mice first.
- Modelling the dark pre-corridor period.
- The neural regression itself (Fig 4 analysis) — this session delivers the
  latents; wiring them to neurons is downstream work.
- Any repo-wide directory restructure.

---

## Edit log

- **2026-05-24 (velocity v3 — real fit + validation)** — Refitted all 16 mice
  with the graded-reward + deterministic-velocity-actor model (`real_fits_v5`)
  and re-ran the per-epoch validation. **Lick channel much improved**: held-out
  CV gain +0.57 nats/bin, **16/16 mice positive** (old model +0.12, 10/16); the
  per-epoch lick change is reproduced for **11/16 mice** (mean Δr 0.70).
  **Velocity improved but still the weaker channel**: per-epoch velocity change
  reproduced for 6/16 mice (mean Δr 0.37, up from 0.28) — the model now produces
  an RZ dip that partially tracks the data instead of flat noise; held-out
  velocity CV is still marginally negative (−0.07). Recovery v6 confirmed the
  parameters (`eta_a` 0.89, `rho` 0.76; perceptual params remain sloppy,
  `kappa_v` now unidentified — a drop candidate). Open: `plot_real_data.py`
  latent export still on v3 — needs the v5 update for the neural regression.

- **2026-05-24 (velocity v2)** — Strengthened the velocity channel. Licking is
  now a *time-limited* Poisson process: the policy sets a lick rate (licks/s)
  and the per-bin count is `Poisson(λ_rate·dt)`. Running fast shrinks `dt` and
  so the lick count, so the agent must slow in the RZ to emit a lick and collect
  reward — velocity now genuinely trades off against licking, and `δ_actor`
  carries a strong reward-backed velocity signal (the `rho·dt` term is the
  reward-rate opportunity cost — "rewards per fixed time"). `lambda_max`
  reinterpreted as a rate (typical 3 → 10).

- **2026-05-24 (velocity v2b)** — Time-limited licking alone did not make
  velocity consequential: with a 9-bin RZ and a binary "first lick" reward the
  agent collects the water at any speed (~20+ expected RZ licks), so it had no
  reason to slow. Reward changed to **graded and saturating**:
  `r_total = reward_magnitude·(1 − exp(−cumulative_RZ_licks / K_REWARD))`,
  `K_REWARD = 10` fixed; the per-bin reward is that total's increment. Running
  fast emits fewer RZ licks and collects less of the drop — a smooth
  velocity/lick/reward-rate trade-off with an interior optimum. `rho` typical
  lowered (0.40 → 0.15) for the new regime.

- **2026-05-24 (velocity v2c)** — Graded reward alone still left the velocity
  actor inert: its *stochastic* policy gradient keys on the covariance of small
  velocity exploration noise with small per-bin outcomes — negligible. That was
  the real bottleneck behind every failed velocity attempt. Replaced with a
  **deterministic gradient**: `w_vel += eta_a·g_vel·b`, `g_vel = ∂δ_actor/∂logv`
  computed analytically (positive in the corridor — faster saves time cost;
  negative in the RZ — faster loses graded reward). Also re-pinned `C_LICK`
  0.15 → 0.025 (kept below the marginal RZ reward `R_max/K_REWARD` so RZ licking
  stays net-positive). Generative sanity: the velocity actor now works — corridor
  speeds up (~+12 cm/s over a session), the RZ slowdown emerges (dip +1 → +9
  cm/s), `a_vel` develops real spatial structure. Recovery re-run as v6, real
  refit as `real_fits_v5`.

- **2026-05-24 (real fit + validation)** — Fitted all 16 task mice with the
  redesigned model (`real_fits_v4/`, interleaved CV) and built the per-epoch
  validation (`scripts/plot_epoch_validation.py`, figures
  `fig_epoch_validation_{lick,vel}.png`). **The redesign meets its goal on the
  lick channel**: the model reproduces the Naive→Expert lick-profile change for
  10/16 mice (mean Δr = 0.66 between the model's epoch-change and the observed
  one) — the old model had frozen latents and could show no epoch structure at
  all. M13 needed a multi-restart refit (`n_restarts=1` had landed in a bad
  basin, nll 26027 → 14240); M01/M07 spot-checked and already optimal, so the
  remaining moderate-Δr mice (M01/M02/M05/M07) are model limitations, not fit
  failures. Velocity remains the weak channel (5/16, Δr 0.28) — the velocity
  actor is structurally weak, as flagged. Held-out interleaved CV is ~on par
  with the old model, but interleaved CV barely tests learning-curve structure;
  the per-epoch comparison is the intended metric. `fitting.py` gained
  `maxiter` / `u_start` knobs; `fit_real_data.py` gained `n_restarts` / refit
  support. Open: `plot_real_data.py` (latent export to `rl_latents.mat`) still
  points at v3 — needs the v4 path/version update before the neural regression.

- **2026-05-24 (recovery)** — Parameter-recovery test (v4, 12 synthetic mice,
  120-trial sessions). Latents recover well (value/lick_rate/v_mean r > 0.99,
  precision 0.96, RPE 0.90) and most parameters recover; but `eta_a` was
  non-identifiable (r = −0.17) — it degenerates with `c_lick`, the product
  `eta_a·c_lick` recovering at r = 0.95. Fix: `c_lick` pinned to the fixed
  constant `C_LICK = 0.15`; `eta_a` carries the suppression-speed degree of
  freedom and becomes identifiable. 16 free parameters. `rho` / `kappa_v`
  remain weakly identified, like the perceptual parameters — not blockers, the
  latents recover regardless. Recovery re-run as v5.

- **2026-05-24 (build)** — Redesign implemented in `config.py` / `agent.py`.
  Two-timescale actor-critic: fast reward-only critic, slow lick-suppression
  actor (reward-modulated three-factor rule), slow velocity actor (policy
  gradient).  Implementation surfaced that speed is inconsequential in a
  bin-grid corridor, so the velocity actor had nothing to optimise; resolved by
  making speed consequential — path-integration noise scales with speed
  (`Q_eff = Q·(1 + kappa_v·v)`, new param `kappa_v`; 17 params total), a
  speed/accuracy trade-off that also makes the `precision` latent evolve across
  trials.  Lick side (fast value lock-on + slow suppression emergence) verified
  on simulated data; velocity side re-checked after the `kappa_v` change.

- **2026-05-24** — Redesign agreed (not yet implemented). Diagnosis: the fitted
  model collapses the within-session learning curve — `value`/`precision` latents
  saturate within a few trials while behaviour improves over tens of trials. Root
  cause: a single fast learning channel (the TD critic) and no actor (the doc's
  policy-gradient actor was never implemented — collapsed to a static `β(V−θ)`
  readout). Redesign to a two-timescale actor-critic: fast critic + fast Kalman
  belief, plus two NEW slow policy-gradient actors (lick-suppression,
  velocity-efficiency) driven by an actor RPE carrying a per-lick cost and a
  `−ρ·dt` time cost that the reward-only critic never predicts away. Decisions
  resolved via a grill-me interview; full design in the `Redesign` section at the
  top. Validation moves to per-epoch (Naive/Intermediate/Expert)
  observed-vs-predicted profiles. Diagnostic figures:
  `figures/fig_diag_epoch_lick_profiles.png`, `fig_diag_epoch_velocity_profiles.png`.

- **2026-05-21** — Real-data fit. Model revised in three ways before fitting the
  16 real task mice: (1) **Poisson lick-count emission** replacing Bernoulli —
  real lick data are counts, not binary; adds `lambda_max` (saturation rate).
  (2) **Per-bin validity mask** so missing behavioural data (NaN licks,
  zero-occupancy bins) is excluded from the likelihood. (3) **`w_init`
  parameter** — a flat elevated initial critic value representing the value
  carried in from the prior random-location automatic-reward task; task-2
  learning carves it down except at the RZ. Sessions are **truncated at the
  behavioural disengagement point** (`change_point_mean`). 13 parameters.
  Re-validated by synthetic recovery (all latents r > 0.95); fit to all 16 mice
  with interleaved cross-validation — the model beats a saturated per-bin null
  on held-out licking for 10/16 mice (+0.12 nats/bin); velocity is the
  constrained channel. Latents exported to `results/rl_latents.mat`. Full
  writeup: ResearchVault `Methods/Belief-State RL Model.md`.
- **2026-05-20** — Initial version. Decisions resolved via grill-me interview:
  model framing (belief-state RL fit + IRC benchmark), joint lick+velocity
  policy with a continuous velocity actor, two-process two-timescale agent
  (perception fast / value slow, explicit RZ-belief dropped for identifiability),
  full latent set exported, per-mouse ML fitting, task mice only to begin with,
  Python+JAX tooling, session deliverable = plan + model + recovery test.
