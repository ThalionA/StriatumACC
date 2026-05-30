# popsim — working notes

Handoff/context notes for the simulated-population generator.

## Purpose
Ground-truth test-bed for cross-area communication analyses (pairs with the
sibling `striatum_cca` package). Three areas, 3–5 latents each, coupling defined
at the latent level so the answer is always known.

## Design decisions
- **Coupling lives at the latent level**, not the population level. The
  observation step (`project_population`) only adds private noise (and optional
  realism nuisance), so the communication subspace is exactly the column space of
  the source loadings mapped through `M`.
- **Lags are non-negative (causal).** A negative-lag relationship for an ordered
  pair `(X, Y)` is produced by adding the reverse edge `Y -> X`; this keeps the
  generator strictly causal and the cross-correlogram sign convention honest
  (`corr(x[t], y[t+k])`, positive peak ⇒ x leads y).
- **`resolve_latents` time-steps in topological order** over zero-lag edges only;
  lagged edges read finalised past values, so lag>0 cycles (e.g. reciprocal
  A↔B at a delay) are allowed and only true instantaneous cycles are rejected.
- **Epoch-varying communication** uses `epochs` masks on edges. The
  reversed-direction edge in `epoch_varying` uses a small lag so A→B and B→A
  never form an instantaneous loop.
- **Determinism**: `simulate`/`simulate_trials` spawn one independent RNG stream
  per area from the master seed, so results are reproducible and adding an area
  does not perturb earlier areas.

## "Partial CCA zeroes it out" (mediated scenario)
With `gain=2.0` and single-link maps A→C→B, marginal `CCA(A,B)` ≈ 0.56 (latents)
/ 0.87 (populations) while `partial CCA(A,B|C)` drops sharply (~0.24 at the
population level). On the *single linked dimensions*, the scalar partial
correlation `partial_correlation(zA0, zB0 | zC)` ≈ 0 — the clean "zeroed-out"
demonstration; the tests assert both.

## Extensions (phases 1-4)
- **Realism** (`realism.py`): per-neuron log-normal gains, shared global
  fluctuation, slow drift, sub-Poisson refractory counts. The refractory model
  is a *moment-matched binomial* (Fano ~ 1 - regularity), not thinning -- any
  independent thinning of a Poisson stays Poisson (Fano 1), so a different count
  distribution is required. All knobs default off -> clean model byte-identical.
- **Trial structure** (`simulate_trials`): `(n_trials, n_bins, n_neurons)` with
  independent, exchangeable trials (per-trial latent draws, shared loadings,
  burn-in sized to the max edge lag) -- the shape the `striatum_cca` compute
  layer consumes and what trial-permutation nulls need.
- **More scenarios**: bidirectional, common_input, rotated_subspace,
  partial_mediation, noise_correlation (+ `SharedNoise` for observation-level
  confounds).
- **Recovery benchmark** (`benchmark.py`, `scripts/recovery_benchmark.py`):
  PC-reduce each area, run lag-0/lagged/partial CCA on populations, score vs
  ground truth. 10/10 scenarios recover.
- **Bridge** (`bridge.py`, `scripts/striatum_cca_demo.py`): drives the *real*
  `striatum_cca` core/lagged/partial on popsim trials; auto-discovers the cca
  src dir (or `STRIATUM_CCA_SRC`), needs h5py.

## Gotcha: partial_mediation at population vs latent level
The graded-mediation grading is visible *per latent dimension* (mediated dim0
collapses under partial CCA, direct dim1 survives), but at the *population* CCA
level the top canonical correlation is carried by the direct channel, so partial
CCA barely drops -- the coupling *survives* partialling C (unlike `mediated`).
The benchmark/test verdicts reflect what is recoverable from populations.

## Environment gotcha
- The bare container `python3` has **no numpy/pytest**. Use the uv venv:
  `uv venv .venv && uv pip install --python .venv numpy scipy matplotlib pytest ruff h5py`,
  then `.venv/bin/python -m pytest`.
- Timescales (`tau`) and cross-correlogram lags are in **bins** (dt defaults 1.0).
- Generated `.npy`/`.png` are git-ignored (repo ignores `*.png` globally); the
  `metadata.json` and `recovery_benchmark.json` files are committed.
