# popsim — working notes

Handoff/context notes for the simulated-population generator.

## Purpose
Ground-truth test-bed for cross-area communication analyses (pairs with the
sibling `striatum_cca` package). Three areas, 3–5 latents each, coupling defined
at the latent level so the answer is always known.

## Design decisions
- **Coupling lives at the latent level**, not the population level. The
  observation step (`project_population`) only adds private noise, so the
  communication subspace is exactly the column space of the source loadings
  mapped through `M`.
- **Lags are non-negative (causal).** A negative-lag relationship for an ordered
  pair `(X, Y)` is produced by adding the reverse edge `Y -> X`; this keeps the
  generator strictly causal and the cross-correlogram sign convention honest
  (`corr(x[t], y[t+k])`, positive peak ⇒ x leads y).
- **`resolve_latents` time-steps in topological order** over zero-lag edges only;
  lagged edges read finalised past values, so lag>0 cycles (e.g. reciprocal
  A↔B at a delay) are allowed and only true instantaneous cycles are rejected.
- **Epoch-varying communication** is implemented with `epochs` masks on edges.
  The reversed-direction edge in `epoch_varying` uses a small lag (default 2) so
  that A→B and B→A never form an instantaneous loop in the dependency graph.
- **Determinism**: `simulate` spawns one independent RNG stream per area from the
  master seed, so results are reproducible and adding an area does not perturb
  earlier areas.

## "Partial CCA zeroes it out" (mediated scenario)
With `gain=2.0` and single-link maps A→C→B, marginal `CCA(A,B)` ≈ 0.56 while
`partial CCA(A,B|C)` ≈ 0.18 (the residual is CCA's known upward bias on the top
canonical correlation). On the *single linked dimensions*, the scalar partial
correlation `partial_correlation(zA0, zB0 | zC)` ≈ 0 — that is the clean
"zeroed-out" demonstration; the test asserts both.

## Gotchas
- The dev container's `tau`/timescales are in **bins**, not seconds (dt defaults
  to 1.0). Cross-correlogram lags are likewise in bins.
- Generated `.npy` arrays and `.png` figures are git-ignored (regenerable from
  the seeded scripts; the repo ignores `*.png` globally). Only the small
  `metadata.json` ground-truth files are committed.
- **No numpy/pytest in the bare `python3`.** Use the uv venv:
  `uv venv .venv && uv pip install --python .venv numpy scipy matplotlib pytest ruff`,
  then `.venv/bin/python -m pytest`.

## Possible next steps
- Add a trial/condition axis (currently a single continuous session) to exercise
  `subtract_condition_mean` in `striatum_cca`.
- Add an adapter that loads `data/generated/<scenario>` straight into
  `striatum_cca.AreaActivity` for an end-to-end recovery benchmark.
