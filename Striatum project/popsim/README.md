# popsim — simulated multi-area neural population activity

A latent-driven generator for population activity in **three neural areas**, built
to provide *ground truth* for communication-subspace analyses (e.g. the sibling
`striatum_cca` package). Activity is generated from a small number of latent
variables per area (**3–5 latents**) that are projected to full populations;
inter-area communication is specified at the **latent level** and is therefore
exactly known.

## What it models

Each area has its own intrinsic latent dynamics (selectable):

| Dynamics       | Description                                                |
|----------------|------------------------------------------------------------|
| `ar1`          | OU-like smooth autocorrelated noise (timescale `tau`).     |
| `lds`          | Rotational linear dynamical system (oscillatory + decay).  |
| `oscillatory`  | Sinusoids at set frequencies plus noise.                   |

Populations are read out from the latents with a per-area observation model:
`gaussian` (firing-rate-like) or `poisson` (spike counts via a softplus rate),
plus optional **biological-realism** knobs (`popsim.RealismParams`, all off by
default): per-neuron log-normal gain heterogeneity, a shared global fluctuation,
slow nonstationary drift, and sub-Poisson refractory regularity.

Communication is a directed, optionally **lagged**, optionally **epoch-gated**
linear map between areas' latents:

```
z_target(t) += gain * M @ z_source(t - lag)     (while the edge is active)
```

A separate `SharedNoise` mechanism adds correlation at the *observation* level
only (a confound with no latent communication).

## Scenarios (`popsim.scenarios`)

| Scenario            | Ground truth                                                                   |
|---------------------|--------------------------------------------------------------------------------|
| `no_coupling`       | Three independent areas.                                                        |
| `zero_lag`          | A → B instantaneously (cross-correlogram peaks at lag 0).                       |
| `lagged`            | A → B at +10 bins, C → B at +25 bins; A→B peaks at −10 for pair (B,A).          |
| `mediated`          | A → C → B, no direct A → B; **partial CCA conditioning on C zeroes it out**.    |
| `epoch_varying`     | A→B, then B→A, then A→B again — changing strength, orientation, and direction.  |
| `bidirectional`     | Reciprocal A↔B at different positive lags on separate latent dims.              |
| `common_input`      | C drives both A and B (shared-input confound); collapses under partial CCA.     |
| `rotated_subspace`  | A→B through a rank-r rotated subspace; CCA recovers exactly r canonical corrs.  |
| `partial_mediation` | Mediated + direct channels; A-B coupling survives partialling C (graded).       |
| `noise_correlation` | Independent latents, but A and B share additive observation noise.              |

These map directly onto the analysis toolkit: lagged CCA, partial CCA, and
epoch-wise statistics.

## Usage

```python
from popsim import scenarios, simulate, simulate_trials

# Continuous session: area -> (n_timesteps, n_neurons)
result = simulate(scenarios.lagged())
result.neural["A"]      # observed activity
result.latents["B"]     # ground-truth latents
result.metadata()       # JSON-serialisable ground-truth coupling description

# Trial-structured: area -> (n_trials, n_bins, n_neurons), independent trials
trials = simulate_trials(scenarios.lagged(), n_trials=80, n_bins=60)
```

Build a custom configuration directly:

```python
from popsim import AreaSpec, SimConfig, CouplingEdge, RealismParams, simulate

cfg = SimConfig(
    areas=[AreaSpec("A", n_latents=4, n_neurons=60),
           AreaSpec("B", n_latents=5, n_neurons=80, observation="poisson",
                    realism=RealismParams(neuron_gain_cv=0.4, refractory=0.5))],
    edges=[CouplingEdge("A", "B", gain=1.2, lag=8)],
    n_timesteps=3000, seed=0,
)
result = simulate(cfg)
```

## Recovery benchmark

`scripts/recovery_benchmark.py` simulates every scenario, reduces each area to
its top PCs (never touching the latents), runs lag-0 / lagged / partial CCA on
the population activity, and scores the recovered coupling against the configured
ground truth — the end-to-end check that the generator and standard analyses
agree. All 10 scenarios recover as configured (10/10).

## Bridge to `striatum_cca`

`popsim.bridge` feeds trial-structured simulations into the *real*
`striatum_cca` compute layer (cross-validated CCA, held-out lagged directionality
curve, partial CCA), so the analysis pipeline can be validated against known
ground truth:

```python
from popsim import scenarios, simulate_trials
from popsim.bridge import analyse_pair

r = simulate_trials(scenarios.lagged(lag_ab=8), n_trials=80, n_bins=60)
out = analyse_pair(r, "A", "B", k=4, partial_area="C", max_lag=25)
out.held_out_cc[0]   # cross-validated canonical correlation
out.peak_lag         # recovered ~ 8
```

This needs the `cca` subproject importable and `h5py` installed
(`pip install -e ".[cca]"`); the bridge auto-discovers the `striatum_cca` src dir
(or honours `STRIATUM_CCA_SRC`). The bridge tests skip automatically if it is
unavailable.

## Reproducing the datasets and figures

```bash
cd "Striatum project/popsim"
uv venv .venv                                 # one-time environment setup
uv pip install --python .venv numpy scipy matplotlib pytest ruff h5py
.venv/bin/python scripts/generate_datasets.py # -> data/generated/<scenario>/
.venv/bin/python scripts/plot_scenarios.py    # -> figures/<scenario>.png
.venv/bin/python scripts/recovery_benchmark.py # -> data/generated/recovery_benchmark.json
.venv/bin/python scripts/striatum_cca_demo.py # drive the real striatum_cca pipeline
.venv/bin/python -m pytest -q                 # ground-truth validation tests
```

Generated arrays (`*.npy`) and figures (`*.png`) are git-ignored because they
regenerate deterministically from the seeded scripts (the repository ignores
`*.png` globally). The small `metadata.json` ground-truth files and the
`recovery_benchmark.json` summary are committed.

## Layout

```
src/popsim/      importable package
  latents.py     intrinsic dynamics (ar1 / lds / oscillatory)
  coupling.py    CouplingEdge + resolve_latents (lags, epochs, mediation)
  observation.py latents -> population (gaussian / poisson) + realism knobs
  realism.py     per-neuron gains, global fluctuation, drift, sub-Poisson counts
  simulate.py    SimConfig / SimResult / TrialResult / simulate / simulate_trials
  scenarios.py   the ten predefined scenarios + SharedNoise
  metrics.py     cross-correlation, CCA, partial CCA, pca_reduce (validation)
  benchmark.py   coupling-recovery benchmark over all scenarios
  bridge.py      adapter into the sibling striatum_cca pipeline
tests/           ground-truth tests (synthetic data, known answers)
scripts/         generate_datasets, plot_scenarios, recovery_benchmark,
                 striatum_cca_demo
```
