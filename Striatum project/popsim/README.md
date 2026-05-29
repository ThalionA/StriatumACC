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
`gaussian` (firing-rate-like) or `poisson` (spike counts via a softplus rate).

Communication is a directed, optionally **lagged**, optionally **epoch-gated**
linear map between areas' latents:

```
z_target(t) += gain * M @ z_source(t - lag)     (while the edge is active)
```

## Scenarios (`popsim.scenarios`)

| Scenario         | Ground truth                                                                  |
|------------------|------------------------------------------------------------------------------|
| `no_coupling`    | Three independent areas.                                                      |
| `zero_lag`       | A → B instantaneously (cross-correlogram peaks at lag 0).                     |
| `lagged`         | A → B at +10 bins, C → B at +25 bins; the A→B edge peaks at −10 for pair (B,A).|
| `mediated`       | A → C → B, no direct A → B; **partial CCA conditioning on C zeroes it out**.  |
| `epoch_varying`  | A→B, then B→A, then A→B again — changing strength, orientation, and direction.|

These map directly onto the analysis toolkit: lagged CCA, partial CCA, and
epoch-wise statistics.

## Usage

```python
from popsim import scenarios, simulate

result = simulate(scenarios.lagged())
result.neural["A"]      # (n_timesteps, n_neurons) observed activity
result.latents["B"]     # (n_timesteps, n_latents) ground-truth latents
result.metadata()       # JSON-serialisable ground-truth coupling description
```

Build a custom configuration directly:

```python
from popsim import AreaSpec, SimConfig, CouplingEdge, simulate

cfg = SimConfig(
    areas=[AreaSpec("A", n_latents=4, n_neurons=60),
           AreaSpec("B", n_latents=5, n_neurons=80, observation="poisson")],
    edges=[CouplingEdge("A", "B", gain=1.2, lag=8)],
    n_timesteps=3000, seed=0,
)
result = simulate(cfg)
```

## Reproducing the datasets and figures

```bash
cd "Striatum project/popsim"
python scripts/generate_datasets.py          # -> data/generated/<scenario>/
python scripts/plot_scenarios.py             # -> figures/<scenario>.png
pytest -q                                     # ground-truth validation tests
```

Generated arrays (`*.npy`) are git-ignored because they regenerate
deterministically from the seeded scripts; the small `metadata.json` ground-truth
files and the validation figures are kept.

## Layout

```
src/popsim/      importable package
  latents.py     intrinsic dynamics (ar1 / lds / oscillatory)
  coupling.py    CouplingEdge + resolve_latents (lags, epochs, mediation)
  observation.py latents -> population (gaussian / poisson)
  simulate.py    SimConfig / SimResult / simulate
  scenarios.py   the five predefined scenarios
  metrics.py     cross-correlation, CCA, partial CCA (validation helpers)
tests/           ground-truth tests (synthetic data, known answers)
scripts/         generate_datasets.py, plot_scenarios.py
```
