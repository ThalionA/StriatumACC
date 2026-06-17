# striatum_tcca — temporal communication-subspace CCA (striatum port)

A faithful port of the TomLearning `tom_cca` **temporal** communication-subspace
pipeline (the "Hippocampus-V1 Communication-Subspace Learning Report", method
contract `CCA_HH_Adapted`) onto the striatum/ACC/V1/CA1/DG dataset.

This is a **fresh port** kept deliberately separate from the mature *spatial*
`cca/` package: the numeric modules (`core`, `lagged`, `surrogate`, `subspace`,
`membership`, `subspace_stats`, `trajectory`, `early_trials`, `kernel_cca`,
`partial`, `subspace_window`) are ported near-verbatim from `tom_cca`; only the
data layer (`dataio`) and `config` are striatum-specific.

## Method, in one line
1 ms spikes → Gaussian smooth (σ=2.5 ms) → time-bin (10 ms primary; 25 ms for the
trajectory; 50 ms robustness) → per-unit z-score over the engaged (in-corridor,
running ≥2 cm/s) reference → residual + partial CCA → held-out whole-trial CV →
held-out CC, n_sig, MI, IFI directionality, Gini, principal-angle rotation —
contrasted across naive/intermediate/expert epochs, a sliding-window learning
trajectory, **engaged vs disengaged**, and Task vs Control corridor.

## Key striatum adaptations (vs tom_cca)
- Data is per-traversal `corridorData.binned_spikes` (1 ms), already corridor-only
  (dark stripped at `trial_world > 6`), in `preprocessed_data2p5cm.mat`.
- **No velocity channel** → speed derived from `trial_position`/`trial_times`
  (a.u.→cm via `AU_TO_CM`), re-zeroing times to corridor onset.
- FS units = `final_neurontypes(:,5) == 2` in *every* area (not a subset).
- Learning point, epochs, disengagement (`change_point_mean`) read per the
  project's MATLAB conventions.
- "Two corridors" = Task (cued) vs Control (blank habituation) — separate
  recordings/cohorts, so a between-cohort contrast (not a within-session
  transition).

## Running tests
```
cd "Striatum project/tcca" && python3 -m pytest -q
```
(`conftest.py` puts `src/` on the path; uses the anaconda python with
numpy/scipy/h5py.)

## Status
Stage 0 (scaffold + config + dataio + tests) — in progress on branch
`claude/temporal-cca-port`. See `NOTES.md` for the running log.
