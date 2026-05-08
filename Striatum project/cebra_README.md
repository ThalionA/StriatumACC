# CEBRA pipeline — quick reference

## What this gives you

A held-out, multi-session CEBRA model trained on the StriatumACC task data
with **position**, **lick rate**, and **lick errors** as joint contrastive
labels. Outputs both a per-mouse decoder R² table (position decoded from
the embedding on held-out trials) and a multi-session consistency matrix
across mice.

Three new files in `Striatum project/`:

- `save_for_cebra.m` — MATLAB exporter
- `cebra_analysis.py` — Python pipeline
- `cebra_README.md` — this file

## Step 1: Export the data (MATLAB)

From `Striatum project/`:

```matlab
save_for_cebra
```

This loads `processed_data/preprocessed_data.mat` if not already in the
workspace, then writes one `.mat` file per mouse to `./cebra_data/`:

- `cebra_mouse{N}_data.mat` for each animal
- Each file holds: neural firing-rate tensor, lick rate, lick errors,
  velocity, learning point, change point, area labels (DMS/DLS/ACC/V1),
  neuron types, and the bin geometry.

You can override the export with `cfg.output_dir`, `cfg.bin_size`, etc.

## Step 2: Set up the Python environment (one-time)

```bash
# from anywhere — installs into the active env
pip install cebra h5py scikit-learn scipy numpy matplotlib
# torch is needed by cebra; install per your CUDA setup
pip install torch  # or: pip install torch --index-url https://download.pytorch.org/whl/cu121
```

CEBRA needs Python 3.9+. If you have multiple Pythons, use the one that
matches your conda/uv environment.

## Step 3: Run

From `Striatum project/`:

```bash
python cebra_analysis.py
```

Defaults (in the `__main__` block at the bottom of the file):

- Scans 16 mouse files, skips missing ones with a warning
- Labels: `('position', 'lick_rate', 'lick_errors')` stacked into the
  contrastive label vector
- Per-area refits: `('all', 'DMS', 'DLS', 'ACC', 'V1')`
- Trial-wise train/test split (25% held out)
- 15 000 iterations per fit, output dim = 3, contrastive temperature auto

To change config, edit the `CebraConfig(...)` call at the bottom of the
file.

## What you'll get

`./cebra_results/`:

- `cebra_results.npz` — Python-side: decoder R²/RMSE tables, consistency
  scores, dataset IDs.
- `cebra_results.mat` — same payload, MATLAB-loadable. Includes per-mouse
  embeddings as `embedding_mouse1`, `embedding_mouse2`, ...
- `cebra_config.json` — exact config used (for reproducibility).
- `cebra_multisession_embeddings.png` — quick QC figure of the
  multi-session 3D embeddings, coloured by position.

A pretty-printed `decoder R² by mouse × area` table is also printed to
stdout at the end of the run.

## Key design choices baked in

1. **Trial-wise train/test split**, not random row split. CEBRA timepoints
   from the same trial would otherwise leak into both halves.
2. **Per-fold StandardScaler** (fit on train only). The legacy CEBRA
   scripts in `legacy/` fit the scaler on the whole dataset before
   splitting, which leaks test data into normalisation.
3. **Held-out linear decoder of position** as the canonical evaluation
   metric (instead of just plotting embeddings and eyeballing). Lets you
   compare CEBRA against the existing TCA / ridge / Poisson decoders.
4. **Per-area refits** so you can ask whether DMS-only, DLS-only, ACC-only,
   V1-only embeddings differ in decoding quality — and whether the
   multi-session consistency aligns more strongly within an area than
   across areas.

## Troubleshooting

- **"V1 ratio is 0/N"** — only some mice have V1 probes; the V1 fits
  silently skip mice with `<5` V1 units. You should see ≈5/16 V1 fits
  populated after running on the full task cohort.
- **"too few <area> units"** — increase `cfg.min_units_per_mouse` or skip
  the area subset. The default threshold matches the project-wide
  `min_units = 5` setting.
- **HDF5 / h5py errors on load** — the loader expects v7.3 .mat files.
  `save_for_cebra.m` always saves with `-v7.3`, so this should be a
  non-issue, but if you exported from a different script, make sure it
  used `-v7.3`.

## Where the legacy versions live

The original three files (`cebra_test.py`, `cebra_multianimal.py`,
`cebra_single_multi_comparison.py`) are in `Striatum project/legacy/`.
They're orphaned and not maintained.
