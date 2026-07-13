# striatum_lfp — voltage-export audit and provisional LFP pipeline

Audits the new 384-channel Neuropixels voltage exports and, once their provenance
and timing are resolved, will analyse band power analogously to unit firing rate.
Downstream MATLAB and temporal-CCA stages are currently gated on provenance.

## Data: verified facts
- 4 mice (1212, 614, 727, 731); one `RawData/LFP/voltage_data_384ch*.mat` each,
  mapped to a mouse by file size (`RawData/LFP/lfp_mapping.txt`).
- HDF5 `data_to_save` = `(n_samples × 384)` float32 stored voltage. Length equals
  each mouse's 1 ms `binned_spikes` length (1212: 11.4 M; others: 8.4 M), making
  a 1000 Hz export grid plausible. Equality of lengths does **not** prove offset
  or sample-accurate VR alignment.
- Channels → depth (Neuropixels geometry) → area via the same µm boundaries the
  spikes use (`Neuropixels_Depth_Data.csv`). Behaviour + learning point reused
  per mouse from `<ID>_raw.mat` / `preprocessed_data2p5cm.mat`.
- Physical units, input band (LF/AP/wideband), gain, referencing, downsampling
  and anti-alias filtering are unknown: no producer script or source metadata is
  present. Call values "stored voltage units", not volts or microvolts.
- Full-file audit: no zero windows during behaviour. Files 614/727/731 have one
  terminal zero-padding block after behaviour; 1212 has effectively no zeros.
- A sharply separated high-amplitude mode recurs every 60 s (614/727/731) or 5 s
  (1212). Its cadence and cross-depth synchrony are instrument-like; its exact
  mechanism is unresolved, so it must be masked rather than interpreted.

## Current gate

Do not position-bin, decode, or run temporal CCA until the export producer or
source `.meta` establishes the exact time offset and signal preprocessing.

## Layout
`src/striatum_lfp/` — configuration, geometry, out-of-core reading, integrity
audit/sanity helpers, provisional feature extraction, and quarantined learning
helpers. `scripts/` contains reproducible audit drivers; `tests/` contains
synthetic-ground-truth pytest checks. The old single-window `qc.py` thresholds
are retained only as tested numerical primitives and are not an analysis gate.

## Running tests
```
cd "Striatum project/lfp" && /opt/anaconda3/bin/python -m pytest -q
```
`conftest.py` puts `src/` on the path. The interpreter is explicit because the
current shell may resolve `python3` to a Homebrew installation without pytest.

See `NOTES.md` for the running log and the full data contract.
