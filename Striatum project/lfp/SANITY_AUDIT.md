# LFP voltage-export sanity audit

**Status:** integrity characterised; physiological signal identity and exact VR
alignment unresolved. Downstream learning, decoding and CCA remain gated.

## Executive conclusion

The four exports are **not 99% empty**. Voltage is continuous throughout every
behavioural session. The earlier result was caused by an absolute `SD > 0.02`
threshold in undocumented units: it was 87× above ordinary 1212 voltage and
7,600–9,700× above ordinary voltage in the other mice, so it selected only a
periodic high-amplitude mode.

The opposite claim—“these are clean LFP files”—is also not established. The
ordinary signal is broadband-dominated, source metadata are missing, and three
sessions contain strong ~75 and ~151 Hz narrowband contamination. Some shared
2–40 Hz structure exists in 614/727/731, but 1212 is qualitatively different.

## Full-session integrity

| Mouse | Nominal duration at 1 kHz | Exact zeros | ≥99%-zero windows during behaviour | Ordinary median RMS (stored units) | High-amplitude one-second bins | Cadence |
|---:|---:|---:|---:|---:|---:|---:|
| 1212 | 190 min | 0.004% | 0 | 2.30e-4 | 17.97% | 5 s |
| 614 | 140 min | 5.34% | 0 | 2.62e-6 | 1.43% | 60 s |
| 727 | 140 min | 3.16% | 0 | 2.57e-6 | 1.44% | 60 s |
| 731 | 140 min | 4.62% | 0 | 2.07e-6 | 1.43% | 60 s |

For 614/727/731 all zero windows form one terminal block after the last VR frame
(448/265/387 s). “High-amplitude bin” means robust z-score >5 on log one-second
RMS; it does not estimate event duration. Prevalence is unchanged at z>8 or z>10
(727 has additional moderate bins at z>3).

## Signal-identity checks

Computed from 40 non-overlapping, session-spanning ordinary corridor windows and 24 depth-spanning
channels per session. Metrics are dimensionless except PSD amplitude.

| Mouse | Lag-1 r (1 ms) | Power 1–100 Hz / 1–499 Hz | PSD slope 2–40 Hz | strongest 40–180 Hz peaks |
|---:|---:|---:|---:|---:|
| 1212 | −0.070 | 15.6% | +0.46 | none sharply resolved |
| 614 | −0.053 | 17.2% | −0.43 | 153.8, 74.2 Hz |
| 727 | −0.108 | 16.0% | −0.66 | 150.9, 74.2 Hz |
| 731 | +0.009 | 18.2% | −0.91 | 150.9, 73.7 Hz |

A conventional strongly low-passed trace would have high positive lag-1
correlation and most power below 100 Hz. These exports instead retain near-white
temporal bandwidth. Nevertheless, 614/727/731 show a negative 2–40 Hz slope and
positive correlations within some nominal striatal depth bands after theta/beta
filtering, consistent with shared LF structure embedded in broadband contamination.
These are descriptive channel correlations, not area-connectivity estimates.
1212 is excluded from area-labelled summaries because its voltage-probe identity
is unverified.

Reference sensitivity is session- and frequency-specific. Common-median
referencing reduced 614's 153.8 Hz peak by 4.54 dB, but changed its 74.2 Hz peak
and both peaks in 727/731 by only −0.06 to +0.03 dB. The persistent ~74 Hz peak
lies inside the planned 30–80 Hz low-gamma band; low-gamma analysis is invalid as
currently specified. Per-peak changes are stored in
`results/signal_identity_summary.csv`.

## Timing

Equal voltage/spike array lengths and VR timestamps fitting inside the nominal
1 ms grid establish **structural compatibility only**. They do not prove offset.

The 60 s events have median voltage-peak offsets from the nearest VR sync edge of
+1.98 s (614), +4.26 s (727) and +1.74 s (731). 1212's 5 s mode is unrelated to
the 60 s VR sync. These deterministic events may share an acquisition clock, but
they do not establish sample-accurate alignment. A several-second uncertainty is
comparable to a corridor traversal and blocks position/trial analyses.

## Provenance audit

No repository code produces `voltage_data_384ch*.mat`. No source `.meta`, physical
gain, input-band label, reference scheme, resampling ratio or anti-alias filter is
available. The HDF5 layout is verified as time × 384 channels in Python (native
MATLAB likely 384 × time), float32, gzip-compressed.

## Claims allowed now

- The exports contain continuous finite voltage throughout behaviour.
- 614/727/731 have terminal zero padding only; 1212 effectively none.
- All sessions contain deterministic, instrument-like high-amplitude events.
- 614/727/731 share broadband structure with a declining 2–40 Hz component and
  ~75/~151 Hz contamination.
- 1212 should not be pooled with the other sessions.

## Claims not allowed now

- Physical power comparisons across mice.
- “Clean LFP” or a specified source band.
- Sample-accurate voltage↔VR alignment.
- Learning-phase effects, position decoding, trial reliability or temporal CCA.
- 30–80 Hz low-gamma effects.
- Area-specific communication claims (referencing/volume-conduction unresolved).

## Required unblock

Recover the producer script or original SpikeGLX `.lf.bin` + `.meta` and establish:

1. source stream (LF/AP/wideband), gain and physical units;
2. referencing and any filtering/downsampling/anti-alias operation;
3. exact sample-zero convention and voltage↔VR offset;
4. identity of the 60 s and 5 s high-amplitude events.

If provenance cannot be recovered, regenerate from `.lf.bin` with the sync channel
retained and document the complete transform. Only then redesign QC and bands.

## Reproducible evidence

- Full scan: `/opt/anaconda3/bin/python scripts/run_sanity_audit.py`
- Figures: `/opt/anaconda3/bin/python scripts/plot_sanity_audit.py`
- Signal identity: `/opt/anaconda3/bin/python scripts/run_signal_identity.py`
- Tests: `/opt/anaconda3/bin/python -m pytest -q` (57 passing at audit completion)
- Figure guide: `figures/README.md`
- Machine-readable summaries: `results/README.md`

No population inferential test is made: there are four sessions, and integrity
metrics are descriptive per session. Trials/windows are repeated measurements,
not independent biological replicates.
