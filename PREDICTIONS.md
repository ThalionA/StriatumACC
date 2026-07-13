# Predictions (newest first)

## 2026-07-13 — LFP audit reproducibility checks

- **Prediction:** Raw within-event voltage peaks will reproduce the documented
  signed offsets from the nearest VR sync edge (within 0.25 s of +1.98/+4.26/+1.74 s
  for 614/727/731), and common-median referencing will alter the ~75/~151 Hz peaks
  by less than 0.2 dB. Confidence: medium-high (~75%).
- **Falsifier:** Any offset misses by >0.25 s, or any peak changes by >=0.2 dB,
  when recomputed by a documented script from the cached event bins and raw voltage.

**Outcome — ✓ timing confirmed; ⚠ referencing claim invalidated.** Robust raw
voltage peaks reproduced +1.983/+4.256/+1.740 s for 614/727/731. However,
common-median referencing reduced 614's 153.8 Hz peak by 4.54 dB. Its 74.2 Hz
peak and both peaks in 727/731 changed by only -0.06 to +0.03 dB. Lesson: do
not generalise a referencing check across frequencies or sessions; persist the
per-peak change in the machine-readable output.

## 2026-07-12 — LFP sanity reanalysis

- **Prediction:** The earlier claim that the four LFP exports are ~99% empty will be invalidated once exact zeros and signal occupancy are measured without the absolute `SD > 0.02` threshold. Confidence: high (~85%). Basis: voltage units are unknown and the files occupy 11–16 GB, close to dense float32 storage.
- **Prediction:** Scale-free diagnostics will still identify intermittent, synchronous broadband artefacts, but ordinary low-amplitude LFP will be present through most corridor epochs. Confidence: medium (~65%).
- **Falsifier:** If sample-level exact-zero fractions are near 99%, or robust within-session amplitude/PSD diagnostics remain absent across channels and corridor epochs independently of threshold and scaling, the empty-export diagnosis stands.

**Outcome — ✓ first prediction confirmed; ↔ second prediction partly confirmed.** Full-file exact-zero fractions were 0.004% (1212), 5.34% (614), 3.16% (727), and 4.62% (731), with zero fully-zero windows during behaviour. The latter three zeros are single terminal padding blocks. Ordinary task voltage is continuous, but not clean conventional LF-band data: lag-1 correlations are near zero, only 16–18% of 1–499 Hz power lies below 100 Hz, and 614/727/731 contain strong ~74–75 and ~151 Hz peaks. Periodic high-amplitude events recur every 60 s or 5 s. Lesson: never threshold undocumented voltage in absolute units; verify exact zeros and signal bandwidth separately.
