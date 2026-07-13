# StriatumACC gotchas

- LFP `data_to_save` units and source band are undocumented: never use an
  absolute amplitude threshold to decide whether signal exists; audit exact
  zeros, finite values and scale-free temporal/spectral structure separately.
- Equal LFP/spike array lengths and VR timestamps fitting inside the nominal
  grid are necessary compatibility checks, not proof of sample-zero alignment.
- Referencing effects are session- and frequency-specific: common-median
  referencing suppresses 614's ~154 Hz peak but not its ~74 Hz peak, and does
  not materially change either peak in 727/731.
