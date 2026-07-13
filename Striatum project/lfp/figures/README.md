# Figure status

Current defensible sanity figures (each has PNG and SVG):

- `sanity_audit_overview_v2` — full-session continuity, behavioural-state gaps,
  periodic high-amplitude mode and robust spectra.
- `sanity_audit_raw_examples_v2` — ordinary versus periodic-event morphology.
- `sanity_audit_event_timing` — periodic-event cadence versus VR sync edges.
- `signal_identity` — bandwidth, temporal smoothness, common-median-reference
  sensitivity and nominal depth-band correlations. Area labels are withheld for
  1212 because its voltage-probe identity is unverified.

`_superseded_absolute_threshold/` contains invalid earlier figures that used
`SD > 0.02` in undocumented voltage units. They are retained only as an audit
trail and must not be presented.

`_quarantined_unaligned_learning/` preserves the old `learning_evolution_*`
figures as an audit trail. They were produced before the voltage-to-VR offset,
source band and 1212 probe identity were established and include a contaminated
30–80 Hz band. They must not be presented as results. The driver now refuses to
run without `--allow-unaligned`, excludes 1212 and omits that band.
