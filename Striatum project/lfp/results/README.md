# Result status

Defensible integrity summaries:

- `sanity_summary.csv`
- `sanity_timing_summary.csv`
- `signal_identity_summary.csv`
- `sanity_figure_examples_and_psd.npz` (exact raw examples and PSD curves plotted)
- `signal_identity_figure_data.npz` (exact spectra, diagnostics and correlations plotted)
- `sanity_windows_<mouse>.csv` / `sanity_audit_<mouse>.npz` (figure source data)

`_quarantined_unaligned_learning/learning_evolution_summary.csv` is preserved
only as an audit trail because exact voltage-to-VR timing, source-band
provenance and 1212 probe identity remain unresolved, and its 30–80 Hz values
are contaminated by the persistent ~74 Hz peak.

`signal_identity_summary.csv` includes per-peak common-median-reference changes
and an explicit area-mapping status. A negative dB change means the peak was
reduced by common-median referencing.
