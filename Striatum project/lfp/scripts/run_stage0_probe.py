"""Retired entry point for the invalid single-window, absolute-threshold QC."""

raise SystemExit(
    "run_stage0_probe.py is retired: its single-window QC produced false "
    "good/dead-channel conclusions. Run scripts/run_sanity_audit.py followed "
    "by scripts/plot_sanity_audit.py and scripts/run_signal_identity.py."
)
