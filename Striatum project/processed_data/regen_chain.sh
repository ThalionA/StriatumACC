#!/bin/zsh
# Full regeneration after the 2026-08-11 cell-type fix. Serial: one MATLAB at
# a time (two concurrent instances OOM'd this machine earlier today).
cd "/Users/theoamvr/Desktop/Experiments/StriatumACC/Striatum project"
M=/Applications/MATLAB_R2026a.app/bin/matlab
# Stop at the first failure: a later step would otherwise read a half-written
# product and the chain would still print COMPLETE.
run() { echo "=== $1 @ $(date +%H:%M) ==="; $M -batch "$1"; rc=$?; echo "--- exit $rc ---"; [ $rc -eq 0 ] || { echo "ABORT at $1"; exit $rc; }; }
run "OrganiseStriatumDataIncV1"
run "OrganiseStriatumDataControlIncV1"
run "ProcessStriatumTask"
run "ProcessStriatumControl"
run "Run_TCA_pipeline"
run "ensemble_analysis"
run "SpatioTemporalActivityEvolution"
# Writes figures/stability_by_animal.csv, the unit reference the LFP
# moving-reliability comparison reads.
run "IntegratedAll_v1"
echo "=== CHAIN COMPLETE @ $(date +%H:%M) ==="
