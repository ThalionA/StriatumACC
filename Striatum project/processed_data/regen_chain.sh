#!/bin/zsh
# Full regeneration after the 2026-08-11 cell-type fix. Serial: one MATLAB at
# a time (two concurrent instances OOM'd this machine earlier today).
cd "/Users/theoamvr/Desktop/Experiments/StriatumACC/Striatum project"
M=/Applications/MATLAB_R2026a.app/bin/matlab
run() { echo "=== $1 @ $(date +%H:%M) ==="; $M -batch "$1"; echo "--- exit $? ---"; }
run "OrganiseStriatumDataIncV1"
run "OrganiseStriatumDataControlIncV1"
run "ProcessStriatumTask"
run "ProcessStriatumControl"
run "Run_TCA_pipeline"
run "ensemble_analysis"
run "SpatioTemporalActivityEvolution"
echo "=== CHAIN COMPLETE @ $(date +%H:%M) ==="
