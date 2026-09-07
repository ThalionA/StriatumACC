#!/bin/zsh
# Resume after the cell-type fix: caches already regenerated at 23:37/23:41.
cd "/Users/theoamvr/Desktop/Experiments/StriatumACC/Striatum project"
M=/Applications/MATLAB_R2026a.app/bin/matlab
run() { echo "=== $1 @ $(date +%H:%M) ==="; $M -batch "$1"; echo "--- exit $? ---"; }
run "Run_TCA_pipeline"
run "ensemble_analysis"
run "SpatioTemporalActivityEvolution"
echo "=== CHAIN COMPLETE @ $(date +%H:%M) ==="
