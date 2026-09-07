#!/bin/zsh
# Re-run after 409/418/703 arrived (08-28) and 1212 was re-exported at full length (08-30/31).
# Recipe from NOTES.md 2026-09-07. Serial; each step must succeed before the next.
cd "/Users/theoamvr/Desktop/Experiments/StriatumACC/Striatum project/lfp"
PY=/opt/anaconda3/bin/python
step() { echo "=== $* @ $(date '+%H:%M:%S') ==="; "$@"; rc=$?; echo "--- exit $rc @ $(date '+%H:%M:%S') ---"; [ $rc -ne 0 ] && { echo "ABORT: step failed"; exit $rc; }; }
step $PY scripts/run_lfp_inventory.py  --jobs 5 --cohort task
step $PY scripts/run_lfp_identity.py   --jobs 5 --cohort task
step $PY scripts/run_lfp_bandpower.py  --jobs 5 --cohort task --only 409,418,703,1212
step $PY scripts/validate_lfp_bandpower.py --cohort task
step $PY scripts/run_lfp_arms.py       --jobs 6 --cohort task
step $PY scripts/run_lfp_group_contrast.py
step $PY scripts/plot_lfp_arms.py --cohort task
step $PY scripts/plot_lfp_task_vs_control.py
echo "=== CHAIN COMPLETE @ $(date '+%H:%M:%S') ==="
