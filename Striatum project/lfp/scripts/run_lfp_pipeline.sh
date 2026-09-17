#!/bin/zsh
# One command that runs the whole LFP chain, in dependency order.
#
# Until now the chain existed only as a recipe in NOTES.md, retyped into a dated
# one-off script each time it had to be re-run (results/rerun_chain_2026-09-07.sh).
# That is how a step gets skipped: the recipe and the script drift, and nothing
# checks the order. This is the single entry point the project convention asks
# for, and every dated re-run should now be a call to it.
#
#   ./scripts/run_lfp_pipeline.sh                     # both cohorts, all steps
#   ./scripts/run_lfp_pipeline.sh --cohort task       # one cohort + cross-cohort steps
#   ./scripts/run_lfp_pipeline.sh --from arms         # resume at a step
#   ./scripts/run_lfp_pipeline.sh --only 409,418      # limit band power to some animals
#   ./scripts/run_lfp_pipeline.sh --list              # print the steps and exit
#
# Steps run in this order; a failure stops the chain rather than letting later
# steps consume a half-written table:
#
#   inventory   per-file audit (size, 1/f slope, mains, adjacency)   ~6 min/cohort
#   identity    verify each filename against that animal's spiking   ~3 min/cohort
#   bandpower   band power per trial x spatial bin (the slow one)   ~5 min/file
#   validate    reproduce the MATLAB bin map, must be 0.0 ms         seconds
#   arms        evolution / decoding / reliability / CCA / behaviour ~4 min/cohort
#   distance    the shank-distance control for cross-area coupling     ~30 s/cohort
#   psi         phase-slope index; RE-READS the voltage exports        ~25 s/file
#   contrast    task vs control, BH-FDR within arm (needs BOTH)      seconds
#   plots       per-cohort figures, then the task-vs-control set     seconds
#
# `bandpower` is the expensive step and is usually the one you want to skip with
# --from when only the downstream analysis changed. It reads the raw voltage
# exports; everything after it reads caches.

set -u
cd "$(dirname "$0")/.."

PY=/opt/anaconda3/bin/python
COHORTS=(task control)
FROM=""
ONLY=""
JOBS=5
LIST=0
STEPS=(inventory identity bandpower validate arms distance psi contrast plots)

usage() { sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cohort) COHORTS=("$2"); shift 2 ;;
    --from)   FROM="$2"; shift 2 ;;
    --only)   ONLY="$2"; shift 2 ;;
    --jobs)   JOBS="$2"; shift 2 ;;
    --list)   LIST=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) print -u2 "unknown argument: $1"; usage; exit 2 ;;
  esac
done

# --from <step>: drop every step before it.
if [[ -n "$FROM" ]]; then
  if [[ ${STEPS[(Ie)$FROM]} -eq 0 ]]; then
    print -u2 "unknown step '$FROM'; steps are: $STEPS"; exit 2
  fi
  STEPS=(${STEPS[$STEPS[(Ie)$FROM],-1]})
fi

# --list after the trim, so it shows the steps this invocation would actually run.
if [[ $LIST -eq 1 ]]; then print -l $STEPS; exit 0; fi

LOG="results/pipeline_$(date +%Y-%m-%d_%H%M).log"
print "logging to $LOG"

run() {
  print "=== $* @ $(date '+%H:%M:%S') ==="
  "$@"
  local rc=$?
  print "--- exit $rc @ $(date '+%H:%M:%S') ---"
  if [[ $rc -ne 0 ]]; then
    print "ABORT: '$*' failed; later steps would read a half-written table."
    exit $rc
  fi
}

{
  print "cohorts: $COHORTS   steps: $STEPS   jobs: $JOBS   only: ${ONLY:-all}"
  for step in $STEPS; do
    case $step in
      contrast)
        # Cross-cohort: needs both arm tables on disk, so it is skipped rather
        # than run against a stale partner when only one cohort was refreshed.
        missing=()
        for c in task control; do
          [[ -f "results/lfp_arms_evolution_${c}.csv" ]] || missing+=($c)
        done
        if (( ${#missing} )); then
          print "SKIP contrast: no arm table for ${missing}. Run those cohorts first."
        else
          run $PY scripts/run_lfp_group_contrast.py
        fi
        ;;
      plots)
        for c in $COHORTS; do run $PY scripts/plot_lfp_arms.py --cohort $c; done
        if [[ -f results/lfp_group_contrast.csv ]]; then
          run $PY scripts/plot_lfp_task_vs_control.py
          run $PY scripts/plot_lfp_combined.py
        fi
        if [[ -f results/lfp_distance_matched_task.csv ]]; then
          run $PY scripts/plot_lfp_distance_control.py
        fi
        if [[ -f results/lfp_psi_task.csv ]]; then
          run $PY scripts/plot_lfp_psi.py
        else
          print "SKIP cross-cohort plots: results/lfp_group_contrast.csv absent."
        fi
        ;;
      *)
        for c in $COHORTS; do
          case $step in
            inventory) run $PY scripts/run_lfp_inventory.py --jobs $JOBS --cohort $c ;;
            identity)  run $PY scripts/run_lfp_identity.py  --jobs $JOBS --cohort $c ;;
            bandpower)
              if [[ -n "$ONLY" ]]; then
                run $PY scripts/run_lfp_bandpower.py --jobs $JOBS --cohort $c --only "$ONLY"
              else
                run $PY scripts/run_lfp_bandpower.py --jobs $JOBS --cohort $c
              fi ;;
            validate)  run $PY scripts/validate_lfp_bandpower.py --cohort $c ;;
            arms)      run $PY scripts/run_lfp_arms.py --jobs $((JOBS + 1)) --cohort $c ;;
            distance)  run $PY scripts/run_lfp_distance_control.py --cohort $c ;;
            psi)       run $PY scripts/run_lfp_psi.py --cohort $c ;;
          esac
        done
        ;;
    esac
  done
  print "=== PIPELINE COMPLETE @ $(date '+%H:%M:%S') ==="
} 2>&1 | tee "$LOG"
