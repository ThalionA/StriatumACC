#!/bin/zsh
# Use the project venv (arm64): the anaconda python3 on PATH is an x86 build
# whose jaxlib aborts on AVX checks on this ARM machine.
cd "/Users/theoamvr/Desktop/Experiments/StriatumACC/Striatum project/rl_model"
export RLMODEL_TIME_BUDGET=900
PY=./.venv/bin/python
for i in $(seq 1 30); do
  echo "=== pass $i @ $(date +%H:%M) ==="
  out=$($PY -m scripts.run_neural_encoding 2>&1)
  echo "$out" | tail -25
  if echo "$out" | grep -q "DONE"; then echo "=== ENCODING DONE ==="; break; fi
  if echo "$out" | grep -q "Traceback"; then echo "=== ABORTING ON ERROR ==="; break; fi
done
