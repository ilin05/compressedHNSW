#!/usr/bin/env bash
# Complete dynamic update experiment; every benchmark process uses one thread.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT="repeated"
DATA_DIR="${DATA_DIR:-$ROOT/datasets/hdf5files}"
GT_ROOT="${GT_ROOT:-$ROOT/script/dynamic_prefix_gt}"
OUT="${OUT:-$ROOT/script/dynamic_updates_${EXPERIMENT}_results}"
BIN="${BIN:-$ROOT/build/dynamic_updates_${EXPERIMENT}/test_dynamic_update_workloads}"
DATASETS="${DATASETS:-sift-128-euclidean mnist-784-euclidean fashion-mnist-784-euclidean gist-960-euclidean deep-image-96-angular}"
EF_SEARCH="${EF_SEARCH:-20 40 80}"
read -r -a datasets <<< "$DATASETS"
read -r -a efs <<< "$EF_SEARCH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 OMP_DYNAMIC=FALSE
args=(--experiment "$EXPERIMENT" --data-dir "$DATA_DIR" --gt-root "$GT_ROOT" --out "$OUT"
      --binary "$BIN" --datasets "${datasets[@]}" --ef "${efs[@]}"
      --batches "${BATCHES:-50}" --search-rounds "${SEARCH_ROUNDS:-3}"
      --M "${M:-16}" --ef-construction "${EF_CONSTRUCTION:-200}" --seed "${SEED:-20261015}"
      --count "${COUNT:-0}" --query-count "${QUERY_COUNT:-0}" --resume "${RESUME:-1}")
if [[ "${SKIP_BUILD:-0}" == 1 ]]; then args+=(--skip-build); fi
if [[ "${KEEP_INITIAL_INDEX:-0}" == 1 ]]; then args+=(--keep-initial-index); fi
if [[ "${VERIFY_SIZE:-0}" == 1 ]]; then args+=(--verify-size); fi
exec python3 -u "$ROOT/script/run_dynamic_update_workloads.py" "${args[@]}"
