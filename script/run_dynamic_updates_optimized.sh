#!/usr/bin/env bash
# Shared implementation for four independent alpha/workload launchers.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
experiment=${1:?Expected repeated or growing}
mode=${2:?Expected alp_a01 or alp_a02}
case "$experiment" in
    repeated) batches=20; ratio=0.03; search_every=2 ;;
    growing) batches=10; ratio=0.05; search_every=1 ;;
    *) echo "Unknown experiment: $experiment" >&2; exit 2 ;;
esac
case "$mode" in
    alp_a01) tag=alpha01 ;;
    alp_a02) tag=alpha02 ;;
    *) echo "Unknown insertion mode: $mode" >&2; exit 2 ;;
esac
DATA_DIR="${DATA_DIR:-$ROOT/datasets/hdf5files}"
GT_ROOT="${GT_ROOT:-$ROOT/script/dynamic_prefix_gt}"
OUT="${OUT:-$ROOT/script/dynamic_updates_${experiment}_${tag}_results}"
BIN="${BIN:-$ROOT/build/dynamic_updates_${experiment}_${tag}/test_dynamic_update_workloads}"
DATASETS="${DATASETS:-sift-128-euclidean mnist-784-euclidean fashion-mnist-784-euclidean gist-960-euclidean deep-image-96-angular}"
read -r -a datasets <<< "$DATASETS"
read -r -a efs <<< "${EF_SEARCH:-40}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 OMP_DYNAMIC=FALSE
args=(--experiment "$experiment" --data-dir "$DATA_DIR" --gt-root "$GT_ROOT" --out "$OUT"
      --binary "$BIN" --datasets "${datasets[@]}" --ef "${efs[@]}"
      --modes raw_f32 "$mode" --insert-decode-cache 1 --insert-upper-tls 1
      --batches "${BATCHES:-$batches}" --repeat-ratio "${REPEAT_RATIO:-$ratio}"
      --search-every "${SEARCH_EVERY:-$search_every}" --search-rounds "${SEARCH_ROUNDS:-3}"
      --M "${M:-16}" --ef-construction "${EF_CONSTRUCTION:-200}" --seed "${SEED:-20261015}"
      --count "${COUNT:-0}" --query-count "${QUERY_COUNT:-0}" --resume "${RESUME:-1}")
if [[ "${SKIP_BUILD:-0}" == 1 ]]; then args+=(--skip-build); fi
if [[ "${KEEP_INITIAL_INDEX:-0}" == 1 ]]; then args+=(--keep-initial-index); fi
if [[ "${VERIFY_SIZE:-0}" == 1 ]]; then args+=(--verify-size); fi
exec python3 -u "$ROOT/script/run_dynamic_update_workloads.py" "${args[@]}"
