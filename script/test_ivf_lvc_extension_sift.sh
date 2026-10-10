#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-$repo_dir/build_lvc_ivf_pq}"
dataset_dir="${DATASET_DIR:-$repo_dir/datasets/hdf5files}"
result_dir="${RESULT_DIR:-$repo_dir/script/ivf_lvc_extension_sift_results}"
train="${TRAIN:-$dataset_dir/sift-128-euclidean_train.fvecs}"
queries="${QUERIES:-$dataset_dir/sift-128-euclidean_test.fvecs}"
truth="${TRUTH:-$dataset_dir/sift-128-euclidean_neighbors.ivecs}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

for path in "$train" "$queries" "$truth"; do
  [[ -f "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done
mkdir -p "$result_dir"
cmake -S "$repo_dir" -B "$build_dir" -DHNSWLIB_EXAMPLES=ON
cmake --build "$build_dir" --target test_ivf_lvc_pq_sift -j "${BUILD_JOBS:-4}"
args=(--train "$train" --queries "$queries" --truth "$truth"
      --output-dir "$result_dir" --nlist "${NLIST:-1000}"
      --nprobes "${NPROBES:-1,2,4,6,8,10,15,20}"
      --pq_alphas "${PQ_ALPHAS:-0.05,0.1}"
      --rounds "${ROUNDS:-1}" --cache-ratio "${CACHE_RATIO:-0.01}"
      --use_pq_filter)
if [[ -n "${TRAIN_LIMIT:-}" ]]; then args+=(--train-limit "$TRAIN_LIMIT"); fi
if [[ -n "${QUERY_LIMIT:-}" ]]; then args+=(--query-limit "$QUERY_LIMIT"); fi
"$build_dir/test_ivf_lvc_pq_sift" "${args[@]}"
