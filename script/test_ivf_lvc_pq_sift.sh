#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-$repo_dir/build_lvc_ivf_pq}"
dataset_dir="${DATASET_DIR:-$repo_dir/datasets/hdf5files}"
result_dir="${RESULT_DIR:-$repo_dir/script/ivf_lvc_pq_sift_results}"
train="${TRAIN:-$dataset_dir/sift-128-euclidean_train.fvecs}"
queries="${QUERIES:-$dataset_dir/sift-128-euclidean_test.fvecs}"
truth="${TRUTH:-$dataset_dir/sift-128-euclidean_neighbors.ivecs}"
nlist="${NLIST:-1000}"
nprobes="${NPROBES:-3,5}"
rounds="${ROUNDS:-1}"
pq_alphas="${PQ_ALPHAS:-0.1,0.2,1.0}"
cache_ratio="${CACHE_RATIO:-0.01}"
use_pq_filter=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

while (($#)); do
  case "$1" in
    --train) train="$2"; shift 2 ;;
    --queries) queries="$2"; shift 2 ;;
    --truth) truth="$2"; shift 2 ;;
    --output-dir) result_dir="$2"; shift 2 ;;
    --nlist) nlist="$2"; shift 2 ;;
    --nprobes) nprobes="$2"; shift 2 ;;
    --rounds) rounds="$2"; shift 2 ;;
    --pq_alpha|--pq_alphas) pq_alphas="$2"; shift 2 ;;
    --cache-ratio) cache_ratio="$2"; shift 2 ;;
    --use_pq_filter) use_pq_filter=1; shift ;;
    --no-pq-filter) use_pq_filter=0; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

for path in "$train" "$queries" "$truth"; do
  [[ -f "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done
mkdir -p "$result_dir"
cmake -S "$repo_dir" -B "$build_dir" -DHNSWLIB_EXAMPLES=ON
cmake --build "$build_dir" --target test_ivf_lvc_pq_sift -j "${BUILD_JOBS:-4}"
args=(--train "$train" --queries "$queries" --truth "$truth"
      --output-dir "$result_dir" --nlist "$nlist" --nprobes "$nprobes"
      --rounds "$rounds" --pq_alphas "$pq_alphas" --cache-ratio "$cache_ratio")
if ((use_pq_filter)); then args+=(--use_pq_filter); fi
"$build_dir/test_ivf_lvc_pq_sift" "${args[@]}"
