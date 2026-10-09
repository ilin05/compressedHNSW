#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-/tmp/hnswlib_ivf_lvc_build}"
result_dir="${RESULT_DIR:-$repo_dir/script/ivf_lvc_chain_results}"
input="${INPUT:-}"
count="${COUNT:-2048}"
nlist="${NLIST:-8}"
cache_ratio="${CACHE_RATIO:-0.01}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

mkdir -p "$result_dir"
if [[ -z "$input" ]]; then
  RESULT_DIR="$result_dir" BUILD_DIR="$build_dir" \
    bash "$repo_dir/script/test_ivf_lvc_compact.sh" --chain-max 2 --cache-ratio "$cache_ratio"
  mv "$result_dir/summary.csv" "$result_dir/summary_ch2.csv"
  input="$result_dir/tiny_train.fvecs"
else
  cmake -S "$repo_dir" -B "$build_dir" -DHNSWLIB_EXAMPLES=ON
  cmake --build "$build_dir" --target test_ivf_lvc_compact -j "${BUILD_JOBS:-4}"
  "$build_dir/test_ivf_lvc_compact" --input "$input" --count "$count" \
    --nlist "$nlist" --chain-max 2 --cache-ratio "$cache_ratio" \
    --output "$result_dir/summary_ch2.csv"
fi

for chain in 4 8 -1; do
  "$build_dir/test_ivf_lvc_compact" --input "$input" --count "$count" \
    --nlist "$nlist" --chain-max "$chain" --cache-ratio "$cache_ratio" \
    --output "$result_dir/summary_ch${chain}.csv"
done
echo "PASS IVF-LVC chain depths and state cache: $result_dir"
