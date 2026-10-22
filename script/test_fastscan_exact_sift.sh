#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build=${BUILD_DIR:-"$repo/build_lvc_ivf_pq"}
data=${DATA_DIR:-"$repo/datasets/hdf5files"}
out=${OUTPUT_DIR:-"$repo/script/fastscan_exact_sift_diagnostic_results"}
index=${INDEX_FILE:-"$repo/script/revision_fastscan_mdim_build_results/sift-128-euclidean/ivfpq_fastscan/index.bin"}
if [[ ${SKIP_BUILD:-0} != 1 ]]; then
    cmake -S "$repo" -B "$build" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$build" --target test_fastscan_exact_diagnostic --parallel "${BUILD_THREADS:-4}"
fi
mkdir -p "$out"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$build/test_fastscan_exact_diagnostic" --index "$index" \
    --base "${BASE_FILE:-$data/sift-128-euclidean_train.fvecs}" \
    --queries "${QUERY_FILE:-$data/sift-128-euclidean_test.fvecs}" \
    --ground-truth "${GT_FILE:-$data/sift-128-euclidean_neighbors.ivecs}" \
    --output "$out/results.csv" --rounds "${ROUNDS:-3}" --batch "${QUERY_BATCH:-32}" \
    --query-count "${QUERY_COUNT:-0}" --fullscan-queries "${FULLSCAN_QUERIES:-100}" \
    --nprobe 1 2 4 6 8 10 15 20 25 30 40 50 60 80 100 \
    2>&1 | tee "$out/run.log"
echo "Diagnostic results: $out/results.csv"
