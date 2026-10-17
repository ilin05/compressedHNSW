#!/usr/bin/env bash
# Rebuild/search only FastScan (M=dim, 4 bit) on the five revision datasets.
set -euo pipefail

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}
DATA_DIR=${DATA_DIR:-"$REPO/datasets/hdf5files"}
BUILD_DIR=${BUILD_DIR:-"$REPO/build_lvc_ivf_pq"}
BUILD_RESULTS=${BUILD_RESULTS:-"$REPO/script/revision_fastscan_mdim_build_results"}
SEARCH_RESULTS=${SEARCH_RESULTS:-"$REPO/script/revision_fastscan_mdim_search_results"}
BUILD_THREADS=${BUILD_THREADS:-32}
ROUNDS=${ROUNDS:-3}
FORCE=${FORCE:-0}
MODE=${1:-all}
DATASETS=(sift-128-euclidean mnist-784-euclidean fashion-mnist-784-euclidean
          gist-960-euclidean deep-image-96-angular)
FORCE_ARGS=()
if [[ "$FORCE" == 1 ]]; then FORCE_ARGS=(--force); fi
if [[ "$MODE" != all && "$MODE" != build && "$MODE" != search ]]; then
    echo "Usage: bash $0 [all|build|search]" >&2
    exit 2
fi

if [[ "$MODE" == all || "$MODE" == build ]]; then
    "$PYTHON" "$REPO/script/build_revision_baselines.py" \
        --algorithms ivfpq_fastscan --datasets "${DATASETS[@]}" \
        --data-dir "$DATA_DIR" --build-dir "$BUILD_DIR" \
        --output-dir "$BUILD_RESULTS" --threads "$BUILD_THREADS" "${FORCE_ARGS[@]}"
fi

if [[ "$MODE" == all || "$MODE" == search ]]; then
    # Search uses one OpenMP thread in test_revision_faiss_search.
    "$PYTHON" "$REPO/script/test_revision_baseline_search.py" \
        --summary "$BUILD_RESULTS/summary.csv" \
        --algorithms ivfpq_fastscan --datasets "${DATASETS[@]}" \
        --data-dir "$DATA_DIR" --build-dir "$BUILD_DIR" \
        --output-dir "$SEARCH_RESULTS" --rounds "$ROUNDS" --k 1 10 \
        --ivf-params 1 2 4 6 8 10 15 20 25 30 40 50 60 80 100 "${FORCE_ARGS[@]}"
fi

echo "Build summary: $BUILD_RESULTS/summary.csv"
echo "Search results: $SEARCH_RESULTS/all_results.csv"
