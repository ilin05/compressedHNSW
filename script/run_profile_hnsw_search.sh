#!/bin/bash
set -e

# Usage examples:
#   bash script/run_profile_hnsw_search.sh
#   bash script/run_profile_hnsw_search.sh sift-128-euclidean
#   bash script/run_profile_hnsw_search.sh gist-960-euclidean 100 10

DATASET=${1:-sift-128-euclidean}
EF=${2:-100}
K=${3:-10}

echo "[Profile] original HNSW retrieval bottleneck analysis"
echo "dataset=${DATASET}, ef=${EF}, k=${K}"

../build/test_profile_hnsw_search \
  --threads 32 \
  --dataset "${DATASET}" \
  --ef "${EF}" \
  --k "${K}" \
  --output_csv "hnsw_profile_${DATASET}.csv"

echo "Saved CSV: hnsw_profile_${DATASET}.csv"