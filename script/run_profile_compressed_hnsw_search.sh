#!/bin/bash
set -e

# Usage examples:
#   bash script/run_profile_compressed_hnsw_search.sh
#   bash script/run_profile_compressed_hnsw_search.sh sift-128-euclidean DeXOR
#   bash script/run_profile_compressed_hnsw_search.sh gist-960-euclidean Elf 100 10

DATASET=${1:-sift-128-euclidean}
ALGO=${2:-DeXOR}
K=${4:-10}

echo "[Profile] compressed HNSW retrieval bottleneck analysis"
echo "dataset=${DATASET}, algorithm=${ALGO}, ef=${EF}, k=${K}"

../build/test_profile_compressed_hnsw_search \
  --threads 32 \
  --dataset "${DATASET}" \
  --algorithm "${ALGO}" \
  --chain_max -1 \
  --use_cache 0 \
  --use_tls 0 \
  --tls_ratio 0.0 \
  --k "${K}" \
  --output_csv "compressed_hnsw_profile_${DATASET}_${ALGO}.csv"

echo "Saved CSV: compressed_hnsw_profile_${DATASET}_${ALGO}.csv"
