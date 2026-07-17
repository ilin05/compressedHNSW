#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BIN="${ROOT_DIR}/build/test_hnswpq_tradeoff"
BASE_DIR="${BASE_DIR:-${ROOT_DIR}/datasets/hdf5files}"
INDEX_DIR="${INDEX_DIR:-${SCRIPT_DIR}}"
OUTPUT="${OUTPUT:-${SCRIPT_DIR}/hnswpq_tradeoff_sift.csv}"

echo "[HNSW+PQ trade-off] dataset=sift-128-euclidean"
echo "[HNSW+PQ trade-off] index_dir=${INDEX_DIR}"
echo "[HNSW+PQ trade-off] output=${OUTPUT}"
echo "[HNSW+PQ trade-off] HNSWPQ_p uses p dimensions/subquantizer: p={1,2,4,8} => m={128,64,32,16}."

"${BIN}" \
  --base_dir "${BASE_DIR}" \
  --index_dir "${INDEX_DIR}" \
  --output "${OUTPUT}" \
  --dataset sift-128-euclidean \
  --pq_dims_per_subq 1 2 4 8 \
  --ef 10 20 40 80 120 200 400 600 800 1000 \
  --k 1 10 \
  --num_rounds 3 \
  "$@"
