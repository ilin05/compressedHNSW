#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

INDEX_DIR="${INDEX_DIR:-.}"
BASE_DIR="${BASE_DIR:-../datasets/hdf5files}"
OUTPUT_CSV="${OUTPUT_CSV:-index_storage_breakdown.csv}"

echo "[Index Storage Breakdown]"
echo "  index_dir=${INDEX_DIR}"
echo "  base_dir=${BASE_DIR}"
echo "  output=${OUTPUT_CSV}"

../build/test_index_storage_breakdown \
  --index_dir "${INDEX_DIR}" \
  --base_dir "${BASE_DIR}" \
  --diff_cache_percent 1 \
  --dataset \
    deep-image-96-angular \
    fashion-mnist-784-euclidean \
    gist-960-euclidean \
    mnist-784-euclidean \
    sift-128-euclidean \
  --output_csv "${OUTPUT_CSV}"

echo "Saved CSV: ${SCRIPT_DIR}/${OUTPUT_CSV}"
