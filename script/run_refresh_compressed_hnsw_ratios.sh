#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BIN_PATH="${BIN_PATH:-${REPO_DIR}/build/test_refresh_compressed_hnsw_ratios}"
NORMAL_CSV="${NORMAL_CSV:-${SCRIPT_DIR}/compressed_hnsw_build_results.csv}"
BIGANN_CSV="${BIGANN_CSV:-${SCRIPT_DIR}/bigann_compressed_hnsw_build_results.csv}"
DATASET_BASE_DIR="${DATASET_BASE_DIR:-${REPO_DIR}/datasets/hdf5files}"
INDEX_DIR="${INDEX_DIR:-${SCRIPT_DIR}}"

if [[ ! -x "${BIN_PATH}" ]]; then
  echo "Binary not found or not executable: ${BIN_PATH}"
  echo "Please build target test_refresh_compressed_hnsw_ratios first."
  exit 1
fi

"${BIN_PATH}" \
  --normal_csv "${NORMAL_CSV}" \
  --bigann_csv "${BIGANN_CSV}" \
  --dataset_base_dir "${DATASET_BASE_DIR}" \
  --index_dir "${INDEX_DIR}"

echo "Compression ratio refresh completed."
