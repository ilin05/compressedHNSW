#!/bin/bash
set -e

DATASET="sift-128-euclidean"
BASE_DIR="../datasets/hdf5files/"
EF=100
OUTPUT="tls_first_stage_retention_sift.csv"

echo "[TLS Retention Diagnostic] dataset=${DATASET}, ef=${EF}"
echo "[TLS Retention Diagnostic] output=${OUTPUT}"

../build/test_tls_retention_diagnostic \
  --base_dir "${BASE_DIR}" \
  --dataset "${DATASET}" \
  --ef "${EF}" \
  --alpha 0.05 0.1 0.2 0.3 0.5 1.0 \
  --output "${OUTPUT}"

echo "[TLS Retention Diagnostic] done"
