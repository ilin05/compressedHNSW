#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="../datasets/hdf5files/"
EF=100
OUTPUT="tls_accuracy_stability.csv"

echo "[TLS Accuracy] alpha sensitivity with repeated end-to-end measurements"
echo "[TLS Accuracy] output=${OUTPUT}"

../build/test_tls_retention_diagnostic \
  --base_dir "${BASE_DIR}" \
  --dataset sift-128-euclidean gist-960-euclidean fashion-mnist-784-euclidean \
  --ef "${EF}" \
  --k 10 \
  --alpha 0.1 0.2 0.3 0.5 1.0 \
  --num_rounds 3 \
  --warmup_rounds 1 \
  --output "${OUTPUT}" \
  "$@"

echo "[TLS Accuracy] done"
