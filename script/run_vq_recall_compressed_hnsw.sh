#!/bin/bash
set -e

echo "[VQ-Recall] Test differential-codec compressed HNSW"

../build/test_vq_recall_compressed_hnsw \
  --base_dir ../datasets/hdf5files/ \
  --dataset sift-128-euclidean deep-image-96-angular mnist-784-euclidean fashion-mnist-784-euclidean gist-960-euclidean \
  --algorithm DeXOR Gorilla Camel Elf \
  --chain_max 2 \
  --tls_ratio 0 0.1 0.2 0.3 0.5 \
  --k 1 10 \
  --num_rounds 10 \
  --output_csv compressed_hnsw_vq_recall_results.csv

echo "Saved CSV: compressed_hnsw_vq_recall_results.csv"
