#!/bin/bash
set -e

echo "[Ablation] Build compressed_hnsw_framework indexes"
../build/test_build_compressed_hnsw_ablation \
  --threads 32 \
  # --dataset fashion-mnist-784-euclidean gist-960-euclidean mnist-784-euclidean sift-128-euclidean \
  --dataset gist-960-euclidean \
  --algorithm DeXOR Gorilla Elf Camel \
  --chain_max 2 5 -1 \
  --output_csv compressed_hnsw_ablation_build_results.csv

echo "Saved CSV: compressed_hnsw_ablation_build_results.csv"
