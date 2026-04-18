#!/bin/bash
set -e

echo "[Ablation] Search compressed_hnsw_framework indexes"
../build/test_search_compressed_hnsw_ablation \
  --threads 32 \
  --k 1 \
  --dataset fashion-mnist-784-euclidean gist-960-euclidean mnist-784-euclidean sift-128-euclidean \
  --algorithm DeXOR Gorilla Elf Camel DeXORPlus \
  --chain_max 2 5 -1 \
  --use_cache 0 1 \
  --use_tls 0 1 \
  --tls_ratio 0.1 0.2 0.3 0.5 \
  --output_csv compressed_hnsw_ablation_search_results.csv

echo "Saved CSV: compressed_hnsw_ablation_search_results.csv"
