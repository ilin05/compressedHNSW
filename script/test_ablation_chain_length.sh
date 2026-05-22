#!/bin/bash
set -e

echo "Testing Ablation Chain Length"
../build/test_ablation_chain_length \
  --dataset mnist-784-euclidean sift-128-euclidean \
  --algorithm DeXOR \
  --chain_max 2 3 4 5 6 7 8 -1 \
  --search_threads 1 \
  --search_ef 200
