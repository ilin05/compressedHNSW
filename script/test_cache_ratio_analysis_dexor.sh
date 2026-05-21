#!/bin/bash
set -e

echo "Testing Cache Ratio Analysis on Compressed HNSW (DeXOR Algorithm)"
echo "=================================================================="
echo ""

# Test DeXOR with SIFT dataset
# Cache ratios: 0.0% (baseline), 0.5%, 1%, 2%, 3%, 5%, 10%, 20%
# Running 10 rounds per configuration for statistical stability

../build/test_cache_ratio_analysis_dexor \
  --dataset sift-128-euclidean_train.fvecs \
  --cache-ratios 0.0 0.5 1.0 2.0 3.0 5.0 10.0 \
  --num-rounds 10

echo ""
echo "✓ DeXOR cache ratio analysis completed!"
echo "Results saved to: cache_ratio_analysis_dexor_results.csv"
