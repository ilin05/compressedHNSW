#!/bin/bash
set -e

echo "Testing Cache Ratio Analysis on HNSWALP"
echo "==========================================="
echo ""

# Measure recall and QPS with different cache ratios
# Running 3 rounds per configuration for more stable results
# Cache ratios: 0.0% (baseline), 0.5%, 1%, 2%, 3%, 5%, 10%

../build/test_cache_ratio_analysis --dataset gist-960-euclidean_train.fvecs sift-128-euclidean_train.fvecs mnist-784-euclidean_train.fvecs --cache-ratios 0.0 0.5 1.0 2.0 3.0 5.0 10.0 --num-rounds 10

echo ""
echo "✓ Cache ratio analysis completed!"
echo "Results saved to: cache_ratio_analysis_results.csv"