#!/bin/bash
set -e

echo "Testing DeXOR root policy, root-cache, and root-ratio experiments"
echo "================================================================="
echo ""

../build/test_cache_ratio_analysis_dexor \
  --dataset sift-128-euclidean_train.fvecs \
  --root-policies random level level0degree \
  --root-ratios 0.0001 0.001 0.01 0.1 \
  --cache-ratios 0.0 0.5 1.0 2.0 5.0 10.0 \
  --chain-max-length -1 \
  --build-threads 32 \
  --search-threads 1 \
  --num-rounds 1

echo ""
echo "DeXOR root-policy, root-cache, and root-ratio experiments completed."
echo "Results saved to:"
echo "  - root_selection_policy_ablation_dexor_results.csv"
echo "  - root_cache_ratio_sensitivity_dexor_results.csv"
echo "  - root_ratio_sensitivity_dexor_results.csv"
