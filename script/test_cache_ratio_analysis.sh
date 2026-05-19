#!/bin/bash
set -e

echo "Testing Cache Ratio Analysis on HNSWALP"
# Measure recall and QPS with the unified VQ driver
../build/test_cache_ratio_analysis --dataset gist-960-euclidean_train.fvecs sift-128-euclidean_train.fvecs mnist-784-euclidean_train.fvecs --cache-ratios 0.5 1.0 2.0 3.0 5.0 10.0