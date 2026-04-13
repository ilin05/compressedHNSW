#!/bin/bash
set -e

echo "Testing bigann HNSW_ALP_PQ Build"
../build/test_build_bigann_hnswalp_pq --dataset bigann_1M bigann_10M

