#!/bin/bash
set -e

echo "Testing bigann HNSW_DEXOR_PQ Build"
../build/test_build_bigann_hnswdexor_pq --dataset bigann_1M bigann_10M

