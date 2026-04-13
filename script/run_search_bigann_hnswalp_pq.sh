#!/bin/bash
set -e

echo "Testing hnswlib bigann HNSW_ALP_PQ Search"
../build/test_search_bigann_hnswalp_pq --dataset bigann_1M bigann_10M 

