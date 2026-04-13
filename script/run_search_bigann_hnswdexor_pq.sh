#!/bin/bash
set -e

echo "Testing hnswlib bigann HNSW_DEXOR_PQ Search"
../build/test_search_bigann_hnswdexor_pq --dataset bigann_1M bigann_10M

