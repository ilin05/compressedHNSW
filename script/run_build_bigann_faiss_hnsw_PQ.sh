#!/bin/bash
set -e

echo "Testing Faiss HNSWPQ Build"
# ../build/test_build_bigann_faiss_hnsw_pq_sq --dataset bigann_1M bigann_10M --pq_m 4 8
../build/test_build_bigann_faiss_hnsw_pq_sq --dataset bigann_1M bigann_10M --pq_m 4
