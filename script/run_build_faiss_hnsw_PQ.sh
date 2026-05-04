#!/bin/bash
set -e

echo "Testing Faiss HNSWPQ Build"
# ../build/test_build_faiss_hnsw_pq_sq --algorithm HNSWPQ --pq_m 4 8
../build/test_build_faiss_hnsw_pq_sq --algorithm HNSWPQ --pq_m 1
