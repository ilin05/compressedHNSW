#!/bin/bash
set -e

echo "Testing Faiss HNSWPQ Search"
# ../build/test_search_faiss_hnsw_pq_sq --algorithm HNSWPQ --pq_m 4 8
../build/test_search_faiss_hnsw_pq_sq --algorithm HNSWPQ --pq_m 4
