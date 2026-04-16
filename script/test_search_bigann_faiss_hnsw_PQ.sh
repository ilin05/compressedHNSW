#!/bin/bash
set -e

echo "Testing Faiss bigann HNSWPQ Search"
# ../build/test_search_bigann_faiss_hnsw_pq_sq --algorithm HNSWPQ --pq_m 4 8 --dataset bigann_1M bigann_10M
../build/test_search_bigann_faiss_hnsw_pq_sq --algorithm HNSWPQ --pq_m 4 --dataset bigann_1M bigann_10M