#!/bin/bash
set -e

echo "Testing Faiss HNSWSQ Search"
../build/test_search_faiss_hnsw_pq_sq --algorithm HNSWSQ --sq_nbits 4 8

