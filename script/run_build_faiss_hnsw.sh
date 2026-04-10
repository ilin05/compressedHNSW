#!/bin/bash
set -e

echo "Testing Faiss Build"
nohup ../build/test_build_faiss_hnsw_pq_sq > test_build_faiss_hnsw_pq_sq_output.log 2>&1

# ../../build/test_build_faiss_hnsw_pq_sq

# echo "Testing Faiss Search..."
# ../../build/test_search_faiss_hnsw_pq_sq
