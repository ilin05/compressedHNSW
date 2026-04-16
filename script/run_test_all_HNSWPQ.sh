#!/bin/bash
set -e

echo "Running all HNSW tests"

echo "==================================="
echo "1. Running Build HNSWPQ Tests"
echo "==================================="
bash run_build_faiss_hnsw_PQ.sh
bash run_build_bigann_faiss_hnsw_PQ.sh

echo "==================================="
echo "2. Running Search HNSWPQ Tests"
echo "==================================="
bash run_search_hnswdexor_pq.sh
bash test_search_bigann_faiss_hnsw_PQ.sh

echo "==================================="
echo "All tests execution finished successfully!"
echo "==================================="
