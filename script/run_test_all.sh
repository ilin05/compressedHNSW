#!/bin/bash
set -e

echo "Running all HNSW tests"

echo "==================================="
echo "1. Running Build Tests"
echo "==================================="
# bash run_build_hnsw.sh
# bash run_build_hnswalp_pq.sh
# bash run_build_hnswdexor_pq.sh
# bash run_build_faiss_hnsw_PQ.sh
# bash run_build_faiss_hnsw_SQ.sh
# bash test_build_compressed_hnsw_framework/run_build_compressed_hnsw_framework.sh

echo "==================================="
echo "2. Running Search Tests"
echo "==================================="
# bash run_search_hnsw.sh
# bash run_search_hnswalp_pq.sh
bash run_search_hnswdexor_pq.sh
# bash test_search_faiss_hnsw_PQ.sh
# bash test_search_faiss_hnsw_SQ.sh
# bash test_search_compressed_hnsw_framework/run_search_compressed_hnsw_framework.sh
../build/test_search_faiss_hnsw_pq_sq --algorithm HNSWPQ HNSWSQ --pq_m 4 8 ----sq_nbits 4 8

echo "==================================="
echo "All tests execution finished successfully!"
echo "==================================="
