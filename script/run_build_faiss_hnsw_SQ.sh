#!/bin/bash
set -e

echo "Testing Faiss HNSWSQ Build"
../build/test_build_faiss_hnsw_pq_sq --algorithm HNSWSQ --sq_nbits 4 8

