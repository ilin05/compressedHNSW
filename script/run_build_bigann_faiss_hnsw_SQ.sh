#!/bin/bash
set -e

echo "Testing Faiss HNSWSQ Build"
../build/test_build_bigann_faiss_hnsw_pq_sq --dataset bigann_1M bigann_10M --sq_nbits 4 8

