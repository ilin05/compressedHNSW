#!/bin/bash
set -e

echo "Testing Faiss bigann HNSWSQ Search"
../build/test_search_bigann_faiss_hnsw_pq_sq --algorithm HNSWSQ --sq_nbits 4 8 --dataset bigann_1M bigann_10M

