#!/bin/bash
set -e

cd ../build
cmake ..
make -j32

cd ../examples/cpp/

echo "Testing Faiss Build..."
../../build/test_build_faiss_hnsw_pq_sq

echo "Testing Faiss Search..."
../../build/test_search_faiss_hnsw_pq_sq
