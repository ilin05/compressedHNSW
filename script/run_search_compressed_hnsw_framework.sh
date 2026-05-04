#!/bin/bash
set -e

echo "Testing hnswlib compressed_hnsw_framework Search"
../build/test_search_compressed_hnsw --use_tls 1 \
    --tls_ratio 0.2 \
    --dataset fashion-mnist-784-euclidean gist-960-euclidean mnist-784-euclidean sift-128-euclidean \
    --algorithm Gorilla Camel Elf

