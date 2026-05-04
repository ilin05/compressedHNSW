#!/bin/bash
set -e

echo "Testing hnswlib compressed_hnsw_framework Search"
../build/test_search_bigann_compressed_hnsw --use_tls 1 \
    --tls_ratio 0.2 \
    --dataset bigann_1M bigann_10M \
    --algorithm Gorilla Camel Elf

