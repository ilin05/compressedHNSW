#!/bin/bash
set -e

echo "Testing bigann compressed_hnsw_framework Build"
../build/test_build_bigann_compressed_hnsw_framework \
    --dataset bigann_1M bigann_10M \
    --algorithm Gorilla Camel Elf

