#!/bin/bash
set -e

echo "Testing hnswlib bigann compressed HNSW framework Build"
../build/test_build_bigann_compressed_hnsw --algorithm Camel Gorilla Elf --dataset bigann_1M bigann_10M

