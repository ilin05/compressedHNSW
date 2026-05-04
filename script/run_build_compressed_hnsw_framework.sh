#!/bin/bash
set -e

echo "Testing compressed_hnsw_framework Build"
../build/test_build_compressed_hnsw_framework \
    --dataset fashion-mnist-784-euclidean gist-960-euclidean mnist-784-euclidean sift-128-euclidean bigann_1M bigann_10M \
    --algorithm Gorilla Camel Elf

