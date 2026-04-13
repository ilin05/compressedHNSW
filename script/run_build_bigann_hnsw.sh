#!/bin/bash
set -e

echo "Testing hnswlib bigann HNSW Build"
../build/test_build_bigann_hnsw --dataset bigann_1M bigann_10M

