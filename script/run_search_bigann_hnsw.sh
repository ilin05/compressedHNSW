#!/bin/bash
set -e

echo "Testing hnswlib bigann HNSW Search"
../build/test_search_bigann_hnsw --dataset bigann_1M bigann_10M

