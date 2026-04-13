#!/bin/bash
set -e

echo "Testing hnswlib bigann compressed HNSW framework Search"
../build/test_search_bigann_compressed_hnsw --algorithm Camel Elf Gorilla --dataset bigann_1M bigann_10M

