#!/bin/bash
set -e

echo "Testing hnswlib compressed HNSW framework Search"
../build/test_search_compressed_hnsw --algorithm Camel Elf Gorilla

