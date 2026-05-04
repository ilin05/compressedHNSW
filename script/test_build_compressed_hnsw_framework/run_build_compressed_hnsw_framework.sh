#!/bin/bash
set -e

echo "Testing hnswlib compressed HNSW framework Build"
../build/test_build_compressed_hnsw --algorithm Camel Gorilla Elf

