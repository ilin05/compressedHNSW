#!/bin/bash
set -e

echo "Testing Faiss NSG Search"
# Use all datasets and baseline IVF only (no PQ)
../build/test_search_faiss_nsg \
	--dataset fashion-mnist-784-euclidean gist-960-euclidean mnist-784-euclidean sift-128-euclidean \
