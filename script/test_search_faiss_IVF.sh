#!/bin/bash
set -e

echo "Testing Faiss IVF Search"
# Use all datasets and baseline IVF only (no PQ)
../build/test_search_faiss_ivf \
	--dataset fashion-mnist-784-euclidean gist-960-euclidean mnist-784-euclidean sift-128-euclidean bigann_1M bigann_10M \
	--algorithm IVFFlat \
	--nlist 1024 \
	--nprobe 1 2 4 8 16 32 48 64
