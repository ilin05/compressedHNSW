#!/bin/bash
set -e

echo "Testing Faiss IVF Build"
# Use all datasets and baseline IVF only (no PQ)
../build/test_build_faiss_ivf \
	--dataset fashion-mnist-784-euclidean gist-960-euclidean mnist-784-euclidean sift-128-euclidean bigann_1M bigann_10M \
	--algorithm IVFFlat \
	--nlist 1024 \
	--train_size 200000
