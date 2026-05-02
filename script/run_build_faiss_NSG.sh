#!/bin/bash
set -e

echo "Testing Faiss NSG Build"
# Build NSG indexes for the four in-memory datasets
../build/test_build_faiss_nsg \
	--dataset fashion-mnist-784-euclidean gist-960-euclidean mnist-784-euclidean sift-128-euclidean \
	--R 32 \
	--normal_base_dir ../datasets/hdf5files/
