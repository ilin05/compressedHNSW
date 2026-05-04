#!/bin/bash
set -e

echo "Testing Faiss HNSW Build"
# Build original Faiss HNSW indexes with the same M / efConstruction as hnswlib HNSW
../build/test_build_faiss_hnsw \
	--dataset fashion-mnist-784-euclidean_train.fvecs gist-960-euclidean_train.fvecs mnist-784-euclidean_train.fvecs sift-128-euclidean_train.fvecs \
	--base_dir ../datasets/hdf5files/
