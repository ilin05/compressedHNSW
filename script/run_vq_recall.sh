#!/bin/bash
set -e

bash run_build_faiss_HNSW.sh

echo "Testing VQ-Recall on HNSW / Compressed-HNSW / IVF / NSG"
# Measure recall and QPS with the unified VQ driver
../build/test_vq_recall \
	--base_dir ../datasets/hdf5files/ \
	--dataset fashion-mnist-784-euclidean gist-960-euclidean mnist-784-euclidean sift-128-euclidean
