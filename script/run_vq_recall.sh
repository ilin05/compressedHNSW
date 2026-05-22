#!/bin/bash
set -e

echo "Testing VQ-Recall on HNSW / Compressed-HNSW / IVF / NSG"
# Measure recall and QPS with the unified VQ driver
../build/test_vq_recall --base_dir ../datasets/hdf5files/ \
    --dataset sift-128-euclidean deep-image-96-angular \
    --algorithm HNSWALP HNSWALP_0.1 HNSWALP_0.2 HNSWALP_0.3 HNSWALP_0.5 "HNSW(hnswlib)" "HNSW(faiss)" "IVF(faiss)" "NSG(faiss)" HNSWPQ1 HNSWSQ8 --num_rounds 10

    # --dataset fashion-mnist-784-euclidean gist-960-euclidean mnist-784-euclidean sift-128-euclidean deep-image-96-angular \