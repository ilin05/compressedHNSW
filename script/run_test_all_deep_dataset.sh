#!/bin/bash
set -e

echo "Running all HNSW tests"

echo "==================================="
echo "1. Running Build Tests"
echo "==================================="
# bash run_build_hnsw.sh
echo "Testing HNSW Build"
../build/test_build_hnsw_float --dataset deep-image-96-angular_train.fvecs
echo "Testing HNSW_ALP_PQ Build"
../build/test_build_hnswalp_pq --dataset deep-image-96-angular_train.fvecs
bash run_build_compressed_hnsw_ablation.sh
echo "Testing Faiss HNSW Build"
../build/test_build_faiss_hnsw --dataset deep-image-96-angular --base_dir ../datasets/hdf5files/
echo "Testing Faiss HNSWPQ Build"
../build/test_build_faiss_hnsw_pq_sq --algorithm HNSWPQ --pq_m 1 --dataset deep-image-96-angular_train.fvecs
echo "Testing Faiss HNSWSQ Build"
../build/test_build_faiss_hnsw_pq_sq --algorithm HNSWSQ --sq_nbits 8 --dataset deep-image-96-angular_train.fvecs
echo "Testing Faiss NSG Build"
../build/test_build_faiss_nsg --dataset deep-image-96-angular --R 32 --normal_base_dir ../datasets/hdf5files/
echo "Testing Faiss IVF Build"
../build/test_build_faiss_ivf --dataset deep-image-96-angular --algorithm IVFFlat --nlist 1024 --train_size 200000
# bash run_build_hnswdexor_pq.sh
# bash run_build_faiss_hnsw_PQ.sh
# bash run_build_faiss_hnsw_SQ.sh
# bash test_build_compressed_hnsw_framework/run_build_compressed_hnsw_framework.sh

echo "==================================="
echo "2. Running Search Tests"
echo "==================================="
echo "Testing VQ-Recall on HNSW / Compressed-HNSW / IVF / NSG"
# Measure recall and QPS with the unified VQ driver
../build/test_vq_recall \
	--base_dir ../datasets/hdf5files/ \
	--dataset deep-image-96-angular \
	--algorithm HNSWALP HNSWALP_0.1 HNSWALP_0.2 HNSWALP_0.3 HNSWALP_0.5 "HNSW(hnswlib)" "HNSW(faiss)" "IVF(faiss)" "NSG(faiss)" HNSWPQ1 HNSWSQ8


echo "==================================="
echo "All tests execution finished successfully!"
echo "==================================="
