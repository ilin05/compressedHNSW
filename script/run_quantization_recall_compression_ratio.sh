#!/bin/bash
set -e

BASE_DIR=${BASE_DIR:-../datasets/hdf5files/}
DATASETS=${DATASETS:-"gist-960-euclidean sift-128-euclidean"}
METHODS=${METHODS:-"SQ PQ"}
COMPRESSION_RATIOS=${COMPRESSION_RATIOS:-"1 2 4 8 16"}
EFS=${EFS:-"10 20 40 80 120 200 400"}
K=${K:-1}
NUM_ROUNDS=${NUM_ROUNDS:-1}
BUILD_THREADS=${BUILD_THREADS:-8}
SEARCH_THREADS=${SEARCH_THREADS:-1}
OUTPUT_CSV=${OUTPUT_CSV:-quantization_recall_compression_ratio_results.csv}

echo "Testing FAISS HNSW quantization recall vs compression ratio"
echo "Datasets: ${DATASETS}"
echo "Methods: ${METHODS}"
echo "Compression ratios: ${COMPRESSION_RATIOS}"
echo "efs: ${EFS}"

../build/test_quantization_recall_compression_ratio \
    --base_dir "${BASE_DIR}" \
    --dataset ${DATASETS} \
    --methods ${METHODS} \
    --compression_ratios ${COMPRESSION_RATIOS} \
    --efs ${EFS} \
    --k "${K}" \
    --num_rounds "${NUM_ROUNDS}" \
    --build_threads "${BUILD_THREADS}" \
    --search_threads "${SEARCH_THREADS}" \
    --output_csv "${OUTPUT_CSV}"
