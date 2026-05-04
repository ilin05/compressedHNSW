#!/bin/bash
set -e

echo "Running all ablation tests"

echo "==================================="
echo "1. Running Build ablation tests"
echo "==================================="
bash run_build_compressed_hnsw_ablation.sh


echo "==================================="
echo "2. Running Search ablation tests"
echo "==================================="
bash run_search_compressed_hnsw_ablation.sh


echo "==================================="
echo "All tests execution finished successfully!"
echo "==================================="
