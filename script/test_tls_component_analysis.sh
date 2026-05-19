#!/bin/bash
set -e

echo "=========================================="
echo "Two-Level Search Component Ablation Study"
echo "=========================================="
echo ""

# Run C++ benchmark with multiple ef values, datasets, and TLS ratios
echo "Step 1: Running ablation study benchmark..."
echo "Command: ../build/test_tls_component_analysis"
echo "Parameters:"
echo "  - Datasets: fashion-mnist-784-euclidean, gist-960-euclidean, mnist-784-euclidean, sift-128-euclidean"
echo "  - EF values: 10 20 30 40 50 60 80 100 120 150 200 300"
echo "  - TLS ratios: 0.1 0.2 0.3"
echo "  - Recall@K: 1"
echo ""

../build/test_tls_component_analysis \
  --dataset fashion-mnist-784-euclidean gist-960-euclidean mnist-784-euclidean sift-128-euclidean \
  --efs 10 20 30 40 50 60 80 100 120 150 200 300 \
  --tls-ratios 0.1 0.2 0.3 \
  --k 1

echo ""
echo "✓ Benchmark completed!"
echo "  Results saved to: hnswalp_tls_component_analysis_results.csv"
echo ""