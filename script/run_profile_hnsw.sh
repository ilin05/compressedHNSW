#!/bin/bash
set -e

echo "Profiling hnswlib node-level statistics"
../build/test_profile_hnsw_node_stats "$@"

echo "Generating plots"
python3 ../script/plot_hnsw_level_stats.py --input_glob "hnsw_node_level_access_degree_*.csv"
