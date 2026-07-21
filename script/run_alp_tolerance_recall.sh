#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

"${SCRIPT_DIR}/../build/test_alp_tolerance_recall" \
  --base_dir ../datasets/hdf5files/ \
  --dataset deep-image-96-angular \
  --output_csv alp_tolerance_deep_results.csv \
  --M 16 \
  --ef_construction 200 \
  --build_threads 32 \
  --search_threads 1 \
  --num_rounds 3 \
  --k 1 10 \
  --ef 10 20 40 80 120 200 400 600 800 1000 \
  --tolerance 1e-4 1e-10 \
  "$@"
