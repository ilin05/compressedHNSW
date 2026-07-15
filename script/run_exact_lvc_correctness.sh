#!/usr/bin/env bash
set -euo pipefail

echo "[Exact-LVC] Bitwise payload validation and search-equivalence test"
echo "[Exact-LVC] TLS is disabled. Payload validation never gates top-k comparison."

../build/test_exact_lvc_correctness \
  --base_dir ../datasets/hdf5files/ \
  --dataset sift-128-euclidean gist-960-euclidean \
  --algorithm DeXOR Gorilla Camel Elf \
  --chain_max 2 \
  --cache_ratio 0.01 \
  --ef 80 \
  --k 1 10 \
  --output_prefix exact_lvc_correctness \
  "$@"

echo "Saved: exact_lvc_correctness_payload.csv"
echo "Saved: exact_lvc_correctness_search.csv"
