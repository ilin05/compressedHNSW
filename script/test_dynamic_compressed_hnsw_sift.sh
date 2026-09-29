#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/datasets/hdf5files}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/build/dynamic_sift}"
BASE="${BASE:-512}"
INSERT="${INSERT:-32}"
QUERIES="${QUERIES:-100}"
CXX="${CXX:-g++}"
train="$DATA_DIR/sift-128-euclidean_train.fvecs"
test="$DATA_DIR/sift-128-euclidean_test.fvecs"
for path in "$train" "$test"; do
  [[ -s "$path" ]] || { echo "Missing SIFT input: $path" >&2; exit 1; }
done
mkdir -p "$OUTPUT_DIR"
"$CXX" -std=c++17 -O2 -fopenmp -DTWO_LEVEL_SEARCH -I "$ROOT" \
  "$ROOT/examples/cpp/test_dynamic_compressed_hnsw.cpp" \
  "$ROOT/encoding_algorithms/elf/elf64_utils.cpp" \
  -o "$OUTPUT_DIR/test_dynamic_compressed_hnsw"
OMP_NUM_THREADS=1 "$OUTPUT_DIR/test_dynamic_compressed_hnsw" \
  "$train" "$test" "$BASE" "$INSERT" "$QUERIES" "$OUTPUT_DIR" \
  | tee "$OUTPUT_DIR/results.log"
