#!/usr/bin/env bash
# Isolate per-insert decode caching and upper-layer TLS, single threaded.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/datasets/hdf5files}"
OUT="${OUT:-$ROOT/script/alp_insert_optimizations_sift_results}"
BIN="${BIN:-$ROOT/build/dynamic_update_batches/test_alp_insert_optimizations}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT" "$(dirname "$BIN")"
python3 "$ROOT/script/prepare_alp_insert_sift_subset.py" \
  --train "$DATA_DIR/sift-128-euclidean_train.fvecs" \
  --queries "$DATA_DIR/sift-128-euclidean_test.fvecs" \
  --output "$OUT/subset" --count "${COUNT:-100000}" \
  --query-count "${QUERY_COUNT:-0}" --seed "${SEED:-20261022}"
"${CXX:-g++}" -std=c++17 -O3 -g -fopenmp -I "$ROOT" \
  "$ROOT/examples/cpp/test_alp_insert_optimizations.cpp" -o "$BIN"
args=(--input "$OUT/subset/train.fvecs" --queries "$OUT/subset/test.fvecs"
      --gt "$OUT/subset/neighbors.ivecs" --output "$OUT"
      --rounds "${ROUNDS:-10}" --delete-ratio 0.05
      --search-rounds "${SEARCH_ROUNDS:-3}" --ef 40
      --seed "${SEED:-20261022}" --insert-pq-alpha "${BASE_ALPHA:-0.2}"
      --upper-alpha "${UPPER_ALPHA:-0.2}" --insert-use-pq-filter "${BASE_TLS:-1}"
      --query-pq-alpha 1)
if [[ "${KEEP_INDEX:-0}" == 1 ]]; then args+=(--keep-index); fi
python3 - "$OUT/config.json" "${args[@]}" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps(dict(arguments=sys.argv[2:], threads=1,
    query_tls=False, recall="exact subset GT, current version labels mapped to subset IDs"), indent=2)+"\n")
PY
"$BIN" "${args[@]}" 2>&1 | tee "$OUT/run.log"
python3 "$ROOT/script/summarize_alp_insert_optimizations.py" "$OUT"
