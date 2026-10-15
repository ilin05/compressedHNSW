#!/usr/bin/env bash
# Full SIFT 1M by default. COUNT/QUERY_COUNT are explicit smoke-test overrides.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/datasets/hdf5files}"
OUT="${OUT:-$ROOT/script/alp_insert_filter_sift_results}"
BIN="${BIN:-$ROOT/build/dynamic_update_batches/test_alp_insert_filter}"
GT_FILE="${GT_FILE:-$DATA_DIR/sift-128-euclidean_neighbors.ivecs}"
INSERT_ALPHA="${INSERT_ALPHA:-0.2}"
QUERY_ALPHA="${QUERY_ALPHA:-0.2}"
EF_SEARCH="${EF_SEARCH:-20 40 80}"
read -r -a ef_values <<< "$EF_SEARCH"
mkdir -p "$OUT" "$(dirname "$BIN")"
"${CXX:-g++}" -std=c++17 -O3 -g -fopenmp -I "$ROOT" \
  "$ROOT/examples/cpp/test_alp_insert_filter.cpp" -o "$BIN"
args=(--input "$DATA_DIR/sift-128-euclidean_train.fvecs"
      --queries "$DATA_DIR/sift-128-euclidean_test.fvecs" --gt "$GT_FILE" --output "$OUT"
      --rounds 2 --delete-ratio 0.05 --search-rounds "${SEARCH_ROUNDS:-3}"
      --seed "${SEED:-20261015}" --insert-pq-alpha "$INSERT_ALPHA"
      --insert-use-pq-filter "${INSERT_FILTER:-1}"
      --query-pq-alpha "$QUERY_ALPHA" --count "${COUNT:-0}" --query-count "${QUERY_COUNT:-0}"
      --ef "${ef_values[@]}")
if [[ "${KEEP_INDEX:-0}" == 1 ]]; then args+=(--keep-index); fi
python3 - "$OUT/config.json" "${args[@]}" <<'PY'
import json, struct, sys
from pathlib import Path
args = sys.argv[2:]
def value(flag):
    return args[args.index(flag) + 1]
shapes = {}
for flag in ['--input', '--queries']:
    path = Path(value(flag))
    with path.open('rb') as handle:
        dim, = struct.unpack('<i', handle.read(4))
    assert dim > 0 and path.stat().st_size % (4*(dim+1)) == 0, path
    shapes[flag] = (path.stat().st_size // (4*(dim+1)), dim)
if value('--count') == '0':
    assert shapes['--input'] == (1000000, 128), f'Expected full SIFT 1M: {shapes}'
print(f'Input shapes: {shapes}; COUNT=0 means full SIFT 1M', flush=True)
Path(sys.argv[1]).write_text(json.dumps({"arguments": sys.argv[2:],
    "input_shapes": shapes,
    "modes": ["raw_f32", "alp_off", "alp_on", "alp_alpha1"],
    "recall": "original source-ID GT; reject inactive version labels before mapping"}, indent=2)+"\n")
PY
OMP_NUM_THREADS="${BUILD_THREADS:-1}" "$BIN" "${args[@]}" 2>&1 | tee "$OUT/run.log"
python3 - "$OUT" <<'PY'
import csv, statistics, sys
from collections import defaultdict
from pathlib import Path
root = Path(sys.argv[1])
rows = list(csv.DictReader((root / "metrics.csv").open()))
groups = defaultdict(list)
for row in rows:
    groups[row['mode'], row['batch'], row['ef_search']].append(row)
assert {r['mode'] for r in rows} == {'raw_f32','alp_off','alp_on','alp_alpha1'}
for mode in {'raw_f32','alp_off','alp_on','alp_alpha1'}:
    assert {int(r['batch']) for r in rows if r['mode'] == mode} == {0,1,2}
    for row in rows:
        if row['mode'] != mode or int(row['batch']) == 0: continue
        if mode in {'raw_f32','alp_off','alp_alpha1'}:
            assert int(row['filter_searches']) == 0, (mode, row)
        if mode == 'alp_on':
            if row['insert_filter'] == '1' and float(row['insert_alpha']) < 1:
                assert int(row['filter_searches']) > 0 and int(row['pq_candidates']) > 0
                assert int(row['exact_candidates']) > 0
for path in root.glob('alp_off_b*.labels'):
    other = root / path.name.replace('alp_off_', 'alp_alpha1_', 1)
    assert other.is_file() and path.read_bytes() == other.read_bytes(), f'alpha=1 label mismatch: {path}'
    dist = path.with_suffix('.distances')
    assert dist.read_bytes() == other.with_suffix('.distances').read_bytes(), f'alpha=1 L2 mismatch: {path}'
assert len(list(root.glob('alp_off_b*.labels'))) == sum(len(v) for k,v in groups.items() if k[0]=='alp_off')
fields = ['mode','batch','ef_search','index_file_bytes','active_nodes','physical_nodes','deleted_nodes',
          'insert_mean_us','delete_mean_us','recall_at_1','recall_at_1_ties','qps','mean_query_us',
          'filter_searches','pq_candidates','exact_candidates','fallback_searches','insert_payload_decodes','max_insert_abs_error',
          'verified_live_vectors','max_live_abs_error']
with (root/'summary.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for key, samples in groups.items():
        row = {field:samples[0][field] for field in fields}
        for field in ['recall_at_1','recall_at_1_ties','qps','mean_query_us']:
            row[field] = statistics.mean(float(s[field]) for s in samples)
        writer.writerow(row)
        print(f"{key}: insert_us={row['insert_mean_us']} recall@1={row['recall_at_1']:.6f} QPS={row['qps']:.3f}")
print('PASS SIFT repeated updates, deletion exclusion, F32 reconstruction and off/alpha=1 equivalence')
print(f'Results: {root / "summary.csv"}')
PY
