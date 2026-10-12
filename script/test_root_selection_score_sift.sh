#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="${BUILD_DIR:-$repo/build_lvc_root_score}"
data="${DATA_DIR:-$repo/datasets/hdf5files}"
default_result="$repo/script/root_selection_score_results"
[[ "${SMOKE:-0}" == 1 ]] && default_result="$repo/script/root_selection_score_smoke_results"
out="${RESULT_DIR:-$default_result}"
dataset=sift-128-euclidean
mkdir -p "$out"

if [[ "${SMOKE:-0}" == 1 ]]; then
  data="$out/smoke_data"
  mkdir -p "$data"
  DATA_OUT="$data" python3 - <<'PY'
import os, random, struct
from pathlib import Path
p = Path(os.environ['DATA_OUT'])
rng = random.Random(20261012)
base = [[float(rng.randrange(256)) for _ in range(128)] for _ in range(512)]
queries = [[v + float((j % 3) - 1) for j, v in enumerate(base[i])]
           for i in range(10)]
def write_fvecs(path, rows):
    with path.open('wb') as f:
        for row in rows:
            f.write(struct.pack('<i', len(row)))
            f.write(struct.pack('<%df' % len(row), *row))
write_fvecs(p / 'sift-128-euclidean_train.fvecs', base)
write_fvecs(p / 'sift-128-euclidean_test.fvecs', queries)
with (p / 'sift-128-euclidean_neighbors.ivecs').open('wb') as f:
    for q in queries:
        best = min(range(len(base)), key=lambda i: sum((a-b)**2 for a,b in zip(q,base[i])))
        f.write(struct.pack('<ii', 1, best))
PY
fi
for suffix in train.fvecs test.fvecs neighbors.ivecs; do
  [[ -s "$data/${dataset}_${suffix}" ]] || { echo "Missing SIFT input: $data/${dataset}_${suffix}" >&2; exit 1; }
done

if [[ "${SKIP_CMAKE:-0}" != 1 ]]; then
  cmake -S "$repo" -B "$build" -DHNSWLIB_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build "$build" --target test_root_access_trace test_build_compressed_hnsw_ablation test_search_compressed_hnsw_ablation -j "${BUILD_JOBS:-4}"
fi

if [[ ! -s "$out/access_counts.txt" || ! -s "$out/eval_query_ids.txt" || ! -s "$out/shared_raw_graph.bin" || "${FORCE_TRACE:-0}" == 1 ]]; then
  "$build/test_root_access_trace" \
    --train "$data/${dataset}_train.fvecs" --queries "$data/${dataset}_test.fvecs" \
    --output "$out" --threads "${BUILD_THREADS:-32}" --M 16 --ef-construction 200 --ef 100 \
    --seed 20261012 > "$out/access_trace.log" 2>&1
fi

for policy in level degree access random; do
  (
    cd "$out"
    verify=(--verify_sample "${VERIFY_SAMPLE:-4096}")
    [[ "${SMOKE:-0}" == 1 || "${VERIFY_ALL:-0}" == 1 ]] && verify=(--verify_vectors)
    "$build/test_build_compressed_hnsw_ablation" \
      --base_dir "$data" --dataset "$dataset" --algorithm DeXOR Gorilla Elf \
      --chain_max 2 -1 --root_policy "$policy" --f32_lossless \
      --shared_graph_file "$out/shared_raw_graph.bin" --access_counts "$out/access_counts.txt" \
      "${verify[@]}" --output_csv "build_${policy}.csv" > "build_${policy}.log" 2>&1
    "$build/test_search_compressed_hnsw_ablation" \
      --base_dir "$data" --dataset "$dataset" --algorithm DeXOR Gorilla Elf \
      --chain_max 2 -1 --root_policy "$policy" --use_cache 0 --use_tls 0 \
      --ef 40 --k 1 --num_rounds 3 --enable_profiling 1 --no_early_stop \
      --query_ids_file "$out/eval_query_ids.txt" --output_csv "search_${policy}.csv" \
      > "search_${policy}.log" 2>&1
  )
done

OUT_DIR="$out" python3 - <<'PY'
import csv, os
from pathlib import Path
out = Path(os.environ['OUT_DIR'])
for kind in ('build', 'search'):
    rows = []
    fields = None
    for policy in ('level', 'degree', 'access', 'random'):
        with (out / f'{kind}_{policy}.csv').open(newline='') as f:
            reader = csv.DictReader(f)
            if fields is None: fields = reader.fieldnames
            assert reader.fieldnames == fields
            part = list(reader)
            assert len(part) == 6, (kind, policy, len(part))
            rows.extend(part)
    with (out / f'{kind}_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'{kind}: {len(rows)} configurations -> {out / (kind + "_summary.csv")}')

expected_roots = {}
for row in csv.DictReader((out / 'build_summary.csv').open(newline='')):
    forest = out / (row['IndexFile'] + '.forest.csv')
    n = int(row['Vectors'])
    parents = [-2] * n
    depths = [-1] * n
    with forest.open(newline='') as f:
        for node in csv.DictReader(f):
            i = int(node['internal_id'])
            assert 0 <= i < n and parents[i] == -2, (forest, i)
            parents[i] = -1 if not node['parent_internal_id'] else int(node['parent_internal_id'])
            depths[i] = int(node['depth'])
    assert all(p != -2 for p in parents), forest
    roots = {i for i, p in enumerate(parents) if p == -1}
    assert len(roots) == int(row['Roots']), forest
    assert max(depths) == int(row['MaxDepth']), forest
    for i, parent in enumerate(parents):
        if parent == -1:
            assert depths[i] == 0, (forest, i)
        else:
            assert 0 <= parent < n and depths[i] == depths[parent] + 1, (forest, i)
    if row['ChainMaxLength'] == '2':
        assert max(depths) <= 1, forest
    policy = row['RootPolicy']
    if policy in expected_roots:
        assert roots == expected_roots[policy], (forest, policy)
    else:
        expected_roots[policy] = roots
with (out / 'access_counts.txt').open() as f:
    counts = [int(line.split()[1]) for line in f]
assert len(counts) == n
top_access = set(sorted(range(n), key=lambda i: (-counts[i], i))[:len(expected_roots['access'])])
assert expected_roots['access'] == top_access, 'Access-frequency roots do not match query trace'
print('PASS: all 24 forests have consistent roots, valid parent depths, and chain bounds')
print('PASS: access-frequency roots are the highest-scoring traced nodes')
PY
