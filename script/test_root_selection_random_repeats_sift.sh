#!/usr/bin/env bash
set -euo pipefail

# Same SIFT graph and 1% roots as test_root_selection_score_sift.sh.
# Only random roots change; no query, access-trace, or search executable is run.
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="${BUILD_DIR:-$repo/build_lvc_root_score}"
data="${DATA_DIR:-$repo/datasets/hdf5files}"
graph="${SHARED_GRAPH_FILE:-$repo/script/root_selection_score_results/shared_raw_graph.bin}"
out="${RESULT_DIR:-$repo/script/root_selection_random_repeats_results/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$out"

if [[ "${SMOKE:-0}" == 1 ]]; then
  data="$out/smoke_data"
  mkdir -p "$data"
  DATA_OUT="$data" python3 - <<'PY'
import os, random, struct
from pathlib import Path
rng = random.Random(20261116)
path = Path(os.environ['DATA_OUT']) / 'sift-128-euclidean_train.fvecs'
with path.open('wb') as f:
    for _ in range(512):
        f.write(struct.pack('<i128f', 128, *(float(rng.randrange(256)) for _ in range(128))))
PY
else
  [[ -s "$graph" ]] || { echo "Missing original shared graph: $graph (set SHARED_GRAPH_FILE)." >&2; exit 1; }
fi
[[ -s "$data/sift-128-euclidean_train.fvecs" ]] || { echo "Missing SIFT train.fvecs in $data" >&2; exit 1; }
[[ ! -e "$out/summary.csv" ]] || { echo "Results already exist; use a new RESULT_DIR: $out" >&2; exit 1; }

if [[ "${SKIP_CMAKE:-0}" != 1 ]]; then
  cmake -S "$repo" -B "$build" -DHNSWLIB_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build "$build" --target test_build_compressed_hnsw_ablation -j "${BUILD_JOBS:-4}"
fi

RUNNER_BUILD="$build" RUNNER_DATA="$data" RUNNER_GRAPH="$graph" RUNNER_OUT="$out" python3 - <<'PY'
import csv
import math
import os
import re
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

out = Path(os.environ['RUNNER_OUT']).resolve()
data = Path(os.environ['RUNNER_DATA']).resolve()
graph = Path(os.environ['RUNNER_GRAPH']).resolve()
binary = Path(os.environ['RUNNER_BUILD']).resolve() / 'test_build_compressed_hnsw_ablation'
runs = int(os.environ.get('RUNS', '20'))
start_seed = int(os.environ.get('START_SEED', '20261116'))
threads = int(os.environ.get('BUILD_THREADS', '32'))
algorithms = os.environ.get('ALGORITHMS', 'DeXOR').split()
chains = [int(v) for v in os.environ.get('CHAIN_MAX', '-1').split()]
smoke = os.environ.get('SMOKE', '0') == '1'
if runs < 1 or threads < 1 or not 0 <= start_seed <= start_seed + runs - 1 <= 2**32 - 1:
    raise ValueError('RUNS/BUILD_THREADS must be positive and all seeds must fit uint32')
if (not algorithms or len(set(algorithms)) != len(algorithms)
        or not set(algorithms) <= {'DeXOR', 'Gorilla', 'Elf'}):
    raise ValueError('ALGORITHMS must contain unique names from DeXOR Gorilla Elf')
if not chains or len(set(chains)) != len(chains) or any(v != -1 and v < 2 for v in chains):
    raise ValueError('CHAIN_MAX must contain unique values: -1 (unbounded), or >=2')
if not binary.is_file():
    raise FileNotFoundError(binary)

fields = ['Run', 'Seed', 'Dataset', 'Algorithm', 'ChainMaxLength',
          'PayloadRatioF32', 'FullIndexRatioF32']
groups = defaultdict(list)
root_signatures = defaultdict(set)
raw_csv = out / 'current_build.csv'
with (out / 'summary.csv').open('x', newline='') as result:
    writer = csv.DictWriter(result, fieldnames=fields)
    writer.writeheader()
    result.flush()
    for run in range(1, runs + 1):
        seed = start_seed + run - 1
        command = [str(binary), '--base_dir', str(data), '--dataset', 'sift-128-euclidean',
                   '--algorithm', *algorithms, '--chain_max', *(str(v) for v in chains),
                   '--root_policy', 'random', '--root_seed', str(seed), '--f32_lossless',
                   '--ratios_only', '--threads', str(threads), '--M', '16',
                   '--ef_construction', '200', '--output_csv', str(raw_csv)]
        if smoke and run == 1:
            # Build a tiny raw graph once, then reuse it in every other run.
            command += ['--reuse_graph']
        else:
            command += ['--shared_graph_file', str(graph)]
        if smoke:
            command += ['--verify_sample', '32']
        log_path = out / ('run_%02d_seed%d.log' % (run, seed))
        print('RUN %d/%d seed=%d codecs=%s chains=%s' % (run, runs, seed, algorithms, chains), flush=True)
        with log_path.open('w') as log:
            log.write('COMMAND: ' + repr(command) + '\n')
            log.flush()
            completed = subprocess.run(command, cwd=out, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError('Build failed (exit %d); see %s' % (completed.returncode, log_path))
        if smoke and run == 1:
            graph = out / 'sift-128-euclidean_shared_raw_graph.bin'
        with raw_csv.open(newline='') as f:
            rows = list(csv.DictReader(f))
        expected = {(a, c) for a in algorithms for c in chains}
        actual = {(r['Algorithm'], int(r['ChainMaxLength'])) for r in rows}
        if len(rows) != len(expected) or actual != expected:
            raise RuntimeError('Incomplete codec/chain matrix: %s' % raw_csv)
        for row in rows:
            if row['RootPolicy'] != 'random' or int(row['Vectors']) < 1:
                raise RuntimeError('Unexpected build row: %r' % row)
            # Recompute from byte counts, avoiding the builder CSV's 6-digit rounding.
            payload_ratio = int(row['RawF32Bytes']) / int(row['PayloadBytes'])
            index_ratio = int(row['RawF32IndexBytes']) / int(row['CompressedIndexBytes'])
            if not all(math.isfinite(x) and x > 0 for x in (payload_ratio, index_ratio)):
                raise RuntimeError('Invalid compression ratio')
            writer.writerow(dict(Run=run, Seed=seed, Dataset=row['Dataset'],
                Algorithm=row['Algorithm'], ChainMaxLength=row['ChainMaxLength'],
                PayloadRatioF32='%.12f' % payload_ratio, FullIndexRatioF32='%.12f' % index_ratio))
            groups[(row['Algorithm'], int(row['ChainMaxLength']))].append((payload_ratio, index_ratio))
        result.flush()
        os.fsync(result.fileno())
        signatures = re.findall(r'ROOT_SAMPLE seed=(\d+) roots=(\d+) hash=(\d+)', log_path.read_text())
        if len(signatures) != len(expected) or any(int(s) != seed for s, _, _ in signatures):
            raise RuntimeError('Missing root-seed diagnostics; rebuild the executable: %s' % log_path)
        for row, (_, count, signature) in zip(rows, signatures):
            if int(count) != int(row['Roots']):
                raise RuntimeError('Root count mismatch')
            root_signatures[int(row['ChainMaxLength'])].add(signature)
        raw_csv.unlink()

with (out / 'statistics.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Algorithm', 'ChainMaxLength', 'Runs', 'Metric',
                     'Mean', 'SampleStd', 'Min', 'Max', 'RelativeSpreadPercent'])
    for (algorithm, chain), values in sorted(groups.items()):
        for i, metric in enumerate(['PayloadRatioF32', 'FullIndexRatioF32']):
            xs = [v[i] for v in values]
            mean = statistics.mean(xs)
            writer.writerow([algorithm, chain, len(xs), metric,
                '%.12f' % mean, '%.12f' % (statistics.stdev(xs) if len(xs) > 1 else 0),
                '%.12f' % min(xs), '%.12f' % max(xs), '%.9f' % ((max(xs)-min(xs))/mean*100)])
for chain, signatures in sorted(root_signatures.items()):
    print('chain=%d distinct root sets=%d/%d' % (chain, len(signatures), runs), flush=True)
    if runs > 1 and len(signatures) != runs:
        raise RuntimeError('Repeated root sets detected; inspect run logs before interpreting variance')
print('Results:', out / 'summary.csv', flush=True)
print('Statistics:', out / 'statistics.csv', flush=True)
PY
