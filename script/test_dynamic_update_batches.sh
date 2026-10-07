#!/usr/bin/env bash
# Usage: DATA_DIR=/path/to/fvecs OUT=/path/to/results PERF_SUDO=1 bash script/test_dynamic_update_batches.sh
# To retry perf without rebuilding/searching: sudo -v; PERF_ONLY=1 PERF_SUDO=1 bash script/test_dynamic_update_batches.sh
# Overrides: BASE=4096 BATCH=256 BATCHES=3 QUERIES=100 PERF_INSERT=1024 SEED=20261007 PERF=1.
# perf samples only the first insertion batch of a separate, identically sampled index.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/datasets/hdf5files}"
OUT="${OUT:-$ROOT/build/dynamic_update_batches}"
BASE="${BASE:-4096}"
BATCH="${BATCH:-256}"
BATCHES="${BATCHES:-3}"
QUERIES="${QUERIES:-100}"
PERF_INSERT="${PERF_INSERT:-1024}"
SEED="${SEED:-20261007}"
PERF="${PERF:-1}"
PERF_ONLY="${PERF_ONLY:-0}"
CXX="${CXX:-g++}"
PERF_BIN="${PERF_BIN:-perf}"
PERF_SUDO="${PERF_SUDO:-0}"
DATASETS="${DATASETS:-sift-128-euclidean mnist-784-euclidean fashion-mnist-784-euclidean gist-960-euclidean}"
CODECS="${CODECS:-raw_f64 raw_f32 dexor gorilla elf alp}"
perf_cmd=("$PERF_BIN")
if [[ "$PERF_SUDO" == 1 ]]; then perf_cmd=(sudo -n "$PERF_BIN"); fi
mkdir -p "$OUT"
if [[ "$PERF_ONLY" == 1 ]]; then
  [[ "$PERF" == 1 && -x "$OUT/test_dynamic_update_batches" && -f "$OUT/all_metrics.csv" ]] || {
    echo "PERF_ONLY requires PERF=1 and an existing completed output directory" >&2; exit 1;
  }
  [[ -f "$OUT/all_perf_breakdown.csv" ]] ||
    printf 'dataset,codec,total_samples,decode_samples,decode_pct,addpoint_samples,addpoint_pct\n' > "$OUT/all_perf_breakdown.csv"
  [[ -f "$OUT/status.csv" ]] || printf 'dataset,codec,normal_status,perf_status\n' > "$OUT/status.csv"
else
  "$CXX" -std=c++17 -O2 -g -fno-omit-frame-pointer -fopenmp -DTWO_LEVEL_SEARCH \
    -I "$ROOT" "$ROOT/examples/cpp/test_dynamic_update_batches.cpp" \
    "$ROOT/encoding_algorithms/elf/elf64_utils.cpp" -o "$OUT/test_dynamic_update_batches"
  printf 'dataset,codec,stage,batch,active,deleted,index_bytes,recall_at_1,mean_query_us,qps,insert_mean_us,delete_mean_us,graph_mean_us,encoding_mean_us,max_insert_abs_error\n' > "$OUT/all_metrics.csv"
  printf 'dataset,codec,total_samples,decode_samples,decode_pct,addpoint_samples,addpoint_pct\n' > "$OUT/all_perf_breakdown.csv"
  printf 'dataset,codec,normal_status,perf_status\n' > "$OUT/status.csv"
fi
if [[ "$PERF" == 1 && "$PERF_SUDO" == 1 ]]; then
  sudo -n true || { echo "sudo credential unavailable; run sudo -v first" >&2; exit 1; }
fi
update_status() {
  local dataset="$1" codec="$2" normal="$3" perf="$4"
  awk -F, -v d="$dataset" -v c="$codec" 'NR==1 || $1!=d || $2!=c' "$OUT/status.csv" > "$OUT/status.tmp"
  mv "$OUT/status.tmp" "$OUT/status.csv"
  printf '%s,%s,%s,%s\n' "$dataset" "$codec" "$normal" "$perf" >> "$OUT/status.csv"
}
for dataset in $DATASETS; do
  train="$DATA_DIR/${dataset}_train.fvecs"
  query="$DATA_DIR/${dataset}_test.fvecs"
  [[ -s "$train" && -s "$query" ]] || { echo "Missing dataset files: $train or $query" >&2; exit 1; }
  for codec in $CODECS; do
    run_dir="$OUT/$dataset/$codec"
    mkdir -p "$run_dir"
    echo "Running $dataset $codec"
    if [[ "$PERF_ONLY" == 1 ]]; then
      [[ -f "$run_dir/metrics.csv" ]] || { echo "Missing metrics: $run_dir/metrics.csv" >&2; continue; }
    else
      if ! OMP_NUM_THREADS=1 "$OUT/test_dynamic_update_batches" "$train" "$query" "$codec" \
        "$BASE" "$BATCH" "$BATCHES" "$QUERIES" "$SEED" "$run_dir" normal > "$run_dir/normal.log" 2>&1; then
        update_status "$dataset" "$codec" FAIL SKIPPED
        echo "FAILED $dataset $codec; see $run_dir/normal.log" >&2
        continue
      fi
      tail -n +2 "$run_dir/metrics.csv" | sed "s/^/$dataset,/" >> "$OUT/all_metrics.csv"
    fi
    if [[ "$PERF" != 1 ]]; then
      update_status "$dataset" "$codec" PASS DISABLED
      continue
    fi
    profile_dir="$run_dir/perf"
    mkdir -p "$profile_dir"
    rm -f "$profile_dir/READY" "$profile_dir/GO" "$profile_dir/DONE" "$profile_dir/EXIT"
    OMP_NUM_THREADS=1 "$OUT/test_dynamic_update_batches" "$train" "$query" "$codec" \
      "$BASE" "$PERF_INSERT" 1 1 "$SEED" "$profile_dir" profile > "$profile_dir/program.log" 2>&1 &
    program_pid=$!
    for ((i=0;i<3600;i++)); do
      [[ -e "$profile_dir/READY" ]] && break
      if ! kill -0 "$program_pid" 2>/dev/null; then cat "$profile_dir/program.log" >&2; exit 1; fi
      sleep 0.1
    done
    [[ -e "$profile_dir/READY" ]] || { echo "Timed out waiting for READY" >&2; exit 1; }
    "${perf_cmd[@]}" record -F 99 -g --call-graph fp -o "$profile_dir/perf.data" \
      -p "$program_pid" > "$profile_dir/record.log" 2>&1 &
    perf_pid=$!
    sleep 1
    if ! kill -0 "$perf_pid" 2>/dev/null; then
      echo "perf attach failed for $dataset $codec; see $profile_dir/record.log" >&2
      touch "$profile_dir/GO" "$profile_dir/EXIT"
      wait "$program_pid" || true
      update_status "$dataset" "$codec" PASS ATTACH_FAILED
      continue
    fi
    touch "$profile_dir/GO"
    for ((i=0;i<3600;i++)); do
      [[ -e "$profile_dir/DONE" ]] && break
      if ! kill -0 "$program_pid" 2>/dev/null; then cat "$profile_dir/program.log" >&2; exit 1; fi
      sleep 0.1
    done
    [[ -e "$profile_dir/DONE" ]] || { echo "Timed out waiting for DONE" >&2; exit 1; }
    kill -INT "$perf_pid" 2>/dev/null || true
    wait "$perf_pid" || true
    touch "$profile_dir/EXIT"
    wait "$program_pid"
    "${perf_cmd[@]}" report --stdio --no-children -i "$profile_dir/perf.data" > "$profile_dir/report.txt" 2> "$profile_dir/report.err" || true
    "${perf_cmd[@]}" script -i "$profile_dir/perf.data" > "$profile_dir/stacks.perf" 2> "$profile_dir/script.err" || true
    awk -F, -v d="$dataset" -v c="$codec" 'NR==1 || $1!=d || $2!=c' \
      "$OUT/all_perf_breakdown.csv" > "$OUT/all_perf_breakdown.tmp"
    mv "$OUT/all_perf_breakdown.tmp" "$OUT/all_perf_breakdown.csv"
    python3 - "$dataset" "$codec" "$profile_dir/stacks.perf" "$OUT/all_perf_breakdown.csv" <<'PY'
import sys
from pathlib import Path
name, codec, source, target = sys.argv[1:]
blocks = Path(source).read_text(errors='replace').split('\n\n')
stacks = [b for b in blocks if b.strip() and not b.lstrip().startswith('#')]
count = len(stacks)
decode = sum('getOriginalDataByInternalId' in b for b in stacks)
add = sum('addPoint' in b for b in stacks)
with open(target, 'a') as out:
    out.write(f'{name},{codec},{count},{decode},{100*decode/count if count else 0:.3f},{add},{100*add/count if count else 0:.3f}\n')
PY
    samples="$(tail -n 1 "$OUT/all_perf_breakdown.csv" | cut -d, -f3)"
    if [[ "$samples" == 0 ]]; then
      update_status "$dataset" "$codec" PASS NO_SAMPLES
    else
      update_status "$dataset" "$codec" PASS PASS
    fi
    rm -f "$profile_dir/stacks.folded" "$profile_dir/flamegraph.svg"
    if [[ "$samples" -gt 0 && -x "$HOME/FlameGraph/stackcollapse-perf.pl" &&
          -x "$HOME/FlameGraph/flamegraph.pl" ]]; then
      if "$HOME/FlameGraph/stackcollapse-perf.pl" "$profile_dir/stacks.perf" > "$profile_dir/stacks.folded" &&
         [[ -s "$profile_dir/stacks.folded" ]]; then
        "$HOME/FlameGraph/flamegraph.pl" "$profile_dir/stacks.folded" > "$profile_dir/flamegraph.svg" || {
          rm -f "$profile_dir/flamegraph.svg"
          echo "FlameGraph failed for $dataset $codec; perf report and stacks are preserved" >&2
        }
      else
        echo "No usable folded stacks for $dataset $codec; perf report and stacks are preserved" >&2
      fi
    fi
  done
done
echo "Metrics: $OUT/all_metrics.csv"
echo "Perf breakdown: $OUT/all_perf_breakdown.csv"
