#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
BIN="${BIN:-${BUILD_DIR}/test_perf_breakdown_hnsw}"
BASE_DIR="${BASE_DIR:-${ROOT_DIR}/datasets/hdf5files}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/perf_breakdown_hnsw_sift}"

DATASET="${DATASET:-sift-128-euclidean}"
EF="${EF:-80}"
K="${K:-1}"
NUM_ROUNDS="${NUM_ROUNDS:-5}"
TLS_RATIO="${TLS_RATIO:-0.2}"

mkdir -p "${OUT_DIR}"

SUMMARY_CSV="${OUT_DIR}/perf_breakdown_hnsw_summary.csv"
rm -f "${SUMMARY_CSV}"

MODES=(
  dexor_noopt
  dexor_opt
  hnswalp_tls
  hnsw
)

PERF_EVENTS="${PERF_EVENTS:-task-clock,cycles,instructions,branches,branch-misses,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses}"

if [[ ! -x "${BIN}" ]]; then
  echo "Executable not found: ${BIN}"
  echo "Build it first, for example: cmake --build ${BUILD_DIR} --target test_perf_breakdown_hnsw -j"
  exit 1
fi

echo "[Perf Breakdown] dataset=${DATASET}, ef=${EF}, rounds=${NUM_ROUNDS}, threads=1"
echo "[Perf Breakdown] output=${OUT_DIR}"

cd "${SCRIPT_DIR}"

append=0
for mode in "${MODES[@]}"; do
  echo
  echo "=== perf stat: ${mode} ==="
  stat_csv="${OUT_DIR}/${mode}_perf_stat.csv"
  log_file="${OUT_DIR}/${mode}.log"

  OMP_NUM_THREADS=1 perf stat \
    -x, \
    -o "${stat_csv}" \
    -e "${PERF_EVENTS}" \
    -- "${BIN}" \
      --base_dir "${BASE_DIR}" \
      --dataset "${DATASET}" \
      --mode "${mode}" \
      --ef "${EF}" \
      --k "${K}" \
      --num_rounds "${NUM_ROUNDS}" \
      --tls_ratio "${TLS_RATIO}" \
      --output_csv "${SUMMARY_CSV}" \
      --append "${append}" \
      > "${log_file}" 2>&1

  append=1
  cat "${log_file}"
done

for mode in "${MODES[@]}"; do
  echo
  echo "=== perf record/report: ${mode} ==="
  data_file="${OUT_DIR}/${mode}.perf.data"
  report_file="${OUT_DIR}/${mode}.perf.report.txt"
  record_log="${OUT_DIR}/${mode}.perf_record.log"

  OMP_NUM_THREADS=1 perf record \
    -F 99 \
    --call-graph dwarf,16384 \
    -o "${data_file}" \
    -- "${BIN}" \
      --base_dir "${BASE_DIR}" \
      --dataset "${DATASET}" \
      --mode "${mode}" \
      --ef "${EF}" \
      --k "${K}" \
      --num_rounds "${NUM_ROUNDS}" \
      --tls_ratio "${TLS_RATIO}" \
      --write_csv 0 \
      > "${record_log}" 2>&1

  perf report \
    --stdio \
    --sort comm,dso,symbol \
    --percent-limit 0.5 \
    -i "${data_file}" \
    > "${report_file}"

  echo "Saved ${data_file}"
  echo "Saved ${report_file}"
done

echo
echo "Saved summary CSV: ${SUMMARY_CSV}"
