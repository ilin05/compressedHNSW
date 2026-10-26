#!/usr/bin/env bash
# Measure the new M=dim/4bit FastScan indexes, replace old FastScan rows, and plot.
set -euo pipefail

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}
BUILD_DIR=${BUILD_DIR:-"$REPO/build_lvc_ivf_pq"}
FASTSCAN_SUMMARY=${FASTSCAN_SUMMARY:-"$REPO/script/revision_fastscan_mdim_build_results/summary.csv"}
BASELINE_CSV=${BASELINE_CSV:-"$REPO/script/revision_baseline_build_results/storage_breakdown.csv"}
HNSW_CSV=${HNSW_CSV:-"$REPO/script/index_storage_breakdown.csv"}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO/script/revision_storage_breakdown_results"}
FIGURE_DIR=${FIGURE_DIR:-"$REPO/script/index_size_revision_plots"}
INDEX_DIR=${INDEX_DIR:-"$REPO/script"}
BASE_DIR=${BASE_DIR:-"$REPO/datasets/hdf5files"}
REPROFILE_HNSW=${REPROFILE_HNSW:-0}
SKIP_COMPILE=${SKIP_COMPILE:-0}
PLOT=${PLOT:-1}
mkdir -p "$OUTPUT_DIR"

# Rebase paths against the new summary's directory, never against the old
# baseline directory. Preserve the original build summary and statistics.
"$PYTHON" - "$FASTSCAN_SUMMARY" "$OUTPUT_DIR/fastscan_inputs.csv" <<'PY'
import csv
import sys
from pathlib import Path

source, output = map(Path, sys.argv[1:])
expected = {"sift-128-euclidean", "mnist-784-euclidean",
            "fashion-mnist-784-euclidean", "gist-960-euclidean",
            "deep-image-96-angular"}
with source.open(newline="") as handle:
    reader = csv.DictReader(handle)
    fields = reader.fieldnames
    rows = [r for r in reader if r["algorithm"] == "ivfpq_fastscan"]
if len(rows) != len(expected) or {r["dataset"] for r in rows} != expected:
    raise ValueError("FastScan summary must contain one index per revision dataset")
for row in rows:
    if int(row["pq_m"]) != int(row["dim"]) or int(row["pq_bits"]) != 4:
        raise ValueError("Expected M=dim, 4bit FastScan: " + row["dataset"])
    local = source.resolve().parent / row["dataset"] / "ivfpq_fastscan/index.bin"
    path = local if local.is_file() else Path(row["index_path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(row["index_file_bytes"]):
        raise ValueError("Index size differs from build summary: " + str(path))
    row["index_path"] = str(path.resolve())
    print("FastScan input:", row["dataset"], "M=" + row["pq_m"],
          "bits=" + row["pq_bits"], row["index_path"], flush=True)
with output.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
PY

if [[ "$SKIP_COMPILE" != 1 ]]; then
    cmake -S "$REPO" -B "$BUILD_DIR" -DHNSWLIB_EXAMPLES=ON
    TARGETS=(test_revision_baseline_storage_breakdown)
    if [[ "$REPROFILE_HNSW" == 1 ]]; then TARGETS+=(test_index_storage_breakdown); fi
    cmake --build "$BUILD_DIR" --target "${TARGETS[@]}" -j 4
fi

if [[ "$REPROFILE_HNSW" == 1 ]]; then
    HNSW_CSV="$OUTPUT_DIR/index_storage_breakdown.csv"
    "$BUILD_DIR/test_index_storage_breakdown" \
        --index_dir "$INDEX_DIR" --base_dir "$BASE_DIR" --diff_cache_percent 1 \
        --lvc_overview_csv "$OUTPUT_DIR/lvc_codec_index_size_overview.csv" \
        --dataset deep-image-96-angular fashion-mnist-784-euclidean \
            gist-960-euclidean mnist-784-euclidean sift-128-euclidean \
        --output_csv "$HNSW_CSV" 2>&1 | tee "$OUTPUT_DIR/hnsw_profile.log"
fi

"$PYTHON" "$REPO/script/profile_revision_baseline_storage.py" \
    --summary "$OUTPUT_DIR/fastscan_inputs.csv" \
    --output "$OUTPUT_DIR/fastscan_storage_breakdown.csv" \
    --build-dir "$BUILD_DIR" --skip-compile \
    2>&1 | tee "$OUTPUT_DIR/fastscan_profile.log"

# Replace, rather than append, the old FastScan rows. All other baseline
# components retain their original byte counts and provenance.
"$PYTHON" - "$BASELINE_CSV" "$OUTPUT_DIR/fastscan_storage_breakdown.csv" \
    "$OUTPUT_DIR/baseline_storage_breakdown.csv" <<'PY'
import csv
import sys
from collections import defaultdict
from pathlib import Path

old, new, output = map(Path, sys.argv[1:])
with old.open(newline="") as handle:
    reader = csv.DictReader(handle)
    fields = reader.fieldnames
    rows = [r for r in reader if r["IndexType"] != "ivfpq_fastscan"]
with new.open(newline="") as handle:
    reader = csv.DictReader(handle)
    if reader.fieldnames != fields:
        raise ValueError("Storage breakdown CSV schemas differ")
    fast = list(reader)
if not fast or any(r["IndexType"] != "ivfpq_fastscan" for r in fast):
    raise ValueError("New breakdown must contain only FastScan")
rows.extend(fast)
groups = defaultdict(list)
for row in rows:
    groups[row["Dataset"], row["IndexType"]].append(row)
datasets = {r["Dataset"] for r in fast}
expected = {(d, a) for d in datasets
            for a in ("ivf_flat", "ivfpq_fastscan", "nsg", "scann")}
if set(groups) != expected:
    raise ValueError("Expected all four baselines for each FastScan dataset")
for key, parts in groups.items():
    sizes = {int(r["FileSizeBytes"]) for r in parts}
    if (len(sizes) != 1 or len({r["Component"] for r in parts}) != len(parts)
            or any(int(r["Bytes"]) < 0 for r in parts)
            or sum(int(r["Bytes"]) for r in parts) != next(iter(sizes))):
        raise ValueError("Invalid component accounting: " + str(key))
temporary = output.with_suffix(".tmp.csv")
with temporary.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
temporary.replace(output)
print("PASS storage accounting:", len(groups), "dataset/index combinations")
print("Merged baseline breakdown:", output)
PY

if [[ "$PLOT" == 1 ]]; then
    MPLBACKEND=Agg "$PYTHON" "$REPO/script/plot_index_storage_breakdown_revision.py" \
        --csv "$HNSW_CSV" --baseline-csv "$OUTPUT_DIR/baseline_storage_breakdown.csv" \
        --out_dir "$FIGURE_DIR"
fi

echo "FastScan breakdown: $OUTPUT_DIR/fastscan_storage_breakdown.csv"
echo "Baseline breakdown: $OUTPUT_DIR/baseline_storage_breakdown.csv"
echo "HNSW breakdown: $HNSW_CSV"
if [[ "$PLOT" == 1 ]]; then echo "Figure: $FIGURE_DIR/index_storage_breakdown_revision.pdf"; fi
