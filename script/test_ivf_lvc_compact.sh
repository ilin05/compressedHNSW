#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-/tmp/hnswlib_ivf_lvc_build}"
result_dir="${RESULT_DIR:-$repo_dir/script/ivf_lvc_compact_results}"
input=""
count=2048
nlist=8
seed=20261009
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

while (($#)); do
  case "$1" in
    --input) input="$2"; shift 2 ;;
    --count) count="$2"; shift 2 ;;
    --nlist) nlist="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$result_dir"
if [[ -z "$input" ]]; then
  input="$result_dir/tiny_train.fvecs"
  python3 - "$input" <<'PY'
import random
import struct
import sys

rng = random.Random(20261009)
with open(sys.argv[1], "wb") as output:
    for i in range(2048):
        group = i % 8
        values = [float(group * 20 + rng.randrange(64)) for _ in range(128)]
        output.write(struct.pack("<i", 128))
        output.write(struct.pack("<128f", *values))
PY
  count=2048
  nlist=8
fi

cmake -S "$repo_dir" -B "$build_dir" -DHNSWLIB_EXAMPLES=ON
cmake --build "$build_dir" --target test_ivf_lvc_compact -j "${BUILD_JOBS:-4}"
"$build_dir/test_ivf_lvc_compact" \
  --input "$input" --count "$count" --nlist "$nlist" --seed "$seed" \
  --output "$result_dir/summary.csv"
