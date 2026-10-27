#!/usr/bin/env bash
# Repeated delete/reinsert: 1% per batch, 20 batches, insertion TLS alpha=0.1.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BATCHES="${BATCHES:-20}"
export REPEAT_RATIO="${REPEAT_RATIO:-0.01}"
export OUT="${OUT:-$ROOT/script/dynamic_updates_repeated_1pct_alpha01_results}"
export BIN="${BIN:-$ROOT/build/dynamic_updates_repeated_1pct_alpha01/test_dynamic_update_workloads}"
exec bash "$ROOT/script/run_dynamic_updates_optimized.sh" repeated alp_a01
