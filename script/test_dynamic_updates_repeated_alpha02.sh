#!/usr/bin/env bash
set -euo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_dynamic_updates_optimized.sh" repeated alp_a02
