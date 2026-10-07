# Dynamic update batch results

`script/test_dynamic_update_batches.sh` writes per-dataset and per-codec metrics, perf reports, and aggregate CSVs here by default. The benchmark executable is built under `build/dynamic_update_batches/`.

When `RESUME=1` or `PERF_ONLY=1` is used and this directory has no aggregate metrics yet, the script copies existing results from `build/dynamic_update_batches/` automatically. It leaves the compiled executable under `build/`.
