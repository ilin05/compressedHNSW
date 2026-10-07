# Dynamic update batch results

`script/test_dynamic_update_batches.sh` writes per-dataset and per-codec metrics, perf reports, and aggregate CSVs here by default. The benchmark executable is built under `build/dynamic_update_batches/`.

To continue an earlier run stored under `build/dynamic_update_batches/`, copy its result files into this directory before using `RESUME=1` or `PERF_ONLY=1`.
