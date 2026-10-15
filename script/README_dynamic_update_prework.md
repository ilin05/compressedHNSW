# ALP insertion filtering and prefix ground truth

Run commands from the hnswlib repository root. Python tools require NumPy.

## Full SIFT 1M validation

```bash
bash script/test_alp_insert_filter_sift.sh
```

Defaults: all 1,000,000 train vectors, all queries, two batches, 5% of the
original vectors per batch, `efSearch=20,40,80`, three query rounds. `M=16`,
`efConstruction=200`, initial graph seed 100, update seed 20261015. The script
checks the train shape when COUNT is unset/zero, preventing an accidental subset
run. Each batch marks current versions deleted and reinserts the same source
vectors under fresh labels. Across batches the active logical dataset stays
unchanged, while physical records and tombstones accumulate. Raw F32 and all ALP
cases use the identical update list in `updates.csv`.

Four cases are run: raw F32, ALP insertion filtering off, ALP insertion filtering
on, ALP insertion alpha=1. The three ALP cases load the same saved initial index
so graph and PQ training are shared. ALP query filtering stays enabled with the
same query alpha in all three cases. No query filtering is applied to Raw F32.
An initial 1% of ALP hubs stay uncompressed. Dynamic neighbor lists use fixed
capacity. No deleted slots are reused and no compaction is performed.

```bash
INSERT_ALPHA=0.1 QUERY_ALPHA=0.2 EF_SEARCH="40 80" \
  OUT=script/alp_insert_filter_sift_alpha01 \
  bash script/test_alp_insert_filter_sift.sh

# Disable the configured insertion-filter case (the exact controls still run).
INSERT_FILTER=0 OUT=script/alp_insert_filter_sift_disabled \
  bash script/test_alp_insert_filter_sift.sh
```

Library API in `hnswalg_simplified_alp_PQ.h`:

```cpp
index.setInsertUsePQFilter(true);  // default true; applies only after compression
index.setInsertPQAlpha(0.2f);      // default 0.2; valid range (0,1]
// Query settings are independent:
index.setUseTLS(true);
index.setTLSRatio(0.2f);
```

Filtering applies at insertion level 0. PQ ranks unvisited neighbors, then the
shortlisted nodes receive exact distances. A usable connection pool is first
filled without filtering; therefore the measured exact/PQ candidate ratio need
not equal alpha. Too few resulting candidates trigger an exact search. Upper
layers and connection pruning retain their existing distance calculations.
Disabled filtering and alpha=1 retain the original traversal order. These
runtime settings are reapplied after loading, not added to the file format.

Results default to `script/alp_insert_filter_sift_results/`:

- `metrics.csv`: per query round, batch, efSearch and mode; update times,
  serialized index size, node counts, insertion-filter counters, reconstruction
  error, recall@1, tie-aware recall@1, QPS and latency.
- `summary.csv`: query-round means; batch update times are repeated across ef
  rows and must not be summed across them.
- `*.labels`, `*.distances`: actual version labels (`labeltype`, native size)
  and returned F32 distances. The wrapper requires off/alpha=1 bitwise equality.
- `updates.csv`, `config.json`, `run.log`: update schedule, settings and progress.

An inactive version returned by search is an error, even if it maps to the
correct source vector. Recall is measured against original source IDs from the
input GT. Strict recall checks GT's first ID; the separate tie-aware column also
accepts another original vector at the same nearest distance. All live vectors
are decoded and checked at each checkpoint, outside the update/query timers.
ANN recall need
not equal 1 for correctness checks to pass. On/off filtering can produce
different graphs and results. Reconstruction is checked against the original
F32 source with an absolute error bound of 1.1e-4 (the ALP tolerance is 1e-4).
This does not claim bitwise losslessness for general F32 inputs.

Index size is obtained by actually saving each checkpoint, outside update/query
timers; it includes serialized PQ and metadata, but is not total process RSS or
allocated memory. Index files are removed after each case unless KEEP_INDEX=1.
This two-batch correctness tool intentionally does not implement the later
50-batch main study or persistent decoded-vector caching.

`pq_candidates` and `exact_candidates` describe neighbor candidates evaluated
by construction base-layer searches (the latter also counts upper-layer
construction searches). `insert_payload_decodes` counts actual ALP decoding
calls throughout the entire insertion, including upper routing and link
pruning, without per-decode timing. Warmup, validation and query decoding are
excluded from this counter. Raw F32 has no ALP decoding and reports zero.

`COUNT`, `QUERY_COUNT`, `SEARCH_ROUNDS`, `GT_FILE`, `DATA_DIR`, `BIN`, `CXX`,
`BUILD_THREADS` and `SEED` can be overridden. When using COUNT for a smoke test,
provide a matching subset GT through GT_FILE. The default GT is the full SIFT
`neighbors.ivecs`, and must not be used for a smaller prefix.

The optional CMake target is `test_alp_insert_filter` (HNSWLIB_EXAMPLES=ON).
The shell wrapper also compiles the tool directly without Faiss dependencies.

## Offline prefix GT for all five datasets

```bash
python3 script/compute_dynamic_prefix_gt.py --threads 8 --resume
```

Defaults to `datasets/hdf5files/` and all SIFT, MNIST, Fashion-MNIST, GIST and
DEEP train/test rows. Outputs are in `script/dynamic_prefix_gt/<dataset>/`:

- `gt_prefix050.ivecs`, `gt_prefix055.ivecs`, ..., `gt_prefix100.ivecs`: K=1
  original train-row IDs. Prefix size is `floor(N * percent / 100)`.
- Corresponding `.fvecs`: F32 squared L2 distances, not Euclidean distances.
- `*_squared_l2.npy`: the computed F64 squared L2 distances.
- `manifest.json`: source sizes/mtimes, dimensions, counts, settings and timing.

Vectors are scanned once in increasing source-ID order. At each checkpoint,
only the newly added vectors are exhaustively compared with all queries and
the running nearest neighbors are updated. F32 data are promoted to F64;
distances use `||q||² + ||x||² - 2 q·x`, with negative roundoff clamped to zero.
Exactly equal computed distances retain the smallest source ID. This is an
exhaustive floating-point computation, not an approximate ANN GT. Very close
floating-point ties may depend on numerical rounding. Inputs remain disk-mapped;
temporary distance matrices are bounded by train/query block sizes.

```bash
python3 script/compute_dynamic_prefix_gt.py \
  --datasets sift-128-euclidean --threads 8 \
  --train-block 8192 --query-block 256 --resume
```

Resume skips only complete datasets with matching source/config manifests. An
interrupted dataset is recomputed from the beginning. `--train-limit` and
`--query-limit` are explicit smoke-test options; their values are reflected in
the manifest. GT preparation is offline and is excluded from update timings.

Independent direct-distance regression test, including prefix changes,
cross-block duplicate ties, file formats and resume:

```bash
python3 script/test_dynamic_prefix_gt.py
```
