# Complete dynamic-update experiments

Run from the HNSW repository root. All construction, PQ training, updates,
warmup and measured searches in these benchmarks use one thread. The scripts
compile their C++ driver directly; Faiss is not required. Python orchestration
uses only the standard library.

## Two independent launch scripts

Experiment 1, Repeated Insert/Delete:

```bash
nohup bash script/test_dynamic_updates_repeated.sh \
  > test_dynamic_updates_repeated.log 2>&1 &
```

Experiment 2, Growing Index (prepare prefix GT first):

```bash
python3 script/compute_dynamic_prefix_gt.py --threads 1 --resume
nohup bash script/test_dynamic_updates_growing.sh \
  > test_dynamic_updates_growing.log 2>&1 &
```

Run the two measured experiments sequentially to avoid contention. Defaults use
all train/query rows of SIFT, MNIST, Fashion-MNIST, GIST and DEEP, in that order.
Dataset cardinality and dimension come from the actual fvecs files.

Repeated starts at 100% of the dataset. Each of 50 batches randomly selects 5%
of source vectors without replacement within that batch, marks their current
versions deleted, and appends the same vectors with fresh labels. Selection can
repeat across batches. Active logical vectors remain N; physical records reach
N + 50*floor(0.05N). The original full `*_neighbors.ivecs` stays valid after
mapping returned active version labels back to original source IDs.

Growing starts with the first floor(0.5N) vectors. Ten batches append successive
prefixes of 55%, 60%, ..., 100%; no deletions are performed. Each checkpoint uses
`GT_ROOT/<dataset>/gt_prefixPPP.ivecs`. The launcher validates the GT manifest,
source file sizes/mtimes, row counts and dimensions before running that dataset.
GT preparation is offline and is excluded from update/query timings.

## Insertion and search configurations

| Insertion mode | Payload | Insertion filtering | Query modes per checkpoint |
|---|---|---|---|
| raw_f32 | Raw F32 | Original HNSW | tls_off |
| alp_a01 | ALP | alpha=0.1 | tls_off and tls_0.2 |
| alp_a02 | ALP | alpha=0.2 | tls_off and tls_0.2 |
| alp_off | ALP | Disabled | tls_off and tls_0.2 |

The three ALP policies load the same saved initial graph, compressed records,
PQ codes and codebook. Initial ALP graph/PQ/compression work runs once per
dataset and experiment. Growing trains PQ only on the initial 50%; the codebook
is frozen and new vectors are encoded with that codebook. All policies receive
the same source-ID update schedule, verified by batch hashes.

Settings: M=16, efConstruction=200, graph seed 100, update seed 20261015,
efSearch={20,40,80}, K=1, three measured query rounds with warmup. Query alpha is
independent of insertion alpha. The initial 1% ALP hub vectors stay uncompressed;
that set is frozen. Its fraction of active nodes falls to about 0.5% by the end
of Growing. ALP tolerance is 1e-4; this is not a general F32 bitwise-lossless claim.

Both indexes start with capacity equal to their initial node count. Each batch
expands capacity to the required physical record count. Expansion is included
in `insert_seconds` / `insert_mean_us` and is also reported as `resize_seconds`.
`graph_mean_us` excludes capacity expansion and final record encoding;
`encoding_mean_us` includes both ALP payload encoding and new-vector PQ coding.
`payload_decodes` counts actual ALP decodes throughout insertion, including
connection pruning, without a per-decode timer.
This avoids reserving 3.5N raw vector slots before the first repeated batch.
No space reclamation or deleted-slot reuse is performed. Updates and queries
are sequential and do not run concurrently.

## Results and resuming

Output directories:

```text
script/dynamic_updates_repeated_results/
script/dynamic_updates_growing_results/
```

Each completed dataset atomically publishes three central CSVs:

- `all_updates.csv`: one row per insertion policy and checkpoint. Update times,
  physical/active/deleted counts, exact serialized index bytes, capacity,
  decoder/filter counters, reconstruction checks and initial construction cost.
- `all_search.csv`: each measured query round, query mode and efSearch; strict
  Recall@1, tie-aware Recall@1, QPS and latency.
- `all_metrics.csv`: query-round averages joined with the checkpoint's update
  metrics. Update time appears on multiple query rows: do not sum it across rows.

Each dataset also has `updates.csv`, `search.csv`, `summary.csv`, `config.json`,
and per-policy subdirectories containing incrementally flushed CSVs and
`run.log`. A policy is marked complete only after all checkpoints are validated.
Completion also verifies that the update hashes agree across all four policies.
Default per-dataset row counts:

| Experiment | Update rows | Search rows | Summary rows |
|---|---:|---:|---:|
| Repeated | 204 | 3213 | 1071 |
| Growing | 44 | 693 | 231 |

Default RESUME=1 reuses completed insertion policies with matching configuration.
An interrupted policy restarts from its initial index; individual batches are
not resumed. Previously completed datasets remain in the central CSVs. Keep the
same settings to resume, or choose a different OUT for a changed configuration.
After a dataset completes its shared initial index is deleted unless
KEEP_INITIAL_INDEX=1; failure leaves it available to retry missing policies.

Serialized index size is computed without writing a multi-GB index every batch.
ALP's initial size is checked against the actual saved file. VERIFY_SIZE=1 also
saves initial/final checkpoints temporarily to verify size accounting. The
record buffer's allocated capacity is separately reported. Process RSS includes
source-file mappings and allocator overhead, and is not an isolated index size;
peak RSS also includes earlier phases of that process. Each policy runs in its
own process to release memory before the next policy.

All newly inserted vectors and a deterministic old-record sample are checked
outside timers. Deleted versions cannot count as hits even if their source IDs
match the GT. All query outputs are checked against the current version mapping.
ANN recall is reported rather than required to equal 1. No per-vector decoding
timer or perf sampling is enabled in the main timing runs.

## Overrides

```bash
DATASETS="sift-128-euclidean" EF_SEARCH="40 80" \
  bash script/test_dynamic_updates_repeated.sh

DATA_DIR=/path/to/fvecs GT_ROOT=/path/to/dynamic_prefix_gt \
  OUT=script/dynamic_growing_custom_results \
  bash script/test_dynamic_updates_growing.sh
```

Other overrides: SEARCH_ROUNDS, BATCHES (Repeated only), M, EF_CONSTRUCTION,
SEED, RESUME, KEEP_INITIAL_INDEX, VERIFY_SIZE, BIN, SKIP_BUILD, CXX.
COUNT/QUERY_COUNT are explicit smoke-test overrides, zero meaning all rows.
Repeated subsets require a separate DATA_DIR with matching train/query/full GT
files. Growing subsets require prefix GT generated with the matching train limit.

For a fast end-to-end check, use >=512 synthetic/subset train rows, prepare all
eleven prefix GT checkpoints, and set BATCHES=2, SEARCH_ROUNDS=1 and VERIFY_SIZE=1.
Growing still runs all ten growth steps so that GT switching is exercised.

The optional CMake target is `test_dynamic_update_workloads`.
