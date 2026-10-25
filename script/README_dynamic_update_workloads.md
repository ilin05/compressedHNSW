# Complete dynamic-update experiments

## Optimized experiments: four independent launchers

The following four scripts can be started together. Each has its own default
OUT directory, executable, initial ALP checkpoint, case logs and resume markers:

```bash
nohup bash script/test_dynamic_updates_repeated_alpha01.sh > repeated_alpha01.log 2>&1 &
nohup bash script/test_dynamic_updates_repeated_alpha02.sh > repeated_alpha02.log 2>&1 &
nohup bash script/test_dynamic_updates_growing_alpha01.sh > growing_alpha01.log 2>&1 &
nohup bash script/test_dynamic_updates_growing_alpha02.sh > growing_alpha02.log 2>&1 &
```

Each script runs its own `raw_f32` baseline plus one ALP insertion policy on all
five full datasets. Per-insert decode caching is enabled; **both upper greedy
TLS and Base TLS use the script's alpha (0.1 or 0.2)**. Query TLS remains
independent: off and on(alpha=0.2) for ALP, off for raw. All processes, including
construction/PQ training and measured updates/queries, use one thread. M=16,
efConstruction=200, efSearch=40, K=1 and three query rounds remain fixed defaults.

| Workload | Update batches | Query checkpoints |
|---|---|---|
| Repeated | 20 batches, each delete/reinsert floor(0.03N) | 0,2,4,6,8,10,12,14,16,18,20 |
| Growing | Initial 50%, ten batches adding 5% through 100% | Every batch 0 through 10 |

Growing reuses the existing eleven offline prefix GT files; the 3% setting is
for Repeated only. Repeated keeps N active vectors and ends with
N+20*floor(0.03N) physical records, approximately 1.6N. Its original full-dataset
GT remains valid under the stable source-ID/version-label mapping.

Default outputs are `script/dynamic_updates_{repeated,growing}_alpha{01,02}_results/`.
Update/reconstruction statistics are flushed **every batch**, including batches
without full query evaluation. Query CSVs contain only the specified checkpoints;
`all_metrics.csv` therefore has no query rows for odd Repeated batches. Per-dataset
default counts: Repeated has 42 update rows, 99 search rows and 33 summary rows;
Growing has 22 update rows, 99 search rows and 33 summary rows. Each completed
policy is immediately published to the dataset/central CSVs, with `progress.json`
identifying a partially completed dataset; `complete.json` means all requested
policies passed. The initial index is shared only inside that one script's run.

New update fields record decode-cache hits, peak cached-vector bytes/nodes,
and upper greedy PQ/exact candidates. Peak bytes count decoded vector capacities,
excluding the temporary cache map's allocator overhead, and reset each batch.
The dataset config records the modes, ratio, query checkpoints and optimization
switches; this schema differs from the old runs, so use the new output directories.
RESUME skips complete policies, not individual interrupted batches.

The common shell implementation is `run_dynamic_updates_optimized.sh`. It
accepts the usual DATA_DIR, GT_ROOT, DATASETS, OUT, BIN, SEARCH_ROUNDS, RESUME,
SKIP_BUILD, KEEP_INITIAL_INDEX and VERIFY_SIZE overrides. Also supported are
EF_SEARCH, BATCHES (Repeated only), REPEAT_RATIO (Repeated only), SEARCH_EVERY,
M, EF_CONSTRUCTION, SEED, COUNT and QUERY_COUNT. SEARCH_EVERY always includes
batch 0 and the final batch. Do not assign the same OUT or a concurrently rebuilt
BIN to two scripts. Concurrent execution is supported, but CPU/memory-bandwidth
contention can change measured latency/QPS; report the concurrency condition or
repeat the final comparison sequentially if needed.

End-to-end correctness check (compiles the driver once, generates tiny exact GT,
and launches all four scripts concurrently):

```bash
python3 script/test_dynamic_updates_optimized.py
```

## Legacy experiment defaults

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
