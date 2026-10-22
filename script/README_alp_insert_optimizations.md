# HNSW+ALP insertion optimizations: SIFT pilot

Run from the repository root:

```bash
bash script/test_alp_insert_optimizations_sift.sh
```

Defaults: fixed random 100,000-vector subset of SIFT train, all 10,000 queries,
10 batches, delete/reinsert 5% of active vectors per batch, single thread,
M=16, efConstruction=200, efSearch=40, 3 query rounds. Query TLS is **off**.
Each reinsert gets a new version label. Search results are validated against the
current live version and mapped back to the stable subset ID for recall@1.

`prepare_alp_insert_sift_subset.py` writes the sampled original IDs and computes
exhaustive F64 L2 ground truth **over the sampled dataset**. Full-SIFT GT is not
valid for this subset. GT preparation is separate from measured update time;
the subset and GT are reused when source paths, sizes, mtimes, count and seed match.

| Mode | Per-insert decode cache | Upper greedy TLS | Level-0 TLS |
|---|---|---|---|
| raw_f32 | — | — | — |
| alp_base | off | off | alpha=0.2 |
| alp_cache | on | off | alpha=0.2 |
| alp_upper | off | alpha=0.2 | alpha=0.2 |
| alp_joint | on | alpha=0.2 | alpha=0.2 |
| alp_exact | on | off | off |
| alp_alpha1 | on | alpha=1 | alpha=1 |

All ALP cases load the same initial graph/PQ checkpoint and replay identical
updates. Raw F32 is built with the same insertion order, M, efConstruction and
seed. Initial ALP raw hubs remain 1%; neither new optimization changes this.
Upper TLS follows query `searchKnn`'s top-alpha rule: PQ-rank a routing node's
neighbors, retain `min(count, max(2, floor(alpha * count)))`, then compare exact
distances. With alpha=1 the original neighbor traversal order is used, preserving
the filter-off path. Set `BASE_TLS=0` to disable base filtering independently in the first four ALP
cases while retaining the upper-layer ablation. Upper-layer connection search and neighbor pruning continue
to use exact distances; the new upper switch controls **greedy navigation**.

## Correctness checks

- At every batch, check all live payloads, deletion/version exclusion, physical
  and tombstone counts, PQ code count, graph edge bounds/levels/duplicates.
- At every batch, compare complete serialized indexes for base/cache,
  upper/joint, and exact/alpha1. The Python summary also compares returned IDs
  and L2 bits for every query round of these equivalent pairs.
- Check cache clearing after insertion and on rejected duplicate-label insertion.
- SIFT integer-valued vectors should reconstruct exactly; the general ALP
  validation retains the existing absolute tolerance 1.1e-4.
- Upper TLS can change graph connections and recall; equality with upper TLS
  disabled is not required. No recall improvement or speedup threshold is assumed.

## Outputs

Default directory: `script/alp_insert_optimizations_sift_results/`.

- `metrics.csv`: flushed after each query round, including all per-batch update
  measurements and counters. Repeated query-round rows repeat the update counters;
  **do not sum these rows directly**.
- `summary.csv`: one row per mode/batch/efSearch, query metrics averaged over rounds.
- `comparison.csv`: update latency, speedup relative to `alp_base`, slowdown relative
  to HNSW, actual decode count, cache hits, upper PQ/exact candidate counts and recall.
- `config.json`, `updates.csv`, `run.log`, `subset/`, query ID/distance binary files.

`decode_cache_peak_bytes` counts cached decoded vector capacities, not allocator
or unordered-map overhead. `decode_cache_peak_nodes` is also reported. Counters
reset each batch; payload decodes exclude validation and query evaluation.
`graph_us` includes navigation, pruning, ADC and cache lookup/population;
`encoding_us` includes ALP/PQ encoding. Total insertion time also includes lifecycle
cleanup and call overhead. Per-distance timers are not added.

Full temporary comparison snapshots are deleted on successful completion unless
`KEEP_INDEX=1`. Peak snapshot disk usage is approximately 11 ALP index files
plus the current snapshot and initial checkpoint with the default 10 batches.
Results are overwritten on rerun; choose a fresh OUT to preserve a previous run.

Overrides:

```bash
DATA_DIR=/path/to/hdf5files OUT=/path/to/results \
  COUNT=100000 QUERY_COUNT=10000 ROUNDS=10 SEARCH_ROUNDS=3 \
  BASE_ALPHA=0.2 UPPER_ALPHA=0.2 \
  bash script/test_alp_insert_optimizations_sift.sh
```

API switches (independent of query TLS): `setInsertDecodeCache(bool)`,
`setInsertUpperTLS(bool)`, `setInsertUpperAlpha(float)`,
`setInsertBaseTLS(bool)` / existing `setInsertUsePQFilter(bool)`, and
`setInsertPQAlpha(float)`. For backward compatibility decode cache and upper TLS
default to off; existing base-filter defaults are unchanged. Callers can test
upper-only TLS by disabling base TLS. Cached vectors live only for one compressed
`addPoint`; map buckets can be reused. Concurrent searches during dynamic updates
remain unsupported, as in the existing implementation.

The executable also has the CMake target `test_alp_insert_optimizations`; the shell
wrapper compiles it directly to avoid depending on optional baseline libraries.

## Local verification

The complete wrapper passed on 512 synthetic 128-dimensional integer vectors,
24 queries and 10 update batches. Complete-index/query-bit equivalence passed
for all three pairs. A separate base-TLS-off run exercised upper TLS independently.
This is a smoke test; real SIFT performance must be measured on the server.

AddressSanitizer found an existing one-past-neighbor read in raw HNSW's SSE
prefetch at `hnswalg.h` (the `datal + j + 1` expression on the last neighbor).
That baseline code was not changed. A two-batch sanitizer run with
`-DNO_MANUAL_VECTORIZATION` and
`ASAN_OPTIONS=detect_leaks=0:handle_segv=0` passed, including cache lifecycle and
serialization checks. This sanitizer run covers scalar distance paths; the
ordinary ten-batch run used the normal SIMD build.
