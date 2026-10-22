# Personal FastScan vs shared-partition F32 diagnostic

This is an independent diagnostic, not a revision baseline. It does not retrain
or modify the original index, baseline results or paper.

```bash
nohup bash script/test_fastscan_exact_sift.sh > test_fastscan_exact_sift.log 2>&1 &
```

The wrapper builds the new CMake target `test_fastscan_exact_diagnostic`, then
loads the existing M=dim, 4-bit SIFT FastScan index and the complete SIFT F32
base/query/ground-truth files. It requires the dataset's row IDs to match the
stored index IDs; it checks that the lists contain each base row ID exactly once.
This does not independently prove the base file is identical to its build input.

Defaults:

- Index: `script/revision_fastscan_mdim_build_results/sift-128-euclidean/ivfpq_fastscan/index.bin`.
- Inputs: `datasets/hdf5files/sift-128-euclidean_{train,test}.fvecs` and
  `sift-128-euclidean_neighbors.ivecs`.
- One search/OpenMP/BLAS thread, K=1, 3 rounds, query batch size 32.
- All queries at nprobe=1,2,4,6,8,10,15,20,25,30,40,50,60,80,100.
- Additional nprobe=nlist test on the first 100 queries. The CSV explicitly
  records QueryCount; compare modes within a row pair, not different query sets.

Both modes use exactly the same ordered coarse list IDs/distances, computed once:

1. `IVFPQ-FastScan`: unmodified `search_preassigned`, using the existing PQ codes
   and normal FastScan implementation. For each nprobe, the first query batch is
   checked against native `search`; IDs and output distances must agree.
2. `SharedPartition-Exact-F32`: scan **every ID in those same lists**, fetch its
   original F32 vector by row ID, and compute exact F32 L2. This is not a rerank
   of the FastScan shortlist. Exact distance ties select the smallest ID.

No new IVF-Flat is trained. The exact mode does not use PQ vectors, but the
original FastScan index remains in memory. It is not a production IVF-Flat memory
or latency baseline: vector storage/access differs from a contiguous IVF-Flat
layout and both modes share the diagnostic process.

## Output and interpretation

`script/fastscan_exact_sift_diagnostic_results/results.csv` is flushed after every
mode/nprobe. `run.log` contains progress and any native/preassigned or full-scan
checks. Recall values are fractions (0..1), not percentages.

| Column | Meaning |
|---|---|
| RecallAt1 | Returned ID equals the first provided GT ID |
| GTInListsAt1 | Fraction of queries whose first GT ID is in the probed lists |
| DistanceRecallAt1 | Returned vector's original F32 L2 equals GT vector's L2 within `1e-5 * max(1, abs(GT distance))`; accommodates ties |
| CoarseLatencyUS | Shared coarse list-selection time per query, measured once |
| ScanLatencyUS | Mean scan time per query across the requested rounds |
| LatencyUS / QPS | CoarseLatencyUS + ScanLatencyUS, and its reciprocal |
| MeanCandidates | Mean number of original vector IDs in the selected lists |

The full-scan exact mode must have DistanceRecallAt1=1; otherwise the program
fails and asks to check GT/data/metric/ID alignment. Strict ID recall can be
slightly below 1 when multiple nearest vectors tie. Coarse timing is not repeated
per round, and timings are diagnostic, not directly interchangeable with existing
revision baseline measurements.

- Exact-F32 recall high, FastScan low: discrepancy is in PQ/distance scoring,
  not missing coarse lists. This two-mode test **does not separate** vector PQ
  error from FastScan LUT quantization error.
- Both low and close to list coverage: missing coarse candidates are a major
  limitation at that nprobe.
- Full-scan Exact-F32 reaches GT, FastScan does not: further evidence of scoring
  loss after removing coarse pruning.

Overrides: `INDEX_FILE`, `DATA_DIR`, `BASE_FILE`, `QUERY_FILE`, `GT_FILE`,
`BUILD_DIR`, `BUILD_THREADS`, `OUTPUT_DIR`, `ROUNDS`, `QUERY_BATCH`, `QUERY_COUNT`,
`FULLSCAN_QUERIES`, `SKIP_BUILD`. QUERY_COUNT=0 uses all queries;
FULLSCAN_QUERIES=0 disables the extra full-scan point. Every invocation overwrites
results in its output directory; use a different OUTPUT_DIR to retain repeats.

Example quick run:

```bash
QUERY_COUNT=1000 ROUNDS=1 bash script/test_fastscan_exact_sift.sh
```

Dataset-independent correctness check using a small real Faiss index:

```bash
python3 script/test_fastscan_exact_diagnostic.py \
  --binary build_lvc_ivf_pq/test_fastscan_exact_diagnostic
```

This check exercises native/preassigned agreement, shared list scanning, strict
vs equal-distance recall, a quantization-induced gap, query-count limits, and
rejection of deliberately corrupted GT. Synthetic results are not SIFT results.
