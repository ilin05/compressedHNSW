# IVFPQ-FastScan: M=dim, 4 bit

Run from the repository root:

```bash
nohup bash script/run_revision_fastscan_mdim.sh > fastscan_mdim_output.log 2>&1 &
```

The script builds and searches only FastScan on SIFT, MNIST, Fashion-MNIST,
GIST, and DEEP. Build and search can also be launched separately:

```bash
bash script/run_revision_fastscan_mdim.sh build
bash script/run_revision_fastscan_mdim.sh search
```

## Configuration

- PQ: one subquantizer per coordinate (`M=dim`), 4 bits per subquantizer.
  For these datasets the code size is `dim/2` bytes per vector, or **8x**
  compression relative to the F32 vector payload. IDs, centroids, codebooks,
  and block padding reduce the overall index compression ratio.
- Residual encoding enabled, block size 32. Other build settings are unchanged:
  `nlist=floor(sqrt(N))`; a seeded random training sample targeting 15% of N,
  capped at 1,500,000 and raised to at least `40*nlist` (at most N).
- Build threads: 32. Search: one thread, K=1 and 10, three timed rounds after
  warmup. nprobe: 1,2,4,6,8,10,15,20,25,30,40,50,60,80,100.
- Search reuses the existing ground-truth recall implementation and large-LUT
  batching safeguard. Recall improvement is measured rather than assumed.

## Outputs and resume

- `script/revision_fastscan_mdim_build_results/summary.csv`: build time, PQ
  configuration, serialized index bytes, and index paths, updated per index.
- `script/revision_fastscan_mdim_search_results/all_results.csv`: Dataset,
  IndexType, Param, K, Recall, QPS, Latency(us), IndexSizeKB, VectorsPerKB, VQ.
- Per-dataset build/search logs and metrics are saved under the same directories.

These directories are separate from the previous baseline results. Re-running
reuses compatible build metrics and matching search configurations. The general
builder also rebuilds old FastScan configurations whose M differs from dim.
Set `FORCE=1` to rebuild/retest all five FastScan indexes in the selected stage.

Optional environment variables: `PYTHON`, `DATA_DIR`, `BUILD_DIR`,
`BUILD_RESULTS`, `SEARCH_RESULTS`, `BUILD_THREADS`, `ROUNDS`, `FORCE`.
Paths default to the script's repository, so changing the repository location
does not require editing the script.
