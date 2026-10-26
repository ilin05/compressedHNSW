# Revision storage breakdown including FastScan M=dim, 4bit

Run from the repository root:

```bash
bash script/run_index_storage_breakdown_revision.sh
```

The default run measures the five new FastScan indexes in
`script/revision_fastscan_mdim_build_results/summary.csv`. It reuses the existing
HNSW/LVC breakdown and IVF-Flat/NSG/ScaNN breakdown, replaces the old FastScan
rows, validates byte totals, and redraws the revision storage figure with
IVFPQ-FastScan enabled. It does not build indexes or run recall benchmarks.

To also rerun the original HNSW/HNSW+PQ/HNSW+SQ/LVC storage profiling:

```bash
REPROFILE_HNSW=1 bash script/run_index_storage_breakdown_revision.sh
```

The original indexes must be under `script/`, as for
`run_index_storage_breakdown.sh`; override `INDEX_DIR` and `BASE_DIR` if needed.
The new script compiles the existing profiling targets in `build_lvc_ivf_pq`;
no CMake changes are needed. Use `SKIP_COMPILE=1` for already-built binaries.

Outputs under `script/revision_storage_breakdown_results/`:

- `fastscan_inputs.csv`: validated input summary with resolved index paths.
- `fastscan_storage_breakdown.csv`: new FastScan component measurements.
- `baseline_storage_breakdown.csv`: four baselines, with old FastScan replaced.
- `fastscan_profile.log`: FastScan profiling output.
- When `REPROFILE_HNSW=1`: `index_storage_breakdown.csv`,
  `lvc_codec_index_size_overview.csv`, and `hnsw_profile.log`.

Figure: `script/index_size_revision_plots/index_storage_breakdown_revision.pdf`.
Original input CSVs are preserved. Use `OUTPUT_DIR` / `FIGURE_DIR` to change
output locations; `PLOT=0` performs measurement and merging without plotting.
`FASTSCAN_SUMMARY`, `BASELINE_CSV`, `HNSW_CSV`, `BUILD_DIR`, and `PYTHON` can also
be overridden. Colocated FastScan indexes are preferred over old absolute paths
in the build summary when a repository has moved to another server.

Accounting follows the existing serialized-index profiler: FastScan codes
(including SIMD block padding) are Vector Payloads; IDs and Original Metadata
are Metadata; PQ codebooks and coarse centroids are Codebook. This is a
serialized storage breakdown, not a measurement of process RSS or allocator
capacity. Runtime LVC Cache remains the existing separately accounted component.

`M=dim, 4bit` stores `dim/2` code bytes per vector: **8x** compression versus F32
payload, before IDs, codebooks, centroids and block padding. The script checks
the summary configuration and file size; actual component sizes are read from
the serialized indexes by `test_revision_baseline_storage_breakdown`.
