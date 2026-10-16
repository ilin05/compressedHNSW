# R1.O3: sample-based codec selection

## Run on the server

From the HNSW repository:

```bash
nohup bash script/test_codec_selection.sh \
  --data-dir "$PWD/datasets/hdf5files" \
  --build-threads 32 \
  > codec_selection_output.log 2>&1 &
```

Defaults: full datasets SIFT/MNIST/Fashion-MNIST/GIST/DEEP; M=16,
efConstruction=200, graph RNG seed=100; level roots=1%; sample=ceil(N*1%)
without replacement (seed=20261016); chain length=2 and -1; three encoder
measurements and median time. Forest preparation and sample profiling use one
thread, independently of raw graph build threads. ALP tolerance=1e-4 and raw
level-based hubs=1% (matching SELECT_HUBS_FOR_CACHE).

One raw F64 framework HNSW graph is built/saved per dataset if missing. Both
forests and every codec use this same graph. NO complete compressed indexes are
built, PQ is NOT trained, and query search is NOT run. Loading an existing raw
framework graph validates N/dimension/M/efConstruction/no deletions. Ensure it
actually contains the matching dataset: matching shape alone cannot prove data
identity. Standard HNSW/F32 or already compacted files are not supported inputs.

To reuse existing raw framework indexes, put them (or symlinks) under a directory
as `DATASET.bin`, then pass `--graph-dir /path/to/raw_graphs`. For example the root
selection experiment's SIFT `shared_raw_graph.bin` is reusable with a symlink
named `sift-128-euclidean.bin`.

```bash
bash script/test_codec_selection.sh --resume --build-threads 32
python3 script/check_codec_selection.py
# Recalculate selection/ranks without profiling:
bash script/test_codec_selection.sh --analyze-only
# A single dataset:
bash script/test_codec_selection.sh --datasets sift-128-euclidean --resume --build-threads 32
```

Keep the original profiling arguments when using --resume. It validates train
path/size/mtime, graph path/size/mtime and profiling configuration. Existing raw
graphs are reused even without --resume; only the sampling run is repeated.
Use --binary /path/to/test_codec_selection to skip CMake, or --skip-build to reuse
the default build directory. Only Python standard-library dependencies are used.

## Measurement and selection

`prepareCompressionForest` is extracted from the production `compress_dataset`
implementation, including propagation, fallback, and parent/depth/reference
statistics. Each graph-wide forest is built once, shared by DeXOR/Gorilla/Elf.
The exact same uniformly drawn sample IDs are used for both chain bounds and all
codecs. Ancestors are support records, NOT additional sampled target nodes.

Differential trial encoding resets dimension-wise states for each target,
replays the complete ancestor chain into a scratch stream, and encodes/aligned
writes the target into its real record. Only the latter contributes to payload
size; both contribute to measured time. ALP independently compresses the sampled
F32 vectors, retaining raw hubs according to its own policy. ALP has no roots or
chain depth; its identical measurement is reused in both chain comparisons.

Prediction is N/sample_count times the actual sample bytes and median encoding
time. Differential candidate time additionally includes the measured forest
preparation time; ALP candidate time does not. `SelectionWallSeconds` includes
sampling, ALP hub selection, trial encoders/repeats and BOTH complete forests;
it excludes graph loading/building. It is an observed profiling cost, not the
estimated one-pass encoding time.

For each chain condition, rank predicted payload size and predicted
forest+encoding time within (1) all four codecs and (2) differential codecs only.
Select the non-expanding candidate minimizing, lexicographically:

1. max(space_rank, time_rank)
2. space_rank + time_rank
3. compressed payload size
4. codec name (deterministic exact tie)

Return raw when every candidate expands. Ranks refer to the entire candidate
pool, including expanding codecs. Joint-top-2 may be impossible; the output
reports achievable and success separately. All-codec and differential candidate
pools are not mixed when evaluating ranks.

## Historical validation boundaries

The supplied `hnswalp_pq_build_compression_results.csv` and
`compressed_hnsw_ablation_build_results.csv` are the offline oracle; they never
enter the prediction/selection decision. The former is reused for both chain
bounds, and the latter is matched by Dataset/Algorithm/ChainMaxLength. Duplicate candidate keys cause an error. Missing keys do not block prediction;
rank/success fields for that incomplete pool are left blank (missing_oracle).
In the supplied CSV, DEEP chain=2 lacks all three differential results. Its
predictions are still produced; unbounded results are never substituted. Full historical CT is used to rank
time. Report the selected historical payload ratio, time/space ranks, joint-top-2,
Pareto status, and payload regret versus the best candidate or raw fallback.

The ARCHIVED CSV ratios use an F64 raw-size convention (also reflected in the
paper table). Defaults --oracle-alp-raw-bytes=8 and
--oracle-differential-raw-bytes=8 convert both to F32 ratios by multiplying by 4/8.
**The CURRENT ALP builder uses an F32 numerator.** If replacing the archived ALP
CSV with new output, pass --oracle-alp-raw-bytes 4. A modern differential CSV's
explicit PayloadRatioF32 column takes precedence over its legacy ratio column.
Index compression ratios cannot be converted by simply dividing by two, so this
experiment does not claim to predict/validate full-index compression ratios.

The old CSVs do not identify the shared graph, root policy, all codec versions or
threads. Default level roots match the legacy ablation script's default, but
validation remains CROSS-RUN HISTORICAL evidence, not a same-graph oracle.
Changing --root-policy degree is supported but further changes this comparison.

The time prediction is explicitly PARTIAL: it includes actual ancestor replay,
encoding and forest work, but excludes PQ training/encoding, compaction/graph
copying, F32 exact-patch construction and index I/O. Historical CT includes
additional production phases and different execution conditions. Its time-rank
validation measures how useful this proxy is; do not interpret prediction/CT as
an end-to-end speedup or silently calibrate it against the same oracle data.
OH uses historical ALP BT as a shared denominator/proxy for all candidates of
the same dataset. The measured current raw build time is reported separately.
Historical own-BT OH is also retained.
Neither proxy guarantees equal timing conditions across old/new runs.

To match historical codec semantics, this profiler uses legacy differential
records without F32 lossless patches. ALP uses the current configured tolerance,
not a claim of bitwise recovery. Comparing newer F32 exact-patch results would
require extending the profiler to include patch bytes and verification time.

## Outputs

All results are under `script/codec_selection_results` (not build/):

- `predictions.csv`: all datasets' codec byte/time predictions and forest stats.
- `candidate_validation.csv`: every candidate's prediction vs archived oracle.
- `selection_summary.csv`: both candidate pools, chain bounds, selected codec,
  actual ranks, joint-top-2 feasibility/success, regret and profiling cost.
- `DATASET/sample_ids.csv`: reproducible internal IDs and external labels.
- `DATASET/profiling.csv`: graph build/reuse and both-forest selection wall time.
- `DATASET/config.json`, `predictions.csv`, `profile.log`, `shared_raw_graph.bin`.

Aggregates are written atomically after each completed dataset. Failures stop
the run, leaving previous dataset summaries intact. Raw graph files/logs are
ignored by Git; CSV/JSON results can be committed. The separate output checker
validates accounting and selection logic; it does not assert the scientific
hypothesis that sampling always achieves both top-two ranks.
