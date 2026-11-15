# SIFT random-root compression repeats

Run from the compressedhnsw repository:

```bash
nohup bash script/test_root_selection_random_repeats_sift.sh \
  > test_root_selection_random_repeats_sift_output.log 2>&1 &
```

Defaults: SIFT 1M, DeXOR, unbounded chains (`CHAIN_MAX=-1`), 1% roots, 20 distinct
root seeds starting at 20261116, and F32-exact patch accounting, matching the
compression-ratio definitions of `test_root_selection_score_sift.sh`.

The script requires the original experiment's
`script/root_selection_score_results/shared_raw_graph.bin`; it reuses this graph
for every seed. It requires only the training fvecs file, without query vectors,
ground truth, or an access-frequency trace. It builds only the compression
executable and runs no retrieval benchmark. PQ preparation remains in the
existing compression implementation; codec encoding and F32 correction logic
are unchanged.

Results go to a timestamped directory beneath
`script/root_selection_random_repeats_results/`:

- `summary.csv`: one row per seed / codec / chain, with `PayloadRatioF32` and
  `FullIndexRatioF32`; rows are flushed after each completed seed.
- `statistics.csv`: mean, sample standard deviation, min, max, and relative
  spread `(max-min)/mean*100` for both ratios.
- `run_XX_seedYYYY.log`: build diagnostics, including root count and root-set
  signature. No compressed index or forest CSV is saved.

The ratios are recalculated from the existing build CSV's byte counts with
12 decimal places. Payload ratio is `RawF32Bytes / PayloadBytes` (codec payload
plus F32 correction values); full-index ratio is
`RawF32IndexBytes / CompressedIndexBytes`, using the existing ablation's
accounted-index convention with decode-state-cache bytes excluded. This is not
a measurement of process RSS or serialized file size.

Optional settings:

```bash
# All three differential codecs and both depth settings, 20 seeds per setting.
ALGORITHMS="DeXOR Gorilla Elf" CHAIN_MAX="2 -1" \
  bash script/test_root_selection_random_repeats_sift.sh

# Custom paths; the script configures/builds the modified executable by default.
SHARED_GRAPH_FILE=/path/to/shared_raw_graph.bin \
DATA_DIR=/path/to/hdf5files RESULT_DIR=/path/to/new_results \
  bash script/test_root_selection_random_repeats_sift.sh

# Synthetic 512-vector smoke check: 20 seeds; bitwise-check 32 vectors per case.
SMOKE=1 BUILD_THREADS=2 RESULT_DIR=/tmp/lvc_random_root_smoke \
  bash script/test_root_selection_random_repeats_sift.sh
```

Other controls follow the existing runner: `BUILD_DIR`, `BUILD_JOBS`,
`BUILD_THREADS`, `SKIP_CMAKE=1`; `RUNS` and `START_SEED` control repetitions.
Use a new result directory for another run to avoid mixing experiments.

Implementation changes required on the server:

- `hnswlib/compressed_hnsw_framework.h`: optional independent root RNG through
  `setCompressionRootSeed`; omitted seed preserves the previous behavior.
- `examples/cpp/test_build_compressed_hnsw_ablation.cpp`: `--root_seed` selects
  that RNG, and `--ratios_only` skips compressed-index saves and forest exports.
- `script/test_root_selection_random_repeats_sift.sh`: the new runner.

The root RNG does not alter PQ training or the original graph's random state.
For each explicit seed, codecs and chain settings start from the same sampled
root ordering. Repeating a seed on the same shared graph reproduces its roots.
