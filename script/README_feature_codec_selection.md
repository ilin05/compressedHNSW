# Fixed-budget feature-based codec recommendation

This experiment extracts features only. It does not run a codec, build/load a
HNSW graph, build a compression forest, or train PQ. Existing historical codec
measurements are offline supervision. An online recommendation needs only a
saved JSON model and a train `.fvecs` file.

## Server run

From the HNSW repository, with Python 3.8+ and NumPy installed:

```bash
bash script/test_feature_codec_selection_smoke.sh
nohup bash script/test_feature_codec_selection.sh \
  --data-dir "$PWD/datasets/hdf5files" \
  > feature_codec_selection_output.log 2>&1 &
```

Expected files are `DATASET_train.fvecs` for SIFT, MNIST, Fashion-MNIST, GIST,
and DEEP, using the same names as the earlier codec-selection experiment.
Defaults evaluate all five datasets, samples=256/512/1024, chain=2/unbounded,
and pools=all/differential. There is no CMake or C++ build step.

```bash
# Reuse features (checks file path/size/mtime, settings, sample counts, seeds).
bash script/test_feature_codec_selection.sh --resume
# Tune a different parameter grid without reading vector data again.
bash script/test_feature_codec_selection.sh --analyze-only \
  --ridges 0.001 0.01 0.1 1 10 \
  --space-weights 0.1 0.25 0.5 0.65 0.75 0.85 0.9
```

If nondefault extraction arguments or data-dir were used, repeat them on resume.
Analyze-only uses the saved features; sampling settings, datasets and seed
partitions must match. It is legitimate to explore the grid using the tuning
partition. Do not adjust the grid after inspecting test seeds and continue to
call those same seeds an untouched test; use new test seeds for confirmation.

## Sampling and features

- Draw at most 1024 distinct vector IDs with Floyd sampling, O(1024) work and
  storage independent of N. No full-N shuffle, ID allocation, or hub sort.
- For a seed, 256 and 512 use prefixes of the same shuffled 1024-ID pool.
  Dimensions and anchors stay identical, isolating sample-size effects.
- Read at most 32 randomly chosen dimensions via two-axis memory-map indexing.
  Do not materialize B full vectors or scan the whole file.
- The first 32 sampled vectors are anchors **inside** the 256/512/1024 budget.
  Each vector chooses its nearest sampled anchor using the selected dimensions;
  anchors exclude their self match. This is a numerical locality proxy, not the
  actual nearest root, graph neighbor, or predecessor forest.
- Features: zero ratio, integer ratio, decimal complexity, exponent spread,
  reference equal ratio, reference XOR width, and d/(d+128). Decimal complexity
  averages the smallest q in 0..8 that round-trips nonzero values to the original
  F32 bits (failure=9); it does not alter codec precision. Exponent spread is
  P95-P5 divided by64 and clipped to1. XOR width uses actual promoted-F64 bits
  by default, matching HNSW differential codec inputs.
- Nonfinite sampled values are rejected rather than given an uncalibrated score.

The work is bounded by the configured B, K and anchor count A: feature work is
O(B*K*A), inference is constant for seven features and four codecs. Memory-map
opening/page faults still affect wall time; this is bounded-size sampling,
not an exact O(1) census of the original dataset.

## Model, parameter search, and fairness

Separate models for chain2/unbounded predict log(payload/F32 bytes) and
log(full historical compression seconds/(N*d)) for each codec. Weighted ridge
regression gives each dataset equal total weight. Features are standardized
using training data only; dataset names are never prediction inputs.

Default train seeds:1001..1005; tuning seeds:2001..2003; untouched test seeds:
3001..3010. The three partitions cannot overlap. Default grid is ridge=0.01,
0.1,1,10 and space weight=0.25,0.5,0.75,0.9. The score is:

```
J(codec) = space_weight * predicted_size / fixed_size_normalizer
         + (1-space_weight) * predicted_time / fixed_time_normalizer
```

Normalizers come from training targets only and remain fixed online. Candidates
with predicted payload ratio below1.02 are excluded; if all fail, select raw.
Raw is a fallback rather than a zero-encoding-time competitor to every codec.
Ridge and space weight are shared across datasets for each chain/pool condition,
not selected separately by dataset name.

Tune lexicographically on: unexpected expansion, excess max(space-rank,time-rank)
over the best feasible candidate, joint-top2 failure where achievable,
within-size instability, cross-size majority disagreement, payload regret, time
slowdown. Ground-truth ranks use all candidates in the selected pool. Payload
regret compares to the best ratio or raw, and time slowdown compares to the
fastest codec in that pool. This mirrors the earlier balanced ranking goal.
For MNIST/chain2/all, simultaneous top2 may be impossible; this is reported,
not forced into a success.

A separate leave-one-dataset-out run **refits and tunes without that dataset**.
Its test rows are an explicit generalization diagnostic. Held-out-seed stability
alone covers new samples of known workloads, not new-dataset generalization.
Repeated seeds never become independent datasets.

## Outputs

Written under `script/feature_codec_selection_results/` by default:

| File | Meaning |
|---|---|
| `features.csv`, `DATASET/features.csv` | Sample counts, seeds, seven features, extraction timings; saved after each dataset |
| `feature_config.json`, `DATASET/features.json` | Extraction identity/settings and per-file hash; supports resume |
| `calibration_config.json` | Grid, seed split and historical CSV identities |
| `model_chain2_all.json` etc. | Deployable coefficients, scaling, selected weights, settings and provenance |
| `model_weights.csv` | Human-readable standardized coefficient rows |
| `parameter_search.csv` | Every tuning setting, selected flag and objective components; includes LODO folds |
| `recommendations.csv` | Untouched-test-seed predictions, historical ranks/regret/Pareto and timing |
| `stability_summary.csv` | Agreement, quality and mean timing per dataset/chain/pool/sample size |
| `stability_targets.csv` | Whether the stability and recommendation goals actually passed |
| `always_alp_baseline.csv` | Fixed-ALP quality for the same all-pool test conditions |
| `leave_dataset_out.csv`, `leave_dataset_out_summary.csv` | Dataset-held-out diagnostics |
| `run_summary.json` | Overall pass/fail and offline calibration wall time |

Default stability requires **10/10 test seeds agree at each sample size**, and
majorities must also agree between256/512/1024. `QualityTargetPassed` additionally
requires Pareto optimality, no unexpected expansion, best achievable max-rank,
and simultaneous top2 whenever feasible. `RecommendationTargetPassed` requires
both quality and stability. Failure is preserved in the CSV; a stable but poor
constant choice does not count as a successful recommendation.

Feature collection uses the largest sample once per seed; `ProbeReadSeconds`
is therefore a conservative maximum-sample read cost for smaller sizes.
`OnlineSecondsUpperBound` adds that read cost, the size-specific feature time,
and inference. `ProbeLoadedVectors` identifies that read count explicitly.
`OnlineToHistoricalBuild` and `OnlineToSelectedCompression` report ratios
against historical ALP build time and selected full compression time; they
are cross-run diagnostics, not a same-machine construction-overhead guarantee.
Standalone recommendation times its actual selected sample read.
Both exclude Python startup. `CalibrationSeconds` is offline fitting/tuning,
not online selection overhead. Do not subtract sampling IO from overhead.

## Online recommendation and adjustable weights

```bash
python3 script/feature_codec_recommendation.py recommend \
  --model script/feature_codec_selection_results/model_chainunbounded_all.json \
  --data-file datasets/hdf5files/sift-128-euclidean_train.fvecs \
  --samples 512 --seed 4001 \
  --output script/feature_codec_selection_results/sift_recommendation.json

# Prefer space more strongly. A single weight infers the other as 1-weight.
python3 script/feature_codec_recommendation.py recommend \
  --model script/feature_codec_selection_results/model_chain2_all.json \
  --data-file datasets/hdf5files/gist-960-euclidean_train.fvecs \
  --samples 256 --space-weight 0.8 --time-weight 0.2
```

JSON coefficients can be inspected/overridden, but modified coefficients or
preferences need revalidation. Do not claim that calibration-selected quality
is unchanged after hand editing. The model inherits level roots=1%, the legacy
codec implementations, and ALP tolerance1e-4; it does not promise bitwise ALP
recovery or accurate timings on a different machine.

## Historical-result boundaries

The existing `test_codec_selection.py` loader is reused only during offline
calibration. Archived ALP/differential CSVs use an F64 numerator, converted to
F32 by default. For newly generated ALP data with an F32 numerator, pass
`--oracle-alp-raw-bytes 4`; explicit differential `PayloadRatioF32` takes priority.
N/dimension are checked wherever historical rows provide them. Graph/codec
versions, threading and full timing scopes remain unverified cross-run;
this experiment recommends against historical results, not a same-graph oracle.
The original files do not measure forest time separately, so predictions of
full compression time include it implicitly rather than estimating it anew.

The five complete train files are absent in the development workspace. Local
unit/integration checks use clearly marked synthetic fixtures and do not
establish five-dataset recommendation quality. The shell script must run on
the server to find real coefficients and certify (or reject) the target.
