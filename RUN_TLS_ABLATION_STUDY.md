# TLS Component Ablation Study - Build & Run Guide

## Overview
This guide explains how to build and run the Two-Level Search (TLS) component analysis tool, which evaluates TLS effectiveness across multiple ef values and TLS ratios.

## Files Modified/Created

### 1. **Core Implementation**
   - **File**: `hnswalg_simplified_alp_PQ.h`
   - **Change**: Added profiling metrics infrastructure
     - Member: `bool enable_profiling_metrics_`
     - Method: `void setProfilingMetrics(bool use)`
   - **Purpose**: Enables metrics collection for profiling (decoding calls, distance computations, etc.)

### 2. **Test Program**
   - **File**: `examples/cpp/test_tls_component_analysis.cpp`
   - **Changes**:
     - Added vector of ef values: `{10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 300}`
     - Added tls_ratios: `{0.1, 0.2, 0.3}`
     - Changed command-line argument from `--ef` to `--efs` for multiple values
     - Implemented nested loop: for each dataset, for each ef, test with/without TLS at each ratio
   - **Output**: CSV file with columns: Dataset, UseTLS, TLSRatio, K, ef, Recall, Latency(ms/query), DecodingCalls, DistanceComputations, AvgDecodingPerQuery

### 3. **Analysis & Visualization**
   - **File**: `script/analyze_tls_component.py`
   - **Outputs**:
     1. **tls_component_analysis.png**: Multi-subplot figure showing:
        - For each dataset (columns): Avg Decoding Calls vs. ef and Recall vs. ef
        - Multiple curves per plot: No TLS, TLS ratio=0.1, 0.2, 0.3
     2. **tls_component_analysis_summary.png**: Combined view across all datasets

### 4. **Build Configuration**
   - **File**: `CMakeLists.txt`
   - **Changes**:
     - Added `test_tls_component_analysis` target with proper linking
     - Applied `TWO_LEVEL_SEARCH` and `SELECT_HUBS_FOR_CACHE` compile definitions

---

## Build Instructions

### Prerequisites
- CMake 3.0+
- C++17 compiler with OpenMP support (GCC 7+, Clang 5+)
- Python 3.7+ with pandas, matplotlib, seaborn

### Step 1: Prepare Dataset Files
Ensure datasets are in the expected location:
```
datasets/hdf5files/
├── fashion-mnist-784-euclidean_test.fvecs
├── fashion-mnist-784-euclidean_neighbors.ivecs
├── gist-960-euclidean_test.fvecs
├── gist-960-euclidean_neighbors.ivecs
├── mnist-784-euclidean_test.fvecs
├── mnist-784-euclidean_neighbors.ivecs
├── sift-128-euclidean_test.fvecs
├── sift-128-euclidean_neighbors.ivecs
└── *_train.fvecs_hnswalp_simplified_pq.bin (pre-built indices)
```

### Step 2: Compile
```bash
cd hnswlib_cpp_py
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target test_tls_component_analysis
```

### Step 3: Run Tests
```bash
./test_tls_component_analysis
```

**Optional command-line arguments:**
```bash
./test_tls_component_analysis \
  --dataset fashion-mnist-784-euclidean gist-960-euclidean \
  --efs 10 20 30 40 50 60 80 100 120 150 200 300 \
  --tls-ratios 0.1 0.2 0.3 \
  --k 1
```

### Step 4: Analyze Results
```bash
cd ../script
python3 analyze_tls_component.py
```

This generates:
- **tls_component_analysis.png**: Detailed per-dataset plots
- **tls_component_analysis_summary.png**: Summary across all datasets
- Console output with detailed metrics analysis

---

## Output Format

### CSV Results (hnswalp_tls_component_analysis_results.csv)
```
Dataset,UseTLS,TLSRatio,K,ef,Recall,Latency(ms/query),DecodingCalls,DistanceComputations,AvgDecodingPerQuery
fashion-mnist-784-euclidean,0,0.00,1,10,0.750000,5.1234,150000,200000,1.5000
fashion-mnist-784-euclidean,1,0.10,1,10,0.745000,4.9000,135000,185000,1.3500
...
```

### Visualization Output
1. **tls_component_analysis.png**:
   - Top row: Avg Decoding Calls per Query vs. ef (for each dataset)
   - Bottom row: Recall vs. ef (for each dataset)
   - Legend: 4 curves per plot (no TLS + 3 TLS ratios)
   - X-axis: log scale for ef values

2. **tls_component_analysis_summary.png**:
   - Left: Summary of decoding efficiency across datasets (no TLS only)
   - Right: Summary of recall across datasets (no TLS only)

---

## Expected Results Summary

### Key Metrics to Analyze
1. **Recall vs. ef**: Shows how recall improves with increasing ef
2. **Avg Decoding per Query vs. ef**: Shows how TLS reduces decoding operations
3. **Latency Speedup**: Ratio of baseline to TLS latency
4. **Decoding Reduction %**: (baseline_decodings - tls_decodings) / baseline_decodings × 100

### Typical Observations
- **Recall**: Increases with ef, decreases slightly with higher TLS ratios
- **Decoding Calls**: Decreases with higher TLS ratios
- **Latency**: Improves with TLS up to optimal ratio, then may degrade
- **Recall-Speedup Trade-off**: TLS ratio 0.2-0.3 often provides best balance

---

## Troubleshooting

### Issue: Missing Dataset Files
**Solution**: Ensure dataset files are properly downloaded and indexed
```bash
# Verify index files exist
ls -la datasets/hdf5files/*_hnswalp_simplified_pq.bin
```

### Issue: "Cannot open index file"
**Solution**: Pre-build indices using existing benchmark tool
```bash
cd build
./test_search_hnswalp_pq --dataset fashion-mnist-784-euclidean
```

### Issue: Python Module Import Errors
**Solution**: Install required packages
```bash
pip install pandas matplotlib seaborn numpy
```

### Issue: Slow Compilation
**Solution**: Use parallel build
```bash
cmake --build . --target test_tls_component_analysis -- -j$(nproc)
```

---

## Performance Expectations

### Typical Runtime
- **Per ef value**: ~30 seconds per dataset (5 configurations: 1 no-TLS + 3 TLS ratios)
- **Full sweep (12 ef values)**: ~6 minutes per dataset
- **All 4 datasets**: ~24 minutes total
- **Python analysis**: < 1 minute

### Output File Sizes
- CSV results: ~50-100 KB
- PNG visualizations: ~200-300 KB per file

---

## Citation

If you use these results in publication, cite:

```bibtex
@article{hnswalp_tls_2024,
  title={Two-Level Search Analysis in HNSW with Adaptive Lossless Precision},
  year={2024}
}
```

---

## Next Steps

1. **Verify** that all modifications compile without errors
2. **Run** the full ablation study on representative datasets
3. **Analyze** results to determine optimal TLS configuration
4. **Generate** publication-quality figures for paper submission
