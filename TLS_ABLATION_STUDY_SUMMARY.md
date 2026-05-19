# TLS Component Ablation Study - Modification Summary

## Project Context
**Goal**: Create comprehensive Two-Level Search (TLS) ablation study tool to evaluate TLS effectiveness across multiple ef values and demonstrate recall-performance trade-offs.

**Project**: HNSWALP (Hierarchical Navigable Small World with Adaptive Lossless Precision) with PQ quantization

---

## Modified Files

### 1. **hnswalg_simplified_alp_PQ.h** (Core Index Implementation)
**Location**: `hnswlib_cpp_py/hnswlib/`

**Changes Made**:
- Added member variable: `bool enable_profiling_metrics_ = false;` (~line 101)
- Added method: `void setProfilingMetrics(bool use) { enable_profiling_metrics_ = use; }` (~line 2345)
- Modified metric collection in `getOriginalDataByInternalId()` to check both `collect_metrics` parameter AND `enable_profiling_metrics_` flag

**Purpose**: Enable external profiling control for metrics collection without requiring template parameter changes

**Key Metrics Collected**:
- `decoding_call_count`: Number of PQ decoding operations
- `decoding_time`: Total time spent decoding
- `metric_distance_computations`: Distance computations count

---

### 2. **examples/cpp/test_tls_component_analysis.cpp** (Ablation Study Program)
**Location**: `hnswlib_cpp_py/examples/cpp/`

**Major Changes**:

#### a) Added Multiple EF Values Vector (line ~268)
```cpp
std::vector<size_t> efs = {10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 300};
```

#### b) Updated TLS Ratios (line ~267)
```cpp
std::vector<double> tls_ratios = {0.1, 0.2, 0.3};  // Was {0.1, 0.2}
```

#### c) Modified Command-Line Argument Parsing (lines ~275-285)
- Removed: Single `--ef` parameter (was `ef = std::stoull(argv[++i])`)
- Added: `--efs` parameter accepting multiple values
```cpp
else if (arg == "--efs" && i + 1 < argc) {
    efs.clear();
    while (i + 1 < argc && argv[i + 1][0] != '-') {
        efs.push_back(std::stoull(argv[++i]));
    }
}
```

#### d) Restructured Main Test Loop (lines ~290+)
- **Previous**: Single ef=100, loop over datasets then TLS ratios
- **New**: Nested loop structure
  ```cpp
  for (dataset in datasets)
    for (ef in efs)  // NEW outer ef loop
      test without TLS (ef)
      for (ratio in tls_ratios)
        test with TLS (ef, ratio)
  ```

#### e) Improved Console Output
- More compact per-test reporting showing ef value explicitly
- Format: `ef=XX`, `WITHOUT TLS: Recall=X.XXXX Latency=XXms AvgDecode=XX.XX`

**Output CSV**: `hnswalp_tls_component_analysis_results.csv`
- Columns: Dataset, UseTLS, TLSRatio, K, ef, Recall, Latency(ms/query), DecodingCalls, DistanceComputations, AvgDecodingPerQuery
- One row per test configuration (dataset × ef × use_tls × tls_ratio)

---

### 3. **script/analyze_tls_component.py** (Analysis & Visualization)
**Location**: `hnswlib_cpp_py/script/`

**Complete Rewrite** - Now generates publication-quality two-plot visualization

**Output 1: tls_component_analysis.png**
- Layout: 2 rows × N columns (N = number of datasets)
- **Top row**: Avg Decoding Calls per Query vs. ef (for each dataset)
- **Bottom row**: Recall vs. ef (for each dataset)
- **Each plot** includes 4 curves with different line styles and markers:
  - `无 TLS` (No TLS) - solid line, square marker
  - `TLS ratio=0.1` - dashed line, circle marker
  - `TLS ratio=0.2` - dotted line, triangle marker
  - `TLS ratio=0.3` - dash-dot line, diamond marker
- **X-axis**: Log scale (ef values)
- **Y-axis**: Decoding calls (top) / Recall (bottom)

**Output 2: tls_component_analysis_summary.png**
- Layout: 1 row × 2 columns
- **Left**: Avg Decoding Calls per Query vs. ef (No TLS only, all datasets)
- **Right**: Recall vs. ef (No TLS only, all datasets)
- Each dataset shown as separate curve with distinct color

**Console Output**: Detailed metrics analysis per ef value per dataset

---

### 4. **CMakeLists.txt** (Build Configuration)
**Location**: `hnswlib_cpp_py/`

**Changes Made** (lines ~195-196, ~314):
```cmake
# Add executable target
add_executable(test_tls_component_analysis 
    examples/cpp/test_tls_component_analysis.cpp 
    ${ENCODING_ALGO_SRC})
target_link_libraries(test_tls_component_analysis hnswlib)

# Add to compressed HNSW targets (applies compile definitions)
# Automatically gets TWO_LEVEL_SEARCH and SELECT_HUBS_FOR_CACHE flags
```

**Compile Definitions Applied**:
- `TWO_LEVEL_SEARCH=ON`: Enables TLS functionality
- `SELECT_HUBS_FOR_CACHE=ON`: Enables hub-aware caching

---

### 5. **script/test_tls_component_analysis.sh** (Execution Script)
**Location**: `hnswlib_cpp_py/script/`

**Complete Rewrite**: Now runs full pipeline with reporting

**Changes**:
1. Added structured parameter display
2. Updated command-line arguments to use `--efs` instead of `--ef`
3. Added default dataset: `mnist-784-euclidean`
4. Calls Python analysis script automatically
5. Better console output with progress markers and file listing

---

### 6. **NEW: RUN_TLS_ABLATION_STUDY.md**
**Location**: `hnswlib_cpp_py/`

Complete guide including:
- Overview of all modifications
- Detailed build instructions
- Command-line usage examples
- Output format documentation
- Troubleshooting section
- Performance expectations
- Citation format

---

## Test Configuration

### Default Parameters
| Parameter | Values |
|-----------|--------|
| **Datasets** | fashion-mnist-784-euclidean, gist-960-euclidean, mnist-784-euclidean, sift-128-euclidean |
| **EF Values** | 10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 300 |
| **TLS Ratios** | 0.0 (no TLS), 0.1, 0.2, 0.3 |
| **K (Recall@K)** | 1 |
| **Total Configs** | 4 datasets × 12 ef values × 4 tls_configs = 192 test runs |

### Expected Runtime
- ~6 minutes per dataset
- ~24 minutes total for all datasets
- Python analysis: <1 minute
- **Total: ~25 minutes**

---

## Key Technical Improvements

### 1. **Metrics Collection Control**
- Before: Metrics collection only via template parameters
- After: Runtime control via `setProfilingMetrics()` method
- Benefit: Can enable/disable metrics without recompilation

### 2. **Comprehensive Sweep**
- Before: Single ef value testing
- After: Full sweep across 12 ef values
- Benefit: Shows complete recall-performance curve

### 3. **Publication-Ready Visualizations**
- Before: Generic 4-plot layout (latency, recall, decoding, distance)
- After: User-specified 2-plot layout with multiple TLS configurations
- Benefit: Directly illustrates TLS effectiveness claims

### 4. **Automated Analysis Pipeline**
- Before: Manual CSV analysis
- After: Automatic analysis + visualization generation
- Benefit: Faster iteration, consistent formatting

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────┐
│ test_tls_component_analysis (C++ executable)│
├─────────────────────────────────────────────┤
│ For each: dataset, ef, use_tls, tls_ratio   │
│   1. Load index (hnswalp_simplified_pq.bin) │
│   2. Load queries & ground truth (fvecs)    │
│   3. Configure TLS & profiling metrics      │
│   4. Execute searchKnn() & collect metrics  │
│   5. Calculate recall vs. k                 │
│   6. Store result: CSV row                  │
└──────────────────────┬──────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────┐
        │ CSV Result File               │
        │ (192 rows × 10 columns)       │
        └──────────────┬────────────────┘
                       │
                       ↓
        ┌──────────────────────────────┐
        │ analyze_tls_component.py      │
        │ (Python analysis script)      │
        └──────────────┬────────────────┘
                       │
          ┌────────────┴────────────┐
          ↓                         ↓
    ┌───────────────────┐  ┌──────────────────────┐
    │ Console Output    │  │ Visualization PNGs   │
    │ (metrics analysis)│  │ (2 files, 300 DPI)   │
    └───────────────────┘  └──────────────────────┘
```

---

## Compilation & Execution Checklist

- [ ] All source files modified as per above specification
- [ ] CMakeLists.txt updated with new target
- [ ] Python environment has pandas, matplotlib, seaborn
- [ ] Dataset files present in `datasets/hdf5files/`
- [ ] Index files pre-built (`.bin` files exist)
- [ ] Build directory created: `mkdir build && cd build`
- [ ] CMake configured: `cmake .. -DCMAKE_BUILD_TYPE=Release`
- [ ] Compilation successful: `cmake --build . --target test_tls_component_analysis`
- [ ] Executable created: `./test_tls_component_analysis`
- [ ] Test script is executable: `chmod +x script/test_tls_component_analysis.sh`
- [ ] Full pipeline runs: `bash script/test_tls_component_analysis.sh`

---

## Publication Readiness

This ablation study framework provides:
1. ✓ Comprehensive evaluation across multiple hyperparameters
2. ✓ Publication-quality visualizations (300 DPI PNG)
3. ✓ Detailed metrics for performance analysis
4. ✓ Reproducible configuration (all parameters documented)
5. ✓ Automated analysis pipeline
6. ✓ Complete documentation and build instructions

**Next Step**: Run full benchmark and generate results for SIGMOD'27 submission.

---

## Reference Code Patterns Used

**Pattern 1**: Multi-value parameter sweeping (from `test_vq_recall.cpp`)
```cpp
std::vector<size_t> efs = {10, 20, 30, ..., 300};
for (size_t ef : efs) { ... }
```

**Pattern 2**: Single ef configuration (from `test_search_hnswalp_pq.cpp`)
```cpp
test_tls_configuration(dataset, base_dir, k, ef, use_tls, tls_ratio);
```

**Pattern 3**: Nested test loops (custom)
```cpp
for (dataset) for (ef) for (tls_ratio) { test() }
```

---

**Last Updated**: 2024  
**Status**: Complete and Ready for Testing  
**Target**: SIGMOD'27 TLS Component Analysis
