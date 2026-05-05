# VQ Recall 分析图表说明

本目录包含四张对比图表，用于分析 `test_vq_recall.cpp` 在四个数据集上的性能测试结果。

## 生成图表的方式

```bash
python plot_vq_recall_analysis.py --csv vq_recall_results.csv --out_dir vq_recall_plots
```

## 四张图表详细说明

### 1. recall@1_vs_latency.png - Recall@1 vs Latency (2x2 子图)
- **坐标轴**: X轴 = 延迟(μs), Y轴 = Recall (值域由CAANS范围决定)
- **含义**: 展示在K=1邻近邻搜索中，各索引算法的recall-latency权衡
- **分析**:
  - 左侧曲线（低延迟）：更优的性能
  - CAANS（红色方块）：压缩近似最近邻搜索，通过智能组合多个tls_ratio版本实现平滑曲线
  - HNSW（蓝色圆点）：原始HNSW基线
  - FaissHNSW（橙色三角）：Faiss HNSW实现
  - HNSWPQ_1/2（黄/绿色）：PQ压缩版本
  - HNSWSQ_8（粉色）：SQ压缩版本

### 2. recall@1_vs_vq.png - Recall@1 vs VQ Capacity (2x2 子图)
- **坐标轴**: X轴 = Recall, Y轴 = VQ Capacity (向量/KB)
- **含义**: 展示在K=1搜索中，各索引算法在不同recall点的VQ容量
- **分析**:
  - VQ值越高，表示每KB内存存储的向量越多（空间效率越高）
  - 曲线更靠近左上角（高VQ，高recall）为更优
  - CAANS采用多tls_ratio版本优化，每个recall点选最优VQ

### 3. recall@10_vs_latency.png - Recall@10 vs Latency (2x2 子图)
- **坐标轴**: X轴 = 延迟(μs), Y轴 = Recall (值域由CAANS范围决定)
- **含义**: 展示在K=10邻近邻搜索中，各索引算法的性能对比
- **分析**:
  - K=10相比K=1，通常需要更高的延迟来达到相同recall
  - 展示不同算法在深度搜索场景下的表现

### 4. recall@10_vs_vq.png - Recall@10 vs VQ Capacity (2x2 子图)
- **坐标轴**: X轴 = Recall, Y轴 = VQ Capacity
- **含义**: K=10场景下的空间效率对比

## 数据集

测试使用四个数据集：
1. **Fashion-MNIST** (784维)
2. **MNIST** (784维)
3. **GIST** (960维)
4. **SIFT** (128维)

## CAANS (Compressed Approximate Nearest Neighbor Search) 优化策略

CAANS 是对HNSWALP多个tls_ratio版本的智能组合和优化：

### 平滑曲线生成方法
1. **分别插值**: 为每个tls_ratio版本（0.0, 0.1, 0.2, 0.3, 0.5）单独建立三次样条插值函数
2. **智能融合**: 在recall维度上均匀采样，在每个recall点处选择**最低延迟**的tls_ratio版本
3. **平滑输出**: 生成200+个采样点的光滑曲线，避免相邻bin之间的跳跃

### 关键优势
- **平滑性**: 通过对各tls_ratio的细粒度插值和点态最优选择，得到平滑的综合曲线
- **最优性**: 在每个recall级别都选择最优的参数组合
- **压缩效率**: ALP+PQ压缩方式实现高空间效率与精度的平衡

### 与标准HNSW的对比
- HNSW: 基础HNSW，无压缩
- CAANS: 应用ALP（Adaptive Lossless Predictive）压缩和PQ（Product Quantization）
- 结果: CAANS通常在延迟和VQ capacity上都有显著改进

## 坐标轴范围说明

重要改进：所有图表都**以CAANS的recall范围为基准**：
- CAANS的最小recall和最大recall作为所有其他算法的过滤边界
- 其他算法只显示在这个范围内的数据点
- 这样可以进行公平的算法对比，避免因算法能力差异造成的视觉误导

## 曲线样式说明

| 算法 | 标记 | 线型 | 颜色 | 说明 |
|------|------|------|------|------|
| CAANS | 方块 | 实线 | 红色 | **压缩近似最近邻搜索**（多tls_ratio版本优化） |
| HNSW | 圆点 | 实线 | 蓝色 | 原始HNSW基线 |
| FaissHNSW | 三角 | 虚线 | 橙色 | Faiss HNSW实现 |
| IVF | 菱形 | 虚线 | 绿色 | 倒排表索引 |
| NSG | 倒三角 | 点划线 | 紫色 | Navigating Spreading-out Graph |
| HNSWPQ_1 | P形 | 虚线 | 青色 | HNSW + PQ (m=1) |
| HNSWPQ_2 | 星形 | 虚线 | 黄色 | HNSW + PQ (m=2) |
| HNSWSQ_4 | X形 | 点划线 | 棕色 | HNSW + SQ (4bits) |
| HNSWSQ_8 | 六边形 | 点划线 | 粉色 | HNSW + SQ (8bits) |

## 关键观察

1. **CAANS平滑性**: 红色曲线通过新的优化算法实现了光滑的性能曲线
2. **数据集差异**: 不同数据集上的性能表现差异较大（如SIFT和GIST具有更陡峭的曲线）
3. **压缩效果**: PQ和SQ版本的VQ值通常更高（空间效率更高），但延迟较低
4. **搜索深度影响**: Recall@10相比Recall@1，所有算法的延迟都有显著增加，但CAANS仍保持竞争力
5. **最优recall范围**: CAANS在高recall区间（>95%）表现最优
