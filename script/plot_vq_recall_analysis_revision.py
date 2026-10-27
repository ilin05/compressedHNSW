#!/usr/bin/env python3
"""Revision of the VQ/recall panels with exact LVC and new index baselines.

The filtered curve follows the original greedy operating-point selection, but
uses positive filtering ratios only. Exact LVC uses precisely HNSWALP0.000000.
Missing dataset/index combinations are reported and never synthesized.
FastScan uses the separate M=dim, 4-bit results, keeping only nprobe>=10;
its minimum recall can extend each panel's original lower recall boundary.
Baseline points up to recall=1.0 are shown with the legacy x-axis limits.
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.transforms import nonsingular


SCRIPT_DIR = Path(__file__).resolve().parent
FONT_SIZE = 25
AXIS_FONT_SIZE = 25
LEGEND_FONT_SIZE = 25
HIGH_RECALL_DATASETS = {"mnist-784-euclidean", "fashion-mnist-784-euclidean"}
RECALL10_MIN_FOR_HIGH_RECALL_DATASETS = 0.90
DATASET_ORDER = (
    "mnist-784-euclidean", "fashion-mnist-784-euclidean",
    "sift-128-euclidean", "gist-960-euclidean", "deep-image-96-angular",
)
DATASET_NAMES = {
    "mnist-784-euclidean": "MNIST",
    "fashion-mnist-784-euclidean": "Fashion-MNIST",
    "sift-128-euclidean": "SIFT",
    "gist-960-euclidean": "GIST",
    "deep-image-96-angular": "DEEP",
}
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Linux Libertine O", "Libertinus Serif", "Times New Roman",
                   "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
    "font.size": FONT_SIZE,
    "axes.labelsize": AXIS_FONT_SIZE,
    "xtick.labelsize": AXIS_FONT_SIZE,
    "ytick.labelsize": AXIS_FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
NEW_BASELINES = ("IVF-Flat", "IVFPQ-FastScan", "NSG", "ScaNN")
CURVE_ORDER = (
    "HNSW-Filtered-LVC", "HNSW-Exact-LVC", "HNSW", "HNSW+PQ",
    "HNSW+SQ", *NEW_BASELINES,
)
STYLES = {
    "HNSW-Filtered-LVC": ("D", "-", "#C84A24", 2.8),
    "HNSW-Exact-LVC": ("X", "-", "#174A7E", 2.8),
    "HNSW": ("o", "--", "#626B73", 2.5),
    "HNSW+PQ": ("s", "--", "#91B6D4", 2.3),
    "HNSW+SQ": ("^", "-.", "#9FC9B4", 2.3),
    "IVF-Flat": ("P", "--", "#9A568F", 2.2),
    "IVFPQ-FastScan": ("h", ":", "#735AA6", 2.2),
    "NSG": ("v", "-.", "#A58AC0", 2.2),
    "ScaNN": ("*", "-", "#629E7A", 2.2),
}
DATA_KEYS = {
    "HNSW": "HNSW", "HNSW+PQ": "HNSWPQ_1", "HNSW+SQ": "HNSWSQ_8",
    **{name: name for name in NEW_BASELINES},
}


def load_data(vq_csv, baseline_csv, fastscan_csv):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(vq_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            index_type = row["IndexType"].strip()
            if index_type.startswith("HNSWALP"):
                key = "HNSWALP"
                ratio = float(index_type[len("HNSWALP"):])
            else:
                key = index_type
                ratio = None
            data[row["Dataset"].strip()][int(row["K"])][key].append({
                "recall": float(row["Recall"]),
                "latency_us": float(row["Latency(us)"]),
                "vq": float(row["VQ"]),
                "param": row["Param"],
                "original_index_type": index_type,
                "tls_ratio": ratio,
            })
    counts = {name: 0 for name in NEW_BASELINES}
    with open(baseline_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row["IndexType"].strip()
            if name not in counts or name == "IVFPQ-FastScan":
                continue
            dataset = row["Dataset"].strip()
            k = int(row["K"])
            data[dataset][k][name].append({
                "recall": float(row["Recall"]),
                "latency_us": float(row["Latency(us)"]),
                "vq": float(row["VQ"]),
                "param": row["Param"],
            })
            counts[name] += 1
    # Never mix the old M=dim/4 baseline with the rebuilt M=dim FastScan index.
    with open(fastscan_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["IndexType"].strip() != "IVFPQ-FastScan":
                continue
            match = re.fullmatch(r"nprobe\s*=\s*(\d+)", row["Param"].strip())
            if match is None:
                raise ValueError(f"Invalid FastScan nprobe parameter: {row['Param']}")
            if int(match.group(1)) < 10:
                continue
            data[row["Dataset"].strip()][int(row["K"])]["IVFPQ-FastScan"].append({
                "recall": float(row["Recall"]),
                "latency_us": float(row["Latency(us)"]),
                "vq": float(row["VQ"]),
                "param": row["Param"],
            })
            counts["IVFPQ-FastScan"] += 1
    print("New baseline rows:", ", ".join(f"{name}={counts[name]}" for name in NEW_BASELINES))
    return data


def sort_and_dedup(points, key):
    result = []
    seen = set()
    for point in sorted(points, key=lambda point: point[key]):
        signature = (round(point[key], 10), round(point["recall"], 10))
        if signature not in seen:
            result.append(point)
            seen.add(signature)
    return result


def filtered_path(points):
    """Original script's greedy slope path, restricted to positive filter ratios."""
    by_ratio = defaultdict(list)
    for point in points:
        ratio = point["tls_ratio"]
        if ratio is not None and ratio > 0:
            by_ratio[ratio].append((point["recall"], point["latency_us"], point["vq"]))
    for values in by_ratio.values():
        values.sort(key=lambda point: point[0])
    ratios = [ratio for ratio in (0.1, 0.2, 0.3, 0.5) if by_ratio[ratio]]
    if not ratios:
        return []
    curve, index = 0, 0
    visited = set()
    result = []
    while curve < len(ratios) and (curve, index) not in visited:
        visited.add((curve, index))
        current = by_ratio[ratios[curve]][index]
        result.append({"recall": current[0], "latency_us": current[1], "vq": current[2]})
        candidates = []
        if index + 1 < len(by_ratio[ratios[curve]]):
            candidates.append((curve, index + 1))
        if curve + 1 < len(ratios):
            candidates.extend((curve + 1, next_index)
                              for next_index, point in enumerate(by_ratio[ratios[curve + 1]])
                              if point[0] > current[0])
        if not candidates:
            break
        def slope(target):
            point = by_ratio[ratios[target[0]]][target[1]]
            recall_gain = point[0] - current[0]
            latency_gain = point[1] - current[1]
            return (float("inf") if recall_gain > 0 else float("-inf")) if latency_gain == 0 else recall_gain / latency_gain
        curve, index = max(candidates, key=slope)
    return result


def curves_for_dataset(data, dataset, k):
    curves = {}
    filtered = filtered_path(data[dataset][k].get("HNSWALP", []))
    if filtered:
        curves["HNSW-Filtered-LVC"] = filtered
    exact = [
        point for point in data[dataset][k].get("HNSWALP", [])
        if point["original_index_type"] == "HNSWALP0.000000"
    ]
    if exact:
        curves["HNSW-Exact-LVC"] = exact
    for display, key in DATA_KEYS.items():
        points = data[dataset][k].get(key, [])
        if points:
            curves[display] = points
    return curves


def plot_panel(ax, curves, dataset, k, metric):
    handles = {}
    filtered = curves.get("HNSW-Filtered-LVC", [])
    if k == 10 and dataset in HIGH_RECALL_DATASETS:
        filtered = [point for point in filtered
                    if point["recall"] >= RECALL10_MIN_FOR_HIGH_RECALL_DATASETS]
    min_recall = min((point["recall"] for point in filtered), default=0.0)
    max_recall = max((point["recall"] for point in filtered), default=1.0)
    x_key = "latency_us" if metric == "latency" else "vq"
    # Reproduce the pre-FastScan auto limits from the old curves and recall cut.
    # Expanding the recall range for FastScan must not expand either x-axis.
    reference_x = []
    for name in CURVE_ORDER:
        if name == "IVFPQ-FastScan":
            continue
        points = filtered if name == "HNSW-Filtered-LVC" else [
            point for point in curves.get(name, [])
            if min_recall <= point["recall"] <= max_recall
        ]
        reference_x.extend(point[x_key] / 1000 if metric == "latency" else point[x_key]
                           for point in points)
    # Preserve the old lower axis bound unless the new FastScan data needs more room.
    y_min = min_recall - 0.02
    if k == 10 and dataset in HIGH_RECALL_DATASETS:
        y_min = max(RECALL10_MIN_FOR_HIGH_RECALL_DATASETS, y_min)
    fastscan = curves.get("IVFPQ-FastScan", [])
    if fastscan:
        fastscan_min = min(point["recall"] for point in fastscan)
        min_recall = min(min_recall, fastscan_min)
        y_min = min(y_min, fastscan_min)
    # Keep higher-recall baseline points; reference_x above preserves x limits.
    for name in CURVE_ORDER:
        points = filtered if name == "HNSW-Filtered-LVC" else [
            point for point in curves.get(name, [])
            if min_recall <= point["recall"] <= 1.0
        ]
        if not points:
            continue
        points = sort_and_dedup(points, key=x_key)
        x = [p[x_key] / 1000 if metric == "latency" else p[x_key] for p in points]
        y = [p["recall"] for p in points]
        marker, linestyle, color, linewidth = STYLES[name]
        (line,) = ax.plot(x, y, marker=marker, linestyle=linestyle, color=color,
                          linewidth=linewidth, markersize=15.0,
                          markerfacecolor=color, markeredgecolor="white",
                          markeredgewidth=0.8, alpha=0.98, label=name)
        handles[name] = line
    if reference_x:
        x_low, x_high = nonsingular(min(reference_x), max(reference_x), expander=0.05)
        margin = plt.rcParams["axes.xmargin"] * (x_high - x_low)
        ax.set_xlim(x_low - margin, x_high + margin)
    ax.set_ylim(y_min, 1.005)
    if metric == "latency":
        ax.set_title(DATASET_NAMES.get(dataset, dataset), fontsize=FONT_SIZE - 1)
    ax.set_xlabel("Latency (ms)" if metric == "latency" else "VQ Capacity",
                  fontsize=AXIS_FONT_SIZE - 1)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.tick_params(labelsize=AXIS_FONT_SIZE - 2)
    if dataset in HIGH_RECALL_DATASETS:
        ax.set_yticks([y_min, (y_min + 1.0) / 2, 1.0])
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    return handles


def plot_combined(data, output, k):
    datasets = [d for d in DATASET_ORDER if d in data and k in data[d]]
    if not datasets:
        raise ValueError(f"No Recall@{k} results in the input CSV files")
    fig, axes = plt.subplots(2, len(datasets), figsize=(5.7 * len(datasets), 8.1),
                             squeeze=False)
    handles = {}
    for col, dataset in enumerate(datasets):
        curves = curves_for_dataset(data, dataset, k)
        for row, metric in enumerate(("latency", "vq")):
            handles.update(plot_panel(axes[row][col], curves, dataset, k, metric))
            if col == 0:
                axes[row][col].set_ylabel(f"Recall@{k}", fontsize=AXIS_FONT_SIZE - 1)
    present = [name for name in CURVE_ORDER if name in handles]
    fig.legend([handles[name] for name in present], present, loc="upper center",
               ncol=5, frameon=False, fontsize=LEGEND_FONT_SIZE,
               bbox_to_anchor=(0.5, 1.01),
               columnspacing=1.0, handletextpad=0.4)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.80, bottom=0.09,
                        wspace=0.32, hspace=0.52)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")
    for name in NEW_BASELINES:
        missing = [DATASET_NAMES.get(d, d) for d in datasets
                   if name not in data[d][k]]
        if missing:
            print(f"Recall@{k}: {name} missing for {', '.join(missing)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=SCRIPT_DIR / "vq_recall_results.csv")
    parser.add_argument("--baseline-csv", "--baseline_csv", type=Path,
                        default=SCRIPT_DIR / "revision_baseline_search_results/all_results.csv")
    parser.add_argument("--fastscan-csv", type=Path,
                        default=SCRIPT_DIR / "revision_fastscan_mdim_search_results/all_results.csv")
    parser.add_argument("--out-dir", "--out_dir", type=Path,
                        default=SCRIPT_DIR / "vq_recall_revision_plots")
    args = parser.parse_args()
    data = load_data(args.csv, args.baseline_csv, args.fastscan_csv)
    for k, filename in ((10, "in_memory_recall@10_combined_revision.pdf"),
                        (1, "appendix_recall@1_combined_revision.pdf")):
        plot_combined(data, args.out_dir / filename, k)


if __name__ == "__main__":
    main()
