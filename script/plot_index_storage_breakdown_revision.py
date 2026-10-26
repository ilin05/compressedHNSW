"""Plot storage components for HNSW/LVC and the revision baselines."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from plot_index_storage_breakdown import (
    DATASET_ORDER,
    COMPONENT_COLORS as ORIGINAL_COLORS,
    format_size,
    normalize_dataset_name,
    pretty_dataset_name,
)


SCRIPT_DIR = Path(__file__).resolve().parent
INDEX_ORDER = (
    "HNSW",
    "HNSW+PQ",
    "HNSW+SQ",
    "LVC",
    "IVF-Flat",
    "IVFPQ-FastScan",
    "NSG",
    "ScaNN",
)
INDEX_NAMES = {
    "ivf_flat": "IVF-Flat",
    "ivfpq_fastscan": "IVFPQ-FastScan",
    "nsg": "NSG",
    "scann": "ScaNN",
}
COMPONENT_ORDER = (
    "Vector Payloads",
    "Graph Structure",
    "Metadata",
    "Compression Metadata",
    "Codebook",
    "Cache",
)
COLORS = {
    "Vector Payloads": ORIGINAL_COLORS["Vector Payloads"],
    "Graph Structure": ORIGINAL_COLORS["Graph Structure"],
    "Metadata": ORIGINAL_COLORS["Original Metadata"],
    "Compression Metadata": ORIGINAL_COLORS["Compression Metadata"],
    "Codebook": ORIGINAL_COLORS["Codebook"],
    "Cache": ("#4b9b94", "#2e6863"),
}

# The component names in the two input CSVs describe their source formats.
# These mappings describe the categories shown in the revision figure.
COMMON_COMPONENTS = {
    "Vector Payloads": "Vector Payloads",
    "Quantized Vector Codes": "Vector Payloads",
    "Graph Structure": "Graph Structure",
    "Original Metadata": "Metadata",
    "Compression Metadata": "Compression Metadata",
    "Quantizer Codebook": "Codebook",
    "Lightweight Codebook": "Codebook",
    "Cache": "Cache",
}
BASELINE_COMPONENTS = {
    "IVF-Flat": {
        "Vector IDs": "Metadata",
        "Coarse Centroids": "Metadata",
    },
    "IVFPQ-FastScan": {
        "Vector IDs": "Metadata",
        "Coarse Centroids": "Codebook",
    },
    "NSG": {},
    "ScaNN": {
        "Partition Assignments": "Metadata",
        "Partition Structure": "Metadata",
        "Quantized Vector Codes": "Codebook",
    },
}


def load_rows(path, baseline, totals, file_sizes):
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            dataset = normalize_dataset_name(row["Dataset"])
            raw_index = row["IndexType"].strip()
            index = INDEX_NAMES.get(raw_index, raw_index) if baseline else raw_index
            if index not in INDEX_ORDER:
                continue
            component = row["Component"].strip()
            mapping = BASELINE_COMPONENTS[index] if baseline else {}
            category = mapping.get(component, COMMON_COMPONENTS.get(component))
            if category is None:
                raise ValueError(f"Unmapped component {component!r} in {path}: {dataset}/{index}")
            byte_count = int(row["Bytes"])
            if byte_count < 0:
                raise ValueError(f"Negative byte count in {path}: {dataset}/{index}/{component}")
            totals[dataset, index][category] += byte_count
            key = dataset, index
            file_size = int(row["FileSizeBytes"])
            if key in file_sizes and file_sizes[key] != file_size:
                raise ValueError(f"Inconsistent file size for {dataset}/{index}")
            file_sizes[key] = file_size


def load_breakdown(original_csv, baseline_csv):
    totals = defaultdict(lambda: defaultdict(int))
    file_sizes = {}
    load_rows(original_csv, False, totals, file_sizes)
    load_rows(baseline_csv, True, totals, file_sizes)
    datasets = [dataset for dataset in DATASET_ORDER if any((dataset, index) in totals for index in INDEX_ORDER)]
    if not datasets:
        raise ValueError("No datasets found in the storage breakdown CSVs")
    for dataset in datasets:
        for index in INDEX_ORDER:
            key = dataset, index
            if key not in totals:
                raise ValueError(f"Missing storage breakdown for {dataset}/{index}")
            total = sum(totals[key].values())
            runtime_cache = totals[key].get("Cache", 0) if total > file_sizes[key] else 0
            if total - runtime_cache != file_sizes[key]:
                raise ValueError(
                    f"Storage components do not match file size for {dataset}/{index}: "
                    f"{total} component bytes, {file_sizes[key]} file bytes"
                )
    return datasets, totals


def draw_breakdown(datasets, totals, output, subplot_width, subplot_height):
    fig, axes = plt.subplots(
        1, len(datasets),
        figsize=(subplot_width * len(datasets), subplot_height),
        sharey=True, squeeze=False,
    )
    axes = axes[0]
    positions = list(range(len(INDEX_ORDER)))
    # positions = [i*0.8 for i in range(len(INDEX_ORDER))]  # alternative spacing
    gib = 1024 ** 3

    for panel, (ax, dataset) in enumerate(zip(axes, datasets)):
        maximum = max(sum(totals[dataset, index].values()) for index in INDEX_ORDER) / gib
        for y, index in enumerate(INDEX_ORDER):
            left = 0.0
            for component in COMPONENT_ORDER:
                value = totals[dataset, index].get(component, 0) / gib
                if value == 0:
                    continue
                face, edge = COLORS[component]
                ax.barh(
                    y, value, left=left, height=0.62, color=face,
                    edgecolor=edge, linewidth=0.75, alpha=0.94, zorder=3,
                )
                left += value
            ax.text(
                left + maximum * 0.018, y, format_size(left),
                ha="left", va="center", fontsize=11,
                color="#1f1f1f", fontweight="bold",
            )
        ax.set_xlim(0, max(maximum * 1.35, 0.02))
        ax.set_yticks(positions)
        if panel == 0:
            ax.set_yticklabels(INDEX_ORDER, fontsize=15)
        else:
            ax.tick_params(axis="y", labelleft=False, left=False)
        ax.invert_yaxis()
        ax.set_title(pretty_dataset_name(dataset), fontsize=16, pad=8)
        ax.grid(axis="x", linestyle="--", alpha=0.26, zorder=0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(axis="x", labelsize=13)

    handles = [
        Patch(facecolor=COLORS[name][0], edgecolor=COLORS[name][1],
              linewidth=0.75, alpha=0.94, label=name)
        for name in COMPONENT_ORDER
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=len(handles), frameon=False,
        bbox_to_anchor=(0.5, 0.985), handletextpad=0.5,
        columnspacing=1.0, borderpad=0.0,
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.93], w_pad=0.8)
    fig.subplots_adjust(bottom=0.18)
    fig.supxlabel("Index Size (GiB)", fontsize=20, y=0.065)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=SCRIPT_DIR / "index_storage_breakdown.csv")
    parser.add_argument(
        "--baseline-csv", type=Path,
        default=SCRIPT_DIR / "revision_baseline_build_results" / "storage_breakdown.csv",
    )
    parser.add_argument("--out_dir", type=Path, default=SCRIPT_DIR / "index_size_revision_plots")
    parser.add_argument("--output", default="index_storage_breakdown_revision.pdf")
    parser.add_argument("--subplot_width", type=float, default=3.7)
    parser.add_argument("--subplot_height", type=float, default=5.6)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    datasets, totals = load_breakdown(args.csv, args.baseline_csv)
    output = args.out_dir / args.output
    draw_breakdown(datasets, totals, output, args.subplot_width, args.subplot_height)
    print(f"Saved figure: {output}")


if __name__ == "__main__":
    main()
