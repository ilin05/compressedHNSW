#!/usr/bin/env python3
"""Plot SIFT IVF-LVC Recall@1 against latency and VQ capacity.

VQ capacity = QPS * (number of indexed vectors / full index size in KiB).
The full index sizes in search.csv include PQ codes and decoding state cache.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


CURVES = (
    ("IVF-Flat", "ivf_flat", 0, 0, "1", "#626B73", "o", "--", 2.5),
    ("IVF-PQ", "ivfpq_m", 0, 0, "1", "#9FC9B4", "^", "-.", 2.3),
    ("ALP Exact", "alp", 0, 0, "1", "#174A7E", "D", "-", 2.8),
    ("ALP Filtered", "alp", 0, 1, "0.05", "#174A7E", "X", "--", 2.8),
    ("DeXOR Exact", "dexor", 2, 0, "1", "#C84A24", "o", "-", 2.5),
    ("DeXOR Filtered", "dexor", 2, 1, "0.05", "#C84A24", "v", "--", 2.5),
)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Linux Libertine O", "Libertinus Serif", "Times New Roman",
                   "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
    "font.size": 25,
    "axes.labelsize": 25,
    "xtick.labelsize": 25,
    "ytick.labelsize": 25,
    "legend.fontsize": 25,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_curves(path):
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"No search results in {path}")
    with (path.parent / "compression.csv").open(newline="", encoding="utf-8") as source:
        compression = list(csv.DictReader(source))
    nodes = int(next(row["nodes"] for row in compression if row["mode"] == "ivf_flat"))

    curves = {}
    all_probes = None
    for label, mode, chain, filtered, alpha, *_ in CURVES:
        points = [row for row in rows if row["mode"] == mode
                  and int(row["chain_max"]) == chain
                  and int(row["pq_filter"]) == filtered
                  and float(row["pq_alpha"]) == float(alpha)]
        if not points:
            raise ValueError(f"Missing search results for {label}")
        probes = [int(row["nprobe"]) for row in points]
        if len(probes) != len(set(probes)):
            raise ValueError(f"Duplicate nprobe values for {label}")
        points.sort(key=lambda row: int(row["nprobe"]))
        probes = {int(row["nprobe"]) for row in points}
        if all_probes is None:
            all_probes = probes
        elif probes != all_probes:
            raise ValueError(f"nprobe sweep differs for {label}")

        curves[label] = []
        for row in points:
            size_kib = int(row["index_bytes"]) / 1024
            curves[label].append({
                "recall": float(row["recall_at_1"]),
                "latency_ms": float(row["mean_us"]) / 1000,
                "vq": float(row["qps"]) * nodes / size_kib,
            })
    return curves


def plot_panel(ax, curves, x_key):
    handles = []
    for label, _, _, _, _, color, marker, linestyle, linewidth in CURVES:
        points = curves[label]
        line, = ax.plot([point[x_key] for point in points],
                        [point["recall"] for point in points],
                        label=label, color=color, marker=marker, linestyle=linestyle,
                        linewidth=linewidth, markersize=15.0, markerfacecolor=color,
                        markeredgecolor="white", markeredgewidth=0.8, alpha=0.98)
        handles.append(line)

    ax.set_xscale("log")
    ax.set_xticks([0.1, 1, 10, 100] if x_key == "latency_ms"
                  else [100, 1000, 10000, 100000])
    if x_key == "vq":
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value / 1000:g}k" if value >= 1000 else f"{value:g}"))
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_ylim(0.25, 1.005)
    ax.set_xlabel("Latency (ms)" if x_key == "latency_ms" else "VQ Capacity", fontsize=24)
    ax.set_ylabel("Recall@1")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.tick_params(axis="both", labelsize=23)
    return handles


def add_legend(fig, handles, columns, anchor_x, fontsize):
    fig.legend(handles, [line.get_label() for line in handles],
               loc="upper center", ncol=columns, frameon=False,
               bbox_to_anchor=(anchor_x, 1.0), fontsize=fontsize,
               handletextpad=0.35, columnspacing=0.7,
               handlelength=1.3, borderpad=0.0)


def plot_single(curves, x_key, output):
    fig, ax = plt.subplots(figsize=(5.7, 5.8))
    handles = plot_panel(ax, curves, x_key)
    add_legend(fig, handles, 2, 0.49, 20.5)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.75, bottom=0.13)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


def plot_combined(curves, output):
    # Match the panel proportions of plot_lvc_codec_comparison.py; the extra
    # figure height accommodates one shared legend above both panels.
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9))
    handles = plot_panel(axes[0], curves, "latency_ms")
    plot_panel(axes[1], curves, "vq")
    axes[0].set_title("(a) Recall@1 vs. Latency", fontsize=24, pad=2)
    axes[1].set_title("(b) Recall@1 vs. VQ Capacity", fontsize=24, pad=2)
    add_legend(fig, handles, 3, 0.514, 25)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.70, bottom=0.15, wspace=0.30)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


def plot_stacked(curves, output):
    """Two readable panels with one shared legend for a paper column."""
    fig, axes = plt.subplots(2, 1, figsize=(5.7, 9.0))
    handles = plot_panel(axes[0], curves, "latency_ms")
    plot_panel(axes[1], curves, "vq")
    axes[0].set_title("(a) Recall@1 vs. Latency", fontsize=24, pad=2)
    axes[1].set_title("(b) Recall@1 vs. VQ Capacity", fontsize=24, pad=2)
    add_legend(fig, handles, 2, 0.49, 20.5)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.76, bottom=0.08, hspace=0.53)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=script_dir / "ivf_lvc_extension_sift_results" / "search.csv")
    parser.add_argument("--out-dir", type=Path,
                        default=script_dir.parent.parent / "VectorCompress" / "SIGMOD27_VectorCompress" / "figures")
    args = parser.parse_args()
    curves = load_curves(args.results)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_single(curves, "latency_ms", args.out_dir / "ivf_lvc_sift_recall@1_latency.pdf")
    plot_single(curves, "vq", args.out_dir / "ivf_lvc_sift_recall@1_vq_capacity.pdf")
    plot_combined(curves, args.out_dir / "ivf_lvc_sift_recall@1_combined.pdf")
    plot_stacked(curves, args.out_dir / "ivf_lvc_sift_recall@1_stacked.pdf")


if __name__ == "__main__":
    main()
