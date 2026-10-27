#!/usr/bin/env python3
"""Plot optimized SIFT updates separately for insertion alpha=0.1 and 0.2.

Each 2x4 figure follows plot_dynamic_updates_sift.py: repeated/growing rows,
insertion latency, serialized index size, Recall@1, and query latency columns.
Updates use every measured batch; searches use only measured checkpoints.
The default query TLS alpha is 0.2 independently of the insertion alpha.

Usage:
    python3 script/plot_dynamic_updates_sift_optimized.py
    python3 script/plot_dynamic_updates_sift_optimized.py --alphas 0.1
    python3 script/plot_dynamic_updates_sift_optimized.py --query-mode tls_off
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Match plot_dynamic_updates_sift.py without requiring that script at runtime.
FONT_SIZE = 25
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Linux Libertine O", "Libertinus Serif", "Times New Roman",
                   "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
    "font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE, "pdf.fonttype": 42, "ps.fonttype": 42,
})


DATASET = "sift-128-euclidean"
ALPHA_MODES = {"0.1": "alp_a01", "0.2": "alp_a02"}
LABELS = {"raw_f32": "HNSW", "lvc": "HNSW+LVC"}
CURVE_STYLES = {
    "raw_f32": dict(marker="o", linestyle="--", color="#626B73", linewidth=2.5),
    "lvc": dict(marker="D", linestyle="-", color="#C84A24", linewidth=2.8),
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def load_results(directory, experiment, insert_mode, ef_search, query_mode):
    """Keep update and search samples separate; never interpolate checkpoints."""
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if config["dataset"] != DATASET or config["experiment"] != experiment:
        raise ValueError("Unexpected experiment configuration: {}".format(directory))
    if insert_mode not in config["insert_modes"]:
        raise ValueError("Missing {} configuration: {}".format(insert_mode, directory))
    modes = ("raw_f32", insert_mode)
    updates = {}
    for row in read_csv(directory / "updates.csv"):
        if (row["dataset"] != DATASET or row["experiment"] != experiment or
                row["insert_mode"] not in modes):
            continue
        key = (row["insert_mode"], int(row["batch"]))
        if key in updates:
            raise ValueError("Duplicate update row: {} {}".format(directory, key))
        updates[key] = row

    searches = {}
    for row in read_csv(directory / "summary.csv"):
        mode = row["insert_mode"]
        wanted_query = "tls_off" if mode == "raw_f32" else query_mode
        if (row["dataset"] != DATASET or row["experiment"] != experiment or
                mode not in modes or int(row["ef_search"]) != ef_search or
                row["query_mode"] != wanted_query):
            continue
        key = (mode, int(row["batch"]))
        if key in searches:
            raise ValueError("Duplicate averaged search row: {} {}".format(directory, key))
        if key not in updates:
            raise ValueError("Search has no matching update: {} {}".format(directory, key))
        if int(row["active_nodes"]) != int(updates[key]["active_nodes"]):
            raise ValueError("Search/update node count mismatch: {} {}".format(directory, key))
        searches[key] = row

    expected_updates = list(range(int(config["batches"]) + 1))
    expected_searches = sorted(int(b) for b in config["search_checkpoints"])
    result = {}
    for mode in modes:
        batches = sorted(b for m, b in updates if m == mode)
        checkpoints = sorted(b for m, b in searches if m == mode)
        if batches != expected_updates:
            raise ValueError("Missing update batches: {} {}: {}".format(directory, mode, batches))
        if checkpoints != expected_searches:
            raise ValueError("Missing efSearch={} {} checkpoints: {} {}: {}".format(
                ef_search, query_mode, directory, mode, checkpoints))

        def point(batch):
            row = updates[(mode, batch)]
            return dict(batch=batch, x=batch if experiment == "repeated" else
                        int(row["active_nodes"]) / 1e6)

        update_points = []
        for batch in batches:
            row = updates[(mode, batch)]
            p = point(batch)
            p.update(insert_ms=float(row["insert_mean_us"]) / 1000,
                     index_mib=int(row["index_file_bytes"]) / (1024 ** 2))
            update_points.append(p)
        search_points = []
        for batch in checkpoints:
            row = searches[(mode, batch)]
            recall = float(row["recall_at_1"])
            if not 0 <= recall <= 1:
                raise ValueError("Invalid Recall@1: {} {}".format(directory, (mode, batch)))
            p = point(batch)
            p.update(recall_percent=recall * 100, query_us=float(row["mean_query_us"]))
            search_points.append(p)
        for p in update_points + search_points:
            if not all(math.isfinite(v) and v >= 0 for v in p.values()):
                raise ValueError("Invalid plotted value: {} {} {}".format(directory, mode, p))
        result["raw_f32" if mode == "raw_f32" else "lvc"] = {
            "updates": update_points, "searches": search_points,
        }
    print("Loaded {}: {} update batches, {} search checkpoints per mode".format(
        directory, len(expected_updates), len(expected_searches)))
    return result, config


def draw_curve(ax, points, metric, mode):
    style = CURVE_STYLES[mode]
    return ax.plot([p["x"] for p in points], [p[metric] for p in points],
                   label=LABELS[mode], **style, markersize=15,
                   markevery=max(1, (len(points) - 1) // 10),
                   markeredgecolor="white", markeredgewidth=0.8,
                   markerfacecolor=style["color"], alpha=0.98)[0]


def plot_results(results, configs, output, insert_alpha):
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.10, top=0.80,
                        wspace=0.40, hspace=0.80)
    columns = (
        ("insert_ms", "Insert latency", "Insertion (ms/vector)"),
        ("index_mib", "Index size", "Index size (MiB)"),
        ("recall_percent", "Recall@1", "Recall@1 (%)"),
        ("query_us", "Query latency", r"Query latency ($\mu$s)"),
    )
    handles = {}
    for row_index, experiment in enumerate(("repeated", "growing")):
        for col, (metric, title, ylabel) in enumerate(columns):
            ax = axes[row_index, col]
            for mode in LABELS:
                source = "searches" if metric in ("recall_percent", "query_us") else "updates"
                points = results[experiment][mode][source]
                if metric == "insert_ms":
                    points = [p for p in points if p["batch"] > 0]
                handles[mode] = draw_curve(ax, points, metric, mode)
                if metric == "recall_percent" and any(p[metric] < 80 for p in points):
                    print("Warning: {} {} recall below the requested 80% axis minimum".format(
                        experiment, mode))
            ax.set_title("({}) {}".format(chr(ord("a") + 4 * row_index + col), title),
                         fontsize=FONT_SIZE, fontweight="normal")
            ax.set_xlabel("Update batch" if experiment == "repeated" else "Active vectors (M)")
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.tick_params(axis="both", labelsize=FONT_SIZE)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=experiment == "repeated"))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.set_ylim((80, 100) if metric == "recall_percent" else (0, None))

    repeated_percent = 100 * float(configs["repeated"]["repeat_ratio"])
    initial_percent = float(configs["growing"]["initial_percent"])
    fig.text(0.53, 0.865, "Repeated insert/delete ({:g}% vectors per batch)".format(
        repeated_percent), ha="center", fontsize=FONT_SIZE)
    fig.text(0.53, 0.415, "Growing index ({:g}% to 100%)".format(initial_percent),
             ha="center", fontsize=FONT_SIZE)
    fig.suptitle(r"SIFT (insertion $\alpha=" + insert_alpha + "$)",
                 x=0.53, y=1.04, fontsize=FONT_SIZE)
    fig.legend([handles[m] for m in LABELS], list(LABELS.values()),
               loc="upper center", bbox_to_anchor=(0.53, 0.995), ncol=2,
               frameon=False, handletextpad=0.5, columnspacing=1.1, borderpad=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = output.with_suffix(suffix)
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        print("Saved:", path)
    plt.close(fig)


def main():
    script = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=script,
                        help="Directory containing the four alpha01/alpha02 result directories")
    parser.add_argument("--output-dir", type=Path, default=script / "dynamic_updates_optimized_plots")
    parser.add_argument("--alphas", nargs="+", choices=tuple(ALPHA_MODES), default=list(ALPHA_MODES),
                        help="Insertion TLS alpha values to plot (default: both)")
    parser.add_argument("--ef-search", type=int, default=40)
    parser.add_argument("--query-mode", choices=("tls_0.2", "tls_off"), default="tls_0.2",
                        help="LVC query mode, independent of insertion alpha; HNSW always uses TLS off")
    args = parser.parse_args()
    for alpha in dict.fromkeys(args.alphas):
        mode = ALPHA_MODES[alpha]
        tag = "alpha" + mode[len("alp_a"):]
        results, configs = {}, {}
        for experiment in ("repeated", "growing"):
            directory = args.results_root / ("dynamic_updates_{}_{}_results".format(
                experiment, tag)) / DATASET
            results[experiment], configs[experiment] = load_results(
                directory, experiment, mode, args.ef_search, args.query_mode)
        name = "dynamic_updates_sift_{}_ef{}_{}_2x4".format(
            tag, args.ef_search, args.query_mode.replace(".", ""))
        plot_results(results, configs, args.output_dir / name, alpha)
        print("Insertion alpha={}; efSearch={}; query mode={}; Recall field=recall_at_1".format(
            alpha, args.ef_search, args.query_mode))


if __name__ == "__main__":
    main()
