#!/usr/bin/env python3
"""Plot the 1% repeated-update SIFT experiment as a single-column 1x3 figure.

Panels show insertion latency, Recall@1, and query latency. Insertion alpha
defaults to 0.1; queries show TLS off/on(0.2), sharing one insertion trajectory.
Index size and other supporting numbers are exported to the analysis JSON.
Growing-index results are not loaded. No benchmark or manuscript edits occur.

Usage:
    python3 script/plot_dynamic_updates_sift_optimized.py
    python3 script/plot_dynamic_updates_sift_optimized.py --results-dir PATH
    python3 script/plot_dynamic_updates_sift_optimized.py --query-mode tls_off
"""

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, MaxNLocator, NullLocator, ScalarFormatter

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
LABELS = {"raw_f32": "HNSW", "lvc": "HNSW+LVC", "lvc_off": "HNSW+LVC (TLS off)"}
CURVE_STYLES = {
    "raw_f32": dict(marker="o", linestyle="--", color="#626B73", linewidth=2.5),
    "lvc": dict(marker="D", linestyle="-", color="#C84A24", linewidth=2.8),
    "lvc_off": dict(marker="^", linestyle=":", color="#72558B", linewidth=2.5),
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
    if (experiment != "repeated" or int(config["batches"]) != 20 or
            not math.isclose(float(config["repeat_ratio"]), 0.01) or
            config["threads"] != 1 or config["nodes"] != 1000000 or
            config["search_rounds"] != 3 or not config["insert_decode_cache"] or
            not config["insert_upper_tls"]):
        raise ValueError("Expected full-SIFT 1% repeated updates: 20 batches, one thread, "
                         "three query rounds, upper TLS and insertion decode cache")
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
            return dict(batch=batch, x=batch)

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
    # Both methods must receive the same deletions/reinsertions, with tombstones
    # retained. Check the measured three-round means before plotting summaries.
    for batch in expected_updates:
        raw, lvc = (updates[(mode, batch)] for mode in modes)
        for key in ("update_hash", "active_nodes", "physical_nodes", "deleted_nodes",
                    "inserted_nodes", "marked_nodes"):
            if raw[key] != lvc[key]:
                raise ValueError("Unpaired update workload: batch={} {}".format(batch, key))
        if (int(raw["active_nodes"]) != 1000000 or int(raw["deleted_nodes"]) != batch * 10000 or
                int(raw["inserted_nodes"]) != (10000 if batch else 0)):
            raise ValueError("Wrong 1% update population at batch={}".format(batch))
    rounds = {}
    for row in read_csv(directory / "search.csv"):
        if (row["insert_mode"] not in modes or int(row["ef_search"]) != ef_search or
                row["query_mode"] != ("tls_off" if row["insert_mode"] == "raw_f32" else query_mode)):
            continue
        key = (row["insert_mode"], int(row["batch"]))
        group = rounds.setdefault(key, {})
        round_id = int(row["search_round"])
        if round_id in group:
            raise ValueError("Duplicate query round: {} {}".format(key, round_id))
        group[round_id] = row
    if set(rounds) != set(searches):
        raise ValueError("Raw query rounds and summary checkpoints disagree")
    for key, row in searches.items():
        if set(rounds[key]) != {0, 1, 2}:
            raise ValueError("Missing three query rounds: {}".format(key))
        for metric in ("recall_at_1", "qps", "mean_query_us"):
            mean = statistics.mean(float(r[metric]) for r in rounds[key].values())
            if not math.isclose(mean, float(row[metric]), rel_tol=1e-8, abs_tol=1e-8):
                raise ValueError("Query summary disagrees with raw rounds: {} {}".format(key, metric))
    print("Loaded {}: initial state + {} update batches, {} search checkpoints per mode".format(
        directory, len(expected_updates) - 1, len(expected_searches)))
    return result, config


def draw_curve(ax, points, metric, mode):
    style = CURVE_STYLES[mode]
    return ax.plot([p["x"] for p in points], [p[metric] for p in points],
                   label=LABELS[mode], **style, markersize=15,
                   markevery=max(1, math.ceil((len(points) - 1) / 10)),
                   markeredgecolor="white", markeredgewidth=0.8,
                   markerfacecolor=style["color"], alpha=0.98)[0]


def plot_results(results, config, output):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.17, top=0.72, wspace=0.48)
    columns = (
        ("insert_ms", "Insertion", "Insertion (ms/vector)"),
        ("recall_percent", "Recall@1", "Recall@1 (%)"),
        ("query_us", "Query latency", r"Query latency ($\mu$s)"),
    )
    first = next(iter(results.values()))
    for col, (metric, title, ylabel) in enumerate(columns):
        ax = axes[col]
        source = "updates" if metric == "insert_ms" else "searches"
        curves = [("raw_f32", first["raw_f32"][source])]
        if metric == "insert_ms":
            # Query TLS does not change the insertion trajectory.
            style = "lvc_off" if list(results) == ["tls_off"] else "lvc"
            curves.append((style, first["lvc"][source]))
        else:
            curves.extend(("lvc_off" if query == "tls_off" else "lvc", values["lvc"][source])
                          for query, values in results.items())
        for mode, points in curves:
            if metric == "insert_ms":
                points = [p for p in points if p["batch"] > 0]
            if metric == "recall_percent" and any(p[metric] < 80 for p in points):
                raise ValueError("Recall axis would hide values below 80%")
            draw_curve(ax, points, metric, mode)
        ax.set_title("({}) {}".format(chr(ord("a") + col), title), fontsize=FONT_SIZE, pad=8)
        ax.set_xlabel("Update batch", labelpad=3)
        ax.set_ylabel(ylabel, labelpad=4)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_xlim(-0.9, int(config["batches"]) + 0.9)
        if metric == "query_us":
            values = [p[metric] for _, points in curves for p in points]
            ax.set_yscale("log")
            ax.set_ylim(min(values) * .78, max(values) * 1.2)
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_locator(NullLocator())
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            maximum = max(p[metric] for _, points in curves for p in points)
            ax.set_ylim((80, 100) if metric == "recall_percent" else (0, maximum * 1.18))
    labels = [("raw_f32", "HNSW")]
    labels += [("lvc_off" if q == "tls_off" else "lvc", "HNSW+LVC (TLS {})".format(
        "off" if q == "tls_off" else "on")) for q in results]
    handles = [Line2D([], [], **CURVE_STYLES[m], markersize=12) for m, _ in labels]
    fig.legend(handles, [label for _, label in labels], loc="upper center",
               bbox_to_anchor=(0.53, 1.055), ncol=len(labels), frameon=False,
               handletextpad=0.4, columnspacing=0.9, handlelength=1.6, borderpad=0)
    fig.text(0.53, 0.865, "Repeated insert/delete ({:g}% vectors per batch)".format(
        100 * float(config["repeat_ratio"])), ha="center", fontsize=FONT_SIZE)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = output.with_suffix(suffix)
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        print("Saved:", path)
    plt.close(fig)


def write_analysis(directory, results, config, output, insert_mode, ef_search):
    updates = read_csv(directory / "updates.csv")
    rows = {mode: [r for r in updates if r["insert_mode"] == mode and int(r["batch"]) > 0]
            for mode in ("raw_f32", insert_mode)}
    raw = {int(r["batch"]): r for r in rows["raw_f32"]}
    lvc = {int(r["batch"]): r for r in rows[insert_mode]}
    ratios = [float(lvc[b]["insert_seconds"]) / float(raw[b]["insert_seconds"]) for b in raw]
    final = int(config["batches"])
    report = dict(dataset=DATASET, experiment="repeated", repeat_ratio=config["repeat_ratio"],
        batches=final, ef_search=ef_search, insertion_mode=insert_mode, threads=config["threads"],
        insert_ratio_total=sum(float(r["insert_seconds"]) for r in lvc.values()) /
                           sum(float(r["insert_seconds"]) for r in raw.values()),
        insert_batch_ratio_min=min(ratios), insert_batch_ratio_max=max(ratios),
        lvc_delete_mean_us=statistics.mean(float(r["delete_mean_us"]) for r in lvc.values()),
        final_raw_serialized_bytes=int(raw[final]["index_file_bytes"]),
        final_lvc_serialized_bytes=int(lvc[final]["index_file_bytes"]),
        final_size_reduction_percent=100 * (1 - int(lvc[final]["index_file_bytes"]) /
                                            int(raw[final]["index_file_bytes"])),
        queries={}, footprint_scope="serialized layout including tombstones; not RSS",
        update_scope="one update trajectory; three timing rounds apply only to queries",
        source_sha256={name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                       for name in ("config.json", "updates.csv", "search.csv", "summary.csv")})
    for query, values in results.items():
        pairs = list(zip(values["raw_f32"]["searches"], values["lvc"]["searches"]))
        report["queries"][query] = dict(
            max_recall_deficit_pp=max(a["recall_percent"] - b["recall_percent"] for a, b in pairs),
            latency_ratio_min=min(b["query_us"] / a["query_us"] for a, b in pairs),
            latency_ratio_max=max(b["query_us"] / a["query_us"] for a, b in pairs),
            final_raw_recall_percent=pairs[-1][0]["recall_percent"],
            final_lvc_recall_percent=pairs[-1][1]["recall_percent"],
            final_raw_query_us=pairs[-1][0]["query_us"], final_lvc_query_us=pairs[-1][1]["query_us"])
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print("Insertion ratio={:.4f}x; final serialized-size reduction={:.2f}%".format(
        report["insert_ratio_total"], report["final_size_reduction_percent"]))


def main():
    script = Path(__file__).absolute().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=script,
                        help="Directory containing dynamic_updates_repeated_1pct_alpha*_results")
    parser.add_argument("--results-dir", type=Path, help="One SIFT directory containing config/updates/search/summary")
    parser.add_argument("--output-dir", type=Path, default=script / "dynamic_updates_optimized_plots")
    parser.add_argument("--alphas", nargs="+", choices=tuple(ALPHA_MODES), default=["0.1"],
                        help="Insertion TLS alpha values to plot (default: 0.1)")
    parser.add_argument("--ef-search", type=int, default=40)
    parser.add_argument("--query-mode", choices=("both", "tls_0.2", "tls_off"), default="both",
                        help="Query TLS modes; insertion is identical for both query curves")
    parser.add_argument("--paper-figure", type=Path, help="Also replace this paper PDF; never edit its .tex source")
    args = parser.parse_args()
    if (args.results_dir or args.paper_figure) and len(set(args.alphas)) != 1:
        parser.error("--results-dir/--paper-figure require one insertion alpha")
    for alpha in dict.fromkeys(args.alphas):
        mode = ALPHA_MODES[alpha]
        tag = "alpha" + mode[len("alp_a"):]
        directory = args.results_dir or args.results_root / ("dynamic_updates_repeated_1pct_{}_results".format(tag)) / DATASET
        results = {}
        for query in (("tls_off", "tls_0.2") if args.query_mode == "both" else (args.query_mode,)):
            results[query], config = load_results(directory, "repeated", mode, args.ef_search, query)
        name = "dynamic_updates_sift_{}_ef{}_{}_1x3".format(
            tag, args.ef_search, args.query_mode.replace(".", ""))
        output = args.output_dir / name
        plot_results(results, config, output)
        write_analysis(directory, results, config, output, mode, args.ef_search)
        if args.paper_figure:
            args.paper_figure.parent.mkdir(parents=True, exist_ok=True)
            backup = args.output_dir / "previous_paper_dynamic_updates_sift_revision.pdf"
            if args.paper_figure.exists() and not backup.exists():
                shutil.copyfile(args.paper_figure, backup)
            shutil.copyfile(output.with_suffix(".pdf"), args.paper_figure)
            print("Replaced paper figure:", args.paper_figure)
        print("Insertion alpha={}; efSearch={}; query mode={}; Recall field=recall_at_1".format(
            alpha, args.ef_search, args.query_mode))


if __name__ == "__main__":
    main()
