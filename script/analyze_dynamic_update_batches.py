#!/usr/bin/env python3
"""Summarize HNSW dynamic-update batches without treating failed runs as results.

Usage: python3 script/analyze_dynamic_update_batches.py [--input DIR] [--output DIR]
The default input is script/dynamic_update_batches_results and output is its
analysis subdirectory. Only Python's standard library is required.
"""

import argparse
import csv
import re
import statistics
from pathlib import Path


CODECS = ("raw_f64", "raw_f32", "alp", "dexor", "gorilla", "elf")
LOSSLESS = {"raw_f64", "raw_f32", "dexor", "gorilla", "elf"}
DATASETS = (
    "sift-128-euclidean", "mnist-784-euclidean",
    "fashion-mnist-784-euclidean", "gist-960-euclidean",
)


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=2):
    return f"{value:.{digits}f}"


def analyze(input_dir, output_dir, base, batch, batches, queries):
    metrics = read_csv(input_dir / "all_metrics.csv")
    statuses = read_csv(input_dir / "status.csv")
    perf = read_csv(input_dir / "all_perf_breakdown.csv")
    status_by_key = {(r["dataset"], r["codec"]): r for r in statuses}
    perf_by_key = {(r["dataset"], r["codec"]): r for r in perf}
    if len(status_by_key) != len(statuses) or len(perf_by_key) != len(perf):
        raise ValueError("Duplicate dataset/codec in status or perf CSV")

    rows_by_key = {}
    for row in metrics:
        key = (row["dataset"], row["codec"])
        rows_by_key.setdefault(key, []).append(row)

    summary, stages, failures = [], [], []
    for dataset in DATASETS:
        for codec in CODECS:
            key = (dataset, codec)
            status = status_by_key.get(key, {})
            rows = rows_by_key.get(key, [])
            expected = [("before", 0)] + [("after", i) for i in range(1, batches + 1)]
            observed = [(r["stage"], int(r["batch"])) for r in rows]
            complete = status.get("normal_status") == "PASS" and observed == expected
            if not complete:
                log = input_dir / dataset / codec / "normal.log"
                detail = ""
                if log.is_file():
                    errors = re.findall(r"^FAIL:.*$", log.read_text(errors="replace"), re.M)
                    detail = errors[0] if errors else "No FAIL line in normal.log"
                failures.append({"dataset": dataset, "codec": codec,
                                 "normal_status": status.get("normal_status", "MISSING"),
                                 "perf_status": status.get("perf_status", "MISSING"),
                                 "observed_stages": len(rows), "detail": detail})
                continue

            if int(rows[0]["active"]) != base or int(rows[0]["deleted"]) != 0:
                raise ValueError(f"Unexpected initial active/deleted counts: {key}")
            for i, row in enumerate(rows[1:], 1):
                if int(row["active"]) != base or int(row["deleted"]) != batch * i:
                    raise ValueError(f"Unexpected active/deleted counts: {key}, batch {i}")
                if codec in LOSSLESS and float(row["max_insert_abs_error"]) != 0:
                    raise ValueError(f"Lossless insert error in completed run: {key}, batch {i}")
            before, after = rows[0], rows[-1]
            updates = rows[1:]
            mean = lambda field: statistics.mean(float(r[field]) for r in updates)
            insert_us = mean("insert_mean_us")
            graph_us = mean("graph_mean_us") if codec not in ("raw_f64", "raw_f32") else None
            encoding_us = mean("encoding_mean_us") if graph_us is not None else None
            p = perf_by_key.get(key, {})
            samples = int(p.get("total_samples", 0))
            if status.get("perf_status") == "PASS" and samples == 0:
                raise ValueError(f"PASS perf status without samples: {key}")
            item = {
                "dataset": dataset, "codec": codec, "batches": batches,
                "insertions": batch * batches, "logical_deletions": batch * batches,
                "active_before": int(before["active"]), "active_after": int(after["active"]),
                "deleted_after": int(after["deleted"]),
                "insert_mean_us": insert_us, "delete_mean_us": mean("delete_mean_us"),
                "graph_mean_us": graph_us if graph_us is not None else "",
                "encoding_mean_us": encoding_us if encoding_us is not None else "",
                "graph_pct_of_insert": 100 * graph_us / insert_us if graph_us is not None else "",
                "encoding_pct_of_insert": 100 * encoding_us / insert_us if encoding_us is not None else "",
                "index_before_bytes": int(before["index_bytes"]),
                "index_after_bytes": int(after["index_bytes"]),
                "index_growth_pct": 100 * (int(after["index_bytes"]) / int(before["index_bytes"]) - 1),
                "recall_before": float(before["recall_at_1"]),
                "recall_after": float(after["recall_at_1"]),
                "query_before_us": float(before["mean_query_us"]),
                "query_after_us": float(after["mean_query_us"]),
                "qps_before": float(before["qps"]), "qps_after": float(after["qps"]),
                "max_insert_abs_error": max(float(r["max_insert_abs_error"]) for r in updates),
                "perf_samples": samples, "decode_stack_samples": int(p.get("decode_samples", 0)),
                "decode_stack_pct": 100 * int(p.get("decode_samples", 0)) / samples if samples else "",
                "perf_status": status.get("perf_status", "MISSING"),
            }
            summary.append(item)
            for row in rows:
                stages.append(row)

    for item in summary:
        baseline = next(x for x in summary if x["dataset"] == item["dataset"]
                        and x["codec"] == "raw_f64")
        item["insert_vs_raw_f64_x"] = item["insert_mean_us"] / baseline["insert_mean_us"]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "summary.csv", summary, list(summary[0]))
    write_csv(output_dir / "stages.csv", stages, list(stages[0]))
    write_csv(output_dir / "failures.csv", failures,
              ["dataset", "codec", "normal_status", "perf_status", "observed_stages", "detail"])

    lines = ["# HNSW dynamic-update batch analysis", "",
             f"Complete combinations: {len(summary)}/{len(DATASETS) * len(CODECS)}. "
             f"Failures: {len(failures)}.", "",
             f"Each combination starts with {base} randomly sampled active nodes. "
             f"Each of {batches} batches logically deletes and inserts {batch} nodes "
             f"({100 * batch / base:.2f}% of the active set). "
             f"Recall@1 uses {queries} queries and exact nearest neighbors among the "
             "current sampled active nodes, not full-dataset ground truth.", "",
             "Insert/delete times are means over batches; index size is serialized "
             "footprint and includes retained deleted records and new inserts. "
             "Graph and encoding are instrumented portions of addPoint; raw modes "
             "have no corresponding breakdown. perf uses a separate 1024-insert "
             "run: decode-stack percentage is the fraction of sampled call stacks "
             "containing getOriginalDataByInternalId, not exclusive CPU time.", "",
             "| Dataset | Codec | Insert ms | Graph % | Encode % | Delete us | "
             "Index MiB before→after | Recall@1 before→after | Query ms before→after | "
             "Decode stacks / samples |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    short = {"sift-128-euclidean": "SIFT", "mnist-784-euclidean": "MNIST",
             "fashion-mnist-784-euclidean": "F-MNIST", "gist-960-euclidean": "GIST"}
    for x in summary:
        g = fmt(x["graph_pct_of_insert"], 1) if x["graph_pct_of_insert"] != "" else "--"
        e = fmt(x["encoding_pct_of_insert"], 1) if x["encoding_pct_of_insert"] != "" else "--"
        decode = (f"{x['decode_stack_samples']}/{x['perf_samples']}"
                  if x["perf_samples"] else "--")
        lines.append(f"| {short[x['dataset']]} | {x['codec']} | {fmt(x['insert_mean_us']/1000)} | "
                     f"{g} | {e} | {fmt(x['delete_mean_us'], 3)} | "
                     f"{fmt(x['index_before_bytes']/1048576)}→{fmt(x['index_after_bytes']/1048576)} | "
                     f"{fmt(x['recall_before'])}→{fmt(x['recall_after'])} | "
                     f"{fmt(x['query_before_us']/1000)}→{fmt(x['query_after_us']/1000)} | {decode} |")
    if failures:
        lines += ["", "## Failed or incomplete combinations", ""]
        lines += [f"- {x['dataset']} / {x['codec']}: {x['detail'] or x['normal_status']}"
                  for x in failures]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(summary)} complete combinations and {len(failures)} failures to {output_dir}")


def main():
    default = Path(__file__).resolve().parent / "dynamic_update_batches_results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--queries", type=int, default=100)
    args = parser.parse_args()
    analyze(args.input, args.output or args.input / "analysis", args.base, args.batch,
            args.batches, args.queries)


if __name__ == "__main__":
    main()
