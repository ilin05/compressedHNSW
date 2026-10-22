#!/usr/bin/env python3
"""Validate ablation outputs, summarize per-batch data and relative improvements."""
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def main():
    root = Path(sys.argv[1])
    with (root / "metrics.csv").open() as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    modes = {"raw_f32", "alp_base", "alp_cache", "alp_upper", "alp_joint", "alp_exact", "alp_alpha1"}
    assert {r["mode"] for r in rows} == modes
    groups = defaultdict(list)
    for row in rows:
        groups[row["mode"], int(row["batch"]), int(row["ef_search"])].append(row)
        assert math.isfinite(float(row["insert_mean_us"]))
        assert float(row["max_live_abs_error"]) <= 1.1e-4
        assert float(row["query_alpha"]) == 1
        assert int(row["ef_search"]) == 40
        if row["decode_cache"] == "0":
            assert int(row["decode_cache_hits"]) == 0
        if row["upper_tls"] == "0" or float(row["upper_alpha"]) == 1:
            assert int(row["upper_pq_candidates"]) == 0
        if row["insert_filter"] == "0" or float(row["insert_alpha"]) == 1:
            assert int(row["pq_candidates"]) == 0
            assert int(row["filter_searches"]) == 0
    batches = {int(r["batch"]) for r in rows}
    assert batches == set(range(max(batches) + 1)) and len(batches) > 1
    for mode in modes:
        assert {b for m, b, _ in groups if m == mode} == batches
    for cached, reference in (("alp_cache", "alp_base"), ("alp_joint", "alp_upper")):
        for batch in batches - {0}:
            c, r = groups[cached, batch, 40][0], groups[reference, batch, 40][0]
            assert int(c["insert_payload_decodes"]) < int(r["insert_payload_decodes"]), (cached, batch)
    # Whole-index equivalence is already checked in C++; verify query bits too.
    for reference, other in (("alp_base", "alp_cache"), ("alp_upper", "alp_joint"),
                             ("alp_exact", "alp_alpha1")):
        for path in root.glob(reference + "_b*.labels"):
            other_path = path.with_name(path.name.replace(reference + "_", other + "_", 1))
            assert path.read_bytes() == other_path.read_bytes(), (path, other_path)
            assert path.with_suffix(".distances").read_bytes() == other_path.with_suffix(".distances").read_bytes()
        assert len(list(root.glob(reference + "_b*.labels"))) == sum(
            len(samples) for (m, _, _), samples in groups.items() if m == reference)
    summary = []
    for (mode, batch, ef), samples in groups.items():
        row = dict(samples[0])
        row["search_round"] = "mean"
        for field in ("recall_at_1", "recall_at_1_ties", "qps", "mean_query_us"):
            row[field] = statistics.mean(float(r[field]) for r in samples)
        summary.append(row)
    with (root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    aggregate = []
    baseline = statistics.mean(float(r["insert_mean_us"]) for r in summary
                               if r["mode"] == "alp_base" and int(r["batch"]) > 0)
    raw = statistics.mean(float(r["insert_mean_us"]) for r in summary
                         if r["mode"] == "raw_f32" and int(r["batch"]) > 0)
    for mode in sorted(modes):
        updates = [r for r in summary if r["mode"] == mode and int(r["batch"]) > 0]
        latency = statistics.mean(float(r["insert_mean_us"]) for r in updates)
        decodes = sum(int(r["insert_payload_decodes"]) for r in updates)
        hits = sum(int(r["decode_cache_hits"]) for r in updates)
        if mode in {"alp_cache", "alp_joint", "alp_exact", "alp_alpha1"}:
            assert hits > 0 and decodes > 0, f"Cache not exercised: {mode}"
        row = dict(mode=mode, insert_mean_us=latency, speedup_vs_alp_base=baseline / latency,
                   slowdown_vs_hnsw=latency / raw,
                   insert_payload_decodes=decodes, decode_cache_hits=hits,
                   decode_cache_peak_bytes=max(int(r["decode_cache_peak_bytes"]) for r in updates),
                   upper_pq_candidates=sum(int(r["upper_pq_candidates"]) for r in updates),
                   upper_exact_candidates=sum(int(r["upper_exact_candidates"]) for r in updates),
                   recall_at_1_mean=statistics.mean(float(r["recall_at_1"]) for r in updates),
                   qps_mean=statistics.mean(float(r["qps"]) for r in updates),
                   final_index_bytes=updates[-1]["index_file_bytes"])
        if mode in {"alp_upper", "alp_joint"} and float(updates[0]["upper_alpha"]) < 1:
            assert row["upper_pq_candidates"] > 0, f"Upper TLS not exercised: {mode}"
        aggregate.append(row)
        print(f"{mode}: insert={latency:.3f} us, speedup={baseline / latency:.3f}x, "
              f"decodes={decodes}, hits={hits}, recall@1={row['recall_at_1_mean']:.6f}")
    with (root / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    print(f"PASS index/query equivalence and update invariants; results: {root}")


if __name__ == "__main__":
    main()
