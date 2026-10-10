#!/usr/bin/env python3
"""Measure single-thread ScaNN recall and latency using the VQ CSV schema."""

import argparse
import csv
import struct
import time
from pathlib import Path

import numpy as np


def read_vectors(path, value_dtype):
    with Path(path).open("rb") as source:
        width = struct.unpack("<i", source.read(4))[0]
    if width <= 0:
        raise ValueError(f"invalid vector width: {path}")
    row_bytes = 4 + width * np.dtype(value_dtype).itemsize
    size = Path(path).stat().st_size
    if size % row_bytes:
        raise ValueError(f"invalid vector file length: {path}")
    rows = np.memmap(path, mode="r", dtype=np.dtype([
        ("width", "<i4"), ("values", value_dtype, (width,))]),
        shape=(size // row_bytes,))
    if np.any(rows["width"] != width):
        raise ValueError(f"inconsistent vector width: {path}")
    return np.ascontiguousarray(rows["values"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--nlist", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 10])
    parser.add_argument("--params", type=int, nargs="+",
                        default=[1, 2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100])
    parser.add_argument("--candidates", type=int, default=100)
    args = parser.parse_args()
    if args.nodes < 1 or args.nlist < 0 or args.rounds < 1 or args.candidates < 1:
        raise ValueError("invalid search configuration")

    import scann

    queries = read_vectors(args.queries, "<f4")
    truth = read_vectors(args.ground_truth, "<u4")
    if not len(queries) or len(queries) != len(truth):
        raise ValueError("query and ground-truth row counts differ")
    index_bytes = sum(file.stat().st_size for file in args.index.rglob("*")
                      if file.is_file())
    if index_bytes <= 0:
        raise ValueError("empty ScaNN index")
    index_kb = index_bytes / 1024
    vectors_per_kb = args.nodes / index_kb
    searcher = scann.scann_ops_pybind.load_searcher(str(args.index))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["Dataset", "IndexType", "Param", "K", "Recall", "QPS",
                         "Latency(us)", "IndexSizeKB", "VectorsPerKB", "VQ"])
        for k in args.k:
            if k < 1 or k > truth.shape[1]:
                raise ValueError("K exceeds ground-truth width")
            for param in args.params:
                if param < 1 or (args.nlist and param > args.nlist):
                    continue
                leaves = param if args.nlist else None
                candidates = args.candidates if args.nlist else param
                if candidates < k:
                    continue
                setting = (f"leaves={leaves};pre_reorder={candidates}"
                           if leaves else f"pre_reorder={candidates}")
                options = dict(final_num_neighbors=k,
                               pre_reorder_num_neighbors=candidates,
                               leaves_to_search=leaves)
                searcher.search_batched(queries[:min(64, len(queries))], **options)
                elapsed = 0.0
                recall = None
                for round_no in range(args.rounds):
                    start = time.perf_counter()
                    ids, _ = searcher.search_batched(queries, **options)
                    elapsed += time.perf_counter() - start
                    if round_no == 0:
                        ids = np.asarray(ids)
                        if ids.shape != (len(queries), k):
                            raise ValueError(f"unexpected ScaNN result shape: {ids.shape}")
                        hits = np.any((ids[:, :, None] >= 0) &
                                      (ids[:, :, None] == truth[:, None, :k]), axis=2)
                        recall = float(hits.sum()) / (len(queries) * k)
                latency_us = elapsed * 1e6 / (args.rounds * len(queries))
                qps = 1e6 / latency_us
                writer.writerow([args.dataset, "ScaNN", setting, k,
                                 f"{recall:.6f}", f"{qps:.6f}",
                                 f"{latency_us:.6f}", f"{index_kb:.6f}",
                                 f"{vectors_per_kb:.6f}",
                                 f"{vectors_per_kb * qps:.6f}"])
                target.flush()
                print(f"{args.dataset} ScaNN {setting} K={k} "
                      f"recall={recall:.6f} QPS={qps:.1f}", flush=True)


if __name__ == "__main__":
    main()
