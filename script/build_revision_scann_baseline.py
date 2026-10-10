#!/usr/bin/env python3
"""Build and serialize one ScaNN L2 baseline from an fvecs dataset."""

import argparse
import csv
import json
import math
import struct
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.15)
    parser.add_argument("--train-cap", type=int, default=1500000)
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()

    try:
        import scann
    except ImportError as exc:
        raise SystemExit("ScaNN Python package missing; install a compatible scann wheel") from exc
    with open(args.input, "rb") as source:
        dim = struct.unpack("<i", source.read(4))[0]
    if dim <= 0 or args.train_fraction <= 0 or args.train_fraction > 1:
        raise ValueError("invalid dimension or train fraction")
    row_bytes = 4 + 4 * dim
    file_bytes = Path(args.input).stat().st_size
    if file_bytes % row_bytes:
        raise ValueError("invalid fvecs file size")
    count = file_bytes // row_bytes
    dtype = np.dtype([("dim", "<i4"), ("vector", "<f4", (dim,))])
    rows = np.memmap(args.input, dtype=dtype, mode="r", shape=(count,))
    if np.any(rows["dim"] != dim):
        raise ValueError("inconsistent fvecs dimensions")
    vectors = np.ascontiguousarray(rows["vector"], dtype=np.float32)
    leaves = math.floor(math.sqrt(count)) if count >= 100000 else 0
    train_count = min(count, max(40 * leaves,
        min(args.train_cap, math.ceil(args.train_fraction * count))))
    builder = scann.scann_ops_pybind.builder(vectors, 1, "squared_l2")
    if leaves:
        builder = builder.tree(
            num_leaves=leaves, num_leaves_to_search=min(10, leaves),
            training_sample_size=train_count)
    builder = builder.score_ah(2, hash_type="lut16",
                               training_sample_size=train_count).reorder(100)
    builder.set_n_training_threads(args.threads)
    start = time.perf_counter()
    searcher = builder.build()
    build_seconds = time.perf_counter() - start
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    searcher.serialize(str(output))
    serialize_seconds = time.perf_counter() - start
    index_bytes = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    config = {
        "distance": "squared_l2", "nodes": count, "dim": dim,
        "num_leaves": leaves, "training_sample_size": train_count,
        "dimensions_per_block": 2, "hash_type": "lut16",
        "reordering_num_neighbors": 100, "reordering_precision": "float32",
        "threads": args.threads,
    }
    (output.parent / "scann_config.json").write_text(json.dumps(config, indent=2) + "\n")
    with open(args.metrics, "w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["dataset", "algorithm", "nodes", "dim", "nlist",
                         "train_vectors", "pq_m", "pq_bits", "by_residual", "bbs",
                         "sample_seconds", "train_seconds", "add_seconds", "serialize_seconds",
                         "build_seconds", "index_file_bytes", "index_path"])
        writer.writerow([args.dataset, "scann", count, dim, leaves, train_count,
                         0, 0, 0, 0, "", "", "", serialize_seconds, build_seconds,
                         index_bytes, str(output)])
    print(f"scann {args.dataset} nodes={count} dim={dim} leaves={leaves} "
          f"bytes={index_bytes} build_seconds={build_seconds:.3f}")


if __name__ == "__main__":
    main()
