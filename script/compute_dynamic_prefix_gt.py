#!/usr/bin/env python3
"""Exact top-1 L2 ground truth for dataset prefixes, using original row IDs.

Requires NumPy. Full datasets and all queries are used unless a limit is given.
F32 inputs are promoted to F64 for blocked exhaustive distance evaluation.
"""

import argparse
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["sift-128-euclidean", "mnist-784-euclidean",
            "fashion-mnist-784-euclidean", "gist-960-euclidean", "deep-image-96-angular"]


def open_fvecs(path):
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype="<i4", count=1)
    if len(header) != 1 or header[0] <= 0:
        raise ValueError(f"Invalid fvecs header: {path}")
    dim = int(header[0])
    if path.stat().st_size % (4 * (dim + 1)):
        raise ValueError(f"Truncated fvecs: {path}")
    words = np.memmap(path, dtype="<f4", mode="r").reshape(-1, dim + 1)
    if not len(words):
        raise ValueError(f"Empty fvecs: {path}")
    return words, dim


def fvecs_block(words, start, end, dim):
    rows = words[start:end]
    if not np.all(rows[:, 0].copy().view("<i4") == dim):
        raise ValueError(f"Inconsistent fvecs dimensions at rows {start}:{end}")
    values = np.asarray(rows[:, 1:], dtype=np.float64, order="C")
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite vector at rows {start}:{end}")
    return values


def signature(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def write_checkpoint(directory, percent, ids, distances):
    stem = directory / f"gt_prefix{percent:03d}"
    ivecs = np.empty((len(ids), 2), dtype="<i4")
    ivecs[:, 0], ivecs[:, 1] = 1, ids
    fvecs = np.empty((len(ids), 2), dtype="<f4")
    fvecs[:, 0].view("<i4")[:] = 1
    fvecs[:, 1] = distances
    for suffix, values in [(".ivecs", ivecs), (".fvecs", fvecs)]:
        temporary = Path(str(stem) + suffix + ".tmp")
        values.tofile(temporary)
        temporary.replace(str(stem) + suffix)
    temporary = Path(str(stem) + "_squared_l2.npy.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, distances)
    temporary.replace(str(stem) + "_squared_l2.npy")


def generate(args, dataset):
    train_path = args.data_dir / f"{dataset}_train.fvecs"
    query_path = args.data_dir / f"{dataset}_test.fvecs"
    train_words, dim = open_fvecs(train_path)
    query_words, query_dim = open_fvecs(query_path)
    if dim != query_dim:
        raise ValueError(f"Train/query dimension mismatch: {dataset}")
    count = min(len(train_words), args.train_limit) if args.train_limit else len(train_words)
    qcount = min(len(query_words), args.query_limit) if args.query_limit else len(query_words)
    if count < 2 or count > np.iinfo(np.int32).max:
        raise ValueError("Need 2..INT32_MAX vectors for ivecs row IDs")
    directory = args.out_dir / dataset
    directory.mkdir(parents=True, exist_ok=True)
    config = {"schema_version": 1, "dataset": dataset, "train": signature(train_path),
              "queries": signature(query_path), "vectors": count, "query_count": qcount,
              "dimension": dim, "prefix_percentages": args.percentages,
              "id_space": "zero-based original train row ID", "k": 1,
              "distance": "squared L2; F64 exhaustive norm/dot evaluation",
              "ties": "smallest row ID for exactly equal computed distances",
              "train_block": args.train_block, "query_block": args.query_block,
              "threads": args.threads}
    manifest_path = directory / "manifest.json"
    if args.resume and manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        complete = all((directory / f"gt_prefix{p:03d}{suffix}").is_file()
                       for p in args.percentages
                       for suffix in (".ivecs", ".fvecs", "_squared_l2.npy"))
        if old.get("config") == config and complete:
            for p in args.percentages:
                for suffix in (".ivecs", ".fvecs"):
                    if (directory / f"gt_prefix{p:03d}{suffix}").stat().st_size != qcount * 8:
                        raise ValueError(f"Truncated existing GT for {dataset}/{p}")
                distances = np.load(directory / f"gt_prefix{p:03d}_squared_l2.npy", mmap_mode="r")
                if distances.shape != (qcount,) or distances.dtype != np.dtype("float64"):
                    raise ValueError(f"Invalid existing distances for {dataset}/{p}")
            print(f"Reuse {dataset}: {manifest_path}", flush=True)
            return
    # Never leave a stale completion marker while overwriting a different run.
    manifest_path.unlink(missing_ok=True)
    queries = fvecs_block(query_words, 0, qcount, dim)
    qnorms = np.einsum("ij,ij->i", queries, queries)
    best_dist = np.full(qcount, np.inf, dtype=np.float64)
    best_ids = np.full(qcount, -1, dtype=np.int64)
    checkpoints = []
    started = time.perf_counter()
    processed = 0
    for percent in args.percentages:
        target = count * percent // 100
        if target == 0:
            raise ValueError("A prefix contains no vectors")
        while processed < target:
            end = min(target, processed + args.train_block)
            vectors = fvecs_block(train_words, processed, end, dim)
            norms = np.einsum("ij,ij->i", vectors, vectors)
            for qstart in range(0, qcount, args.query_block):
                qend = min(qcount, qstart + args.query_block)
                distances = queries[qstart:qend] @ vectors.T
                distances *= -2.0
                distances += qnorms[qstart:qend, None]
                distances += norms[None, :]
                np.maximum(distances, 0.0, out=distances)
                if not np.isfinite(distances).all():
                    raise ValueError("Non-finite squared distances")
                local_ids = distances.argmin(axis=1)
                local_dist = distances[np.arange(qend - qstart), local_ids]
                improved = local_dist < best_dist[qstart:qend]
                best_dist[qstart:qend][improved] = local_dist[improved]
                best_ids[qstart:qend][improved] = processed + local_ids[improved]
            processed = end
        write_checkpoint(directory, percent, best_ids, best_dist)
        elapsed = time.perf_counter() - started
        checkpoints.append({"percent": percent, "vectors": target, "elapsed_seconds": elapsed,
                            "ids": f"gt_prefix{percent:03d}.ivecs",
                            "squared_l2": f"gt_prefix{percent:03d}_squared_l2.npy"})
        print(f"GT {dataset} prefix={percent}% vectors={target} queries={qcount} "
              f"seconds={elapsed:.3f}", flush=True)
    manifest = {"config": config, "checkpoints": checkpoints,
                "numpy_version": np.__version__, "seconds": time.perf_counter() - started,
                "train_block": args.train_block, "query_block": args.query_block,
                "threads": args.threads}
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(manifest_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets/hdf5files")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "script/dynamic_prefix_gt")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--percentages", nargs="+", type=int, default=list(range(50, 101, 5)))
    parser.add_argument("--train-block", type=int, default=8192)
    parser.add_argument("--query-block", type=int, default=256)
    parser.add_argument("--train-limit", type=int, default=0, help="Smoke tests only; default uses all rows")
    parser.add_argument("--query-limit", type=int, default=0, help="Default uses all queries")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (args.train_block <= 0 or args.query_block <= 0 or args.threads <= 0 or
            args.train_limit < 0 or args.query_limit < 0 or
            not args.percentages or args.percentages != sorted(set(args.percentages)) or
            min(args.percentages) < 1 or max(args.percentages) > 100):
        parser.error("Invalid block size, threads, limits or prefix percentages")
    # Configure BLAS before importing NumPy; no threadpoolctl dependency is needed.
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                 "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = str(args.threads)
    global np
    import numpy as np
    for dataset in args.datasets:
        generate(args, dataset)


if __name__ == "__main__":
    main()
