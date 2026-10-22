#!/usr/bin/env python3
"""Fixed random SIFT subset and exact subset GT; never reuse full-SIFT GT."""
import argparse
import json
from pathlib import Path

import numpy as np


def read_fvecs(path):
    words = np.memmap(path, dtype="<i4", mode="r")
    dim = int(words[0])
    if dim <= 0 or words.size % (dim + 1):
        raise ValueError(f"Invalid fvecs: {path}")
    rows = words.reshape(-1, dim + 1)
    if not np.all(rows[:, 0] == dim):
        raise ValueError(f"Inconsistent dimensions: {path}")
    return rows[:, 1:].view("<f4")


def write_fvecs(path, values):
    rows = np.empty((len(values), values.shape[1] + 1), dtype="<f4")
    rows[:, 0].view("<i4")[:] = values.shape[1]
    rows[:, 1:] = values
    rows.tofile(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100000)
    parser.add_argument("--query-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()
    if args.count < 256 or args.query_count < 0:
        parser.error("count must be >=256; query-count must be >=0")
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = {}
    for name, path in (("train", args.train), ("queries", args.queries)):
        stat = path.stat()
        inputs[name] = [str(path.resolve()), stat.st_size, stat.st_mtime_ns]
    config = dict(inputs=inputs, count=args.count, query_count=args.query_count, seed=args.seed,
                  gt="float64 exhaustive squared L2; first subset ID breaks ties")
    marker = args.output / "subset.json"
    files = [args.output / name for name in ("train.fvecs", "test.fvecs", "neighbors.ivecs", "source_ids.npy")]
    if marker.exists() and json.loads(marker.read_text()) == config and all(p.is_file() for p in files):
        print(f"Reuse subset + exact GT: {args.output}", flush=True)
        return
    marker.unlink(missing_ok=True)
    base, queries = read_fvecs(args.train), read_fvecs(args.queries)
    if args.count > len(base) or args.query_count > len(queries) or base.shape[1] != queries.shape[1]:
        raise ValueError("Requested subset/query count exceeds input, or dimensions differ")
    ids = np.random.default_rng(args.seed).choice(len(base), args.count, replace=False)
    # Sorting gives deterministic source order without changing the sampled set.
    ids.sort()
    x = np.asarray(base[ids], dtype="<f4")
    q = np.asarray(queries[:args.query_count or len(queries)], dtype="<f4")
    if not np.isfinite(x).all() or not np.isfinite(q).all():
        raise ValueError("Non-finite input")
    np.save(files[3], ids)
    write_fvecs(files[0], x)
    write_fvecs(files[1], q)
    x64 = x.astype(np.float64)
    norms = np.einsum("ij,ij->i", x64, x64)
    truth = np.empty((len(q), 2), dtype="<i4")
    truth[:, 0] = 1
    for start in range(0, len(q), 64):
        batch = q[start:start + 64].astype(np.float64)
        distances = norms[None, :] + np.einsum("ij,ij->i", batch, batch)[:, None]
        distances -= 2 * (batch @ x64.T)
        truth[start:start + len(batch), 1] = distances.argmin(axis=1)
        print(f"Exact subset GT: {start + len(batch)}/{len(q)}", flush=True)
    truth.tofile(files[2])
    marker.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Prepared {len(x)} vectors, {len(q)} queries, dim={x.shape[1]}", flush=True)


if __name__ == "__main__":
    main()
