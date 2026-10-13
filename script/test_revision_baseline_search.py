#!/usr/bin/env python3
"""Run recall/latency sweeps for the four revision baselines."""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


ALGORITHMS = ("ivf_flat", "ivfpq_fastscan", "nsg", "scann")
FIELDS = ("Dataset", "IndexType", "Param", "K", "Recall", "QPS",
          "Latency(us)", "IndexSizeKB", "VectorsPerKB", "VQ")
DEFAULT_PARAMS = (1, 2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100)
SMALL_SCANN_PARAMS = (10, 20, 30, 50, 80, 100, 150, 200)


def collect(output_dir):
    rows = []
    for path in sorted(output_dir.glob("*/*/results.csv")):
        with path.open(newline="") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError(f"unexpected result schema: {path}")
            rows.extend(reader)
    destination = output_dir / "all_results.csv"
    with destination.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path,
                        default=repo / "script/revision_baseline_build_results/summary.csv")
    parser.add_argument("--data-dir", type=Path, default=repo / "datasets/hdf5files")
    parser.add_argument("--output-dir", type=Path,
                        default=repo / "script/revision_baseline_search_results")
    parser.add_argument("--build-dir", type=Path, default=repo / "build_lvc_ivf_pq")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS,
                        default=list(ALGORITHMS))
    parser.add_argument("--k", type=int, nargs="+", default=[1, 10])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--ivf-params", type=int, nargs="+", default=list(DEFAULT_PARAMS))
    parser.add_argument("--nsg-params", type=int, nargs="+", default=list(DEFAULT_PARAMS))
    parser.add_argument("--scann-params", type=int, nargs="+")
    parser.add_argument("--scann-candidates", type=int, default=100)
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1 or any(k < 1 for k in args.k):
        raise ValueError("rounds and K must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.summary.open(newline="") as source:
        indexes = [item for item in csv.DictReader(source)
                   if item["algorithm"] in args.algorithms and
                   (args.datasets is None or item["dataset"] in args.datasets)]
    if not indexes:
        raise ValueError("no matching indexes in build summary")

    if not args.skip_compile and any(i["algorithm"] != "scann" for i in indexes):
        for command, logname in [
            (["cmake", "-S", str(repo), "-B", str(args.build_dir),
              "-DHNSWLIB_EXAMPLES=ON"], "cmake_configure.log"),
            (["cmake", "--build", str(args.build_dir), "--target",
              "test_revision_faiss_search", "-j", "4"], "cmake_build.log")]:
            with (args.output_dir / logname).open("w") as log:
                subprocess.run(command, cwd=repo, stdout=log,
                               stderr=subprocess.STDOUT, check=True)

    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1")
    for item in indexes:
        dataset, algorithm = item["dataset"], item["algorithm"]
        directory = args.output_dir / dataset / algorithm
        directory.mkdir(parents=True, exist_ok=True)
        final = directory / "results.csv"
        temporary = directory / "results.tmp.csv"
        index_path = Path(item["index_path"])
        if not index_path.exists():
            raise FileNotFoundError(index_path)
        index_mtime_ns = (max(file.stat().st_mtime_ns for file in index_path.rglob("*")
                              if file.is_file()) if index_path.is_dir()
                          else index_path.stat().st_mtime_ns)
        query_path = args.data_dir / f"{dataset}_test.fvecs"
        truth_path = args.data_dir / f"{dataset}_neighbors.ivecs"
        if not query_path.exists() or not truth_path.exists():
            raise FileNotFoundError(f"queries or ground truth missing: {dataset}")
        if algorithm == "scann":
            params = args.scann_params or (
                DEFAULT_PARAMS if int(item["nlist"]) else SMALL_SCANN_PARAMS)
        elif algorithm == "nsg":
            params = args.nsg_params
        else:
            params = args.ivf_params
        config = {"index_path": str(index_path),
                  "index_bytes": item["index_file_bytes"],
                  "index_mtime_ns": index_mtime_ns,
                  "queries": [str(query_path), query_path.stat().st_size,
                              query_path.stat().st_mtime_ns],
                  "ground_truth": [str(truth_path), truth_path.stat().st_size,
                                   truth_path.stat().st_mtime_ns],
                  "rounds": args.rounds, "k": args.k, "params": list(params),
                  "scann_candidates": args.scann_candidates if algorithm == "scann" else None}
        if algorithm == "ivfpq_fastscan" and int(item["by_residual"]):
            query_count = query_path.stat().st_size // (4 + 4 * int(item["dim"]))
            max_probe = max((p for p in params if p <= int(item["nlist"])), default=0)
            lut_bytes = (query_count * max_probe * int(item["pq_m"]) *
                         (1 << int(item["pq_bits"])) * 5)
            if lut_bytes > (1 << 31):
                config["fastscan_large_lut_batch"] = 64
        config_path = directory / "config.json"
        if final.exists() and config_path.exists() and not args.force:
            saved_config = json.loads(config_path.read_text())
            if algorithm != "ivfpq_fastscan":
                saved_config.pop("fastscan_query_batch", None)
            if saved_config == config:
                print(f"Reuse {dataset} {algorithm}: {final}", flush=True)
                continue
        common = ["--dataset", dataset, "--index", str(index_path),
                  "--queries", str(query_path), "--ground-truth", str(truth_path),
                  "--output", str(temporary), "--rounds", str(args.rounds),
                  "--k", *map(str, args.k)]
        if algorithm == "scann":
            nlist = int(item["nlist"])
            command = [sys.executable, str(repo / "script/test_revision_scann_search.py"),
                       *common, "--nodes", item["nodes"], "--nlist", str(nlist),
                       "--candidates", str(args.scann_candidates),
                       "--params", *map(str, params)]
        else:
            command = [str(args.build_dir / "test_revision_faiss_search"),
                       *common, "--algorithm", algorithm,
                       "--params", *map(str, params)]
        print(f"Running {dataset} {algorithm}", flush=True)
        log_path = directory / "search.log"
        try:
            with log_path.open("w") as log:
                subprocess.run(command, cwd=repo, env=environment, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"{dataset} {algorithm} failed; see {log_path}") from error
        temporary.replace(final)
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        collect(args.output_dir)
        print(f"PASS {dataset} {algorithm}: {final}", flush=True)
    print("Results:", collect(args.output_dir))


if __name__ == "__main__":
    main()
