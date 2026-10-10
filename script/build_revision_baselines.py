#!/usr/bin/env python3
"""Build the IVF-Flat, NSG, IVFPQ-FastScan, and ScaNN revision baselines."""

import argparse
import csv
import struct
import subprocess
import sys
from pathlib import Path


DATASETS = (
    "sift-128-euclidean", "mnist-784-euclidean",
    "fashion-mnist-784-euclidean", "gist-960-euclidean",
    "deep-image-96-angular", "bigann_1M", "bigann_10M",
)
ALGORITHMS = ("ivf_flat", "nsg", "ivfpq_fastscan", "scann")
FIELDS = ("dataset", "algorithm", "nodes", "dim", "nlist", "train_vectors",
          "pq_m", "pq_bits", "by_residual", "bbs", "sample_seconds", "train_seconds",
          "add_seconds", "serialize_seconds", "build_seconds", "index_file_bytes",
          "index_path", "source")


def shape(path):
    with path.open("rb") as source:
        dim = struct.unpack("<i", source.read(4))[0]
    if dim <= 0 or path.stat().st_size % (4 + 4 * dim):
        raise ValueError(f"invalid fvecs file: {path}")
    return path.stat().st_size // (4 + 4 * dim), dim


def run(command, cwd, log):
    print("Running:", " ".join(map(str, command)), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as target:
        subprocess.run(list(map(str, command)), cwd=cwd, stdout=target,
                       stderr=subprocess.STDOUT, check=True)


def read_one(path):
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 1:
        raise ValueError(f"expected one metrics row: {path}")
    return rows[0]


def write_summary(path, results):
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        writer.writeheader()
        for key in sorted(results):
            writer.writerow({field: results[key].get(field, "") for field in FIELDS})


def nsg_build_seconds(dataset, build_csv):
    with build_csv.open(newline="") as input_file:
        matches = [row for row in csv.DictReader(input_file)
                   if row["Dataset"] == dataset]
    if not matches:
        raise ValueError(f"NSG build time missing from {build_csv}: {dataset}")
    return matches[-1]["BuildTime(s)"]


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS[:5]))
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS,
                        default=list(ALGORITHMS))
    parser.add_argument("--data-dir", type=Path, default=repo / "datasets/hdf5files")
    parser.add_argument("--bigann-base-file", type=Path,
                        default=repo / "bigann/bigann_base.bvecs")
    parser.add_argument("--output-dir", type=Path,
                        default=repo / "script/revision_baseline_build_results")
    parser.add_argument("--build-dir", type=Path, default=repo / "build_lvc_ivf_pq")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    args = parser.parse_args()
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.build_dir = args.build_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_compile:
        run(["cmake", "-S", repo, "-B", args.build_dir,
             "-DHNSWLIB_EXAMPLES=ON"], repo, args.output_dir / "cmake_configure.log")
        targets = ["build_revision_faiss_baseline"]
        if "nsg" in args.algorithms:
            targets.append("test_build_faiss_nsg")
        if "ivf_flat" in args.algorithms and any(d.startswith("bigann_") for d in args.datasets):
            targets.append("test_build_faiss_ivf")
        run(["cmake", "--build", args.build_dir, "--target", *targets,
             "-j", args.threads], repo, args.output_dir / "cmake_build.log")

    summary_path = args.output_dir / "summary.csv"
    results = {}
    if summary_path.exists():
        with summary_path.open(newline="") as source:
            for row in csv.DictReader(source):
                results[(row["dataset"], row["algorithm"])] = row
    failures = []
    for dataset in args.datasets:
        is_bigann = dataset.startswith("bigann_")
        if is_bigann:
            n = int(dataset.split("_")[1][:-1]) * 1000000
            with args.bigann_base_file.open("rb") as source:
                dim = struct.unpack("<i", source.read(4))[0]
            if dim <= 0 or args.bigann_base_file.stat().st_size < n * (4 + dim):
                raise ValueError("BIGANN bvecs file is too small")
            input_path = args.bigann_base_file.resolve()
        else:
            input_path = args.data_dir / f"{dataset}_train.fvecs"
            n, dim = shape(input_path)
        for algorithm in args.algorithms:
            if is_bigann and algorithm != "ivf_flat":
                continue
            directory = args.output_dir / dataset / algorithm
            directory.mkdir(parents=True, exist_ok=True)
            metrics = directory / "metrics.csv"
            log = directory / "build.log"
            results.pop((dataset, algorithm), None)
            write_summary(summary_path, results)
            try:
                if algorithm == "nsg":
                    old_index = repo / "script" / f"{dataset}_NSG_R32.bin"
                    old_csv = repo / "script/faiss_nsg_build_results.csv"
                    historical_time = None
                    if old_index.exists() and old_csv.exists() and not args.force:
                        try:
                            historical_time = nsg_build_seconds(dataset, old_csv)
                        except ValueError:
                            pass
                    if historical_time is not None:
                        index_path, build_csv, source = old_index, old_csv, "existing"
                        build_seconds = historical_time
                    else:
                        index_path = directory / f"{dataset}_NSG_R32.bin"
                        build_csv = directory / "faiss_nsg_build_results.csv"
                        if args.force or not index_path.exists() or not build_csv.exists():
                            run([args.build_dir / "test_build_faiss_nsg", "--dataset", dataset,
                                 "--R", 32, "--normal_base_dir", args.data_dir], directory, log)
                        source = "built"
                        build_seconds = nsg_build_seconds(dataset, build_csv)
                    row = {"dataset": dataset, "algorithm": "nsg", "nodes": n,
                           "dim": dim, "nlist": 0, "train_vectors": 0,
                           "pq_m": 0, "pq_bits": 0, "by_residual": 0, "bbs": 0,
                           "sample_seconds": "", "train_seconds": "", "add_seconds": "",
                           "serialize_seconds": "", "build_seconds": build_seconds,
                           "index_file_bytes": index_path.stat().st_size,
                           "index_path": str(index_path), "source": source}
                    with metrics.open("w", newline="") as target:
                        writer = csv.DictWriter(target, fieldnames=FIELDS)
                        writer.writeheader(); writer.writerow(row)
                elif is_bigann:
                    nlist = int(n ** 0.5)
                    train_count = min(n, max(40 * nlist, int(0.15 * n + 0.999999)))
                    index_path = directory / f"{dataset}_IVFFlat_nlist{nlist}.bin"
                    legacy_csv = directory / "faiss_ivf_build_results.csv"
                    if args.force or not index_path.exists() or not legacy_csv.exists():
                        run([args.build_dir / "test_build_faiss_ivf", "--dataset", dataset,
                             "--algorithm", "IVFFlat", "--nlist", nlist,
                             "--train_size", train_count, "--bigann_base_file", input_path],
                            directory, log)
                    with legacy_csv.open(newline="") as source:
                        records = [record for record in csv.DictReader(source)
                                   if record["Dataset"] == dataset]
                    if len(records) != 1:
                        raise ValueError("missing BIGANN IVF-Flat build metrics")
                    old = records[0]
                    row = {"dataset": dataset, "algorithm": "ivf_flat", "nodes": n,
                           "dim": dim, "nlist": nlist, "train_vectors": old["TrainSize"],
                           "pq_m": 0, "pq_bits": 0, "by_residual": 0, "bbs": 0,
                           "sample_seconds": "", "train_seconds": old["TrainTime(s)"],
                           "add_seconds": old["AddTime(s)"], "serialize_seconds": "",
                           "build_seconds": old["TotalTime(s)"],
                           "index_file_bytes": index_path.stat().st_size,
                           "index_path": str(index_path),
                           "source": "built_bigann_prefix_sample"}
                    with metrics.open("w", newline="") as target:
                        writer = csv.DictWriter(target, fieldnames=FIELDS)
                        writer.writeheader(); writer.writerow(row)
                else:
                    index_path = directory / ("index" if algorithm == "scann" else "index.bin")
                    if args.force or not metrics.exists() or not index_path.exists():
                        if algorithm == "scann":
                            command = [sys.executable, repo / "script/build_revision_scann_baseline.py",
                                       "--input", input_path, "--output", index_path,
                                       "--metrics", metrics, "--dataset", dataset,
                                       "--threads", args.threads]
                        else:
                            command = [args.build_dir / "build_revision_faiss_baseline",
                                       "--input", input_path, "--output", index_path,
                                       "--metrics", metrics, "--dataset", dataset,
                                       "--algorithm", algorithm, "--threads", args.threads]
                        run(command, repo, log)
                    row = read_one(metrics)
                    row["source"] = "built"
                results[(dataset, algorithm)] = row
                write_summary(summary_path, results)
                print(f"PASS {dataset} {algorithm}: {row['index_file_bytes']} bytes", flush=True)
            except (OSError, ValueError, subprocess.CalledProcessError) as error:
                failures.append((dataset, algorithm, str(error)))
                print(f"FAIL {dataset} {algorithm}: {error}; see {log}", file=sys.stderr)
    print("Summary:", summary_path)
    if failures:
        raise SystemExit(f"{len(failures)} baseline builds failed")


if __name__ == "__main__":
    main()
