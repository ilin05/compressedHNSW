#!/usr/bin/env python3
"""Run repeated/growing workloads one dataset at a time; publish CSVs atomically."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["sift-128-euclidean", "mnist-784-euclidean", "fashion-mnist-784-euclidean",
            "gist-960-euclidean", "deep-image-96-angular"]
MODES = ["raw_f32", "alp_a01", "alp_a02", "alp_off"]
UPDATE_FIELDS = ("dataset,experiment,insert_mode,batch,prefix_percent,active_nodes,physical_nodes,deleted_nodes,"
    "capacity_nodes,inserted_nodes,marked_nodes,index_file_bytes,record_buffer_capacity_bytes,"
    "insert_seconds,delete_seconds,resize_seconds,insert_mean_us,delete_mean_us,graph_mean_us,"
    "encoding_mean_us,filter_searches,pq_candidates,exact_candidates,fallback_searches,payload_decodes,"
    "decode_cache_hits,decode_cache_peak_bytes,decode_cache_peak_nodes,upper_searches,upper_pq_candidates,upper_exact_candidates,"
    "verified_vectors,max_abs_error,update_hash,current_process_rss_kib,peak_process_rss_kib,"
    "initial_build_seconds,initial_compression_seconds,threads").split(",")
SEARCH_FIELDS = ("dataset,experiment,insert_mode,batch,prefix_percent,query_mode,query_alpha,ef_search,"
    "search_round,queries,recall_at_1,recall_at_1_ties,qps,mean_query_us,threads").split(",")
SUMMARY_FIELDS = UPDATE_FIELDS + ["index_type", "query_mode", "query_alpha", "ef_search", "search_rounds",
    "queries", "recall_at_1", "recall_at_1_ties", "qps", "mean_query_us"]


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def atomic_csv(path, fields, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def file_info(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def shape(path):
    with path.open("rb") as handle:
        data = handle.read(4)
    if len(data) != 4:
        raise ValueError(f"Invalid fvecs: {path}")
    dim, = struct.unpack("<i", data)
    if dim <= 0 or path.stat().st_size % (4 * (dim + 1)):
        raise ValueError(f"Invalid fvecs shape: {path}")
    return path.stat().st_size // (4 * (dim + 1)), dim


def preflight(args, dataset):
    train = args.data_dir / f"{dataset}_train.fvecs"
    query = args.data_dir / f"{dataset}_test.fvecs"
    total, dim = shape(train)
    query_total, query_dim = shape(query)
    n, q = args.count or total, args.query_count or query_total
    if n > total or q > query_total or dim != query_dim or n < 512 or q < 1:
        raise ValueError(f"Invalid full/subset shapes for {dataset}")
    gt_dir = args.gt_root / dataset
    gt = args.data_dir / f"{dataset}_neighbors.ivecs"
    steps = args.batches if args.experiment == "repeated" else 10
    checkpoints = sorted(set(range(0, steps + 1, args.search_every)) | {steps})
    config = {"schema": 2, "dataset": dataset, "experiment": args.experiment,
              "train": file_info(train), "queries": file_info(query), "nodes": n,
              "query_count": q, "dim": dim, "batches": args.batches if args.experiment == "repeated" else 10,
              "step_percent": args.repeat_ratio * 100 if args.experiment == "repeated" else 5,
              "repeat_ratio": args.repeat_ratio, "search_checkpoints": checkpoints,
              "search_every": args.search_every,
              "insert_decode_cache": bool(args.insert_decode_cache), "insert_upper_tls": bool(args.insert_upper_tls),
              "initial_percent": 100 if args.experiment == "repeated" else 50,
              "search_rounds": args.search_rounds, "ef_search": args.ef, "M": args.M,
              "ef_construction": args.ef_construction, "initial_seed": 100, "update_seed": args.seed,
              "insert_modes": args.modes, "alp_query_modes": ["tls_off", "tls_0.2"],
              "hub_ratio_initial": 0.01, "alp_tolerance": 1e-4,
              "capacity_policy": "initial count, then expand to required count at each batch",
              "threads": 1, "verify_size": args.verify_size}
    if args.experiment == "growing":
        manifest_path = gt_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        source = manifest["config"]
        if source["vectors"] != n or source["dimension"] != dim or source["query_count"] < q:
            raise ValueError(f"Prefix GT shape mismatch: {manifest_path}")
        for key, path in [("train", train), ("queries", query)]:
            info = file_info(path)
            if any(source[key][name] != info[name] for name in ("bytes", "mtime_ns")):
                raise ValueError(f"GT source file changed: {path}; regenerate prefix GT")
        prefixes = []
        for percent in range(50, 101, 5):
            path = gt_dir / f"gt_prefix{percent:03d}.ivecs"
            if path.stat().st_size != source["query_count"] * 8:
                raise ValueError(f"Truncated K=1 prefix GT: {path}")
            prefixes.append(file_info(path))
        config["ground_truth"] = prefixes
    else:
        with gt.open("rb") as handle:
            width, = struct.unpack("<i", handle.read(4))
        if width < 1 or gt.stat().st_size % (4 * (width + 1)) or gt.stat().st_size // (4 * (width + 1)) < q:
            raise ValueError(f"Wrong full-dataset GT: {gt}")
        if n != total:
            raise ValueError("Repeated subsets require matching input/GT files in a separate DATA_DIR")
        config["ground_truth"] = file_info(gt)
    return config, train, query, gt, gt_dir


def verify_case(directory, config, mode):
    update = rows(directory / "updates.csv")
    search = rows(directory / "search.csv")
    checkpoints = config["batches"] + 1
    query_modes = ["tls_off"] if mode == "raw_f32" else ["tls_off", "tls_0.2"]
    expected = {(b, tls, ef, r) for b in config["search_checkpoints"] for tls in query_modes
                for ef in config["ef_search"] for r in range(config["search_rounds"])}
    actual = {(int(r["batch"]), r["query_mode"], int(r["ef_search"]), int(r["search_round"])) for r in search}
    if actual != expected or len(actual) != len(search) or len(update) != checkpoints:
        raise ValueError(f"Incomplete/duplicate checkpoints: {directory}")
    for batch, row in enumerate(update):
        prefix = 100 if config["experiment"] == "repeated" else 50 + batch * 5
        active = config["nodes"] * prefix // 100
        deleted = batch * max(1, int(config["nodes"] * config["repeat_ratio"])) if config["experiment"] == "repeated" else 0
        if (int(row["batch"]) != batch or int(row["active_nodes"]) != active or
                int(row["deleted_nodes"]) != deleted or int(row["physical_nodes"]) != active + deleted or
                int(row["threads"]) != 1):
            raise ValueError(f"Invalid record counts: {directory}/{batch}")
        if mode in ["raw_f32", "alp_off"] and int(row["filter_searches"]) != 0:
            raise ValueError("Disabled insertion filtering was unexpectedly exercised")
        if batch and mode in ["alp_a01", "alp_a02"] and int(row["filter_searches"]) == 0:
            raise ValueError("Insertion filtering was never exercised")
        if mode != "raw_f32" and batch and config["insert_decode_cache"] and int(row["decode_cache_hits"]) == 0:
            raise ValueError("Insertion decode cache was never exercised")
        if mode != "raw_f32" and not config["insert_upper_tls"] and int(row["upper_pq_candidates"]) != 0:
            raise ValueError("Disabled upper TLS was exercised")
    for row in search:
        if row["dataset"] != config["dataset"] or row["insert_mode"] != mode or int(row["threads"]) != 1:
            raise ValueError("Search identity/thread mismatch")
    return update, search


def summarize(update, search):
    lookup = {(r["insert_mode"], r["batch"]): r for r in update}
    groups = {}
    for row in search:
        key = row["insert_mode"], row["batch"], row["query_mode"], row["ef_search"]
        groups.setdefault(key, []).append(row)
    summary = []
    for key, samples in groups.items():
        result = dict(lookup[key[:2]])
        result.update(index_type="HNSW" if key[0] == "raw_f32" else "HNSW-ALP",
                      query_mode=key[2], ef_search=key[3], search_rounds=len(samples),
                      query_alpha=samples[0]["query_alpha"], queries=samples[0]["queries"])
        for field in ["recall_at_1", "recall_at_1_ties", "qps", "mean_query_us"]:
            result[field] = statistics.mean(float(s[field]) for s in samples)
        summary.append(result)
    return summary


def publish_dataset(directory, config, partial=False):
    update, search = [], []
    reference = None
    completed = []
    for mode in config["insert_modes"]:
        if partial and not (directory / mode / "complete.json").exists():
            continue
        u, s = verify_case(directory / mode, config, mode)
        signature = [(r["batch"], r["update_hash"], r["active_nodes"], r["physical_nodes"]) for r in u]
        if reference is not None and signature != reference:
            raise ValueError("Insertion policies did not receive the same update schedule")
        reference = signature
        update.extend(u)
        search.extend(s)
        completed.append(mode)
    atomic_csv(directory / "updates.csv", UPDATE_FIELDS, update)
    atomic_csv(directory / "search.csv", SEARCH_FIELDS, search)
    atomic_csv(directory / "summary.csv", SUMMARY_FIELDS, summarize(update, search))
    if completed == config["insert_modes"]:
        atomic_json(directory / "complete.json", {"config": config, "status": "PASS"})
        (directory / "progress.json").unlink(missing_ok=True)
    else:
        atomic_json(directory / "progress.json", {"status": "partial", "completed_modes": completed})


def publish_all(out):
    datasets, updates, search, summaries = [], [], [], []
    for directory in sorted(out.iterdir()):
        if not directory.is_dir() or not any((directory / name).is_file() for name in ("complete.json", "progress.json")):
            continue
        updates.extend(rows(directory / "updates.csv"))
        search.extend(rows(directory / "search.csv"))
        summaries.extend(rows(directory / "summary.csv"))
        datasets.append(directory.name)
    atomic_csv(out / "all_updates.csv", UPDATE_FIELDS, updates)
    atomic_csv(out / "all_search.csv", SEARCH_FIELDS, search)
    atomic_csv(out / "all_metrics.csv", SUMMARY_FIELDS, summaries)
    print(f"Published completed datasets: {datasets}; {out / 'all_metrics.csv'}", flush=True)


def execute(command, log, environment):
    print(f"Running: {' '.join(command)}", flush=True)
    with log.open("w") as handle:
        try:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Benchmark failed; see {log}") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["repeated", "growing"], required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets/hdf5files")
    parser.add_argument("--gt-root", type=Path, default=ROOT / "script/dynamic_prefix_gt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/dynamic_update_workloads/test_dynamic_update_workloads")
    parser.add_argument("--ef", nargs="+", type=int, default=[20, 40, 80])
    parser.add_argument("--search-rounds", type=int, default=3)
    parser.add_argument("--batches", type=int, default=50, help="Repeated workload only; growing always has 10 batches")
    parser.add_argument("--repeat-ratio", type=float, default=.05)
    parser.add_argument("--search-every", type=int, default=1)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--insert-decode-cache", type=int, choices=[0, 1], default=0)
    parser.add_argument("--insert-upper-tls", type=int, choices=[0, 1], default=0)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--count", type=int, default=0, help="Smoke tests only; default uses full data")
    parser.add_argument("--query-count", type=int, default=0)
    parser.add_argument("--resume", type=int, choices=[0, 1], default=1)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--keep-initial-index", action="store_true")
    parser.add_argument("--verify-size", action="store_true", help="Save initial/final indexes to verify size counters")
    args = parser.parse_args()
    if (args.batches < 1 or args.search_rounds < 1 or args.search_every < 1 or
            not 0 < args.repeat_ratio < 1 or len(set(args.modes)) != len(args.modes) or
            args.count < 0 or args.query_count < 0 or
            args.M < 2 or args.ef_construction < args.M or min(args.ef) < 1 or len(set(args.ef)) != len(args.ef)):
        parser.error("Invalid counts/construction/search parameters")
    for name in ["data_dir", "gt_root", "out", "binary"]:
        setattr(args, name, getattr(args, name).resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"]:
        environment[name] = "1"
    environment["OMP_DYNAMIC"] = "FALSE"
    if not args.skip_build:
        args.binary.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([os.environ.get("CXX", "g++"), "-std=c++17", "-O3", "-g", "-fopenmp",
                        "-I", str(ROOT), str(ROOT / "examples/cpp/test_dynamic_update_workloads.cpp"),
                        "-o", str(args.binary)], cwd=ROOT, env=environment, check=True)
    for dataset in args.datasets:
        print(f"Dataset: {dataset} ({args.experiment}, threads=1)", flush=True)
        config, train, query, gt, gt_dir = preflight(args, dataset)
        directory = args.out / dataset
        directory.mkdir(parents=True, exist_ok=True)
        config_path = directory / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError(f"Configuration changed: choose a new OUT directory ({config_path})")
        atomic_json(config_path, config)
        common = [str(args.binary), "--dataset", dataset, "--experiment", args.experiment,
                  "--input", str(train), "--queries", str(query), "--gt", str(gt), "--gt-dir", str(gt_dir),
                  "--count", str(args.count), "--query-count", str(args.query_count),
                  "--batches", str(args.batches), "--search-rounds", str(args.search_rounds),
                  "--repeat-ratio", str(args.repeat_ratio), "--search-every", str(args.search_every),
                  "--insert-decode-cache", str(args.insert_decode_cache),
                  "--insert-upper-tls", str(args.insert_upper_tls),
                  "--M", str(args.M), "--ef-construction", str(args.ef_construction), "--seed", str(args.seed),
                  "--ef", *map(str, args.ef)]
        initial_dir = directory / "initial"
        initial_dir.mkdir(exist_ok=True)
        initial_index = initial_dir / "alp_initial.bin"
        try:
            for mode in args.modes:
                case = directory / mode
                case.mkdir(exist_ok=True)
                marker = case / "complete.json"
                if args.resume and marker.exists():
                    verify_case(case, config, mode)
                    print(f"Reuse {dataset} {mode}", flush=True)
                    continue
                (directory / "complete.json").unlink(missing_ok=True)
                marker.unlink(missing_ok=True)
                if mode != "raw_f32" and (not (initial_dir / "complete.json").exists() or not initial_index.exists()):
                    (initial_dir / "complete.json").unlink(missing_ok=True)
                    execute(common + ["--mode", "prepare_alp", "--output", str(initial_dir),
                            "--initial-index", str(initial_index)], initial_dir / "build.log", environment)
                    atomic_json(initial_dir / "complete.json", {"status": "PASS"})
                command = common + ["--mode", mode, "--initial-index", str(initial_index), "--output", str(case)]
                if args.verify_size:
                    command.append("--verify-size")
                execute(command, case / "run.log", environment)
                verify_case(case, config, mode)
                atomic_json(marker, {"status": "PASS"})
                publish_dataset(directory, config, partial=True)
                publish_all(args.out)
            publish_dataset(directory, config)
            (directory / "failure.json").unlink(missing_ok=True)
            publish_all(args.out)
            if not args.keep_initial_index:
                initial_index.unlink(missing_ok=True)
            print(f"PASS dataset={dataset}: {directory / 'summary.csv'}", flush=True)
        except Exception as error:
            atomic_json(directory / "failure.json", {"dataset": dataset, "mode": mode,
                        "error": str(error), "status": "FAIL"})
            raise


if __name__ == "__main__":
    main()
