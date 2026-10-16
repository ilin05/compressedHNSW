#!/usr/bin/env python3
"""Profile 1% of vectors on full HNSW forests; validate against historical CSVs."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess

DATASETS = ["sift-128-euclidean", "mnist-784-euclidean", "fashion-mnist-784-euclidean",
            "gist-960-euclidean", "deep-image-96-angular"]
CODECS = ["ALP", "DeXOR", "Gorilla", "Elf"]


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def dataset_name(value):
    suffix = "_train.fvecs"
    return value[:-len(suffix)] if value.endswith(suffix) else value


def load_oracle(alp_path, differential_path, alp_bytes, differential_bytes):
    """Raw-byte conventions are explicit: the archived CSVs use F64 DR."""
    result = {}

    def add(dataset, codec, chain, ratio, ct, bt, source, nodes=None, dim=None):
        key = (dataset, codec, chain)
        if key in result:
            raise ValueError(f"Duplicate oracle key: {key}")
        if not all(math.isfinite(v) and v > 0 for v in (ratio, ct, bt)):
            raise ValueError(f"Invalid historical metrics: {key}")
        result[key] = dict(ratio=ratio, ct=ct, bt=bt, source=str(source), nodes=nodes, dim=dim)

    for row in read_csv(alp_path):
        for chain in (2, -1):  # Independent ALP has no chain setting.
            add(dataset_name(row["Dataset"]), "ALP", chain,
                float(row["Data compression ratio"]) * 4 / alp_bytes,
                float(row["Compress time (s)"]), float(row["Build time (s)"]), alp_path)
    for row in read_csv(differential_path):
        codec, chain = row["Algorithm"], int(row["ChainMaxLength"])
        if codec not in CODECS[1:] or chain not in (2, -1):
            continue
        ratio = (float(row["PayloadRatioF32"]) if row.get("PayloadRatioF32") else
                 float(row["DataCompressionRatio"]) * 4 / differential_bytes)
        add(dataset_name(row["Dataset"]), codec, chain, ratio,
            float(row["CompressTime(s)"]), float(row["BuildTime(s)"]), differential_path,
            int(row["Vectors"]) if row.get("Vectors") else None,
            int(row["Dim"]) if row.get("Dim") else None)
    return result


def rank(values, reverse=False):
    # Competition ranks: equal values receive the same rank.
    return {key: 1 + sum(other > value if reverse else other < value
                         for other in values.values()) for key, value in values.items()}


def choose(ratios, times):
    sr, tr = rank(ratios, reverse=True), rank(times)
    feasible = [c for c in ratios if ratios[c] > 1]
    if not feasible:
        return "raw", sr, tr
    selected = min(feasible, key=lambda c: (max(sr[c], tr[c]), sr[c] + tr[c], -ratios[c], c))
    return selected, sr, tr


def analyze(output, datasets, oracle):
    all_predictions, candidates, selections = [], [], []
    for dataset in datasets:
        directory = output / dataset
        predictions = read_csv(directory / "predictions.csv")
        profiling = read_csv(directory / "profiling.csv")[0]
        config = json.loads((directory / "config.json").read_text())
        if not config.get("completed"):
            raise ValueError(f"Incomplete profiling run: {dataset}")
        wall = float(profiling["SelectionWallSeconds"])
        all_predictions.extend(predictions)
        for chain in (2, -1):
            rows = {row["Codec"]: row for row in predictions if int(row["ChainMaxLength"]) == chain}
            if set(rows) != set(CODECS):
                raise ValueError(f"Incomplete predictions: {dataset} chain={chain}")
            reference = {c: oracle.get((dataset, c, chain)) for c in CODECS}
            shape_match = all((r["nodes"] is None or r["nodes"] == int(rows[c]["Nodes"])) and
                              (r["dim"] is None or r["dim"] == int(rows[c]["Dim"]))
                              for c, r in reference.items() if r is not None)
            # A common historical denominator keeps CT and OH rankings consistent.
            common_bt = reference["ALP"]["bt"] if reference["ALP"] else None
            for pool, codecs in (("all", CODECS), ("differential", CODECS[1:])):
                pr = {c: float(rows[c]["PredictedPayloadRatioF32"]) for c in codecs}
                pt = {c: float(rows[c]["PredictedForestEncodeSeconds"]) for c in codecs}
                available = {c: reference[c] for c in codecs if reference[c] is not None}
                complete = len(available) == len(codecs)
                ar = {c: r["ratio"] for c, r in available.items()}
                at = {c: r["ct"] for c, r in available.items()}
                selected, psr, ptr = choose(pr, pt)
                asr, atr = (rank(ar, reverse=True), rank(at)) if complete else ({}, {})
                achievable = (any(asr[c] <= 2 and atr[c] <= 2 and ar[c] > 1 for c in codecs)
                              if complete else None)
                pareto = ([c for c in codecs if not any(
                    ar[d] >= ar[c] and at[d] <= at[c] and (ar[d] > ar[c] or at[d] < at[c])
                    for d in codecs if d != c)] if complete else [])
                for c in codecs:
                    candidates.append(dict(
                        Dataset=dataset, ChainMaxLength=chain, Pool=pool, Codec=c,
                        PredictedPayloadRatioF32=pr[c], ActualPayloadRatioF32=ar.get(c, ""),
                        PayloadRelativeError=pr[c] / ar[c] - 1 if c in ar else "",
                        HistoricalShapeMatch=int(shape_match),
                        PredictedPayloadRank=psr[c], ActualPayloadRank=asr.get(c, ""),
                        PredictedForestEncodeSeconds=pt[c], HistoricalFullCompressSeconds=at.get(c, ""),
                        PredictedTimeRank=ptr[c], ActualTimeRank=atr.get(c, ""),
                        HistoricalCommonBaseOH=at[c] / common_bt if c in at and common_bt else "",
                        HistoricalOwnBaseOH=at[c] / available[c]["bt"] if c in at else "",
                        CommonBuildSeconds=common_bt or "", CommonBuildSource="historical_ALP_BT_proxy",
                        Selected=int(c == selected),
                        HistoricalSource=available[c]["source"] if c in available else "",
                        ValidationStatus="complete" if complete else "missing_oracle",
                        ConfigurationMatch="unverified_historical_graph_and_build_versions",
                        TimeComparison="partial_forest_encode_prediction_vs_historical_full_CT"))
                is_raw = selected == "raw"
                ratio = 1.0 if is_raw else ar.get(selected)
                ct = 0.0 if is_raw else at.get(selected)
                selections.append(dict(
                    Dataset=dataset, ChainMaxLength=chain, Pool=pool, SelectedCodec=selected,
                    SampleCount=next(iter(rows.values()))["SampleCount"],
                    RootPolicy=next(iter(rows.values()))["RootPolicy"],
                    PredictedPayloadRatioF32=1 if is_raw else pr[selected],
                    ActualPayloadRatioF32=ratio if ratio is not None else "",
                    HistoricalShapeMatch=int(shape_match),
                    ActualPayloadRank=asr.get(selected, ""), ActualTimeRank=atr.get(selected, ""),
                    JointTop2Achievable=int(achievable) if complete else "",
                    JointTop2Success=int(not is_raw and asr[selected] <= 2 and atr[selected] <= 2) if complete else "",
                    ParetoOptimal=int(not is_raw and selected in pareto) if complete else "",
                    PayloadSizeRegretVsBest=max(1.0, max(ar.values())) / ratio - 1 if complete else "",
                    TimeSlowdownVsFastest=ct / min(at.values()) if complete else "",
                    HistoricalFullCompressSeconds=ct if ct is not None else "",
                    HistoricalCommonBaseOH=ct / common_bt if ct is not None and common_bt else "",
                    CommonBuildSource="historical_ALP_BT_proxy", BothChainsSelectionWallSeconds=wall,
                    ProfileToSelectedCT=wall / ct if ct else "",
                    ValidationStatus="complete" if complete else "missing_oracle",
                    MissingOracleCodecs=";".join(c for c in codecs if c not in available),
                    ValidationScope="historical_cross_run_not_same_graph",
                    TimeScope="prediction_excludes_PQ_compaction_patches_IO"))
    write_csv(output / "predictions.csv", all_predictions)
    write_csv(output / "candidate_validation.csv", candidates)
    write_csv(output / "selection_summary.csv", selections)
    for row in selections:
        print(f"{row['Dataset']} chain={row['ChainMaxLength']} pool={row['Pool']} "
              f"selected={row['SelectedCodec']} ranks={row['ActualPayloadRank']}/{row['ActualTimeRank']} "
              f"joint_top2={row['JointTop2Success']} achievable={row['JointTop2Achievable']}", flush=True)


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=repo / "datasets/hdf5files")
    parser.add_argument("--output-dir", type=Path, default=repo / "script/codec_selection_results")
    parser.add_argument("--graph-dir", type=Path, help="One raw F64 framework graph per dataset: DATASET.bin")
    parser.add_argument("--build-dir", type=Path, default=repo / "build_lvc_codec_selection")
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--sample-ratio", type=float, default=.01)
    parser.add_argument("--root-ratio", type=float, default=.01)
    parser.add_argument("--root-policy", choices=["level", "degree"], default="level")
    parser.add_argument("--alp-hub-ratio", type=float, default=.01)
    parser.add_argument("--alp-tolerance", type=float, default=1e-4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--build-threads", type=int, default=1)
    parser.add_argument("--build-jobs", type=int, default=4)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--alp-oracle", type=Path, default=repo / "script/hnswalp_pq_build_compression_results.csv")
    parser.add_argument("--differential-oracle", type=Path, default=repo / "script/compressed_hnsw_ablation_build_results.csv")
    parser.add_argument("--oracle-alp-raw-bytes", choices=[4, 8], type=int, default=8,
                        help="Archived ALP CSV uses F64 DR; current builder uses F32, so use 4 for new CSVs")
    parser.add_argument("--oracle-differential-raw-bytes", choices=[4, 8], type=int, default=8)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    oracle = load_oracle(args.alp_oracle, args.differential_oracle,
                         args.oracle_alp_raw_bytes, args.oracle_differential_raw_bytes)
    for dataset in args.datasets:
        if Path(dataset).name != dataset:
            parser.error("Dataset must be a simple filename prefix")
        for c in CODECS:
            for chain in (2, -1):
                if (dataset, c, chain) not in oracle:
                    print(f"WARNING missing historical row: {(dataset, c, chain)}; prediction proceeds, ranks unavailable", flush=True)
    if args.analyze_only:
        analyze(output, args.datasets, oracle)
        return
    build = args.build_dir.resolve()
    binary = args.binary.resolve() if args.binary else build / "test_codec_selection"
    if not args.skip_build and not args.binary:
        subprocess.run(["cmake", "-S", str(repo), "-B", str(build), "-DHNSWLIB_EXAMPLES=ON",
                        "-DCMAKE_BUILD_TYPE=Release"], check=True)
        subprocess.run(["cmake", "--build", str(build), "--target", "test_codec_selection",
                        "-j", str(args.build_jobs)], check=True)
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    completed = []
    for dataset in args.datasets:
        directory = output / dataset
        directory.mkdir(exist_ok=True)
        graph = ((args.graph_dir.resolve() / f"{dataset}.bin") if args.graph_dir else
                 directory / "shared_raw_graph.bin")
        train = args.data_dir.resolve() / f"{dataset}_train.fvecs"
        identity = train.stat()
        config = dict(dataset=dataset, train=str(train), train_bytes=identity.st_size,
                      train_mtime_ns=identity.st_mtime_ns, graph=str(graph), seed=args.seed,
                      sample_ratio=args.sample_ratio, root_ratio=args.root_ratio,
                      root_policy=args.root_policy, rounds=args.rounds, M=args.M,
                      ef_construction=args.ef_construction, build_threads=args.build_threads,
                      alp_hub_ratio=args.alp_hub_ratio, alp_tolerance=args.alp_tolerance)
        if graph.exists():
            stat = graph.stat()
            config.update(graph_bytes=stat.st_size, graph_mtime_ns=stat.st_mtime_ns)
        config_path = directory / "config.json"
        old = json.loads(config_path.read_text()) if config_path.exists() else {}
        reuse = args.resume and old == {**config, "completed": True} and all(
            (directory / name).is_file() for name in ("predictions.csv", "profiling.csv", "sample_ids.csv"))
        if reuse:
            print(f"Reuse profiling: {dataset}", flush=True)
        else:
            config_path.write_text(json.dumps({**config, "completed": False}, indent=2) + "\n")
            command = [str(binary), "--train", str(train), "--graph", str(graph),
                       "--output", str(directory), "--dataset", dataset]
            for name in ("sample_ratio", "root_ratio", "root_policy", "rounds", "seed", "M",
                         "ef_construction", "build_threads", "alp_hub_ratio", "alp_tolerance"):
                command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
            print(f"Profile full forest + sample: {dataset}", flush=True)
            with (directory / "profile.log").open("w") as log:
                subprocess.run(command, cwd=repo, env=environment, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            stat = graph.stat()
            config.update(graph_bytes=stat.st_size, graph_mtime_ns=stat.st_mtime_ns)
            config_path.write_text(json.dumps({**config, "completed": True}, indent=2) + "\n")
        completed.append(dataset)
        analyze(output, completed, oracle)  # Preserve aggregate results after every dataset.
    print(f"Results: {output / 'selection_summary.csv'}")


if __name__ == "__main__":
    main()
