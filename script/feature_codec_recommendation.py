#!/usr/bin/env python3
"""Fixed-budget feature extraction and codec recommendation; never trial-encode.

NumPy is the only dependency. Dataset names are provenance, never model inputs.
The local reference is the nearest sampled anchor, not a graph predecessor.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np

DATASETS = ["sift-128-euclidean", "mnist-784-euclidean", "fashion-mnist-784-euclidean",
            "gist-960-euclidean", "deep-image-96-angular"]
CODECS = ["ALP", "DeXOR", "Gorilla", "Elf"]
FEATURES = ["zero_ratio", "integer_ratio", "decimal_complexity", "exponent_spread",
            "reference_equal_ratio", "reference_xor_width", "dimension_feature"]
SCHEMA = 1


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("Empty results: {}".format(path))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    temporary.replace(path)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sample_ids(n, count, rng):
    """Floyd sampling: O(count) work/storage, even for a billion input rows."""
    if not 0 < count <= n:
        raise ValueError("Need {} distinct entries, available {}".format(count, n))
    selected, result = set(), []
    for j in range(n - count, n):
        t = rng.randrange(j + 1)
        chosen = j if t in selected else t
        selected.add(chosen)
        result.append(chosen)
    rng.shuffle(result)
    return result


def read_probe(path, samples, seed, settings):
    start = time.perf_counter()
    path = Path(path)
    with path.open("rb") as f:
        dim_data = np.fromfile(f, dtype="<i4", count=1)
    if len(dim_data) != 1 or int(dim_data[0]) <= 0:
        raise ValueError("Invalid fvecs dimension: {}".format(path))
    dim = int(dim_data[0])
    n, remainder = divmod(path.stat().st_size, 4 * (dim + 1))
    if remainder or not n:
        raise ValueError("Truncated/empty fvecs: {}".format(path))
    if not settings["references"] < samples <= settings["sample_pool_size"]:
        raise ValueError("Samples must exceed anchor count and fit the configured sample pool")
    rng = random.Random(seed)
    pool = sample_ids(n, settings["sample_pool_size"], rng)
    ids = pool[:samples]
    dimensions = sorted(sample_ids(dim, min(dim, settings["dimensions"]), rng))
    data = np.memmap(str(path), dtype="<f4", mode="r", shape=(n, dim + 1))
    # Two-axis advanced indexing touches only selected coordinates, not N*d or B*d.
    header = np.asarray(data.view("<i4")[ids, 0])
    if not np.all(header == dim):
        raise ValueError("Inconsistent sampled fvecs headers")
    values = np.asarray(data[np.ix_(ids, np.asarray(dimensions) + 1)], dtype=np.float32)
    del data
    if not np.all(np.isfinite(values)):
        raise ValueError("Nonfinite sampled values; no model calibrated for this input")
    identity = hashlib.sha256(np.asarray(ids + dimensions, dtype="<i8").tobytes()).hexdigest()
    return values, values[:settings["references"]], n, dim, time.perf_counter() - start, identity


def bit_length(values):
    values = np.asarray(values, dtype=np.uint64).copy()
    result = np.zeros(values.shape, dtype=np.int16)
    for shift in (32, 16, 8, 4, 2, 1):
        mask = values >= np.uint64(1 << shift)
        values[mask] >>= np.uint64(shift)
        result[mask] += shift
    return result + (values != 0)


def extract_features(values, references, full_dim, settings):
    x = values.astype(np.float64)
    nonzero = x[x != 0]
    # Minimum decimal places q<=8 which reconstruct the original F32 bits.
    # This is a feature, not a codec precision/tolerance change.
    decimal = np.full(nonzero.shape, 9, dtype=np.float64)
    original = nonzero.astype(np.float32).view(np.uint32)
    for q in range(9):
        reconstructed = (np.rint(nonzero * (10.0 ** q)) / (10.0 ** q)).astype(np.float32)
        decimal[(decimal == 9) & (reconstructed.view(np.uint32) == original)] = q
    exponents = np.frexp(np.abs(nonzero))[1] if nonzero.size else np.array([0])
    spread = float(np.percentile(exponents, 95) - np.percentile(exponents, 5))
    ref = references.astype(np.float64)
    nearest = []
    for begin in range(0, len(x), 128):
        chunk = x[begin:begin + 128]
        distances = ((chunk[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2)
        # Anchors belong to the sample budget. Exclude their self matches.
        anchors = np.arange(begin, min(begin + len(chunk), len(ref)))
        distances[anchors - begin, anchors] = np.inf
        nearest.append(ref[np.argmin(distances, axis=1)])
    nearest = np.concatenate(nearest)
    if settings["codec_word_bits"] == 64:
        a, b = x.view(np.uint64), nearest.view(np.uint64)
    else:
        a = x.astype(np.float32).view(np.uint32).astype(np.uint64)
        b = nearest.astype(np.float32).view(np.uint32).astype(np.uint64)
    xor = np.bitwise_xor(a, b)
    active = xor != 0
    lowbit = xor[active] & (~xor[active] + np.uint64(1))
    widths = bit_length(xor[active]) - bit_length(lowbit) + 1
    features = dict(
        zero_ratio=float(np.mean(x == 0)), integer_ratio=float(np.mean(x == np.rint(x))),
        decimal_complexity=float(np.mean(decimal) / 9) if decimal.size else 0.0,
        exponent_spread=min(1.0, spread / 64),
        reference_equal_ratio=float(np.mean(values == nearest)),
        reference_xor_width=float(np.mean(widths) / settings["codec_word_bits"]) if widths.size else 0.0,
        dimension_feature=full_dim / (full_dim + settings["dimension_scale"]),
    )
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in features.values()):
        raise ValueError("Invalid features: {}".format(features))
    return features


def predict(model, features):
    x = np.array([features[f] for f in model["features"]], dtype=float)
    x = np.r_[1.0, (x - np.array(model["mean"])) / np.array(model["scale"])]
    return {c: dict(size=float(np.exp(np.clip(x @ np.array(weights["log_size"]), -30, 30))),
                    time=float(np.exp(np.clip(x @ np.array(weights["log_time"]), -30, 30))))
            for c, weights in model["coefficients"].items()}


def select(model, prediction, space_weight=None, time_weight=None):
    if space_weight is None and time_weight is None:
        sw, tw = model["space_weight"], 1 - model["space_weight"]
    elif time_weight is None:
        sw, tw = space_weight, 1 - space_weight
    elif space_weight is None:
        sw, tw = 1 - time_weight, time_weight
    else:
        sw, tw = space_weight, time_weight
    if min(sw, tw) < 0 or sw + tw <= 0:
        raise ValueError("Weights must be nonnegative and have positive sum")
    sw, tw = sw / (sw + tw), tw / (sw + tw)
    allowed = CODECS if model["pool"] == "all" else CODECS[1:]
    scores = {c: sw * prediction[c]["size"] / model["size_normalizer"] +
              tw * prediction[c]["time"] / model["time_normalizer"] for c in allowed}
    feasible = [c for c in allowed if 1 / prediction[c]["size"] >= model["min_payload_ratio"]]
    choice = min(feasible, key=lambda c: (scores[c], c)) if feasible else "raw"
    return choice, scores


def fit_model(rows, oracle, chain, ridge, settings, pool, space_weight, min_ratio):
    x = np.array([[float(r[f]) for f in FEATURES] for r in rows])
    # Equal dataset weighting: repeated seeds are not independent datasets.
    counts = {d: sum(r["Dataset"] == d for r in rows) for d in {r["Dataset"] for r in rows}}
    weights = np.array([1 / counts[r["Dataset"]] for r in rows])
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weights))
    scale[scale < 1e-8] = 1
    x = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    y = []
    for row in rows:
        cells = []
        for c in CODECS:
            actual = oracle[(row["Dataset"], c, chain)]
            cells.extend([math.log(1 / actual["ratio"]), math.log(actual["ct"] /
                          (int(row["Nodes"]) * int(row["Dim"])))])
        y.append(cells)
    y = np.array(y)
    penalty = np.eye(x.shape[1]) * ridge
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(x.T @ (x * weights[:, None]) + penalty,
                                   x.T @ (y * weights[:, None]))
    model = dict(schema=SCHEMA, features=FEATURES, settings=settings, chain_max=chain,
                 pool=pool, ridge=ridge, space_weight=space_weight,
                 min_payload_ratio=min_ratio, mean=mean.tolist(), scale=scale.tolist(),
                 size_normalizer=float(np.median(np.exp(y[:, 0::2]))),
                 time_normalizer=float(np.median(np.exp(y[:, 1::2]))),
                 coefficients={c: dict(log_size=coefficients[:, 2*i].tolist(),
                                      log_time=coefficients[:, 2*i+1].tolist())
                               for i, c in enumerate(CODECS)},
                 time_scope="historical_full_compression_seconds_per_coordinate",
                 reference_scope="nearest_sample_anchor_not_graph_predecessor",
                 historical_configuration_match="unverified_cross_run_graph_codec_threads",
                 root_policy="level", root_ratio=0.01, alp_tolerance=1e-4,
                 training_datasets=sorted(counts))
    return model


def rank(values, reverse=False):
    return {c: 1 + sum(v > values[c] if reverse else v < values[c]
                       for v in values.values()) for c in values}


def assess(choice, actual, min_ratio):
    sr = rank({c: a["ratio"] for c, a in actual.items()}, reverse=True)
    tr = rank({c: a["ct"] for c, a in actual.items()})
    feasible = [c for c, a in actual.items() if a["ratio"] >= min_ratio]
    best = min(feasible, key=lambda c: (max(sr[c], tr[c]), sr[c] + tr[c],
                                       -actual[c]["ratio"], c)) if feasible else "raw"
    achievable = any(sr[c] <= 2 and tr[c] <= 2 for c in feasible)
    maxrank = 0 if choice == "raw" and not feasible else (
        len(actual) + 1 if choice == "raw" else max(sr[choice], tr[choice]))
    bestrank = 0 if best == "raw" else max(sr[best], tr[best])
    ratio = 1 if choice == "raw" else actual[choice]["ratio"]
    ct = 0 if choice == "raw" else actual[choice]["ct"]
    pareto = choice == "raw" and not feasible or choice != "raw" and not any(
        a["ratio"] >= ratio and a["ct"] <= ct and (a["ratio"] > ratio or a["ct"] < ct)
        for a in actual.values())
    return dict(OracleChoice=best, ActualPayloadRatioF32=ratio,
                ActualCompressSeconds=ct, ActualPayloadRank=sr.get(choice, ""),
                ActualTimeRank=tr.get(choice, ""), JointTop2Achievable=int(achievable),
                JointTop2Success=int(choice in feasible and sr[choice] <= 2 and tr[choice] <= 2),
                OracleMatch=int(choice == best), ParetoOptimal=int(pareto),
                UnexpectedExpansion=int(choice != "raw" and ratio < min_ratio),
                MaxRankRegret=max(0, maxrank - bestrank),
                PayloadRegret=max(1, max(a["ratio"] for a in actual.values())) / ratio - 1,
                TimeSlowdown=ct / min(a["ct"] for a in actual.values()))


def evaluate(model, rows, oracle, phase):
    result = []
    chain = model["chain_max"]
    codecs = CODECS if model["pool"] == "all" else CODECS[1:]
    for row in rows:
        start = time.perf_counter()
        prediction = predict(model, row)
        choice, scores = select(model, prediction)
        inference = time.perf_counter() - start
        actual = {c: oracle[(row["Dataset"], c, chain)] for c in codecs}
        metrics = assess(choice, actual, model["min_payload_ratio"])
        online = float(row["ProbeReadSeconds"]) + float(row["FeatureSeconds"]) + inference
        build = oracle[(row["Dataset"], "ALP", chain)]["bt"]
        result.append(dict(Dataset=row["Dataset"], Phase=phase, ChainMaxLength=chain,
                           Pool=model["pool"], SampleCount=int(row["SampleCount"]),
                           Seed=int(row["Seed"]), SelectedCodec=choice,
                           PredictedPayloadRatioF32=1 if choice == "raw" else 1 / prediction[choice]["size"],
                           PredictedCompressSeconds=0 if choice == "raw" else
                           prediction[choice]["time"] * int(row["Nodes"]) * int(row["Dim"]),
                           ProbeReadSeconds=float(row["ProbeReadSeconds"]),
                           FeatureSeconds=float(row["FeatureSeconds"]), InferenceSeconds=inference,
                           ProbeLoadedVectors=int(row["ProbeLoadedVectors"]),
                           OnlineSecondsUpperBound=online, HistoricalBuildSeconds=build,
                           OnlineToHistoricalBuild=online/build,
                           OnlineToSelectedCompression=online/metrics["ActualCompressSeconds"]
                           if metrics["ActualCompressSeconds"] else "", **metrics))
    return result


def summarize(rows):
    groups = {}
    for row in rows:
        key = tuple(row[f] for f in ("Dataset", "Phase", "ChainMaxLength", "Pool", "SampleCount"))
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, group in sorted(groups.items()):
        choices = {c: sum(r["SelectedCodec"] == c for r in group) for c in CODECS + ["raw"]}
        majority = max(choices, key=lambda c: (choices[c], c))
        achievable = [r for r in group if r["JointTop2Achievable"]]
        summaries.append(dict(zip(("Dataset", "Phase", "ChainMaxLength", "Pool", "SampleCount"), key),
            Runs=len(group), MajorityCodec=majority, RecommendationAgreement=choices[majority] / len(group),
            AllSeedsAgree=int(choices[majority] == len(group)),
            CodecCounts=json.dumps({c: v for c, v in choices.items() if v}, sort_keys=True),
            JointTop2Achievable=int(bool(achievable)),
            JointTop2SuccessRate=float(np.mean([r["JointTop2Success"] for r in achievable])) if achievable else "",
            OracleMatchRate=float(np.mean([r["OracleMatch"] for r in group])),
            ParetoRate=float(np.mean([r["ParetoOptimal"] for r in group])),
            MeanPayloadRegret=float(np.mean([r["PayloadRegret"] for r in group])),
            MeanTimeSlowdown=float(np.mean([r["TimeSlowdown"] for r in group])),
            MeanOnlineSecondsUpperBound=float(np.mean([r["OnlineSecondsUpperBound"] for r in group]))))
    return summaries


def tuning_key(rows):
    mean = lambda name: float(np.mean([r[name] for r in rows]))
    summaries = summarize(rows)
    achievable = [r for r in rows if r["JointTop2Achievable"]]
    miss = 1 - float(np.mean([r["JointTop2Success"] for r in achievable])) if achievable else 0
    by_dataset = {}
    for row in summaries:
        by_dataset.setdefault(row["Dataset"], []).append(row["MajorityCodec"])
    cross_size = float(np.mean([1 - max(v.count(c) for c in set(v)) / len(v)
                                for v in by_dataset.values()]))
    return (mean("UnexpectedExpansion"), mean("MaxRankRegret"), miss,
            float(np.mean([1 - r["RecommendationAgreement"] for r in summaries])),
            cross_size, mean("PayloadRegret"), mean("TimeSlowdown"))


def tune(train, validation, oracle, chain, pool, settings, args):
    if not train or not validation:
        raise ValueError("Empty training or tuning partition")
    models, grid = [], []
    for ridge in args.ridges:
        base = fit_model(train, oracle, chain, ridge, settings, pool, .5, args.min_payload_ratio)
        for weight in args.space_weights:
            model = dict(base, space_weight=weight)
            key = tuning_key(evaluate(model, validation, oracle, "tune"))
            models.append((key, ridge, weight, model))
            grid.append(dict(ChainMaxLength=chain, Pool=pool, Ridge=ridge, SpaceWeight=weight,
                             UnexpectedExpansionRate=key[0], MeanMaxRankRegret=key[1],
                             AchievableJointTop2MissRate=key[2], MeanInstability=key[3],
                             CrossSampleMajorityDisagreement=key[4],
                             MeanPayloadRegret=key[5], MeanTimeSlowdown=key[6]))
    _, _, _, model = min(models, key=lambda item: item[:3])
    for row in grid:
        row["Selected"] = int(row["Ridge"] == model["ridge"] and row["SpaceWeight"] == model["space_weight"])
    return model, grid


def collect(args, settings, output):
    seeds = [("train", s) for s in args.train_seeds] + [("tune", s) for s in args.tune_seeds] + [
        ("test", s) for s in args.test_seeds]
    paths = {d: args.data_dir / (d + "_train.fvecs") for d in args.datasets}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing training fvecs files:\n" + "\n".join(missing))
    identities = {}
    for dataset in args.datasets:
        path = paths[dataset]
        stat = path.stat()  # Fail before partial runs if any dataset is absent.
        identities[dataset] = dict(path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    config = dict(schema=SCHEMA, settings=settings, samples=args.samples, seeds=seeds, inputs=identities)
    config_path = output / "feature_config.json"
    old = json.loads(config_path.read_text()) if config_path.exists() else None
    if args.resume and old is not None and old != json.loads(json.dumps(config)):
        raise ValueError("Resume feature configuration/data identity differs; use a new output directory")
    write_json(config_path, config)
    all_rows = []
    for dataset in args.datasets:
        cached = output / dataset / "features.csv"
        manifest_path = cached.with_suffix(".json")
        expected_config = dict(settings=settings, samples=args.samples, seeds=seeds,
                               input=identities[dataset])
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        can_reuse = (args.resume and old is not None and cached.exists() and
            manifest.get("config") == json.loads(json.dumps(expected_config)) and
            manifest.get("sha256") == hashlib.sha256(cached.read_bytes()).hexdigest())
        if can_reuse:
            rows = read_csv(cached)
            keys = {(r["Split"], int(r["Seed"]), int(r["SampleCount"])) for r in rows}
            expected = {(split, seed, b) for split, seed in seeds for b in args.samples}
            if keys != expected or len(rows) != len(expected):
                raise ValueError("Incomplete cached features: {}".format(cached))
            print("Reuse features:", dataset, flush=True)
        else:
            rows = []
            for split, seed in seeds:
                values, refs, n, dim, read_s, digest = read_probe(
                    identities[dataset]["path"], max(args.samples), seed, settings)
                for count in args.samples:
                    start = time.perf_counter()
                    features = extract_features(values[:count], refs, dim, settings)
                    rows.append(dict(Dataset=dataset, Split=split, Seed=seed, SampleCount=count,
                        Nodes=n, Dim=dim, SampledDimensions=values.shape[1], ProbeIdentity=digest,
                        ProbeLoadedVectors=max(args.samples),
                        ProbeReadSeconds=read_s, FeatureSeconds=time.perf_counter()-start, **features))
            write_csv(cached, rows)
            write_json(manifest_path, dict(config=expected_config,
                sha256=hashlib.sha256(cached.read_bytes()).hexdigest()))
            print("Extracted {}: {} feature probes; no forest/codec execution".format(dataset, len(rows)), flush=True)
        all_rows.extend(rows)
        write_csv(output / "features.csv", all_rows)
    return all_rows


def calibrate(args):
    # Existing parser provides explicit archived F64 -> F32 ratio conversion.
    from test_codec_selection import load_oracle
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    settings = dict(sample_pool_size=max(args.samples), dimensions=args.dimensions,
                    references=args.references, dimension_scale=128.0, codec_word_bits=args.codec_word_bits)
    oracle = load_oracle(args.alp_oracle, args.differential_oracle,
                         args.oracle_alp_raw_bytes, args.oracle_differential_raw_bytes)
    if args.analyze_only:
        saved = json.loads((output / "feature_config.json").read_text())
        seeds = [[split, seed] for split, group in (("train", args.train_seeds),
                 ("tune", args.tune_seeds), ("test", args.test_seeds)) for seed in group]
        if (saved["settings"] != settings or set(saved["inputs"]) != set(args.datasets) or
                saved["seeds"] != seeds or saved["samples"] != args.samples):
            raise ValueError("Analyze-only dataset/settings differ from saved feature extraction")
        rows = read_csv(output / "features.csv")
        expected = {(dataset, split, seed, count) for dataset in args.datasets
                    for split, seed in seeds for count in args.samples}
        keys = {(r["Dataset"], r["Split"], int(r["Seed"]), int(r["SampleCount"])) for r in rows}
        if keys != expected or len(rows) != len(expected):
            raise ValueError("Incomplete/duplicate saved features; finish extraction before analyze-only")
    else:
        rows = collect(args, settings, output)
    for row in rows:
        for chain in args.chains:
            for c in CODECS:
                a = oracle[(row["Dataset"], c, chain)]
                if a["nodes"] is not None and a["nodes"] != int(row["Nodes"]) or (
                        a["dim"] is not None and a["dim"] != int(row["Dim"])):
                    raise ValueError("Historical shape mismatch: {} {}".format(row["Dataset"], c))
    splits = {split: [r for r in rows if r["Split"] == split] for split in ("train", "tune", "test")}
    if any(not v for v in splits.values()):
        raise ValueError("Missing train/tune/test features")
    provenance = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    provenance["oracle_identity"] = {name: dict(path=str(path.resolve()),
        bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
        for name,path in (("ALP",args.alp_oracle),("differential",args.differential_oracle))}
    write_json(output / "calibration_config.json", provenance)
    start = time.perf_counter()
    all_results, all_grids, baselines, cross, certificates, model_paths = [], [], [], [], [], []
    for chain in args.chains:
        for pool in args.pools:
            model, grid = tune(splits["train"], splits["tune"], oracle, chain, pool, settings, args)
            tag = "chain{}_{}".format("unbounded" if chain == -1 else chain, pool)
            model_path = output / ("model_" + tag + ".json")
            write_json(model_path, model)
            model_paths.append(model_path)
            print("{} selected ridge={} space_weight={}".format(
                tag, model["ridge"], model["space_weight"]), flush=True)
            all_grids.extend(dict(Fold="known_datasets", **r) for r in grid)
            result = evaluate(model, splits["test"], oracle, "heldout_seed")
            all_results.extend(result)
            summary = summarize(result)
            for dataset in args.datasets:
                group = [r for r in summary if r["Dataset"] == dataset]
                stable = all(r["RecommendationAgreement"] >= args.stability_threshold for r in group)
                same = len({r["MajorityCodec"] for r in group}) == 1
                tested = [r for r in result if r["Dataset"] == dataset]
                quality = all(not r["UnexpectedExpansion"] and r["MaxRankRegret"] == 0 and
                    r["ParetoOptimal"] and (not r["JointTop2Achievable"] or r["JointTop2Success"])
                    for r in tested)
                cert = dict(Dataset=dataset, ChainMaxLength=chain, Pool=pool,
                    SpaceWeight=model["space_weight"], Ridge=model["ridge"],
                    StabilityThreshold=args.stability_threshold, StableWithinEachSampleSize=int(stable),
                    SameMajorityAcrossSampleSizes=int(same), StabilityTargetPassed=int(stable and same),
                    QualityTargetPassed=int(quality), RecommendationTargetPassed=int(stable and same and quality),
                    MajorityBySampleSize=json.dumps({str(r["SampleCount"]):r["MajorityCodec"] for r in group}))
                certificates.append(cert)
                print("{} {}: stability={} choices={}".format(dataset, tag,
                    cert["StabilityTargetPassed"], cert["MajorityBySampleSize"]), flush=True)
            if pool == "all":
                for r in result:
                    actual = {c: oracle[(r["Dataset"], c, chain)] for c in CODECS}
                    baselines.append(dict(Dataset=r["Dataset"], Phase="always_ALP", ChainMaxLength=chain,
                        Pool=pool, SampleCount=r["SampleCount"], Seed=r["Seed"], SelectedCodec="ALP",
                        **assess("ALP", actual, args.min_payload_ratio)))
            # Refit/tune without the held-out dataset; its data never chooses parameters.
            if not args.skip_lodo and len(args.datasets) > 1:
                for heldout in args.datasets:
                    train = [r for r in splits["train"] if r["Dataset"] != heldout]
                    validation = [r for r in splits["tune"] if r["Dataset"] != heldout]
                    fold, fold_grid = tune(train, validation, oracle, chain, pool, settings, args)
                    tested = [r for r in splits["test"] if r["Dataset"] == heldout]
                    cross.extend(evaluate(fold, tested, oracle, "leave_dataset_out"))
                    all_grids.extend(dict(Fold="leave_out_"+heldout, **r) for r in fold_grid)
            # Preserve completed conditions, not just the last dataset/condition.
            write_csv(output / "recommendations.csv", all_results)
            write_csv(output / "stability_summary.csv", summarize(all_results))
            write_csv(output / "stability_targets.csv", certificates)
            write_csv(output / "parameter_search.csv", all_grids)
            if baselines:
                write_csv(output / "always_alp_baseline.csv", baselines)
            if cross:
                write_csv(output / "leave_dataset_out.csv", cross)
                write_csv(output / "leave_dataset_out_summary.csv", summarize(cross))
    weight_rows = []
    for path in model_paths:
        m = json.loads(path.read_text())
        for codec, coeff in m["coefficients"].items():
            for target in ("log_size", "log_time"):
                for name, value in zip(["intercept"] + FEATURES, coeff[target]):
                    weight_rows.append(dict(Model=path.name, Codec=codec, Target=target, Feature=name,
                                            StandardizedCoefficient=value))
    write_csv(output / "model_weights.csv", weight_rows)
    write_json(output / "run_summary.json", dict(schema=SCHEMA,
        CalibrationSeconds=time.perf_counter()-start,
        HeldoutSeedStabilityPassed=all(r["StabilityTargetPassed"] for r in certificates),
        HeldoutSeedRecommendationPassed=all(r["RecommendationTargetPassed"] for r in certificates),
        PassedConditions=sum(r["StabilityTargetPassed"] for r in certificates),
        TotalConditions=len(certificates), OnlineTrialEncoding=False, OnlineFullForest=False,
        GeneralizationScope="heldout seeds share calibrated datasets; LODO reported separately",
        OracleScope="historical_cross_run_not_same_graph; ALP tolerance unchanged",
        TimingScope="probe/read/features/inference; excludes Python startup; read uses maximum sample size",
        settings=settings))
    print("Results:", output, flush=True)


def recommend(args):
    model = json.loads(args.model.read_text())
    if model.get("schema") != SCHEMA or model["features"] != FEATURES:
        raise ValueError("Unsupported feature/model schema")
    x, refs, n, dim, read_s, identity = read_probe(args.data_file, args.samples, args.seed, model["settings"])
    start = time.perf_counter()
    features = extract_features(x, refs, dim, model["settings"])
    feature_s = time.perf_counter()-start
    start = time.perf_counter()
    prediction = predict(model, features)
    selected, scores = select(model, prediction, args.space_weight, args.time_weight)
    inference_s = time.perf_counter()-start
    result = dict(SelectedCodec=selected, SampleCount=args.samples, Seed=args.seed, Nodes=n, Dim=dim,
        ProbeLoadedVectors=args.samples,
        ChainMaxLength=model["chain_max"], Pool=model["pool"], ProbeIdentity=identity,
        ProbeReadSeconds=read_s, FeatureSeconds=feature_s, InferenceSeconds=inference_s,
        OnlineSeconds=read_s+feature_s+inference_s, Features=features,
        Candidates={c:dict(PredictedPayloadRatioF32=1/v["size"],
                          PredictedCompressSeconds=v["time"]*n*dim, Score=scores.get(c))
                    for c,v in prediction.items()})
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("calibrate", help="Extract features, tune weights, evaluate untouched test seeds")
    run.add_argument("--data-dir", type=Path, default=repo/"datasets/hdf5files")
    run.add_argument("--output-dir", type=Path, default=repo/"script/feature_codec_selection_results")
    run.add_argument("--datasets", nargs="+", default=DATASETS)
    run.add_argument("--samples", nargs="+", type=int, default=[256,512,1024])
    run.add_argument("--train-seeds", nargs="+", type=int, default=list(range(1001,1006)))
    run.add_argument("--tune-seeds", nargs="+", type=int, default=list(range(2001,2004)))
    run.add_argument("--test-seeds", nargs="+", type=int, default=list(range(3001,3011)))
    run.add_argument("--dimensions", type=int, default=32)
    run.add_argument("--references", type=int, default=32)
    run.add_argument("--codec-word-bits", choices=[32,64], type=int, default=64)
    run.add_argument("--chains", nargs="+", type=int, choices=[2,-1], default=[2,-1])
    run.add_argument("--pools", nargs="+", choices=["all","differential"], default=["all","differential"])
    run.add_argument("--ridges", nargs="+", type=float, default=[.01,.1,1,10])
    run.add_argument("--space-weights", nargs="+", type=float, default=[.25,.5,.75,.9])
    run.add_argument("--min-payload-ratio", type=float, default=1.02)
    run.add_argument("--stability-threshold", type=float, default=1.0)
    run.add_argument("--alp-oracle", type=Path, default=repo/"script/hnswalp_pq_build_compression_results.csv")
    run.add_argument("--differential-oracle", type=Path, default=repo/"script/compressed_hnsw_ablation_build_results.csv")
    run.add_argument("--oracle-alp-raw-bytes", type=int, choices=[4,8], default=8)
    run.add_argument("--oracle-differential-raw-bytes", type=int, choices=[4,8], default=8)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--analyze-only", action="store_true")
    run.add_argument("--skip-lodo", action="store_true")
    infer = sub.add_parser("recommend", help="Use saved model; no historical CSV or codec required")
    infer.add_argument("--model", type=Path, required=True)
    infer.add_argument("--data-file", type=Path, required=True)
    infer.add_argument("--samples", type=int, default=512)
    infer.add_argument("--seed", type=int, default=3001)
    infer.add_argument("--space-weight", type=float)
    infer.add_argument("--time-weight", type=float)
    infer.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "calibrate":
        if (any(v <= 0 for v in args.samples + [args.dimensions,args.references]) or
            args.references < 2 or min(args.samples) <= args.references or
            len(set(args.samples)) != len(args.samples) or
            any(v <= 0 or not math.isfinite(v) for v in args.ridges) or
            any(not 0 <= v <= 1 for v in args.space_weights) or
            not 0 < args.stability_threshold <= 1 or not args.min_payload_ratio > 1):
            parser.error("Invalid sampling/weight/regularization parameters")
        groups = [args.train_seeds,args.tune_seeds,args.test_seeds]
        if any(not g or len(set(g)) != len(g) for g in groups) or any(
                set(groups[i]) & set(groups[j]) for i in range(3) for j in range(i)):
            parser.error("Train/tune/test seeds must be nonempty, unique and disjoint")
        for dataset in args.datasets:
            if Path(dataset).name != dataset:
                parser.error("Dataset must be a filename prefix")
        calibrate(args)
    else:
        recommend(args)


if __name__ == "__main__":
    main()
