#!/usr/bin/env python3
"""Append exact serialized storage components for revision baseline indexes."""

import argparse
import csv
import subprocess
from pathlib import Path


FIELDS = ("Dataset", "IndexType", "ActualVariant", "IndexFile", "Component",
          "Bytes", "GiB", "FileSizeBytes", "N", "Dim", "Note")
SCANN_FILES = {
    "dataset.npy": ("Vector Payloads", "full precision reranking vectors; includes NPY header"),
    "hashed_dataset.npy": ("Quantized Vector Codes", "AH codes; includes NPY header"),
    "hashed_dataset_soar.npy": ("Quantized Vector Codes", "SOAR AH codes; includes NPY header"),
    "ah_codebook.pb": ("Quantizer Codebook", "serialized AH codebook"),
    "serialized_partitioner.pb": ("Partition Structure", "partitioner and its centroids"),
    "datapoint_to_token.npy": ("Partition Assignments", "point-to-partition mapping; includes NPY header"),
    "bfloat16_dataset.npy": ("Vector Payloads", "BF16 reranking vectors; includes NPY header"),
    "int8_dataset.npy": ("Quantized Vector Codes", "INT8 vectors; includes NPY header"),
    "int8_multipliers.npy": ("Quantizer Codebook", "INT8 scales; includes NPY header"),
    "dp_norms.npy": ("Original Metadata", "per-point norms; includes NPY header"),
}


def row(dataset, algorithm, path, component, size, total, nodes, dim, note):
    return {"Dataset": dataset, "IndexType": algorithm, "ActualVariant": algorithm,
            "IndexFile": str(path), "Component": component, "Bytes": size,
            "GiB": f"{size / 1024**3:.6f}", "FileSizeBytes": total,
            "N": nodes, "Dim": dim, "Note": note}


def scann_parts(path):
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"empty ScaNN index: {path}")
    parts = {}
    notes = {}
    for file in files:
        component, note = SCANN_FILES.get(
            file.name, ("Original Metadata", "serialized ScaNN configuration or auxiliary file"))
        parts[component] = parts.get(component, 0) + file.stat().st_size
        notes.setdefault(component, set()).add(note)
    return [(name, size, "; ".join(sorted(notes[name])))
            for name, size in parts.items()]


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path,
                        default=repo / "script/revision_baseline_build_results/summary.csv")
    parser.add_argument("--output", type=Path,
                        default=repo / "script/revision_baseline_build_results/storage_breakdown.csv")
    parser.add_argument("--build-dir", type=Path, default=repo / "build_lvc_ivf_pq")
    parser.add_argument("--skip-compile", action="store_true")
    args = parser.parse_args()
    if not args.skip_compile:
        subprocess.run(["cmake", "-S", str(repo), "-B", str(args.build_dir),
                        "-DHNSWLIB_EXAMPLES=ON"], check=True)
        subprocess.run(["cmake", "--build", str(args.build_dir), "--target",
                        "test_revision_baseline_storage_breakdown", "-j", "4"], check=True)

    with args.summary.open(newline="") as source:
        indexes = list(csv.DictReader(source))
    results = []
    for index in indexes:
        dataset, algorithm = index["dataset"], index["algorithm"]
        path = Path(index["index_path"])
        if not path.exists():
            raise FileNotFoundError(path)
        if algorithm == "scann":
            parts = scann_parts(path)
            total = sum(size for _, size, _ in parts)
        else:
            process = subprocess.run(
                [str(args.build_dir / "test_revision_baseline_storage_breakdown"),
                 "--index", str(path), "--algorithm", algorithm],
                text=True, capture_output=True, check=True)
            parsed = list(csv.DictReader(process.stdout.splitlines()))
            if not parsed:
                raise ValueError(f"empty breakdown: {path}")
            parts = [(item["component"], int(item["bytes"]),
                      "serialized Faiss component") for item in parsed]
            total = int(parsed[0]["file_size_bytes"])
        if total != int(index["index_file_bytes"]):
            raise ValueError(f"size changed since build: {path}")
        if sum(size for _, size, _ in parts) != total:
            raise ValueError(f"components do not sum to index size: {path}")
        for component, size, note in parts:
            results.append(row(dataset, algorithm, path, component, size, total,
                               index["nodes"], index["dim"], note))
        print(f"PASS {dataset} {algorithm}: {total} bytes across {len(parts)} components")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)
    print("Storage breakdown:", args.output)


if __name__ == "__main__":
    main()
