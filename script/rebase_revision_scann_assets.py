#!/usr/bin/env python3
"""Make existing ScaNN index asset paths relative to their index directory."""

import argparse
import csv
import os
from pathlib import Path
import tempfile

from google.protobuf import text_format
from scann.scann_ops import scann_assets_pb2


def rebase(index):
    manifest = index / "scann_assets.pbtxt"
    if not manifest.is_file():
        raise FileNotFoundError(f"ScaNN asset manifest missing: {manifest}")
    assets = scann_assets_pb2.ScannAssets()
    text_format.Parse(manifest.read_text(), assets)
    changed = 0
    for asset in assets.assets:
        filename = Path(asset.asset_path).name
        if not filename or not (index / filename).is_file():
            raise FileNotFoundError(f"Asset listed in {manifest} is missing: {filename}")
        if asset.asset_path != filename:
            asset.asset_path = filename
            changed += 1
    if changed:
        # Do not leave a partially written manifest if the process is interrupted.
        with tempfile.NamedTemporaryFile(mode="w", dir=index, prefix=".scann_assets_",
                                         delete=False) as target:
            target.write(text_format.MessageToString(assets))
            temporary = Path(target.name)
        try:
            os.replace(temporary, manifest)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"{index}: rebased {changed}/{len(assets.assets)} asset paths")


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path,
                        default=repo / "script/revision_baseline_build_results/summary.csv")
    parser.add_argument("--index", type=Path,
                        help="Repair one index; by default repair all ScaNN indexes in --summary")
    args = parser.parse_args()
    if args.index:
        indexes = [args.index]
    else:
        with args.summary.open(newline="") as source:
            indexes = [Path(row["index_path"]) for row in csv.DictReader(source)
                       if row["algorithm"] == "scann"]
    if not indexes:
        raise ValueError("No ScaNN indexes found")
    for index in indexes:
        rebase(index)


if __name__ == "__main__":
    main()
