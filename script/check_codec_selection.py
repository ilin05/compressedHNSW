#!/usr/bin/env python3
"""Check completed profiling outputs and selector/oracle bookkeeping (no rebuild)."""
import argparse
import math
from pathlib import Path
from test_codec_selection import CODECS, choose, rank, read_csv


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=repo / "script/codec_selection_results")
    args = parser.parse_args()
    assert choose({"a": 2, "b": .9}, {"a": 2, "b": 1})[0] == "a"
    assert choose({"a": .5, "b": .9}, {"a": 2, "b": 1})[0] == "raw"
    assert rank({"a": 1, "b": 1, "c": 2}) == {"a": 1, "b": 1, "c": 3}
    # An empty joint-top-2 intersection is legitimate, not a selector failure.
    ratio = dict(a=4, b=3, c=2, d=1.5)
    time = dict(a=4, b=3, c=2, d=1)
    _, sr, tr = choose(ratio, time)
    assert not any(sr[c] <= 2 and tr[c] <= 2 for c in ratio)
    predictions = read_csv(args.results / "predictions.csv")
    summaries = read_csv(args.results / "selection_summary.csv")
    datasets = sorted({row["Dataset"] for row in predictions})
    assert len(predictions) == len(datasets) * 8
    assert len(summaries) == len(datasets) * 4
    for dataset in datasets:
        rows = [r for r in predictions if r["Dataset"] == dataset]
        ids = read_csv(args.results / dataset / "sample_ids.csv")
        assert len({int(r["InternalId"]) for r in ids}) == len(ids)
        for chain in (2, -1):
            records = {r["Codec"]: r for r in rows if int(r["ChainMaxLength"]) == chain}
            assert set(records) == set(CODECS)
            for codec, r in records.items():
                n, dim, count = int(r["Nodes"]), int(r["Dim"]), int(r["SampleCount"])
                assert count == len(ids) and 0 < count <= n
                assert all(0 <= int(x["InternalId"]) < n for x in ids)
                payload = float(r["SamplePayloadBytes"]) * n / count
                assert math.isclose(payload, float(r["PredictedPayloadBytes"]), rel_tol=1e-9)
                assert math.isclose(n * dim * 4 / payload,
                                    float(r["PredictedPayloadRatioF32"]), rel_tol=1e-9)
                assert float(r["PredictedEncodeSeconds"]) > 0
                if codec == "ALP":
                    assert float(r["AverageDepth"]) == 0 and int(r["Roots"]) == 0
                    assert float(r["ForestSeconds"]) == 0
                elif chain == 2:
                    assert 0 <= float(r["AverageDepth"]) <= 1
        for pool, codecs in (("all", CODECS), ("differential", CODECS[1:])):
            for chain in (2, -1):
                records = {r["Codec"]: r for r in rows if int(r["ChainMaxLength"]) == chain}
                expected = choose({c: float(records[c]["PredictedPayloadRatioF32"]) for c in codecs},
                                  {c: float(records[c]["PredictedForestEncodeSeconds"]) for c in codecs})[0]
                summary = next(r for r in summaries if r["Dataset"] == dataset and
                               r["Pool"] == pool and int(r["ChainMaxLength"]) == chain)
                assert summary["SelectedCodec"] == expected
                if summary["ValidationStatus"] == "missing_oracle":
                    assert summary["ActualPayloadRank"] == "" and summary["JointTop2Success"] == ""
                elif int(summary["JointTop2Success"]):
                    assert int(summary["ActualPayloadRank"]) <= 2 and int(summary["ActualTimeRank"]) <= 2
        if any(r.get("HistoricalShapeMatch") == "0" for r in summaries if r["Dataset"] == dataset):
            print(f"WARNING {dataset}: sample input shape differs from historical oracle; ranks are not same-size evidence")
        print(f"PASS {dataset}: shared samples, byte extrapolation, chain/ALP semantics, selection")
    print("PASS codec selection output checks")


if __name__ == "__main__":
    main()
