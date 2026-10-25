#!/usr/bin/env python3
"""Run all four launchers concurrently on small real HNSW indexes, not performance data."""
import csv
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


class OptimizedWorkloadTests(unittest.TestCase):
    def test_four_concurrent_launchers_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = Path(os.environ.get("DYNAMIC_TEST_BINARY", root / "benchmark"))
            if "DYNAMIC_TEST_BINARY" not in os.environ:
                subprocess.run([os.environ.get("CXX", "g++"), "-std=c++17", "-O2", "-fopenmp",
                                "-I", str(ROOT), str(ROOT / "examples/cpp/test_dynamic_update_workloads.cpp"),
                                "-o", str(binary)], check=True)
            dataset = "sift-128-euclidean"
            data, gt = root / "data", root / "gt" / dataset
            data.mkdir(); gt.mkdir(parents=True)
            rng = random.Random(42)
            base = [[rng.randrange(256) for _ in range(128)] for _ in range(600)]
            queries = [[rng.randrange(256) for _ in range(128)] for _ in range(16)]
            for name, values in (("train", base), ("test", queries)):
                with (data / f"{dataset}_{name}.fvecs").open("wb") as handle:
                    for vector in values:
                        handle.write(struct.pack("<i128f", 128, *vector))
            distances = [[sum((a-b)**2 for a, b in zip(query, vector)) for vector in base]
                         for query in queries]
            for percent in range(50, 101, 5):
                contents = b"".join(struct.pack("<ii", 1, min(range(600 * percent // 100),
                                   key=lambda i: d[i])) for d in distances)
                (gt / f"gt_prefix{percent:03d}.ivecs").write_bytes(contents)
                if percent == 100:
                    (data / f"{dataset}_neighbors.ivecs").write_bytes(contents)
            def info(name):
                path = data / f"{dataset}_{name}.fvecs"
                return dict(bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
            (gt / "manifest.json").write_text(json.dumps(dict(config=dict(
                train=info("train"), queries=info("test"), vectors=600, dimension=128, query_count=16))))
            jobs = []
            for experiment in ("repeated", "growing"):
                for alpha in ("01", "02"):
                    tag = experiment + "_alpha" + alpha
                    output = root / tag
                    command = ["bash", str(ROOT / "script" / f"test_dynamic_updates_{tag}.sh")]
                    env = dict(os.environ, DATA_DIR=str(data), GT_ROOT=str(root / "gt"),
                               OUT=str(output), BIN=str(binary), DATASETS=dataset, SKIP_BUILD="1", RESUME="1")
                    for key in ("BATCHES", "REPEAT_RATIO", "SEARCH_EVERY", "EF_SEARCH", "SEARCH_ROUNDS",
                                "COUNT", "QUERY_COUNT", "M", "EF_CONSTRUCTION", "SEED", "VERIFY_SIZE"):
                        env.pop(key, None)
                    log = (root / (tag + ".log")).open("w")
                    jobs.append((subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT),
                                 log, command, env, experiment, alpha, output))
            for process, log, command, env, experiment, alpha, output in jobs:
                process.wait(timeout=180)
                log.close()
                self.assertEqual(process.returncode, 0, (root / (experiment + "_alpha" + alpha + ".log")).read_text())
                config = json.loads((output / dataset / "config.json").read_text())
                mode = "alp_a" + alpha
                self.assertEqual(config["insert_modes"], ["raw_f32", mode])
                self.assertTrue(config["insert_decode_cache"] and config["insert_upper_tls"])
                self.assertEqual(config["ef_search"], [40])
                self.assertEqual(config["search_checkpoints"], list(range(0, 21, 2)) if experiment == "repeated" else list(range(11)))
                update, search = rows(output / "all_updates.csv"), rows(output / "all_search.csv")
                self.assertEqual(len(update), 42 if experiment == "repeated" else 22)
                self.assertEqual(len(search), 99)
                for m in ("raw_f32", mode):
                    last = [r for r in update if r["insert_mode"] == m][-1]
                    self.assertEqual(int(last["active_nodes"]), 600)
                    self.assertEqual(int(last["physical_nodes"]), 960 if experiment == "repeated" else 600)
                    self.assertEqual(int(last["deleted_nodes"]), 360 if experiment == "repeated" else 0)
                alp = [r for r in update if r["insert_mode"] == mode and int(r["batch"]) > 0]
                self.assertGreater(sum(int(r["decode_cache_hits"]) for r in alp), 0)
                self.assertGreater(sum(int(r["upper_pq_candidates"]) for r in alp), 0)
                self.assertTrue(all(float(r["max_abs_error"]) == 0 for r in update))
                self.assertEqual({r["query_mode"] for r in search if r["insert_mode"] == mode}, {"tls_off", "tls_0.2"})
                stamps = {path: path.stat().st_mtime_ns for path in (output / dataset).glob("*/updates.csv")}
                resumed = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                self.assertEqual(resumed.returncode, 0, resumed.stdout)
                self.assertEqual(stamps, {path: path.stat().st_mtime_ns for path in stamps})
                self.assertNotIn("Running:", resumed.stdout)


if __name__ == "__main__":
    unittest.main()
