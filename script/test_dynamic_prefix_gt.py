"""Small exhaustive regression test for all prefix-GT checkpoints."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


class PrefixGroundTruthTest(unittest.TestCase):
    def test_incremental_ground_truth_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(20261015)
            train = rng.normal(size=(57, 5)).astype("<f4")
            train[40] = train[3]  # Duplicate across prefix boundaries; smaller ID wins.
            query = np.vstack([train[40], train[-1], rng.normal(size=(5, 5))]).astype("<f4")
            dataset = "sift-128-euclidean"
            for suffix, values in [("train", train), ("test", query)]:
                rows = np.empty((len(values), 6), dtype="<f4")
                rows[:, 0].view("<i4")[:] = 5
                rows[:, 1:] = values
                rows.tofile(root / f"{dataset}_{suffix}.fvecs")
            command = [sys.executable, str(Path(__file__).with_name("compute_dynamic_prefix_gt.py")),
                       "--datasets", dataset, "--data-dir", str(root), "--out-dir", str(root / "gt"),
                       "--train-block", "7", "--query-block", "3", "--threads", "1"]
            subprocess.run(command, check=True, capture_output=True, text=True)
            output = root / "gt" / dataset
            for percent in range(50, 101, 5):
                count = len(train) * percent // 100
                distances = ((query.astype("f8")[:, None] - train[:count].astype("f8")) ** 2).sum(axis=2)
                expected_ids = distances.argmin(axis=1)
                stem = output / f"gt_prefix{percent:03d}"
                ids = np.fromfile(str(stem) + ".ivecs", dtype="<i4").reshape(-1, 2)
                np.testing.assert_array_equal(ids[:, 0], 1)
                np.testing.assert_array_equal(ids[:, 1], expected_ids)
                np.testing.assert_allclose(np.load(str(stem) + "_squared_l2.npy"),
                                           distances.min(axis=1), atol=1e-12, rtol=1e-12)
                packed = np.fromfile(str(stem) + ".fvecs", dtype="<f4").reshape(-1, 2)
                np.testing.assert_array_equal(packed[:, 0].copy().view("<i4"), 1)
                np.testing.assert_allclose(packed[:, 1], distances.min(axis=1), atol=1e-6)
                self.assertTrue(np.all(ids[:, 1] < count))
            manifest = output / "manifest.json"
            self.assertEqual(len(json.loads(manifest.read_text())["checkpoints"]), 11)
            timestamp = manifest.stat().st_mtime_ns
            result = subprocess.run(command + ["--resume"], check=True, capture_output=True, text=True)
            self.assertIn("Reuse", result.stdout)
            self.assertEqual(manifest.stat().st_mtime_ns, timestamp)


if __name__ == "__main__":
    unittest.main()
