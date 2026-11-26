import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

from forecast_agents.runtime.reproducibility import file_hash

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rescore_reports", ROOT / "eval/rescore_reports.py")
rescorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rescorer)


class RescoreTests(unittest.TestCase):
    def fixture(self, root, with_probability=False):
        root.mkdir(parents=True)
        row = {"symbol": "A", "horizon": 1, "origin": "2020-01-03", "fit_cutoff": "2020-01-02",
               "target": 0.02, "prediction": 0.01, "effective_prediction": 0.01,
               "abstain": False, "fallback": False, "risk": 0.0}
        if with_probability:
            row["prob_up"] = 0.8
        report = {"rows": [row], "blocks": [{"fit_cutoff": "2020-01-02", "first_origin": "2020-01-03",
                    "last_origin": "2020-01-03", "origins": 1, "metrics": {}}]}
        for name, value in (("run.json", {"mode": "single", "status": "completed"}),
                            ("test_report.json", report), ("baseline_report.json", report)):
            (root / name).write_text(json.dumps(value))

    def test_legacy_report_gets_mse_without_fabricating_direction_probabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            self.fixture(root)
            before = {p.name: file_hash(p) for p in root.iterdir()}
            result = rescorer.rescore(root, root / "metrics_v2.json")
            condition = result["conditions"][0]
            self.assertAlmostEqual(condition["metrics"]["1"]["raw"]["mse"], 0.0001)
            self.assertIsNone(condition["metrics"]["1"]["direction"]["raw"]["cross_entropy"])
            self.assertAlmostEqual(condition["baseline_metrics"]["1"]["direction"]["raw"]["cross_entropy"], math.log(2))
            self.assertEqual(condition["blocks"][0]["metrics"], condition["metrics"])
            self.assertEqual(before, {name: file_hash(root / name) for name in before})
            with self.assertRaises(FileExistsError):
                rescorer.rescore(root, root / "metrics_v2.json")

    def test_comparison_retains_failed_condition_without_making_up_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root / "single", with_probability=True)
            comparison = {"status": "completed_with_failures", "runs": [
                {"mode": "single", "status": "completed", "output": "single", "repetition": 1},
                {"mode": "multi", "status": "failed", "output": "multi", "repetition": 1}]}
            (root / "comparison.json").write_text(json.dumps(comparison))
            result = rescorer.rescore(root, root / "metrics.json")
            self.assertEqual(result["source_status"], "completed_with_failures")
            self.assertIsNone(result["conditions"][1]["metrics"])
            self.assertAlmostEqual(result["conditions"][0]["metrics"]["1"]["direction"]["raw"]["cross_entropy"], -math.log(0.8))

    def test_running_comparison_is_not_rescored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "comparison.json").write_text('{"status": "running", "runs": []}')
            with self.assertRaisesRegex(ValueError, "finished"):
                rescorer.rescore(root, root / "metrics.json")
            self.assertFalse((root / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
