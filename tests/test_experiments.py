import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from forecast_agents.cli import run_experiment
from forecast_agents.config import DataConfig, ExperimentConfig, load_config
from forecast_agents.runtime.budget import BudgetLimits


class ExperimentTests(unittest.TestCase):
    def config(self):
        return ExperimentConfig(data=DataConfig(stocks=2, sessions=35, train_sessions=15,
                                                validation_sessions=10))

    def test_mock_single_and_multi_complete_with_identical_forecasting_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = {}
            for mode in ("single", "multi"):
                root = Path(directory) / mode
                summary = run_experiment(replace(self.config(), mode=mode), root)
                self.assertEqual(summary["status"], "completed", summary.get("error"))
                self.assertTrue(summary["mock"])
                self.assertEqual(sum(s["total_tokens"] for s in summary["stage_budgets"].values()),
                                 summary["budget"]["total_tokens"])
                outputs[mode] = summary["candidate_hash"]
                events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
                stages = [row["role"] for row in events if row["event"] == "stage_started"]
                self.assertEqual(stages, ["single"] if mode == "single" else
                                 ["diagnosis", "features", "modeling", "auditing"])
                frozen = next(i for i, row in enumerate(events) if row["event"] == "submission_frozen")
                self.assertFalse(any(row["event"] == "model_request" for row in events[frozen:]))
                visible = json.loads((root / "workspace/data/history.json").read_text())
                private = json.loads((root / "private/panel.json").read_text())
                self.assertLess(max(r["date"] for r in visible), max(r["date"] for r in private))
                report = json.loads((root / "test_report.json").read_text())
                self.assertEqual(set(report["metrics"]), {"1", "3", "5"})
                for metric in report["metrics"].values():
                    self.assertIsNotNone(metric["raw"]["mse"])
                    self.assertIsNotNone(metric["direction"]["raw"]["cross_entropy"])
                    self.assertEqual(metric["direction"]["raw"]["probability_count"], metric["count"])
                self.assertTrue(all(row["origin"] > max(r["date"] for r in visible)
                                    for row in report["rows"]))
            self.assertEqual(outputs["single"], outputs["multi"])

    def test_exhausted_budget_is_a_failed_run_not_a_successful_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "failed"
            summary = run_experiment(replace(self.config(), budget=BudgetLimits(max_llm_calls=1)), root)
            self.assertEqual(summary["status"], "failed")
            self.assertIn(summary["error_type"], {"BudgetExceeded", "StageBudgetExceeded"})
            self.assertFalse((root / "test_report.json").exists())
            with self.assertRaises(FileExistsError):
                run_experiment(self.config(), root)

    def test_unknown_config_keys_and_real_backend_local_execution_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            for content in ('unknown = true\n', '[llm]\nunknown = true\n',
                            '[llm]\nbackend = "anthropic"\nbase_url = "http://localhost/v1"\n',
                            '[budget]\nwall_seconds = nan\n'):
                path.write_text(content)
                with self.subTest(content=content), self.assertRaises(ValueError):
                    load_config(path)
