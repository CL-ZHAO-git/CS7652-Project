import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from forecast_agents.config import config_record, load_config
from forecast_agents.runtime.reproducibility import file_hash

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("comparison_driver", ROOT / "eval/run_comparison.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class ComparisonTests(unittest.TestCase):
    def prepare(self, root, repetitions=1):
        # Small trusted fixture run; no network, Docker or restricted data.
        config = config_record(load_config(ROOT / "configs/demo.toml"))
        config["data"].update(stocks=2, sessions=35, train_sessions=15, validation_sessions=10)
        path = root / "config.json"
        path.write_text(json.dumps(config))
        return driver.prepare(path, root / "comparison", repetitions)

    def test_preparation_does_not_load_credentials_docker_or_api_and_orders_alternate(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(driver, "Executor", side_effect=AssertionError("Docker accessed")), \
                    patch.object(driver, "key_environment", side_effect=AssertionError("Credential read")):
                output = self.prepare(Path(directory), repetitions=2)
            protocol = json.loads((output / "protocol.json").read_text())
            self.assertEqual(protocol["orders"], [["single", "multi"], ["multi", "single"]])
            self.assertEqual(json.loads((output / "comparison.json").read_text())["status"], "prepared")
            driver.verify(output, protocol)

    def test_changed_frozen_source_is_rejected_before_any_run(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.prepare(Path(directory))
            source = output / "protocol/src/forecast_agents/agents/runner.py"
            source.write_text(source.read_text() + "\n# changed\n")
            with self.assertRaisesRegex(ValueError, "framework"):
                driver.execute(output)

    def test_json_dataset_is_snapshotted_and_modification_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "panel.json"
            data.write_text('[{"private": "fixture-not-evaluated"}]')
            config = config_record(load_config(ROOT / "configs/demo.toml"))
            config["data"].update(source="json", path=str(data))
            path = root / "config.json"
            path.write_text(json.dumps(config))
            output = driver.prepare(path, root / "comparison", 1)
            frozen = output / "private/panel.json"
            self.assertEqual(file_hash(frozen), file_hash(data))
            data.write_text("original changed")
            protocol = json.loads((output / "protocol.json").read_text())
            driver.verify(output, protocol)  # independent frozen copy
            frozen.write_text("frozen changed")
            with self.assertRaisesRegex(ValueError, "dataset"):
                driver.verify(output, protocol)

    def test_both_conditions_use_frozen_framework_and_matching_config(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.prepare(Path(directory))
            state = driver.execute(output)
            self.assertEqual(state["status"], "completed", state)
            self.assertEqual([r["mode"] for r in state["runs"]], ["single", "multi"])
            summaries = [json.loads((output / r["output"] / "run.json").read_text()) for r in state["runs"]]
            configs = [s["config"] for s in summaries]
            self.assertEqual(configs[0].pop("mode"), "single")
            self.assertEqual(configs[1].pop("mode"), "multi")
            self.assertEqual(configs[0], configs[1])
            self.assertEqual(summaries[0]["candidate_hash"], summaries[1]["candidate_hash"])
            with self.assertRaisesRegex(ValueError, "unstarted"):
                driver.execute(output)

    def test_failed_condition_is_retained_and_does_not_hide_other_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = self.prepare(root)
            def failure(command, **kwargs):
                destination = Path(command[-1])
                destination.mkdir(parents=True)
                (destination / "run.json").write_text(json.dumps({"status": "failed", "error": "fixture failure"}))
                return type("Result", (), {"returncode": 1, "stdout": "{}", "stderr": ""})()
            with patch.object(driver.subprocess, "run", side_effect=failure):
                state = driver.execute(output)
            self.assertEqual(state["status"], "completed_with_failures")
            self.assertEqual(len(state["runs"]), 2)
            self.assertTrue(all(r["status"] == "failed" for r in state["runs"]))

    def test_midrun_protocol_change_invalidates_and_retains_the_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.prepare(Path(directory))
            def changed(command, **kwargs):
                destination = Path(command[-1])
                destination.mkdir(parents=True)
                (destination / "run.json").write_text(json.dumps({"status": "completed"}))
                with (output / "protocol/execution_config.json").open("a") as stream:
                    stream.write("\n ")
                return type("Result", (), {"returncode": 0, "stdout": "{}", "stderr": ""})()
            with patch.object(driver.subprocess, "run", side_effect=changed):
                state = driver.execute(output)
            self.assertEqual(state["status"], "failed")
            self.assertEqual(len(state["runs"]), 1)
            self.assertEqual(state["runs"][0]["status"], "invalid_protocol")
            self.assertTrue((output / state["runs"][0]["output"] / "run.json").is_file())


if __name__ == "__main__":
    unittest.main()
