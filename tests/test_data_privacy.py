import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from forecast_agents.agents.runner import AgentRunner
from forecast_agents.config import load_config
from forecast_agents.cli import run_experiment
from forecast_agents.llm.types import Message, ModelResponse, ToolCall, Usage, as_json
from forecast_agents.runtime.budget import Budget, BudgetLimits
from forecast_agents.runtime.events import EventLog
from forecast_agents.runtime.executor import ExecutionResult
from forecast_agents.runtime.errors import CandidateExecutionError
from forecast_agents.runtime.errors import PredictionContractError
from forecast_agents.tasks.panel import Panel, Bar, TaskSpec, synthetic_panel
from forecast_agents.tools.privacy import DataAccessError, describe_panel, execution_summary
from forecast_agents.tools.workspace import WorkspaceTools

SECRET = "PRIVATE_ROW_SENTINEL_938172"


class PrivacyTests(unittest.TestCase):
    def test_oversized_read_returns_public_bounds_and_recovers_with_pagination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = self.tools(root)
            source = "# Authored source\n" * 700
            tools.execute("write_file", {"path": "solution/pipeline.py", "content": source})
            definition = next(t for t in tools.definitions if t["name"] == "read_file")
            self.assertEqual(definition["parameters"]["properties"]["max_chars"]["maximum"], 8000)
            self.assertIn("8000", definition["description"])
            first = tools.execute("read_file", {"path": "solution/pipeline.py"})
            with self.assertRaises(DataAccessError) as caught:
                tools.execute("read_file", {"path": "solution/pipeline.py", "offset": first["next_offset"], "max_chars": 12000})
            feedback = tools.error_result(caught.exception)
            self.assertIn("8000", feedback["error"])
            self.assertIn("next_offset", feedback["error"])
            self.assertNotIn(SECRET, json.dumps(feedback))
            last = tools.execute("read_file", {"path": "solution/pipeline.py", "offset": first["next_offset"], "max_chars": 8000})
            self.assertEqual(first["text"] + last["text"], source)
            self.assertIsNone(last["next_offset"])
            with self.assertRaises(DataAccessError):
                tools.execute("read_file", {"path": "data/history.json", "max_chars": 8000})

    def tools(self, root, executor=None):
        (root / "data").mkdir()
        (root / "data/history.json").write_text(SECRET)
        (root / "task.json").write_text('{"protocol": "public task"}')
        (root / "PIPELINE_CONTRACT.md").write_text("Public contract")
        panel = synthetic_panel(2, 40)
        task = TaskSpec(panel.dates[19], panel.dates[29])
        budget = Budget(BudgetLimits())
        return WorkspaceTools(root, panel, task, executor, budget, EventLog(root / "events.jsonl"), summaries_only=True)

    def test_raw_data_reports_and_runtime_created_names_are_not_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = self.tools(root)
            (root / "report.json").write_text(SECRET)
            (root / "validation.json").write_text(json.dumps({"rows": [SECRET]}))
            (root / (SECRET + ".txt")).write_text(SECRET)
            for name in ("data/history.json", "report.json", "validation.json"):
                with self.subTest(path=name), self.assertRaises(DataAccessError):
                    tools.execute("read_file", {"path": name})
            self.assertNotIn(SECRET, json.dumps(tools.execute("list_files", {})))

    def test_only_unchanged_agent_source_can_be_read_and_task_snapshot_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = self.tools(root)
            tools.execute("write_file", {"path": "solution/features.py", "content": "VALUE = 1"})
            self.assertEqual(tools.execute("read_file", {"path": "solution/features.py"})["text"], "VALUE = 1")
            (root / "solution/features.py").write_text(SECRET)
            with self.assertRaises(DataAccessError):
                tools.execute("read_file", {"path": "solution/features.py"})
            (root / "task.json").write_text(SECRET)
            self.assertEqual(tools.execute("read_file", {"path": "task.json"})["text"], '{"protocol": "public task"}')

    def test_stdout_traceback_values_are_local_only_but_error_type_and_location_are_returned(self):
        class FakeExecutor:
            def execute(self, *args, **kwargs):
                return ExecutionResult(1, SECRET, '  File "/workspace/check.py", line 3\n    ' + SECRET + '\nValueError: ' + SECRET, False, 0.1)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = self.tools(root, FakeExecutor())
            tools.execute("write_file", {"path": "check.py", "content": "print('local check')"})
            result = tools.execute("run_python", {"path": "check.py"})
            self.assertNotIn(SECRET, json.dumps(result))
            self.assertTrue(result["output_withheld"])
            self.assertEqual(result["error_type"], "ValueError")
            self.assertEqual(result["locations"], [{"path": "check.py", "line": 3}])
            self.assertIn(SECRET, tools.events.path.read_text())  # local audit record

    def test_descriptions_use_only_development_history_and_have_no_price_series(self):
        panel = synthetic_panel(2, 40)
        task = TaskSpec(panel.dates[19], panel.dates[29])
        original = describe_panel(panel, task)
        poisoned = [Bar(r["symbol"], r["date"], r["close"] * (100 if r["date"] > task.validation_end else 1))
                    for r in panel.history(panel.dates[-1])]
        self.assertEqual(original, describe_panel(Panel(poisoned), task))
        self.assertEqual(original["rows"], 60)
        self.assertEqual(original["pooled_daily_returns"]["count"], 58)
        self.assertNotIn("prices", original)
        self.assertNotIn("history", original)
        self.assertNotIn("by_date", original)

    def test_validation_summary_is_cached_from_controller_not_read_from_runtime_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir()
            tools = self.tools(root)
            tools.execute("write_file", {"path": "solution/pipeline.py", "content": "# candidate"})
            report = {"metrics": {"1": {"count": 20, "served": {"mae": 0.0123456789}, "risk_coverage": []}},
                      "blocks": [], "rows": [{"target": SECRET}]}
            with patch("forecast_agents.tools.workspace.evaluate", return_value=report):
                summary = tools.execute("validate_candidate", {})
            self.assertNotIn(SECRET, json.dumps(summary))
            self.assertEqual(summary["metrics"]["1"]["served"]["mae"], 0.012346)
            (root / "validation.json").write_text(SECRET)
            self.assertEqual(summary, tools.execute("read_validation_summary", {}))

    def test_model_requests_do_not_contain_raw_reads_stdout_or_validation_exceptions(self):
        class FakeExecutor:
            def execute(self, *args, **kwargs):
                return ExecutionResult(0, SECRET, SECRET, False, 0.1)

        class Backend:
            name = "privacy-fixture"

            def __init__(self):
                self.requests = []
                self.actions = iter([
                    ("read_file", {"path": "data/history.json"}),
                    ("write_file", {"path": "check.py", "content": "# local check"}),
                    ("run_python", {"path": "check.py"}),
                    ("write_file", {"path": "solution/pipeline.py", "content": "# candidate"}),
                    ("validate_candidate", {}),
                    ("finish_stage", {"status": "rejected", "artifacts": [], "findings": [], "recommendations": []}),
                ])

            def generate(self, request):
                self.requests.append(request)
                name, args = next(self.actions)
                return ModelResponse(Message("assistant", tool_calls=(ToolCall(str(len(self.requests)), name, args),)),
                                     "tool_calls", Usage(100, 1))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir()
            tools = self.tools(root, FakeExecutor())
            backend = Backend()
            runner = AgentRunner(backend, tools, tools.budget, tools.events)
            with patch("forecast_agents.tools.workspace.evaluate", side_effect=RuntimeError(SECRET)):
                runner.run_stage("diagnosis", [])
            self.assertEqual(len(backend.requests), 6)
            serialized = json.dumps([as_json(r) for r in backend.requests])
            self.assertNotIn(SECRET, serialized)
            self.assertIn(SECRET, tools.events.path.read_text())

    def test_json_experiment_automatically_enables_summary_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            panel = synthetic_panel(2, 40)
            path = root / "panel.json"
            path.write_text(json.dumps(panel.history(panel.dates[-1])))
            demo = load_config(Path(__file__).parents[1] / "configs/demo.toml")
            config = replace(demo, data=replace(demo.data, source="json", path=str(path),
                                                train_sessions=20, validation_sessions=10))
            result = run_experiment(config, root / "run")
            self.assertEqual(result["status"], "completed", result.get("error"))
            self.assertEqual(result["llm_data_access"], "summaries_only")
            events = [json.loads(line) for line in (root / "run/events.jsonl").read_text().splitlines()]
            results = [e["result"] for e in events if e["event"] == "tool_result" and "returncode" in e["result"]]
            self.assertTrue(results)
            self.assertTrue(all(e["output_withheld"] and not e["stdout"] and not e["stderr"] for e in results))

    def test_real_development_config_uses_json_docker_and_development_dates(self):
        c = load_config(Path(__file__).parents[1] / "configs/deepseek_crsp_dev.toml")
        self.assertEqual(c.data.source, "json")
        self.assertIn("2005_2014", c.data.path)
        self.assertEqual(c.data.validation_end, "2011-12-31")
        self.assertEqual(c.execution.mode, "docker")
        self.assertEqual(c.budget.wall_seconds, 7200)

    def test_builtin_none_type_errors_return_safe_debugging_hints(self):
        cases = (("must be real number, not NoneType", "numeric_operation_on_none"),
                 ("'<' not supported between instances of 'NoneType' and 'NoneType'", "incompatible_comparison_types"),
                 ("'NoneType' object is not subscriptable", "not_subscriptable"))
        for message, code in cases:
            with self.subTest(message=message):
                result = execution_summary({"returncode": 1, "timed_out": False, "elapsed_seconds": 0.1,
                    "stdout": SECRET, "stderr": "TypeError: " + message}, [])
                self.assertEqual(result["diagnostic_code"], code)
                self.assertIn("NoneType", result["operand_types"])
                self.assertNotIn(SECRET, json.dumps(result))

    def test_nonwhitelisted_or_value_bearing_errors_remain_withheld(self):
        for message in ("must be real number, not NoneType " + SECRET,
                        "'<' not supported between instances of '" + SECRET + "' and 'float'",
                        "private value=" + SECRET):
            with self.subTest(template=message[:15]):
                result = execution_summary({"returncode": 1, "timed_out": False, "elapsed_seconds": 0.1,
                    "stdout": SECRET, "stderr": "TypeError: " + message}, [])
                self.assertNotIn(SECRET, json.dumps(result))
                self.assertNotIn("diagnostic_code", result)

    def test_missing_pipeline_returns_actionable_precondition_without_runtime_values(self):
        with tempfile.TemporaryDirectory() as directory:
            tools = self.tools(Path(directory))
            for tool in ("audit_candidate", "validate_candidate"):
                with self.subTest(tool=tool), self.assertRaises(DataAccessError) as caught:
                    tools.execute(tool, {})
                result = tools.error_result(caught.exception)
                self.assertIn("pipeline.py", result["error"])
                self.assertNotIn(SECRET, json.dumps(result))

    def test_prediction_contract_errors_expose_fixed_actionable_messages_only(self):
        with tempfile.TemporaryDirectory() as directory:
            tools = self.tools(Path(directory))
            result = tools.error_result(PredictionContractError("New candidates must include prob_up"))
            self.assertTrue(result["is_error"])
            self.assertIn("prob_up", result["error"])

    def test_candidate_validation_failure_returns_safe_diagnostics_and_keeps_raw_output_local(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir()
            tools = self.tools(root)
            tools.execute("write_file", {"path": "solution/pipeline.py", "content": "state = {'models': {}}\n"})
            stderr = '  File "/workspace/solution/pipeline.py", line 8\n    ' + SECRET + "\nKeyError: 'models'"
            failure = CandidateExecutionError("fit", ExecutionResult(1, SECRET, stderr, False, 0.1))
            with patch("forecast_agents.tools.workspace.evaluate", side_effect=failure):
                with self.assertRaises(CandidateExecutionError) as caught:
                    tools.execute("validate_candidate", {})
            result = tools.error_result(caught.exception)
            self.assertEqual(result["error_type"], "KeyError")
            self.assertEqual(result["missing_key"], "models")
            self.assertEqual(result["operation"], "fit")
            self.assertEqual(result["locations"], [{"path": "solution/pipeline.py", "line": 8}])
            self.assertNotIn(SECRET, json.dumps(result))
            self.assertIn(SECRET, tools.events.path.read_text())
            self.assertIsNone(tools.last_validation_hash)

    def test_only_unchanged_authored_literals_and_public_horizons_are_debug_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = self.tools(root)
            tools.execute("write_file", {"path": "check.py", "content": "state['models']\n"})
            recorded = {"returncode": 1, "timed_out": False, "elapsed_seconds": 0.1, "stdout": SECRET}
            for key in ("models", "3", SECRET):
                with self.subTest(key=key):
                    result = execution_summary({**recorded, "stderr": "KeyError: " + repr(key)}, [], tools.known_debug_keys())
                    self.assertEqual(result.get("missing_key"), key if key != SECRET else None)
                    self.assertNotIn(SECRET, json.dumps(result))
            (root / "check.py").write_text("state['" + SECRET + "']\n")
            self.assertNotIn("models", tools.known_debug_keys())
            self.assertNotIn(SECRET, tools.known_debug_keys())

    def test_relative_import_diagnostic_matches_only_exact_fixed_message(self):
        recorded = {"returncode": 1, "timed_out": False, "elapsed_seconds": 0.1, "stdout": SECRET}
        message = "attempted relative import with no known parent package"
        result = execution_summary({**recorded, "stderr": "ImportError: " + message}, [])
        self.assertEqual(result["diagnostic_code"], "relative_import_without_package")
        result = execution_summary({**recorded, "stderr": "ImportError: " + message + SECRET}, [])
        self.assertNotIn("diagnostic_code", result)
        self.assertNotIn(SECRET, json.dumps(result))


if __name__ == "__main__":
    unittest.main()
