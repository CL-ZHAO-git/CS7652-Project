import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from platform_support import pid_alive, symlinks_supported

from forecast_agents.agents.runner import AgentRunner
from forecast_agents.llm.types import Message, ModelResponse, ToolCall, Usage
from forecast_agents.orchestration.policies import run_policy
from forecast_agents.runtime.budget import Budget, BudgetExceeded, BudgetLimits
from forecast_agents.runtime.events import EventLog
from forecast_agents.runtime.executor import Executor
from forecast_agents.tools.workspace import WorkspaceTools


class ScriptedBackend:
    name = "scripted-test"

    def __init__(self, responses):
        self.responses = iter(responses)

    def generate(self, request):
        return next(self.responses)


def response(name, arguments, usage=Usage(1, 1), stop_reason="tool_calls"):
    return ModelResponse(Message("assistant", tool_calls=(ToolCall("call", name, arguments),)),
                         stop_reason, usage)


class RuntimeTests(unittest.TestCase):
    def test_shared_call_and_tool_budgets_and_unknown_usage(self):
        budget = Budget(BudgetLimits(max_llm_calls=1, max_tool_calls=1))
        budget.begin_call()
        budget.charge(Usage(2, 3))
        budget.begin_tool()
        with self.assertRaises(BudgetExceeded):
            budget.begin_call()
        with self.assertRaises(BudgetExceeded):
            budget.begin_tool()
        self.assertEqual(budget.total_tokens, 5)
        with self.assertRaisesRegex(BudgetExceeded, "Unknown"):
            budget.charge(Usage())
        self.assertEqual(budget.unknown_usage_calls, 1)

    def test_wall_budget_uses_monotonic_elapsed_time(self):
        now = [10.0]
        budget = Budget(BudgetLimits(wall_seconds=2), clock=lambda: now[0])
        now[0] = 12.1
        with self.assertRaisesRegex(BudgetExceeded, "Wall"):
            budget.begin_call()

    def test_workspace_tools_reject_traversal_and_protected_input_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            tools = WorkspaceTools(root, None, None, None, None, None)
            for path in ("../outside", "/tmp/absolute", "data/file.json", "task.json"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    tools.execute("write_file", {"path": path, "content": "x"})

    @unittest.skipUnless(symlinks_supported(), "creating symlinks needs Developer Mode on Windows")
    def test_workspace_tools_reject_symlink_escape(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            tools = WorkspaceTools(root, None, None, None, None, None)
            (root / "outside").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                tools.execute("read_file", {"path": "outside/file"})

    def test_tool_paths_and_protections_are_the_same_on_every_os(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            tools = WorkspaceTools(root, None, None, None, None, None)
            # Case variants would overwrite inputs on case-insensitive macOS/Windows file systems.
            for path in ("DATA/history.json", "Task.JSON", "solution/PIPELINE_CONTRACT.md",
                         "solution\\pipeline.py", "C:/outside.py", "c:outside.py"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    tools.execute("write_file", {"path": path, "content": "x"})
            tools.execute("write_file", {"path": "solution/lib/helper.py", "content": "a\nb\n"})
            self.assertEqual(tools.execute("list_files", {})["files"], ["solution/lib/helper.py"])
            self.assertEqual((root / "solution/lib/helper.py").read_bytes(), b"a\nb\n")  # no CRLF

    def test_descendant_processes_do_not_outlive_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "spawn.py").write_text(
                "import subprocess, sys\n"
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                "print(child.pid)\n")
            result = Executor().execute(root, "spawn.py", timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            pid = int(result.stdout.strip())
            deadline = time.monotonic() + 5
            while pid_alive(pid) and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertFalse(pid_alive(pid))

    @unittest.skipIf(sys.platform == "darwin", "macOS does not enforce RLIMIT_AS")
    def test_local_memory_limit_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "allocate.py").write_text("block = bytearray(512 * 1024 * 1024)\nprint('allocated')\n")
            result = Executor(memory_mb=256).execute(root, "allocate.py", timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("MemoryError", result.stderr)

    def test_generated_code_prints_utf8_on_every_os(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "unicode.py").write_text("print('\\u2265 \\u2014 \\u00e9')\n")
            result = Executor().execute(root, "unicode.py")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "\u2265 \u2014 \u00e9")

    def test_executor_times_out_and_does_not_forward_api_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sleep.py").write_text("import time\ntime.sleep(10)\n")
            result = Executor().execute(root, "sleep.py", timeout=0.1)
            self.assertTrue(result.timed_out)
            (root / "env.py").write_text("import os\nprint('FA_TEST_SECRET' in os.environ)\n")
            with patch.dict("os.environ", {"FA_TEST_SECRET": "fixture-only"}):
                result = Executor().execute(root, "env.py")
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout.strip(), "False")

    def test_generated_scripts_can_import_local_helpers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "solution").mkdir()
            (root / "solution/features.py").write_text("VALUE = 42\n")
            (root / "solution/train.py").write_text("from features import VALUE\nprint(VALUE)\n")
            result = Executor().execute(root, "solution/train.py")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "42")

    def test_length_stop_and_token_overshoot_do_not_execute_tools(self):
        for model_response in (response("write_file", {"path": "bad", "content": "x"},
                                        stop_reason="length"),
                               response("write_file", {"path": "bad", "content": "x"}, Usage(100, 100))):
            with self.subTest(stop=model_response.stop_reason), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                budget = Budget(BudgetLimits(max_total_tokens=50))
                events = EventLog(root / "events.jsonl")
                tools = WorkspaceTools(root, None, None, None, budget, events)
                runner = AgentRunner(ScriptedBackend([model_response]), tools, budget, events)
                with self.assertRaises(RuntimeError):
                    runner.run_stage("diagnosis", [])
                self.assertFalse((root / "bad").exists())
                self.assertEqual(budget.tool_calls, 0)

    def test_unknown_usage_is_logged_and_stops_before_tool_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            budget = Budget(BudgetLimits())
            events = EventLog(root / "events.jsonl")
            tools = WorkspaceTools(root, None, None, None, budget, events)
            runner = AgentRunner(ScriptedBackend([
                response("write_file", {"path": "bad", "content": "x"}, Usage())]), tools, budget, events)
            with self.assertRaises(BudgetExceeded):
                runner.run_stage("diagnosis", [])
            self.assertFalse((root / "bad").exists())
            records = [json.loads(line) for line in events.path.read_text().splitlines()]
            self.assertTrue(any(row["event"] == "model_response" for row in records))

    def test_revision_limit_stops_cycles(self):
        class RevisingRunner:
            def run_stage(self, role, handoffs):
                return {"role": role, "status": "revise", "next_role": "features"}
        with self.assertRaisesRegex(RuntimeError, "Revision limit"):
            run_policy("multi", RevisingRunner(), max_revisions=1)

    def test_nonfinal_roles_cannot_bypass_audit_by_finishing(self):
        class CompletingRunner:
            def __init__(self):
                self.roles = []

            def run_stage(self, role, handoffs):
                self.roles.append(role)
                return {"role": role, "status": "completed"}
        runner = CompletingRunner()
        run_policy("multi", runner)
        self.assertEqual(runner.roles, ["diagnosis", "features", "modeling", "auditing"])
