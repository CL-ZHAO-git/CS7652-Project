import json
import tempfile
import unittest
from pathlib import Path

from forecast_agents.agents.context import finalization_context, request_context, working_state
from forecast_agents.llm.types import Message, ToolCall
from forecast_agents.runtime.budget import Budget, BudgetLimits
from forecast_agents.runtime.events import EventLog
from forecast_agents.runtime.executor import ExecutionResult
from forecast_agents.tools.workspace import WorkspaceTools


class ContextTests(unittest.TestCase):
    def test_failed_page_read_keeps_safe_feedback_and_previous_successful_preview(self):
        history = [Message("system", "Instructions"), Message("user", "Task"),
                   Message("assistant", tool_calls=(ToolCall("ok", "read_file", {"path": "solution/pipeline.py"}),)),
                   Message("tool", json.dumps({"text": "AUTHORED_SOURCE_SENTINEL", "truncated": True,
                                                "next_offset": 8000}), tool_call_id="ok"),
                   Message("assistant", tool_calls=(ToolCall("bad", "read_file", {"path": "solution/pipeline.py",
                                                                                             "offset": 8000, "max_chars": 12000}),)),
                   Message("tool", json.dumps({"is_error": True, "error": "max_chars must be <= 8000; use next_offset"}), tool_call_id="bad")]
        state = working_state(history)
        self.assertTrue(state["recent_file_reads"][0]["is_error"])
        self.assertIn("8000", state["recent_file_reads"][0]["error"])
        self.assertEqual(state["cached_file_previews"][0]["text"], "AUTHORED_SOURCE_SENTINEL")
        view, info = finalization_context(history, Message("system", "Budget"), "checked-hash")
        text = "".join(m.text for m in view)
        self.assertNotIn("AUTHORED_SOURCE_SENTINEL", text)
        self.assertIn('"truncated": true', text)
        self.assertIn("Partial source pages", text)
        self.assertIn("checked-hash", text)
        self.assertTrue(info["finalization_only"])

    def test_failed_revalidation_replaces_stale_success_in_compacted_memory(self):
        history = [Message("system", "Instructions"), Message("user", "Task")]
        for call_id, name, result in (
                ("v1", "validate_candidate", {"candidate_hash": "old", "metrics": {"1": 0.1}}),
                ("a1", "audit_candidate", {"passed": True}),
                ("v2", "validate_candidate", {"is_error": True, "error_type": "KeyError",
                    "diagnostic_code": "missing_known_key", "missing_key": "models", "operation": "fit"}),
                ("a2", "audit_candidate", {"is_error": True, "error": "Validate the current source first"})):
            history.extend([Message("assistant", tool_calls=(ToolCall(call_id, name, {}),)),
                            Message("tool", json.dumps(result), tool_call_id=call_id)])
        state = working_state(history)
        self.assertNotIn("candidate_hash", state["latest_validation_result"])
        self.assertEqual(state["latest_validation_result"]["missing_key"], "models")
        self.assertEqual(state["latest_validation_result"]["operation"], "fit")
        self.assertTrue(state["latest_audit_result"]["is_error"])
        self.assertNotIn("passed", state["latest_audit_result"])

    def test_data_reads_are_small_previews_and_pages_have_character_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            content = "\u00e9" * 900 + "x" * 31000
            (root / "data/history.json").write_text(content, encoding="utf-8")
            tools = WorkspaceTools(root, None, None, None, None, None)
            first = tools.execute("read_file", {"path": "data/history.json", "max_chars": 8000})
            self.assertEqual(first["text"], content[:1000])
            self.assertTrue(first["truncated"])
            self.assertEqual(first["next_offset"], 1000)
            self.assertIn("Python script", first["hint"])
            second = tools.execute("read_file", {"path": "data/history.json", "offset": 1000, "max_chars": 30})
            self.assertEqual(second["text"], content[1000:1030])
            self.assertEqual(second["next_offset"], 1030)
            # Files stay complete for generated code; only the observation is limited.
            self.assertEqual((root / "data/history.json").read_text(), content)

    def test_source_reads_can_be_paged_and_invalid_ranges_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "code.py").write_text("0123456789")
            tools = WorkspaceTools(root, None, None, None, None, None)
            self.assertEqual(tools.execute("read_file", {"path": "code.py", "offset": 6, "max_chars": 4}),
                             {"text": "6789", "truncated": False, "next_offset": None})
            for values in ({"offset": -1}, {"max_chars": 0}, {"max_chars": 8001}, {"offset": True}):
                with self.subTest(values=values), self.assertRaises(ValueError):
                    tools.execute("read_file", {"path": "code.py", **values})

    def test_compaction_drops_whole_old_exchanges_and_preserves_latest_native_state(self):
        old = Message("assistant", tool_calls=(ToolCall("old", "read_file", {"path": "large.txt"}),))
        native = {"openai_compatible": {"reasoning_content": "opaque state", "content": None,
                  "tool_calls": [{"id": "a", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]}}
        recent = Message("assistant", tool_calls=(ToolCall("a", "read_file", {"path": "a"}),
                                                  ToolCall("b", "read_file", {"path": "b"})), native=native)
        history = [Message("system", "Instructions"), Message("user", "Current task"), old,
                   Message("tool", json.dumps({"text": "x" * 32000}), tool_call_id="old"), recent,
                   Message("tool", '{"text": "a"}', tool_call_id="a"),
                   Message("tool", '{"text": "b"}', tool_call_id="b")]
        original = tuple(history)
        view, info = request_context(history, Message("system", "Current budget"), 4000)
        self.assertEqual(info["omitted_exchanges"], 1)
        self.assertFalse(info["over_target"])
        self.assertNotIn(old, view)
        self.assertIs(view[-3], recent)
        self.assertEqual([m.tool_call_id for m in view[-2:]], ["a", "b"])
        self.assertEqual(view[-3].native, native)
        self.assertEqual(tuple(history), original)
        self.assertEqual(view[2].text, "Current task")

    def test_revisions_keep_current_handoff_and_validation_evidence_without_old_tool_pairs(self):
        history = [Message("system", "Instructions"), Message("user", "Old task"),
                   Message("assistant", tool_calls=(ToolCall("v", "validate_candidate", {}),)),
                   Message("tool", json.dumps({"report": "validation.json", "candidate_hash": "abc", "metrics": {"1": 0.1}}), tool_call_id="v"),
                   Message("user", "Current revision: correct the leakage")]
        view, info = request_context(history, Message("system", "Current budget"), 4000)
        self.assertEqual(info["omitted_exchanges"], 1)
        self.assertEqual(view[2].text, history[-1].text)
        self.assertIn('"candidate_hash": "abc"', view[3].text)
        self.assertFalse(any(m.role == "tool" for m in view))

    def test_oversized_latest_exchange_is_preserved_and_explicitly_reported(self):
        assistant = Message("assistant", tool_calls=(ToolCall("a", "read_file", {"path": "a"}),))
        history = [Message("system", "Instructions"), Message("user", "Task"), assistant,
                   Message("tool", json.dumps({"text": "x" * 10000}), tool_call_id="a")]
        view, info = request_context(history, Message("system", "Budget"), 4000)
        self.assertTrue(info["over_target"])
        self.assertEqual(view[-2:], tuple(history[-2:]))

    def test_trimmed_context_keeps_protocol_source_and_completed_check(self):
        history = [Message("system", "Instructions"), Message("user", "Implement the model")]
        for i, (path, text) in enumerate((("task.json", '{"horizons": [1, 3, 5]}'),
                                          ("PIPELINE_CONTRACT.md", "fit(history, horizons); predict(history, state, horizons)"),
                                          ("solution/features.py", "def row(history, symbol, origin): return [1]"))):
            history.extend([Message("assistant", tool_calls=(ToolCall(str(i), "read_file", {"path": path}),)),
                            Message("tool", json.dumps({"text": text, "truncated": False}), tool_call_id=str(i))])
        history.extend([Message("assistant", tool_calls=(ToolCall("run", "run_python", {"path": "solution/smoke.py"}),)),
                        Message("tool", json.dumps({"returncode": 0, "stdout": "causality mismatches 0; OK", "stderr": ""}), tool_call_id="run")])
        for i in range(5):
            history.extend([Message("assistant", tool_calls=(ToolCall(str(i), "list_files", {}),)),
                            Message("tool", json.dumps({"files": ["file.py"] * 150}), tool_call_id=str(i))])
        view, info = request_context(history, Message("system", "Budget"), 4000)
        self.assertGreater(info["omitted_exchanges"], 0)
        self.assertIn("solution/features.py", info["cached_paths"])
        summary = next(m for m in view if m.text.startswith("Controller working memory"))
        state = json.loads(summary.text.splitlines()[1])
        previews = {entry["path"]: entry["text"] for entry in state["cached_file_previews"]}
        self.assertIn("fit(history", previews["PIPELINE_CONTRACT.md"])
        self.assertIn("def row", previews["solution/features.py"])
        self.assertEqual(state["recent_executions"][0]["returncode"], 0)
        self.assertIn("causality mismatches 0", state["recent_executions"][0]["stdout_preview"])

    def test_working_memory_invalidates_old_pages_and_does_not_call_old_checks_current(self):
        path = "solution/features.py"
        history = [Message("system", "Instructions"), Message("user", "Task")]
        for i, offset in enumerate((0, 8000)):
            history.extend([Message("assistant", tool_calls=(ToolCall(str(i), "read_file", {"path": path, "offset": offset}),)),
                            Message("tool", '{"text": "old code", "truncated": false}', tool_call_id=str(i))])
        history.extend([Message("assistant", tool_calls=(ToolCall("check", "run_python", {"path": "check.py"}),)),
                        Message("tool", '{"returncode": 0, "stdout": "OK"}', tool_call_id="check"),
                        Message("assistant", tool_calls=(ToolCall("edit", "write_file", {"path": path, "content": "new code"}),)),
                        Message("tool", json.dumps({"written": path}), tool_call_id="edit"),
                        Message("assistant", tool_calls=(ToolCall("bad", "write_file", {"path": path, "content": "failed edit"}),)),
                        Message("tool", '{"is_error": true, "error": "write failed"}', tool_call_id="bad"),
                        Message("assistant", tool_calls=(ToolCall("report", "write_file", {"path": "summary.md", "content": "Report"}),)),
                        Message("tool", '{"written": "summary.md"}', tool_call_id="report")])
        state = working_state(history)
        self.assertEqual(state["cached_file_previews"], [{"path": path, "offset": 0, "text": "new code",
                                                         "truncated": False, "source": "successful_write"}])
        self.assertEqual(state["current_write_revision"], 1)
        self.assertEqual(state["recent_executions"][0]["at_write_revision"], 0)

    def test_completion_can_omit_revision_target_but_revise_requires_one(self):
        with tempfile.TemporaryDirectory() as directory:
            tools = WorkspaceTools(Path(directory), None, None, None, None, None)
            args = {"status": "completed", "artifacts": [], "findings": [], "recommendations": []}
            result = tools.execute("finish_stage", args)
            self.assertEqual(result["next_role"], "")
            self.assertEqual(result["status"], "completed")
            with self.assertRaisesRegex(ValueError, "Revision requires"):
                tools.execute("finish_stage", {**args, "status": "revise"})

    def test_large_execution_output_is_logged_but_only_a_preview_reaches_model(self):
        class FakeExecutor:
            def execute(self, *args, **kwargs):
                return ExecutionResult(1, "x" * 20000, "e" * 10000, False, 0.1)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = EventLog(root / "events.jsonl")
            tools = WorkspaceTools(root, None, None, FakeExecutor(), Budget(BudgetLimits()), events)
            preview = tools.execute("run_python", {"path": "probe.py"})
            self.assertEqual(len(preview["stdout"]), 4000)
            self.assertEqual(len(preview["stderr"]), 2000)
            self.assertTrue(preview["stdout_truncated"])
            self.assertTrue(preview["stderr_truncated"])
            self.assertTrue(preview["is_error"])
            recorded = json.loads(events.path.read_text())
            self.assertEqual(recorded["event"], "execution_output")
            self.assertEqual(len(recorded["result"]["stdout"]), 20000)
            self.assertEqual(len(recorded["result"]["stderr"]), 10000)


if __name__ == "__main__":
    unittest.main()
