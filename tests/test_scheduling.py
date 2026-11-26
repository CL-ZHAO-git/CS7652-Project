import json
import tempfile
import unittest
from pathlib import Path

from forecast_agents.agents.runner import AgentRunner
from forecast_agents.config import load_config
from forecast_agents.evaluation.evaluator import freeze_solution
from forecast_agents.llm.types import Message, ModelResponse, ToolCall, Usage
from forecast_agents.runtime.budget import Budget, BudgetLimits
from forecast_agents.runtime.events import EventLog
from forecast_agents.runtime.scheduling import SchedulingConfig, StageAllowance, StageBudgetExceeded
from forecast_agents.tools.workspace import WorkspaceTools
from forecast_agents.templates import ZERO_CODE


class RecordingBackend:
    name = "scheduling-fixture"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return next(self.responses)


def finish(status="completed", next_role=""):
    return ToolCall("done", "finish_stage", {"status": status, "artifacts": [],
                    "findings": ["Fixture evidence"], "recommendations": [], "next_role": next_role})


def response(*calls, usage=Usage(100, 1)):
    return ModelResponse(Message("assistant", tool_calls=calls), "tool_calls", usage)


class SchedulingTests(unittest.TestCase):
    def audited_closing_runner(self, root, backend):
        shared = self.budget()
        runner = self.runner(root, backend, shared, SchedulingConfig(force_tool_choice=True, max_policy_corrections=0))
        runner.tools.execute("write_file", {"path": "solution/pipeline.py", "content": ZERO_CODE})
        digest = freeze_solution(root / "solution", root / "snapshot")
        runner.tools.last_validation_hash = runner.tools.last_audit_hash = digest
        # Reproduce the historical audit-end ledger: only 7,281 role tokens left.
        stage = runner.allowance("auditing")
        stage.total_tokens, shared.total_tokens = 142719, 459015
        stage.last_input_tokens, stage.last_context_chars = 9422, 32000
        stage.llm_calls, shared.llm_calls = 15, 53
        stage.finalizing = True
        runner.histories["auditing"] = [Message("system", "Audit current code and report limitations."),
            Message("user", "Review candidate"),
            Message("assistant", tool_calls=(ToolCall("page", "read_file", {"path": "solution/pipeline.py"}),)),
            Message("tool", json.dumps({"text": "OLD_SOURCE_SENTINEL" * 1500, "truncated": True,
                                         "next_offset": 8000}), tool_call_id="page")]
        return runner, shared

    def test_audited_closing_uses_small_context_and_explicit_finish_within_same_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = RecordingBackend([response(finish(), usage=Usage(500, 100))])
            runner, shared = self.audited_closing_runner(Path(directory), backend)
            stage = runner.allowance("auditing")
            self.assertGreater(stage.estimate_input(32000, 1000), stage.remaining_tokens)
            result = runner.run_stage("auditing", [])
            self.assertEqual(result["status"], "completed")
            request = backend.requests[0]
            self.assertEqual(request.tool_choice, "finish_stage")
            self.assertEqual({t["name"] for t in request.tools}, {"finish_stage"})
            self.assertNotIn("OLD_SOURCE_SENTINEL", "".join(m.text for m in request.messages))
            self.assertIn("Partial source pages", "".join(m.text for m in request.messages))
            self.assertLess(sum(len(m.text) for m in request.messages), 8000)
            self.assertEqual(stage.total_tokens, 142719 + 600)
            self.assertEqual(shared.total_tokens, 459015 + 600)
            self.assertEqual(stage.llm_calls, 16)
            self.assertEqual(shared.tool_calls, 1)
            self.assertEqual(stage.limits.max_total_tokens, 150000)

    def test_audited_compact_context_never_treats_plain_text_as_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = RecordingBackend([ModelResponse(Message("assistant", "Done"), "end", Usage(100, 1))])
            runner, shared = self.audited_closing_runner(Path(directory), backend)
            with self.assertRaisesRegex(StageBudgetExceeded, "without finish_stage"):
                runner.run_stage("auditing", [])
            self.assertEqual(shared.tool_calls, 0)
            events = [json.loads(l) for l in runner.events.path.read_text().splitlines()]
            self.assertFalse(any(e["event"] == "stage_finished" for e in events))

    def test_compact_finalization_still_rechecks_source_hash_at_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            class MutatingBackend(RecordingBackend):
                def generate(self, request):
                    (root / "solution/pipeline.py").write_text("# changed after evidence\n")
                    return super().generate(request)
            backend = MutatingBackend([response(finish())])
            runner, shared = self.audited_closing_runner(root, backend)
            with self.assertRaisesRegex(StageBudgetExceeded, "Final completion requires audit_candidate"):
                runner.run_stage("auditing", [])
            self.assertFalse(runner.tools.audit_is_current())
            events = [json.loads(l) for l in runner.events.path.read_text().splitlines()]
            self.assertFalse(any(e["event"] == "stage_finished" for e in events))

    def test_compact_finalization_does_not_waive_observed_token_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = RecordingBackend([response(finish(), usage=Usage(8000, 0))])
            runner, shared = self.audited_closing_runner(Path(directory), backend)
            with self.assertRaisesRegex(StageBudgetExceeded, "observed token allowance"):
                runner.run_stage("auditing", [])
            self.assertEqual(shared.total_tokens, 459015 + 8000)
            self.assertEqual(shared.tool_calls, 0)

    def budget(self, clock=None):
        limits = BudgetLimits(max_llm_calls=80, max_total_tokens=600000,
                              max_tool_calls=240, wall_seconds=1800)
        return Budget(limits, **({"clock": clock} if clock else {}))

    def runner(self, root, backend, budget, scheduling=None):
        events = EventLog(root / "events.jsonl")
        tools = WorkspaceTools(root, None, None, None, budget, events)
        return AgentRunner(backend, tools, budget, events, scheduling=scheduling)

    def test_development_config_reserves_audit_resources(self):
        config = load_config(Path(__file__).parents[1] / "configs/deepseek.toml")
        self.assertEqual(config.budget.max_total_tokens, 600000)
        self.assertEqual(config.budget.max_llm_calls, 80)
        shared = Budget(config.budget)
        roles = [StageAllowance(role, shared, config.scheduling) for role in
                 ("diagnosis", "features", "modeling", "auditing")]
        self.assertEqual([r.limits.max_total_tokens for r in roles], [90000, 90000, 270000, 150000])
        self.assertEqual(sum(r.limits.max_llm_calls for r in roles[:-1]), 60)
        self.assertEqual(roles[-1].limits.max_llm_calls, 20)
        self.assertEqual(sum(r.limits.max_tool_calls for r in roles[:-1]), 180)
        self.assertEqual(roles[-1].limits.max_tool_calls, 60)

    def test_role_overrun_is_charged_and_blocks_tools_before_borrowing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = RecordingBackend([response(ToolCall("bad", "write_file", {"path": "bad.py", "content": "x"}),
                                                usage=Usage(90001, 0))])
            shared = self.budget()
            runner = self.runner(root, backend, shared)
            with self.assertRaisesRegex(StageBudgetExceeded, "diagnosis.*token"):
                runner.run_stage("diagnosis", [])
            self.assertEqual(shared.total_tokens, 90001)
            self.assertEqual(runner.allowance("diagnosis").total_tokens, 90001)
            self.assertEqual(shared.tool_calls, 0)
            self.assertFalse((root / "bad.py").exists())
            self.assertIs(runner.tools.budget, shared)
            self.assertEqual(runner.allowance("auditing").limits.max_total_tokens, 150000)
            self.assertEqual(runner.allowance("auditing").total_tokens, 0)

    def test_finish_only_restricts_tools_and_output_but_requires_explicit_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = RecordingBackend([response(finish())])
            shared = self.budget()
            runner = self.runner(root, backend, shared)
            # Reserve the final two of this role's twelve calls for handoff.
            shared.llm_calls = runner.allowance("features").llm_calls = 10
            result = runner.run_stage("features", [])
            self.assertEqual(result["status"], "completed")
            self.assertEqual({t["name"] for t in backend.requests[0].tools}, {"finish_stage"})
            self.assertEqual(backend.requests[0].max_output_tokens, 1024)
            self.assertEqual(shared.llm_calls, 11)
            self.assertEqual(runner.allowance("features").llm_calls, 11)

    def test_finish_only_for_modeling_allows_validation_and_blocks_entire_invalid_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = RecordingBackend([response(finish(), ToolCall("bad", "write_file", {"path": "bad.py", "content": "x"}))])
            shared = self.budget()
            runner = self.runner(root, backend, shared, SchedulingConfig(max_policy_corrections=0))
            shared.llm_calls = runner.allowance("modeling").llm_calls = 34
            with self.assertRaisesRegex(StageBudgetExceeded, "finish-only"):
                runner.run_stage("modeling", [])
            self.assertEqual({t["name"] for t in backend.requests[0].tools}, {"finish_stage", "validate_candidate"})
            self.assertEqual(shared.tool_calls, 0)
            self.assertFalse((root / "bad.py").exists())
            events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
            self.assertFalse(any(e["event"] == "stage_finished" for e in events))

    def test_finish_only_text_does_not_automatically_accept_a_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = RecordingBackend([ModelResponse(Message("assistant", "Done"), "end", Usage(100, 1))])
            shared = self.budget()
            runner = self.runner(root, backend, shared, SchedulingConfig(max_policy_corrections=0))
            shared.llm_calls = runner.allowance("auditing").llm_calls = 18
            with self.assertRaisesRegex(RuntimeError, "without finish_stage"):
                runner.run_stage("auditing", [])

    def test_same_role_revisions_share_counters_and_do_not_reset_time(self):
        now = [0.0]
        shared = self.budget(clock=lambda: now[0])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = RecordingBackend([response(finish("revise", "features")), response(finish())])
            runner = self.runner(root, backend, shared)
            runner.run_stage("diagnosis", [])
            first = runner.allowance("diagnosis")
            first.elapsed = 10
            now[0] = 100
            runner.run_stage("diagnosis", [])
            self.assertIs(first, runner.allowance("diagnosis"))
            self.assertEqual(first.llm_calls, 2)
            self.assertEqual(first.total_tokens, 202)
            self.assertEqual(first.elapsed_seconds, 10)
            self.assertEqual(shared.llm_calls, 2)

    def test_insufficient_input_allowance_stops_before_an_api_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = RecordingBackend([])
            shared = self.budget()
            runner = self.runner(root, backend, shared)
            stage = runner.allowance("diagnosis")
            shared.total_tokens = stage.total_tokens = 89000
            stage.last_input_tokens, stage.last_context_chars = 5000, 10000
            with self.assertRaisesRegex(StageBudgetExceeded, "no request sent"):
                runner.run_stage("diagnosis", [])
            self.assertEqual(backend.requests, [])
            self.assertEqual(shared.llm_calls, 0)

    def test_role_call_tool_and_time_caps_do_not_reset_shared_ledger(self):
        now = [0.0]
        shared = self.budget(clock=lambda: now[0])
        stage = StageAllowance("diagnosis", shared, SchedulingConfig())
        stage.start()
        stage.llm_calls = stage.limits.max_llm_calls
        with self.assertRaisesRegex(StageBudgetExceeded, "model-call"):
            stage.begin_call()
        stage.tool_calls = stage.limits.max_tool_calls
        with self.assertRaisesRegex(StageBudgetExceeded, "tool-call"):
            stage.begin_tool()
        self.assertEqual(shared.llm_calls, 0)
        self.assertEqual(shared.tool_calls, 0)
        now[0] = 271
        with self.assertRaisesRegex(StageBudgetExceeded, "wall-time"):
            stage.check()
        stage.stop()
        now[0] = 1000
        self.assertEqual(stage.elapsed_seconds, 271)

    def test_invalid_scheduling_configuration_is_rejected(self):
        for values in ({"enabled": 1}, {"auditing_share": 0.1}, {"diagnosis_share": float("nan")},
                       {"features_share": True}, {"finish_only_fraction": 1},
                       {"finish_call_reserve": 0}, {"finish_output_tokens": False}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                SchedulingConfig(**values)

    def test_finish_only_does_not_reopen_after_the_next_prompt_shrinks(self):
        shared = self.budget()
        stage = StageAllowance("modeling", shared, SchedulingConfig())
        shared.total_tokens = stage.total_tokens = 140000
        stage.last_input_tokens = 100000
        self.assertTrue(stage.finish_only())
        # A compact validation/finish request can be much smaller. That must not
        # reopen model search or reset a closing role's revision allowance.
        stage.last_input_tokens = 100
        self.assertTrue(stage.finish_only())
        self.assertTrue(stage.snapshot()["finish_only"])

    def test_hidden_reads_get_one_feedback_round_without_execution_or_new_allowance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = RecordingBackend([
                response(ToolCall("r1", "read_file", {"path": "never_read.py"}),
                         ToolCall("r2", "read_file", {"path": "also_never_read.py"})),
                response(finish("revise", "modeling")),
            ])
            shared = self.budget()
            runner = self.runner(root, backend, shared, SchedulingConfig(force_tool_choice=True))
            shared.llm_calls = runner.allowance("features").llm_calls = 10
            result = runner.run_stage("features", [])
            self.assertEqual(result["status"], "revise")
            self.assertEqual(result["next_role"], "modeling")
            self.assertEqual(runner.allowance("features").policy_corrections, 1)
            self.assertEqual(runner.allowance("features").llm_calls, 12)
            self.assertEqual(shared.total_tokens, 202)
            self.assertEqual(shared.tool_calls, 3)  # two rejected attempts plus handoff
            self.assertEqual([r.tool_choice for r in backend.requests], ["finish_stage", "finish_stage"])
            feedback = [m for m in backend.requests[1].messages if m.role == "tool"]
            self.assertEqual([m.tool_call_id for m in feedback], ["r1", "r2"])
            self.assertTrue(all(json.loads(m.text)["policy_blocked"] for m in feedback))

    def test_second_policy_violation_fails_without_executing_either_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = RecordingBackend([
                response(ToolCall("r", "read_file", {"path": "never_read.py"})),
                response(ToolCall("w", "write_file", {"path": "bad.py", "content": "x"})),
            ])
            shared = self.budget()
            runner = self.runner(root, backend, shared)
            shared.llm_calls = runner.allowance("features").llm_calls = 10
            with self.assertRaisesRegex(StageBudgetExceeded, "correction limit exhausted"):
                runner.run_stage("features", [])
            self.assertEqual(len(backend.requests), 2)
            self.assertFalse((root / "bad.py").exists())
            self.assertEqual(shared.total_tokens, 202)

    def test_known_failed_feature_check_cannot_be_marked_completed_while_closing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = RecordingBackend([response(finish()), response(finish("revise", "modeling"))])
            shared = self.budget()
            runner = self.runner(root, backend, shared)
            shared.llm_calls = runner.allowance("features").llm_calls = 10
            runner.histories["features"] = [Message("system", "Features"), Message("user", "Implement helpers"),
                Message("assistant", tool_calls=(ToolCall("smoke", "run_python", {"path": "smoke.py"}),)),
                Message("tool", '{"is_error": true, "returncode": 1, "stderr": "AssertionError"}', tool_call_id="smoke")]
            result = runner.run_stage("features", [])
            self.assertEqual(result["status"], "revise")
            self.assertEqual(result["next_role"], "modeling")
            events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
            finished = [e for e in events if e["event"] == "stage_finished"]
            self.assertEqual(len(finished), 1)
            self.assertEqual(finished[0]["handoff"]["status"], "revise")
            self.assertTrue(any(e["event"] == "tool_result" and "Latest feature check failed" in e["result"].get("error", "")
                                for e in events))

    def test_plain_text_closing_gets_feedback_and_still_requires_a_tool_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = RecordingBackend([ModelResponse(Message("assistant", "Done"), "end", Usage(100, 1)),
                                        response(finish("rejected"))])
            shared = self.budget()
            runner = self.runner(root, backend, shared)
            shared.llm_calls = runner.allowance("auditing").llm_calls = 18
            result = runner.run_stage("auditing", [{"role": "modeling", "findings": ["Candidate evidence"]}])
            self.assertEqual(result["status"], "rejected")
            self.assertEqual(len(backend.requests), 2)
            self.assertTrue(any("Candidate evidence" in m.text for m in backend.requests[1].messages if m.role == "user"))

    def test_closing_correction_keeps_required_audit_tool_available(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = RecordingBackend([
                ModelResponse(Message("assistant", "Need to finish"), "end", Usage(100, 1)),
                response(finish("rejected")),
            ])
            shared = self.budget()
            runner = self.runner(root, backend, shared, SchedulingConfig(force_tool_choice=True))
            shared.llm_calls = runner.allowance("auditing").llm_calls = 18
            runner.run_stage("auditing", [])
            self.assertIn("audit_candidate", {t["name"] for t in backend.requests[1].tools})
            self.assertEqual(backend.requests[1].tool_choice, "required")


if __name__ == "__main__":
    unittest.main()
