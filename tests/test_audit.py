import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from forecast_agents.evaluation.audit import audit_candidate, AuditFailure
from forecast_agents.runtime.budget import Budget, BudgetLimits
from forecast_agents.runtime.events import EventLog
from forecast_agents.runtime.executor import Executor
from forecast_agents.tasks.panel import TaskSpec, synthetic_panel
from forecast_agents.templates import FEATURES_CODE, ZERO_CODE
from forecast_agents.tools.workspace import WorkspaceTools


class AuditTests(unittest.TestCase):
    def panel_and_task(self):
        panel = synthetic_panel(2, 40)
        return panel, TaskSpec(panel.dates[19], panel.dates[29])

    def candidate(self, root, feature_code=FEATURES_CODE):
        root.mkdir()
        (root / "pipeline.py").write_text(ZERO_CODE)
        (root / "features.py").write_text(feature_code)
        return root

    def test_fixed_audit_changes_real_future_rows_and_passes_causal_features(self):
        panel, task = self.panel_and_task()
        with tempfile.TemporaryDirectory() as directory:
            solution = self.candidate(Path(directory) / "solution")
            report = audit_candidate(panel, task, solution, Executor())
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["probes"]), 2)
        self.assertTrue(all(p["future_rows_changed"] > 0 for p in report["probes"]))
        self.assertTrue(all(p["origin"] < task.validation_end for p in report["probes"]))
        self.assertTrue(all(p["feature_prefix_invariant"] for p in report["probes"]))

    def test_feature_that_ignores_origin_and_reads_future_is_rejected(self):
        panel, task = self.panel_and_task()
        bad = '''def features_at(history, origin):
    return {s: {"future": sum(r["close"] for r in history if r["symbol"] == s)}
            for s in sorted({r["symbol"] for r in history})}
'''
        with tempfile.TemporaryDirectory() as directory:
            solution = self.candidate(Path(directory) / "solution", bad)
            with self.assertRaisesRegex(AuditFailure, "depend_on_future"):
                audit_candidate(panel, task, solution, Executor())

    def test_unchanged_validation_and_audit_are_required_after_source_edits(self):
        panel, task = self.panel_and_task()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir()
            self.candidate(root / "solution")
            tools = WorkspaceTools(root, panel, task, Executor(), Budget(BudgetLimits(wall_seconds=60)),
                                   EventLog(root.parent / "events.jsonl"))
            self.assertFalse(tools.audit_is_current())
            with self.assertRaisesRegex(ValueError, "Validate"):
                tools.execute("audit_candidate", {})
            tools.execute("validate_candidate", {})
            report = tools.execute("audit_candidate", {})
            self.assertTrue(report["passed"])
            self.assertTrue(tools.audit_is_current())
            (root / "solution/pipeline.py").write_text(ZERO_CODE + "\n# changed\n")
            self.assertFalse(tools.audit_is_current())
            with self.assertRaisesRegex(ValueError, "Validate"):
                tools.execute("audit_candidate", {})
            self.assertIsNone(tools.last_audit_hash)

    def test_audit_reports_only_counts_and_status_not_feature_values(self):
        panel, task = self.panel_and_task()
        code = '''def features_at(history, origin):
    return {s: {"private_constant": 987654321.123456} for s in sorted({r["symbol"] for r in history if r["date"] <= origin})}
'''
        with tempfile.TemporaryDirectory() as directory:
            report = audit_candidate(panel, task, self.candidate(Path(directory) / "solution", code), Executor())
        self.assertNotIn("987654321", json.dumps(report))
        self.assertNotIn("private_constant", json.dumps(report))

    def test_failed_revalidation_invalidates_old_approval_and_summary(self):
        panel, task = self.panel_and_task()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir()
            self.candidate(root / "solution")
            tools = WorkspaceTools(root, panel, task, Executor(), Budget(BudgetLimits(wall_seconds=60)),
                                   EventLog(root.parent / "events.jsonl"))
            tools.execute("validate_candidate", {})
            tools.execute("audit_candidate", {})
            self.assertTrue(tools.audit_is_current())
            with patch("forecast_agents.tools.workspace.evaluate", side_effect=RuntimeError("fixture failure")):
                with self.assertRaises(RuntimeError):
                    tools.execute("validate_candidate", {})
            self.assertIsNone(tools.last_validation_hash)
            self.assertFalse(tools.audit_is_current())
            with self.assertRaisesRegex(ValueError, "No validation summary"):
                tools.execute("read_validation_summary", {})


if __name__ == "__main__":
    unittest.main()
