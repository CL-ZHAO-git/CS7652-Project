import tempfile
import unittest
from pathlib import Path

from forecast_agents.evaluation.evaluator import evaluate, freeze_solution, score, validate_predictions
from forecast_agents.runtime.executor import Executor
from forecast_agents.tasks.panel import Bar, Panel, TaskSpec, synthetic_panel
from forecast_agents.templates import ZERO_CODE


OBSERVATION_CODE = '''from pathlib import Path

def fit(history, horizons):
    Path("fit_only.txt").write_text("must not survive into predict")
    return {"fit_end": max(row["date"] for row in history)}

def predict(history, state, horizons):
    assert not Path("fit_only.txt").exists()
    assert not Path("previous_prediction.txt").exists()
    Path("previous_prediction.txt").write_text("no sharing between origins")
    count = len({row["date"] for row in history})
    assert max(row["date"] for row in history) > state["fit_end"]
    return [{"symbol": symbol, "horizon": h, "prediction": float(count),
             "risk": 0.0, "abstain": False, "fallback": False}
            for symbol in sorted({row["symbol"] for row in history}) for h in horizons]
'''


class EvaluationTests(unittest.TestCase):
    def test_worker_gets_only_causal_prefixes_in_fresh_workspaces(self):
        panel = synthetic_panel(2, 24)
        task = TaskSpec(panel.dates[10], panel.dates[16])
        with tempfile.TemporaryDirectory() as directory:
            solution = Path(directory) / "solution"
            solution.mkdir()
            (solution / "pipeline.py").write_text(OBSERVATION_CODE)
            report = evaluate(panel, task, solution, Executor(), "test")
        for row in report["rows"]:
            self.assertEqual(row["prediction"], panel.dates.index(row["origin"]) + 1)
        self.assertEqual(report["fit_cutoff"], task.train_end)

    def test_future_changes_do_not_change_earlier_predictions(self):
        panel = synthetic_panel(1, 24)
        task = TaskSpec(panel.dates[10], panel.dates[16])
        origin = panel.dates[17]
        altered = Panel([Bar(row["symbol"], row["date"],
                             row["close"] * (2 if row["date"] > origin else 1))
                         for row in panel.history(panel.dates[-1])])
        with tempfile.TemporaryDirectory() as directory:
            solution = Path(directory)
            (solution / "pipeline.py").write_text(OBSERVATION_CODE)
            first = evaluate(panel, task, solution, Executor())
            second = evaluate(altered, task, solution, Executor())
        a = [row for row in first["rows"] if row["origin"] == origin]
        b = [row for row in second["rows"] if row["origin"] == origin]
        self.assertEqual([row["prediction"] for row in a], [row["prediction"] for row in b])
        self.assertNotEqual([row["target"] for row in a], [row["target"] for row in b])

    def test_missing_duplicate_and_nonfinite_predictions_fail(self):
        row = {"symbol": "A", "horizon": 1, "prediction": 0.0, "risk": 0.1,
               "abstain": False, "fallback": False}
        validate_predictions([row], ["A"], [1])
        invalid = [[], [row, row], [{**row, "prediction": float("nan")}],
                   [{**row, "horizon": True}], [{**row, "abstain": True, "fallback": True}]]
        for rows in invalid:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validate_predictions(rows, ["A"], [1])

    def test_abstention_fallback_and_tied_risks_have_distinct_metrics(self):
        rows = [
            {"horizon": 1, "target": 1.0, "prediction": 9.0, "effective_prediction": 0.0,
             "risk": 0.1, "abstain": False, "fallback": True},
            {"horizon": 1, "target": 2.0, "prediction": 2.0, "effective_prediction": 2.0,
             "risk": 0.1, "abstain": True, "fallback": False},
        ]
        metrics = score(rows, [1])["1"]
        self.assertEqual(metrics["coverage"], 0.5)
        self.assertEqual(metrics["fallback_rate"], 0.5)
        self.assertEqual(metrics["served"]["mae"], 1.0)
        self.assertEqual(metrics["raw"]["mae"], 4.0)
        self.assertEqual(len(metrics["risk_coverage"]), 1)
        self.assertEqual(metrics["risk_coverage"][0]["coverage"], 1.0)

    def test_snapshot_hash_changes_with_source_and_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "solution"
            source.mkdir()
            (source / "pipeline.py").write_text(ZERO_CODE)
            first = freeze_solution(source, root / "first")
            (source / "pipeline.py").write_text(ZERO_CODE + "\n# revision\n")
            second = freeze_solution(source, root / "second")
            self.assertNotEqual(first, second)
            (source / "link").symlink_to(root / "first")
            with self.assertRaisesRegex(ValueError, "symlink"):
                freeze_solution(source, root / "third")
            (root / "root_link").symlink_to(source, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                freeze_solution(root / "root_link", root / "fourth")
