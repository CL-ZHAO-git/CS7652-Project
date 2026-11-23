import hashlib
import tempfile
import unittest
from pathlib import Path

from platform_support import symlinks_supported

from forecast_agents.evaluation.evaluator import (evaluate, freeze_solution, score, validate_predictions,
                                                  zero_baseline)
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


FIT_DATE_CODE = '''def fit(history, horizons):
    return {"fit_end": max(row["date"] for row in history)}

def predict(history, state, horizons):
    assert max(row["date"] for row in history) > state["fit_end"]
    # Report which model made the forecast: the last date its fit could see.
    value = float(state["fit_end"].replace("-", ""))
    return [{"symbol": symbol, "horizon": h, "prediction": value, "risk": 0.0,
             "abstain": False, "fallback": False}
            for symbol in sorted({row["symbol"] for row in history}) for h in horizons]
'''


class EvaluationTests(unittest.TestCase):
    def test_walk_forward_refits_each_block_on_history_through_its_cutoff(self):
        panel = synthetic_panel(2, 300)  # weekdays from 2020-01-01 to early 2021
        task = TaskSpec("2020-03-31", "2020-09-30", refit="quarter", validation_stride=10, test_stride=10)
        with tempfile.TemporaryDirectory() as directory:
            solution = Path(directory)
            (solution / "pipeline.py").write_text(FIT_DATE_CODE)
            validation = evaluate(panel, task, solution, Executor(), "validation")
            test = evaluate(panel, task, solution, Executor(), "test")
        self.assertEqual([b["fit_cutoff"] for b in validation["blocks"]], ["2020-03-31", "2020-06-30"])
        self.assertEqual([b["fit_cutoff"] for b in test["blocks"]], ["2020-09-30", "2020-12-31"])
        self.assertEqual(test["protocol"], "walk_forward_quarter")
        for row in validation["rows"] + test["rows"]:
            self.assertEqual(row["prediction"], float(row["fit_cutoff"].replace("-", "")))
            self.assertGreater(row["origin"], row["fit_cutoff"])

    def test_in_process_zero_baseline_matches_running_the_zero_pipeline(self):
        panel = synthetic_panel(2, 300)
        task = TaskSpec("2020-03-31", "2020-09-30", refit="quarter", test_stride=10)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "pipeline.py").write_text(ZERO_CODE)
            executed = evaluate(panel, task, Path(directory), Executor(), "test")
        computed = zero_baseline(panel, task, "test")
        for key in ("origins", "blocks", "metrics", "rows"):
            self.assertEqual(computed[key], executed[key], key)

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

    def test_snapshot_hash_changes_with_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "solution"
            source.mkdir()
            (source / "pipeline.py").write_text(ZERO_CODE)
            first = freeze_solution(source, root / "first")
            (source / "pipeline.py").write_text(ZERO_CODE + "\n# revision\n")
            second = freeze_solution(source, root / "second")
            self.assertNotEqual(first, second)

    def test_snapshot_hash_is_identical_across_operating_systems(self):
        files = {"pipeline.py": b"x = 1\n", "B.py": b"b\n", "a.py": b"a\n", "lib/helper.py": b"h\n"}
        expected = hashlib.sha256()
        # POSIX separators, case-sensitive order of path parts: "B.py" sorts before "a.py".
        for name in ("B.py", "a.py", "lib/helper.py", "pipeline.py"):
            expected.update(name.encode() + b"\0" + files[name] + b"\0")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "solution/lib").mkdir(parents=True)
            for name, content in files.items():
                (root / "solution" / name).write_bytes(content)
            self.assertEqual(freeze_solution(root / "solution", root / "frozen"), expected.hexdigest())

    @unittest.skipUnless(symlinks_supported(), "creating symlinks needs Developer Mode on Windows")
    def test_snapshot_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "solution"
            source.mkdir()
            (source / "pipeline.py").write_text(ZERO_CODE)
            freeze_solution(source, root / "first")
            (source / "link").symlink_to(root / "first")
            with self.assertRaisesRegex(ValueError, "symlink"):
                freeze_solution(source, root / "third")
            (root / "root_link").symlink_to(source, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                freeze_solution(root / "root_link", root / "fourth")
