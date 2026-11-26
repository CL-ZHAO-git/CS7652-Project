import hashlib
import math
import tempfile
import unittest
from pathlib import Path

from platform_support import symlinks_supported

from forecast_agents.evaluation.evaluator import (evaluate, freeze_solution, score, validate_predictions,
                                                  zero_baseline, invoke_candidate)
from forecast_agents.runtime.errors import CandidateExecutionError
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
    def test_worker_supports_relative_imports_for_namespace_and_regular_packages(self):
        # Hand-written fixture only; no live-agent code runs on the host.
        with tempfile.TemporaryDirectory() as directory:
            solution = Path(directory)
            (solution / "features.py").write_text('''from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Features:
    value: float = 0.0

def features_at(history, origin):
    return {"A": {"value": Features().value}}
''')
            (solution / "pipeline.py").write_text('''from .features import Features

def fit(history, horizons):
    return {"value": Features().value}

def predict(history, state, horizons):
    return [{"symbol": "A", "horizon": h, "prediction": state["value"]} for h in horizons]
''')
            for with_init in (False, True):
                with self.subTest(with_init=with_init):
                    if with_init:
                        (solution / "__init__.py").write_text("from .pipeline import fit\n")
                    state = invoke_candidate(solution, Executor(), "fit", [], [1])
                    self.assertEqual(state, {"value": 0.0})
                    prediction = invoke_candidate(solution, Executor(), "predict", [], [1], state=state)
                    self.assertEqual(prediction, [{"symbol": "A", "horizon": 1, "prediction": 0.0}])
                    self.assertEqual(invoke_candidate(solution, Executor(), "features", [], [1], origin="2020-01-01"),
                                     {"A": {"value": 0.0}})

    def test_worker_failure_retains_local_execution_without_raw_exception_string(self):
        with tempfile.TemporaryDirectory() as directory:
            solution = Path(directory)
            (solution / "pipeline.py").write_text("raise ValueError('PRIVATE_FIXTURE_SENTINEL')\n")
            with self.assertRaises(CandidateExecutionError) as caught:
                invoke_candidate(solution, Executor(), "fit", [], [1])
            self.assertEqual(caught.exception.operation, "fit")
            self.assertIn("PRIVATE_FIXTURE_SENTINEL", caught.exception.execution.stderr)
            self.assertNotIn("PRIVATE_FIXTURE_SENTINEL", str(caught.exception))

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
        self.assertEqual(metrics["raw"]["mse"], 32.0)
        self.assertEqual(metrics["served"]["mse"], 1.0)
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

    def test_selected_baseline_prevents_apparent_gain_from_dropping_hard_rows(self):
        rows = [
            {"horizon": 1, "target": 1.0, "prediction": -0.1, "effective_prediction": -0.1,
             "risk": 0.1, "abstain": False, "fallback": False},
            {"horizon": 1, "target": 100.0, "prediction": 0.0, "effective_prediction": 0.0,
             "risk": 1.0, "abstain": True, "fallback": False},
        ]
        m = score(rows, [1])["1"]
        self.assertLess(m["served"]["mae"], m["zero_baseline"]["mae"])
        self.assertGreater(m["served"]["mae"], m["zero_baseline_served"]["mae"])
        self.assertGreater(m["raw"]["mae"], m["zero_baseline"]["mae"])
        self.assertEqual(m["zero_baseline_served"]["mae"], 1.0)
        self.assertEqual(m["served_count"], 1)
        self.assertEqual(m["abstained_count"], 1)

    def test_selected_baseline_is_undefined_if_every_forecast_abstains(self):
        rows = [{"horizon": 1, "target": 1.0, "prediction": 2.0, "effective_prediction": 2.0,
                 "risk": 1.0, "abstain": True, "fallback": False}]
        m = score(rows, [1])["1"]
        self.assertEqual(m["served_count"], 0)
        self.assertEqual(m["zero_baseline_served"], {"mae": None, "mse": None, "rmse": None})

    def test_direction_probabilities_are_optional_but_must_be_valid_and_complete(self):
        row = {"symbol": "A", "horizon": 1, "prediction": 0.0, "risk": 0.0,
               "abstain": False, "fallback": False}
        for p in (0.0, 0.8, 1.0):
            validate_predictions([{**row, "prob_up": p}], ["A"], [1])
        validate_predictions([row], ["A"], [1])  # Legacy interface.
        with self.assertRaisesRegex(ValueError, "must include prob_up"):
            validate_predictions([row], ["A"], [1], require_prob_up=True)
        for p in (None, True, -0.01, 1.01, float("nan"), float("inf"), "0.5"):
            with self.subTest(probability=p), self.assertRaises(ValueError):
                validate_predictions([{**row, "prob_up": p}], ["A"], [1])
        with self.assertRaisesRegex(ValueError, "every row"):
            validate_predictions([{**row, "prob_up": 0.8}, {**row, "horizon": 3}], ["A"], [1, 3])

    def test_direction_bce_uses_positive_return_labels_and_zero_is_non_up(self):
        rows = [{"horizon": 1, "target": y, "prediction": 0.0, "effective_prediction": 0.0,
                 "prob_up": p, "abstain": False, "fallback": False, "risk": 0.0}
                for y, p in ((0.02, 0.8), (-0.01, 0.2), (0.0, 0.2))]
        metric = score(rows, [1])["1"]
        self.assertAlmostEqual(metric["direction"]["raw"]["cross_entropy"], -math.log(0.8))
        self.assertEqual(metric["direction"]["raw"]["accuracy"], 1.0)
        self.assertAlmostEqual(metric["direction"]["half_probability_baseline"]["cross_entropy"], math.log(2))
        self.assertAlmostEqual(metric["raw"]["mse"], 0.0005 / 3)
        self.assertAlmostEqual(metric["raw"]["mse"], metric["raw"]["rmse"] ** 2)
        self.assertEqual(metric["risk_coverage"][-1]["mse"], metric["raw"]["mse"])

    def test_direction_fallback_and_abstention_use_correct_probabilities_and_samples(self):
        rows = [{"horizon": 1, "target": 1.0, "prediction": 1.0, "effective_prediction": 0.0,
                 "prob_up": 0.9, "abstain": False, "fallback": True, "risk": 0.0},
                {"horizon": 1, "target": -1.0, "prediction": 1.0, "effective_prediction": 1.0,
                 "prob_up": 0.9, "abstain": True, "fallback": False, "risk": 1.0}]
        d = score(rows, [1])["1"]["direction"]
        self.assertEqual(d["raw"]["count"], 2)
        self.assertEqual(d["served"]["count"], 1)
        self.assertAlmostEqual(d["raw"]["cross_entropy"], -(math.log(0.9) + math.log(0.1)) / 2)
        self.assertAlmostEqual(d["served"]["cross_entropy"], math.log(2))
        self.assertEqual(d["served"], d["half_probability_baseline_served"])

    def test_missing_probabilities_are_not_inferred_or_selectively_scored(self):
        rows = [{"horizon": 1, "target": 1.0, "prediction": 100.0, "effective_prediction": 100.0,
                 "abstain": False, "fallback": False, "risk": 0.0}]
        d = score(rows, [1])["1"]["direction"]
        self.assertIsNone(d["raw"]["cross_entropy"])
        self.assertIsNone(d["raw"]["accuracy"])
        self.assertEqual(d["raw"]["probability_count"], 0)
        mixed = score([*rows, {**rows[0], "prob_up": 0.9}], [1])["1"]["direction"]["raw"]
        self.assertEqual(mixed["probability_count"], 1)
        self.assertIsNone(mixed["cross_entropy"])
        self.assertIsNone(score([], [1])["1"]["direction"]["raw"]["cross_entropy"])

    def test_extreme_direction_probabilities_produce_finite_clipped_losses(self):
        rows = [{"horizon": 1, "target": t, "prediction": 0.0, "effective_prediction": 0.0,
                 "prob_up": p, "abstain": False, "fallback": False, "risk": 0.0}
                for t, p in ((1.0, 0.0), (-1.0, 1.0))]
        loss = score(rows, [1])["1"]["direction"]["raw"]["cross_entropy"]
        self.assertTrue(math.isfinite(loss))
        self.assertGreater(loss, 30)

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
