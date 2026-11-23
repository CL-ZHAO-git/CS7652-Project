import hashlib
import json
import math
import shutil
import tempfile
import time
from itertools import groupby
from pathlib import Path

from forecast_agents.tasks.panel import Panel, TaskSpec


def freeze_solution(source: Path, destination: Path):
    """Snapshot candidate sources; reject links and oversized/unbounded submissions."""
    if source.is_symlink():
        raise ValueError("Candidate root cannot be a symlink")
    source = source.resolve()
    if not source.is_dir() or not (source / "pipeline.py").is_file():
        raise ValueError("Candidate must contain solution/pipeline.py")
    entries = list(source.rglob("*"))
    if len(entries) > 1000 or any(p.is_symlink() for p in entries):
        raise ValueError("Candidate contains symlinks or too many entries")
    # Case-sensitive ordering and POSIX separators give the same hash on every OS.
    files = sorted((p for p in entries if p.is_file() and "__pycache__" not in p.parts),
                   key=lambda p: p.relative_to(source).parts)
    if sum(p.stat().st_size for p in files) > 10 * 1024 * 1024:
        raise ValueError("Candidate exceeds 10 MiB source/artifact limit")
    destination.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256()
    for source_file in files:
        relative = source_file.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        content = source_file.read_bytes()
        target.write_bytes(content)
        digest.update(relative.as_posix().encode() + b"\0" + content + b"\0")
    return digest.hexdigest()


def validate_predictions(predictions, symbols, horizons):
    if not isinstance(predictions, list):
        raise ValueError("Prediction output must be a list")
    expected = {(symbol, h) for symbol in symbols for h in horizons}
    seen = set()
    for row in predictions:
        if not isinstance(row, dict) or set(row) != {"symbol", "horizon", "prediction", "risk", "abstain", "fallback"}:
            raise ValueError("Prediction row has missing or unexpected fields")
        if not isinstance(row["symbol"], str) or type(row["horizon"]) is not int:
            raise ValueError("Invalid stock or horizon type")
        key = (row["symbol"], row["horizon"])
        if key not in expected or key in seen:
            raise ValueError("Unexpected or duplicate stock-horizon prediction")
        seen.add(key)
        for field in ("prediction", "risk"):
            if type(row[field]) not in (int, float) or not math.isfinite(row[field]):
                raise ValueError("Predictions and risks must be finite numbers")
        if row["risk"] < 0 or any(type(row[f]) is not bool for f in ("abstain", "fallback")):
            raise ValueError("Risk must be nonnegative and decisions must be booleans")
        if row["abstain"] and row["fallback"]:
            raise ValueError("Abstention and fallback are distinct decisions")
    if seen != expected:
        raise ValueError("Incomplete stock-horizon coverage")


def _loss(rows, prediction_key):
    if not rows:
        return {"mae": None, "rmse": None}
    errors = [row[prediction_key] - row["target"] for row in rows]
    return {"mae": sum(abs(e) for e in errors) / len(errors),
            "rmse": math.sqrt(sum(e * e for e in errors) / len(errors))}


def score(rows, horizons):
    result = {}
    for h in horizons:
        group = [row for row in rows if row["horizon"] == h]
        served = [row for row in group if not row["abstain"]]
        curve = []
        # Keep equal-risk observations together to avoid cherry-picking within ties.
        count = 0
        absolute_error = squared_error = 0.0
        for risk, tied_rows in groupby(sorted(group, key=lambda row: row["risk"]),
                                      key=lambda row: row["risk"]):
            for row in tied_rows:
                error = row["prediction"] - row["target"]
                count += 1
                absolute_error += abs(error)
                squared_error += error * error
            curve.append({"risk_threshold": risk, "coverage": count / len(group),
                          "mae": absolute_error / count, "rmse": math.sqrt(squared_error / count)})
        baseline = [{**row, "zero": 0.0} for row in group]
        result[str(h)] = {
            "count": len(group), "coverage": len(served) / len(group) if group else 0,
            "fallback_rate": sum(row["fallback"] for row in group) / len(group) if group else 0,
            "served": _loss(served, "effective_prediction"),
            "raw": _loss(group, "prediction"), "zero_baseline": _loss(baseline, "zero"),
            "risk_coverage": curve,
        }
    return result


def summary_metrics(metrics):
    """Per-horizon metrics without the risk-coverage curves."""
    return {h: {k: v for k, v in metric.items() if k != "risk_coverage"} for h, metric in metrics.items()}


def _report(split, task, blocks, rows_by_block, started):
    rows = [row for block_rows in rows_by_block for row in block_rows]
    return {"split": split, "protocol": task.protocol, "fit_cutoff": blocks[0].fit_cutoff,
            "origins": [origin for block in blocks for origin in block.origins],
            "blocks": [{"fit_cutoff": block.fit_cutoff, "first_origin": block.origins[0],
                        "last_origin": block.origins[-1], "origins": len(block.origins),
                        "metrics": summary_metrics(score(block_rows, task.horizons))}
                       for block, block_rows in zip(blocks, rows_by_block)],
            "metrics": score(rows, task.horizons), "rows": rows,
            "elapsed_seconds": time.monotonic() - started}


def zero_baseline(panel: Panel, task: TaskSpec, split="test"):
    """The zero-return forecast on the same blocks, origins, and labels.

    It is fixed code, so it is scored in-process instead of in per-origin sandboxes.
    """
    started = time.monotonic()
    blocks = panel.blocks(task, split)
    label_cutoff = task.validation_end if split == "validation" else panel.dates[-1]
    rows_by_block = [[{"symbol": symbol, "horizon": h, "prediction": 0.0, "risk": 0.0,
                       "abstain": False, "fallback": False, "origin": origin,
                       "fit_cutoff": block.fit_cutoff, "effective_prediction": 0.0,
                       "target": panel.label(symbol, origin, h, label_cutoff)}
                      for origin in block.origins for symbol in panel.symbols for h in task.horizons]
                     for block in blocks]
    return _report(split, task, blocks, rows_by_block, started)


def evaluate(panel: Panel, task: TaskSpec, solution: Path, executor, split="test",
             timeout_seconds=10.0, wall_seconds=120.0):
    """Fit each block at its cutoff; predict in fresh processes on strictly causal prefixes.

    Final labels remain in this controller. No per-origin workspace contains the
    full test period, a previous origin's files, or files created during fitting.
    """
    started = time.monotonic()
    blocks = panel.blocks(task, split)
    worker = Path(__file__).with_name("worker.py")

    def invoke(operation, history, state=None):
        remaining = wall_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Evaluation wall-time limit exhausted")
        # Every call uses a new folder, so a folder left behind by a transient file lock
        # (for example antivirus scanning on Windows) cannot leak into another origin.
        with tempfile.TemporaryDirectory(prefix="forecast-eval-", ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            freeze_solution(solution, root / "solution")
            shutil.copyfile(worker, root / "worker.py")
            payload = {"operation": operation, "history": history,
                       "state": state, "horizons": task.horizons}
            execution = executor.execute(root, "worker.py", json.dumps(payload),
                                         timeout=min(timeout_seconds, remaining))
            if execution.timed_out or execution.returncode:
                raise RuntimeError(f"Candidate {operation} failed: {execution.stderr[:2000]}")
            path = root / "result.json"
            if not path.is_file() or path.is_symlink() or path.stat().st_size > 10 * 1024 * 1024:
                raise ValueError("Candidate did not produce a bounded JSON result")
            return json.loads(path.read_text(encoding="utf-8"))

    label_cutoff = task.validation_end if split == "validation" else panel.dates[-1]
    rows_by_block = []
    for block in blocks:
        state = invoke("fit", panel.history(block.fit_cutoff))
        rows = []
        for origin in block.origins:
            predictions = invoke("predict", panel.history(origin), state)
            validate_predictions(predictions, panel.symbols, task.horizons)
            for prediction in predictions:
                row = {**prediction, "origin": origin, "fit_cutoff": block.fit_cutoff,
                       "target": panel.label(prediction["symbol"], origin,
                                             prediction["horizon"], label_cutoff)}
                row["effective_prediction"] = 0.0 if row["fallback"] else row["prediction"]
                rows.append(row)
        rows_by_block.append(rows)
    return _report(split, task, blocks, rows_by_block, started)
