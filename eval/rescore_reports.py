"""Recompute aggregate metrics from saved local rows, without rerunning agents."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forecast_agents.evaluation.evaluator import score, summary_metrics
from forecast_agents.runtime.reproducibility import file_hash, tree_hash


def rescore(input_root, output):
    input_root, output = Path(input_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new output file; existing reports are never overwritten")
    comparison_path = input_root / "comparison.json"
    source_hashes = {}
    if comparison_path.is_file():
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        if comparison["status"] in {"prepared", "running"}:
            raise ValueError("Comparison must have finished before rescoring")
        source_hashes["comparison.json"] = file_hash(comparison_path)
        entries = comparison["runs"]
        status = comparison["status"]
    else:
        run = json.loads((input_root / "run.json").read_text(encoding="utf-8"))
        entries = [{"output": ".", "mode": run["mode"], "status": run["status"]}]
        status = run["status"]
    conditions = []
    for entry in entries:
        condition = {k: entry[k] for k in ("mode", "repetition", "status", "output") if k in entry}
        conditions.append(condition)
        run_root = (input_root / entry["output"]).resolve()
        if not run_root.is_relative_to(input_root):
            raise ValueError("Condition path escapes the input root")
        if entry["status"] != "completed":
            condition["metrics"] = None  # A failed run never becomes a successful baseline.
            continue
        reports = {}
        for filename in ("run.json", "test_report.json", "baseline_report.json"):
            path = run_root / filename
            source_hashes[path.relative_to(input_root).as_posix()] = file_hash(path)
            reports[filename] = json.loads(path.read_text(encoding="utf-8"))
        report = reports["test_report.json"]
        if reports["run.json"]["status"] != "completed":
            raise ValueError("Condition status does not match the saved run")
        horizons = sorted({r["horizon"] for r in report["rows"]})
        condition["metrics"] = summary_metrics(score(report["rows"], horizons))
        condition["blocks"] = []
        for block in report["blocks"]:
            rows = [r for r in report["rows"] if r["fit_cutoff"] == block["fit_cutoff"]
                    and block["first_origin"] <= r["origin"] <= block["last_origin"]]
            condition["blocks"].append({**{k: v for k, v in block.items() if k != "metrics"},
                                        "metrics": summary_metrics(score(rows, horizons))})
        baseline = reports["baseline_report.json"]
        # The no-information direction baseline is explicitly p=0.5, separate
        # from the zero-return point forecast and from any legacy missing p.
        baseline_rows = [{**r, "prob_up": 0.5} for r in baseline["rows"]]
        condition["baseline_metrics"] = summary_metrics(score(baseline_rows, horizons))
    result = {"source_status": status, "conditions": conditions,
              "metric_definitions": {"mse": "mean((prediction - target)**2)",
                  "direction_label": "1 if target > 0 else 0 (including zero returns)",
                  "cross_entropy": "mean(-y*ln(p) - (1-y)*ln(1-p)); p clipped to [1e-15, 1-1e-15]",
                  "probability_field": "prob_up", "accuracy_threshold": 0.5,
                  "fallback_probability": 0.5,
                  "missing_probabilities": "cross_entropy/accuracy unavailable; never infer from return or risk"},
              "scorer_sha256": tree_hash(ROOT / "src"), "source_sha256": source_hashes,
              "paid_calls": 0, "candidate_code_executed": False}
    for relative, digest in source_hashes.items():
        if file_hash(input_root / relative) != digest:
            raise ValueError("Source reports changed during rescoring")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path, help="Completed run or comparison directory")
    parser.add_argument("--output", required=True, type=Path, help="New aggregate JSON file")
    args = parser.parse_args(argv)
    try:
        result = rescore(args.input_root, args.output)
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__,
                          "error": "Cannot rescore saved reports; check paths, completion status and report schema."}))
        return 1
    print(json.dumps({"status": "completed", "output": str(args.output),
                      "conditions": len(result["conditions"]), "paid_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
