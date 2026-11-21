"""Offline smoke evaluation, usable without installing the package or dependencies."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forecast_agents.cli import run_experiment
from forecast_agents.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    config = load_config(Path(__file__).resolve().parents[1] / "configs/demo.toml")
    summaries = {mode: run_experiment(replace(config, mode=mode), args.output_root / mode)
                 for mode in ("single", "multi")}
    result = {mode: {"status": s["status"], "candidate_hash": s.get("candidate_hash"),
                     "budget": s["budget"], "metrics": s.get("metrics")}
              for mode, s in summaries.items()}
    (args.output_root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({mode: s["status"] for mode, s in summaries.items()}, indent=2))
    return 0 if all(s["status"] == "completed" for s in summaries.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
