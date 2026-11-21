import argparse
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from forecast_agents.agents.runner import AgentRunner
from forecast_agents.config import config_record, load_config, validate_config
from forecast_agents.evaluation.evaluator import evaluate, freeze_solution
from forecast_agents.llm.backends import AnthropicBackend, OpenAICompatibleBackend
from forecast_agents.llm.mock import MockBackend
from forecast_agents.orchestration.policies import run_policy
from forecast_agents.runtime.budget import Budget
from forecast_agents.runtime.events import EventLog
from forecast_agents.runtime.executor import Executor
from forecast_agents.tasks.panel import Bar, Panel, TaskSpec, synthetic_panel
from forecast_agents.templates import PIPELINE_CONTRACT, ZERO_CODE
from forecast_agents.tools.workspace import WorkspaceTools


def _dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_experiment(config, output: Path):
    validate_config(config)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    events = EventLog(output / "events.jsonl")
    budget = Budget(config.budget)
    summary = {"status": "running", "mode": config.mode, "backend": config.llm.backend,
               "mock": config.llm.backend == "mock", "config": config_record(config),
               "python": sys.version, "started": datetime.now(timezone.utc).isoformat()}
    _dump(output / "run.json", summary)
    try:
        data = config.data
        if data.source == "synthetic":
            panel = synthetic_panel(data.stocks, data.sessions, data.seed)
        else:
            panel = Panel([Bar(**row) for row in json.loads(Path(data.path).read_text())])
        train = data.train_sessions - 1
        validation = train + data.validation_sessions
        if validation >= len(panel.dates) - max(data.horizons):
            raise ValueError("Insufficient data for the requested train/validation/test windows")
        task = TaskSpec(panel.dates[train], panel.dates[validation], data.horizons)
        panel.origins(task, "validation")
        panel.origins(task, "test")
        workspace = output / "workspace"
        (workspace / "data").mkdir(parents=True)
        _dump(workspace / "data/history.json", panel.history(task.validation_end))
        _dump(workspace / "task.json", {**asdict(task), "stock_pool": list(panel.symbols),
                                      "protocol": "fixed_fit_at_train_end"})
        (workspace / "PIPELINE_CONTRACT.md").write_text(PIPELINE_CONTRACT)
        # Kept outside every container mount. Docker never receives this directory.
        private = output / "private"
        private.mkdir(mode=0o700)
        _dump(private / "panel.json", panel.history(panel.dates[-1]))
        executor = Executor(config.execution.mode, config.execution.image,
                            config.execution.cpus, config.execution.memory_mb)
        summary["execution"] = {"mode": executor.mode, "resolved_image": executor.image}
        events.emit("run_started", mode=config.mode, backend=config.llm.backend,
                    task=asdict(task), execution=summary["execution"])
        llm = config.llm
        if llm.backend == "mock":
            backend = MockBackend()
        elif llm.backend == "openai_compatible":
            backend = OpenAICompatibleBackend(llm.model, llm.base_url, llm.api_key_env, llm.token_parameter)
        else:
            backend = AnthropicBackend(llm.model, llm.base_url, llm.api_key_env)
        # The agent tool layer receives a development-only panel, not the full panel.
        development = Panel([Bar(**row) for row in panel.history(task.validation_end)])
        tools = WorkspaceTools(workspace, development, task, executor, budget, events,
                               config.execution.timeout_seconds)
        runner = AgentRunner(backend, tools, budget, events,
                             llm.max_output_tokens, llm.timeout_seconds)
        handoffs = run_policy(config.mode, runner, config.max_revisions)
        _dump(output / "handoffs.json", handoffs)
        candidate_hash = freeze_solution(workspace / "solution", output / "submission")
        if candidate_hash != tools.last_validation_hash:
            raise ValueError("Final candidate differs from the last validated source snapshot")
        budget.check()
        summary["agent_budget"] = budget.snapshot()
        summary["candidate_hash"] = candidate_hash
        events.emit("submission_frozen", candidate_hash=candidate_hash)
        # Final execution has its own common, recorded evaluation budget. No live LLM.
        report = evaluate(panel, task, output / "submission", executor, "test",
                          config.execution.timeout_seconds, config.execution.evaluation_wall_seconds)
        _dump(output / "test_report.json", report)
        baseline_dir = output / "baseline"
        baseline_dir.mkdir()
        (baseline_dir / "pipeline.py").write_text(ZERO_CODE)
        baseline = evaluate(panel, task, baseline_dir, executor, "test",
                            config.execution.timeout_seconds, config.execution.evaluation_wall_seconds)
        _dump(output / "baseline_report.json", baseline)
        summary.update(status="completed", metrics={h: {k: v for k, v in metric.items() if k != "risk_coverage"}
                                                   for h, metric in report["metrics"].items()})
        events.emit("run_completed", candidate_hash=candidate_hash)
    except Exception as exc:
        summary.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        events.emit("run_failed", error_type=type(exc).__name__, error=str(exc))
    finally:
        summary["budget"] = summary.get("agent_budget", budget.snapshot())
        _dump(output / "run.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a reproducible forecasting-agent experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--config", type=Path, default=Path("configs/demo.toml"))
    run.add_argument("--mode", choices=("single", "multi"))
    run.add_argument("--output", type=Path, required=True, help="New run directory; existing paths are never overwritten")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.mode:
            config = replace(config, mode=args.mode)
        summary = run_experiment(config, args.output)
    except (ValueError, OSError, TypeError) as exc:
        print(f"Configuration/run error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": summary["status"], "mode": summary["mode"],
                      "mock": summary["mock"], "output": str(args.output),
                      "error": summary.get("error")}, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
