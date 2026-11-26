"""Prepare frozen inputs, then run sequential, counterbalanced single/multi pairs.

--prepare-only does not load credentials, contact APIs, or access Docker. Execute
that prepared plan exactly once with --execute-prepared. Interrupted comparisons
are preserved and cannot be resumed; failed individual conditions are retained.
"""

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forecast_agents.config import config_record, load_config
from forecast_agents.llm.credentials import key_environment
from forecast_agents.runtime.executor import Executor
from forecast_agents.runtime.reproducibility import file_hash, tree_hash

LAUNCH = ("import sys; sys.path.insert(0,sys.argv.pop(1)); "
          "from forecast_agents.cli import main; raise SystemExit(main())")


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare(config_path, output, repetitions):
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    config = load_config(config_path)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol_dir = output / "protocol"
    protocol_dir.mkdir()
    shutil.copytree(ROOT / "src", protocol_dir / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(Path(__file__), protocol_dir / "run_comparison.py")
    data_info = {"source": config.data.source, "seed": config.data.seed}
    if config.data.source == "json":
        private = output / "private"
        private.mkdir(mode=0o700)
        snapshot = private / "panel.json"
        shutil.copyfile(Path(config.data.path), snapshot)
        data_info.update(sha256=file_hash(snapshot), original_path=config.data.path,
                         snapshot="private/panel.json")
        config = replace(config, data=replace(config.data, path=str(snapshot)))
    frozen_config = protocol_dir / "config.json"
    dump(frozen_config, config_record(config))
    protocol = {"created": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                "framework_sha256": tree_hash(protocol_dir / "src"),
                "driver_sha256": file_hash(protocol_dir / "run_comparison.py"),
                "config_sha256": file_hash(frozen_config), "dataset": data_info,
                "repetitions": repetitions,
                "orders": [["single", "multi"] if i % 2 == 0 else ["multi", "single"] for i in range(repetitions)],
                "model_image_note": "Image is resolved once before execution; provider aliases are not immutable model-weight versions."}
    dump(output / "protocol.json", protocol)
    dump(output / "comparison.json", {"status": "prepared", "runs": [], "protocol_sha256": file_hash(output / "protocol.json")})
    return output


def verify(output, protocol):
    state = json.loads((output / "comparison.json").read_text())
    if file_hash(output / "protocol.json") != state["protocol_sha256"]:
        raise ValueError("Prepared protocol was modified")
    if tree_hash(output / "protocol/src") != protocol["framework_sha256"]:
        raise ValueError("Frozen framework was modified")
    if file_hash(Path(__file__)) != protocol["driver_sha256"]:
        raise ValueError("Comparison driver changed; prepare a new plan")
    if file_hash(output / "protocol/config.json") != protocol["config_sha256"]:
        raise ValueError("Frozen configuration was modified")
    if protocol["dataset"]["source"] == "json":
        if file_hash(output / protocol["dataset"]["snapshot"]) != protocol["dataset"]["sha256"]:
            raise ValueError("Frozen dataset was modified")


def execute(output, key_file=None):
    output = output.resolve()
    protocol = json.loads((output / "protocol.json").read_text())
    state = json.loads((output / "comparison.json").read_text())
    if state["status"] != "prepared":
        raise ValueError("Only an unstarted prepared plan can be executed; choose a new output root")
    verify(output, protocol)
    config = load_config(output / "protocol/config.json")
    # Resolve once before any paid request, then pass the image ID to both arms.
    executor = Executor(config.execution.mode, config.execution.image,
                        config.execution.cpus, config.execution.memory_mb)
    if executor.mode == "docker":
        config = replace(config, execution=replace(config.execution, image=executor.image))
    execution_config = output / "protocol/execution_config.json"
    dump(execution_config, config_record(config))
    execution_hash = file_hash(execution_config)
    with key_environment(config.llm.api_key_env, key_file):
        state.update(status="running", resolved_image=executor.image,
                     execution_config_sha256=execution_hash, started=datetime.now(timezone.utc).isoformat())
        dump(output / "comparison.json", state)
        active_entry = None
        try:
            for repetition, order in enumerate(protocol["orders"], 1):
                for mode in order:
                    verify(output, protocol)
                    if file_hash(execution_config) != execution_hash:
                        raise ValueError("Execution configuration changed between conditions")
                    destination = output / f"repeat-{repetition:02d}" / mode
                    if destination.exists():
                        raise ValueError("Condition output already exists")
                    command = [sys.executable, "-I", "-X", "utf8", "-c", LAUNCH,
                               str(output / "protocol/src"), "run", "--config", str(execution_config),
                               "--mode", mode, "--output", str(destination)]
                    entry = {"repetition": repetition, "mode": mode, "status": "running",
                             "output": str(destination.relative_to(output))}
                    active_entry = entry
                    state["runs"].append(entry)
                    dump(output / "comparison.json", state)
                    print(f"Starting repetition {repetition}/{len(protocol['orders'])}: {mode}", file=sys.stderr, flush=True)
                    completed = subprocess.run(command, capture_output=True, text=True)
                    destination.mkdir(parents=True, exist_ok=True)
                    dump(destination / "controller_console.json", {"stdout": completed.stdout, "stderr": completed.stderr})
                    summary_path = destination / "run.json"
                    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
                    passed = completed.returncode == 0 and summary.get("status") == "completed"
                    entry.update(status="completed" if passed else "failed", returncode=completed.returncode,
                                 budget=summary.get("budget"), metrics=summary.get("metrics"),
                                 error=summary.get("error", "Missing run summary" if not summary else None))
                    verify(output, protocol)
                    if file_hash(execution_config) != execution_hash:
                        entry["status"] = "invalid_protocol"
                        raise ValueError("Execution configuration changed during a condition")
                    dump(output / "comparison.json", state)
                    print(f"Finished {mode}: {entry['status']}", file=sys.stderr, flush=True)
                    active_entry = None
            state["status"] = "completed" if all(r["status"] == "completed" for r in state["runs"]) else "completed_with_failures"
        except Exception as exc:
            state.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            if active_entry is not None:
                active_entry["status"] = "invalid_protocol" if isinstance(exc, ValueError) else "failed"
        except KeyboardInterrupt:
            state.update(status="interrupted", error="Comparison interrupted; not resumable")
            if active_entry is not None and active_entry["status"] == "running":
                active_entry["status"] = "interrupted"
        finally:
            state["ended"] = datetime.now(timezone.utc).isoformat()
            dump(output / "comparison.json", state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--execute-prepared", type=Path)
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.execute_prepared:
            if args.config or args.output_root or args.prepare_only:
                raise ValueError("--execute-prepared cannot be combined with preparation options")
            output = args.execute_prepared
        else:
            if not args.config or not args.output_root:
                raise ValueError("Preparation requires --config and --output-root")
            output = prepare(args.config, args.output_root, args.repetitions)
            if args.prepare_only:
                print(json.dumps({"status": "prepared", "output": str(output), "paid_calls": 0}, indent=2))
                return 0
        result = execute(output, args.key_file)
        print(json.dumps({"status": result["status"], "output": str(output),
                          "runs": [{k: r[k] for k in ("repetition", "mode", "status")} for r in result["runs"]]}, indent=2))
        return 0 if result["status"] == "completed" else 1
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Comparison error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
