import json
from dataclasses import asdict
from pathlib import Path

from forecast_agents.evaluation.evaluator import evaluate, freeze_solution


def _schema(name, description, properties, required):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": required, "additionalProperties": False}}


STRING = {"type": "string"}
STRINGS = {"type": "array", "items": STRING}
TOOL_DEFINITIONS = (
    _schema("list_files", "List relative workspace files.", {}, []),
    _schema("read_file", "Read a workspace text file (bounded output).", {"path": STRING}, ["path"]),
    _schema("write_file", "Create or replace a workspace text file; data/ is read-only.",
            {"path": STRING, "content": STRING}, ["path", "content"]),
    _schema("run_python", "Run a workspace Python script with a bounded deadline.", {"path": STRING}, ["path"]),
    _schema("validate_candidate", "Evaluate solution/ on historical validation only; no test feedback.", {}, []),
    _schema("finish_stage", "Finish this stage or request a bounded revision. Call this tool alone.",
            {"status": {"type": "string", "enum": ["completed", "revise", "rejected"]},
             "artifacts": STRINGS, "findings": STRINGS, "recommendations": STRINGS,
             "next_role": {"type": "string", "enum": ["", "diagnosis", "features", "modeling"]}},
            ["status", "artifacts", "findings", "recommendations", "next_role"]),
)


def validate_arguments(value, schema):
    kind = schema["type"]
    valid = {"object": isinstance(value, dict), "array": isinstance(value, list),
             "string": isinstance(value, str)}[kind]
    if not valid:
        raise ValueError(f"Expected {kind}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError("Unsupported enum value")
    if kind == "object":
        if not set(schema.get("required", ())).issubset(value):
            raise ValueError("Missing required tool arguments")
        if set(value) - set(schema["properties"]):
            raise ValueError("Unexpected tool arguments")
        for key, item in value.items():
            validate_arguments(item, schema["properties"][key])
    elif kind == "array":
        for item in value:
            validate_arguments(item, schema["items"])


class WorkspaceTools:
    definitions = TOOL_DEFINITIONS

    def __init__(self, root, development_panel, task, executor, budget, events,
                 execution_timeout=15.0):
        self.root = Path(root).resolve()
        self.panel, self.task = development_panel, task
        self.executor, self.budget, self.events = executor, budget, events
        self.execution_timeout = execution_timeout
        self.validation_count = 0
        self.last_validation_hash = None

    def path(self, relative, write=False):
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError("A nonempty relative path is required")
        target = (self.root / relative).resolve()
        if not target.is_relative_to(self.root) or target == self.root:
            raise ValueError("Path escapes the workspace")
        if write and (target.is_relative_to(self.root / "data") or target.name in {"task.json", "PIPELINE_CONTRACT.md"}):
            raise ValueError("Task inputs are read-only")
        return target

    def execute(self, name, arguments):
        definitions = {tool["name"]: tool for tool in self.definitions}
        if name not in definitions:
            raise ValueError("Unknown tool")
        validate_arguments(arguments, definitions[name]["parameters"])
        if name == "list_files":
            return {"files": sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*")
                                    if p.is_file() and not p.is_symlink())[:1000]}
        if name == "read_file":
            with self.path(arguments["path"]).open(encoding="utf-8") as stream:
                text = stream.read(32_001)
            return {"text": text[:32_000], "truncated": len(text) > 32_000}
        if name == "write_file":
            if len(arguments["content"].encode()) > 1_000_000:
                raise ValueError("Tool write exceeds 1 MB")
            path = self.path(arguments["path"], write=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(arguments["content"], encoding="utf-8")
            return {"written": arguments["path"]}
        if name == "run_python":
            path = self.path(arguments["path"])
            if path.suffix != ".py":
                raise ValueError("Only Python scripts can be executed")
            result = self.executor.execute(self.root, str(path.relative_to(self.root)),
                                           timeout=min(self.execution_timeout, self.budget.remaining_seconds))
            if result.returncode or result.timed_out:
                return {"is_error": True, **asdict(result)}
            return asdict(result)
        if name == "validate_candidate":
            self.validation_count += 1
            snapshot = self.root.parent / "validation_snapshots" / str(self.validation_count)
            digest = freeze_solution(self.root / "solution", snapshot)
            report = evaluate(self.panel, self.task, snapshot, self.executor, "validation",
                              self.execution_timeout, self.budget.remaining_seconds)
            target = self.root / "validation.json"
            # Validation targets are historical development data, never final targets.
            target.write_text(json.dumps(report, indent=2), encoding="utf-8")
            self.last_validation_hash = digest
            self.events.emit("validation", candidate_hash=digest, metrics=report["metrics"])
            return {"report": "validation.json", "candidate_hash": digest,
                    "metrics": {h: {k: v for k, v in metrics.items() if k != "risk_coverage"}
                                for h, metrics in report["metrics"].items()}}
        if name == "finish_stage":
            for artifact in arguments["artifacts"]:
                if not self.path(artifact).is_file():
                    raise ValueError(f"Missing declared artifact: {artifact}")
            if arguments["status"] == "revise" and not arguments["next_role"]:
                raise ValueError("Revision requires next_role")
            if arguments["status"] != "revise" and arguments["next_role"]:
                raise ValueError("next_role is only valid for revisions")
            return {"finished": True, **arguments}
        raise AssertionError("Unreachable tool")
