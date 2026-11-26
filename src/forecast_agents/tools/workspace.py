import json
import hashlib
import tempfile
import ast
import re
from dataclasses import asdict
from pathlib import Path

from forecast_agents.evaluation.evaluator import evaluate, freeze_solution, summary_metrics
from forecast_agents.evaluation.audit import audit_candidate, AuditFailure
from forecast_agents.tools.privacy import DataAccessError, describe_panel, execution_summary, rounded_summary
from forecast_agents.runtime.errors import CandidateExecutionError, PredictionContractError


def _schema(name, description, properties, required):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": required, "additionalProperties": False}}


STRING = {"type": "string"}
STRINGS = {"type": "array", "items": STRING}
PROTECTED_NAMES = {"task.json", "pipeline_contract.md"}
TOOL_DEFINITIONS = (
    _schema("describe_data", "Compute local, rounded development-panel statistics; returns no raw rows or price series.", {}, []),
    _schema("read_validation_summary", "Read the latest controller-produced aggregate validation metrics, without prediction rows.", {}, []),
    _schema("audit_candidate", "Audit a complete validated pipeline plus features, not features alone. Requires solution/pipeline.py and features.py:features_at(history, origin). Feature-only stages should use local helper tests. Returns aggregate pass/evidence only.", {}, []),
    _schema("list_files", "List relative workspace files.", {}, []),
    _schema("read_file", "Read a text-file preview: offset >= 0 and max_chars from 1 to 8000. "
            "Follow next_offset for subsequent pages; offsets count characters, not lines. "
            "Data previews are capped at 1000 characters; analyze full data using a Python script.",
            {"path": STRING, "offset": {"type": "integer", "minimum": 0},
             "max_chars": {"type": "integer", "minimum": 1, "maximum": 8000}}, ["path"]),
    _schema("write_file", "Create or replace a workspace text file; data/ is read-only.",
            {"path": STRING, "content": STRING}, ["path", "content"]),
    _schema("run_python", "Run a workspace Python script with a bounded deadline.", {"path": STRING}, ["path"]),
    _schema("validate_candidate", "Evaluate solution/ on historical validation only, overall and per "
            "walk-forward fold; no test feedback. Compare raw with zero_baseline (all rows), or served "
            "with zero_baseline_served (identical non-abstained rows). Reports MAE/MSE/RMSE plus direction "
            "cross entropy/accuracy when prob_up exists; compare to the matching half-probability baseline. "
            "raw is not unshrunk predictions.", {}, []),
    _schema("finish_stage", "Finish this stage or request a bounded revision. Call alone. "
            "Omit next_role for completed/rejected; only revise specifies a next_role.",
            {"status": {"type": "string", "enum": ["completed", "revise", "rejected"]},
             "artifacts": STRINGS, "findings": STRINGS, "recommendations": STRINGS,
             "next_role": {"type": "string", "enum": ["", "diagnosis", "features", "modeling"],
                           "description": "Revision target only. Omit for completed/rejected."}},
            ["status", "artifacts", "findings", "recommendations"]),
)


def validate_arguments(value, schema):
    kind = schema["type"]
    valid = {"object": isinstance(value, dict), "array": isinstance(value, list),
             "string": isinstance(value, str), "integer": type(value) is int}[kind]
    if not valid:
        raise ValueError(f"Expected {kind}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError("Unsupported enum value")
    if kind == "integer" and (("minimum" in schema and value < schema["minimum"])
                              or ("maximum" in schema and value > schema["maximum"])):
        # These bounds apply to read_file only. Never echo attempted values or paths.
        raise DataAccessError("read_file requires offset >= 0 and 1 <= max_chars <= 8000. "
                              "Use the returned next_offset to read another page; offsets count characters, not lines.")
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
                 execution_timeout=15.0, summaries_only=False):
        self.root = Path(root).resolve()
        self.panel, self.task = development_panel, task
        self.executor, self.budget, self.events = executor, budget, events
        self.execution_timeout = execution_timeout
        self.validation_count = 0
        self.last_validation_hash = None
        self.last_audit_hash = None
        self.summaries_only = summaries_only
        self.owned_files = {}
        self.owned_keys = {}
        self.trusted_inputs = {name: (self.root / name).read_text(encoding="utf-8")
                               for name in ("task.json", "PIPELINE_CONTRACT.md") if (self.root / name).is_file()}
        self.last_validation_summary = None
        if summaries_only:
            descriptions = {
                "read_file": "Read only original task/contract text or unchanged agent-written files. offset >= 0, max_chars 1..8000; follow next_offset for pages (character offsets, not line numbers). Raw data/reports are blocked; use describe_data/read_validation_summary.",
                "run_python": "Execute local Python against local data. Returns status and sanitized error type/line only; stdout/stderr and reports stay local. Use write_file for source edits.",
                "list_files": "List known task paths and agent-written paths only; runtime-created names are withheld.",
            }
            self.definitions = tuple({**tool, "description": descriptions.get(tool["name"], tool["description"])}
                                     for tool in TOOL_DEFINITIONS)

    def error_result(self, exc):
        if isinstance(exc, CandidateExecutionError):
            recorded = asdict(exc.execution)
            if self.events is not None:
                self.events.emit("candidate_execution_failure", operation=exc.operation, execution=recorded)
            result = execution_summary(recorded, self.owned_files, self.known_debug_keys())
            result.update(error=str(exc), operation=exc.operation)
            return result
        if not self.summaries_only or isinstance(exc, (DataAccessError, PredictionContractError)):
            return {"is_error": True, "error": str(exc)}
        # Candidate exceptions can contain raw values, source excerpts or printed data.
        return {"is_error": True, "error_type": type(exc).__name__,
                "error": "Local tool failed; details remain in local logs. Use controller summaries and execution status for feedback."}

    def known_debug_keys(self):
        # Only strings already authored by the LLM or declared in the public task.
        keys = {str(h) for h in self.task.horizons} if self.task is not None else set()
        for path, values in self.owned_keys.items():
            source = self.root / path
            if source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest() == self.owned_files.get(path):
                keys.update(values)
        return keys

    def audit_is_current(self):
        if self.last_audit_hash is None or self.last_audit_hash != self.last_validation_hash:
            return False
        with tempfile.TemporaryDirectory(prefix="forecast-hash-") as directory:
            return freeze_solution(self.root / "solution", Path(directory) / "source") == self.last_audit_hash

    def path(self, relative, write=False):
        # Forward-slash relative paths only, so tool behavior is identical on every OS.
        if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
                or relative.startswith("/") or "\\" in relative or ":" in relative):
            raise ValueError("A nonempty relative path with forward slashes is required")
        target = (self.root / relative).resolve()
        if not target.is_relative_to(self.root) or target == self.root:
            raise ValueError("Path escapes the workspace")
        # Case-insensitive, because macOS and Windows file systems usually are.
        parts = [part.casefold() for part in target.relative_to(self.root).parts]
        if write and (parts[0] == "data" or parts[-1] in PROTECTED_NAMES):
            raise ValueError("Task inputs are read-only")
        return target

    def execute(self, name, arguments):
        definitions = {tool["name"]: tool for tool in self.definitions}
        if name not in definitions:
            raise ValueError("Unknown tool")
        validate_arguments(arguments, definitions[name]["parameters"])
        if name == "describe_data":
            return describe_panel(self.panel, self.task)
        if name == "read_validation_summary":
            if self.last_validation_summary is None:
                raise DataAccessError("No validation summary yet; call validate_candidate first")
            return self.last_validation_summary
        if name == "list_files":
            if self.summaries_only:
                return {"files": sorted(set(self.trusted_inputs) | set(self.owned_files) | {"data/history.json"}),
                        "runtime_files_withheld": True}
            return {"files": sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob("*")
                                    if p.is_file() and not p.is_symlink())[:1000]}
        if name == "read_file":
            path = self.path(arguments["path"])
            offset, limit = arguments.get("offset", 0), arguments.get("max_chars", 8000)
            if offset < 0 or not 1 <= limit <= 8000:
                raise DataAccessError("read_file requires offset >= 0 and 1 <= max_chars <= 8000. Use next_offset for subsequent pages.")
            if self.summaries_only:
                relative = path.relative_to(self.root).as_posix()
                if relative in self.trusted_inputs:
                    content = self.trusted_inputs[relative]
                elif relative in self.owned_files:
                    raw = path.read_bytes()
                    if hashlib.sha256(raw).hexdigest() != self.owned_files[relative]:
                        raise DataAccessError("Source was modified by local execution. Rewrite code with write_file; runtime-produced content cannot be read into the LLM.")
                    content = raw.decode("utf-8")
                else:
                    raise DataAccessError("Data and runtime-produced reports are local-only. Use describe_data or read_validation_summary; only unchanged agent-written files can be read.")
                text = content[offset:offset + limit]
                truncated = offset + limit < len(content)
                return {"text": text, "truncated": truncated, "next_offset": offset + limit if truncated else None}
            is_data = path.relative_to(self.root).parts[0].casefold() == "data"
            limit = min(limit, 1000) if is_data else limit
            with path.open(encoding="utf-8") as stream:
                # Character offsets, not seek cookies or byte offsets; bound memory.
                remaining = offset
                while remaining:
                    skipped = stream.read(min(remaining, 8192))
                    if not skipped:
                        break
                    remaining -= len(skipped)
                text = stream.read(limit + 1)
            truncated = len(text) > limit
            result = {"text": text[:limit], "truncated": truncated,
                      "next_offset": offset + limit if truncated else None}
            if is_data:
                result["hint"] = "Preview only. Load the complete file in a workspace Python script; print compact statistics."
            return result
        if name == "write_file":
            if len(arguments["content"].encode()) > 1_000_000:
                raise ValueError("Tool write exceeds 1 MB")
            path = self.path(arguments["path"], write=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            # newline="\n": the agent's exact text, not CRLF-translated on Windows.
            path.write_text(arguments["content"], encoding="utf-8", newline="\n")
            self.owned_files[path.relative_to(self.root).as_posix()] = hashlib.sha256(arguments["content"].encode("utf-8")).hexdigest()
            relative = path.relative_to(self.root).as_posix()
            self.owned_keys[relative] = set()
            if path.suffix == ".py":
                try:
                    tree = ast.parse(arguments["content"])
                    self.owned_keys[relative] = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)
                        and isinstance(node.value, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}|[0-9]{1,2}", node.value)}
                except (SyntaxError, ValueError, RecursionError):
                    pass
            return {"written": arguments["path"]}
        if name == "run_python":
            path = self.path(arguments["path"])
            if path.suffix != ".py":
                raise ValueError("Only Python scripts can be executed")
            result = self.executor.execute(self.root, str(path.relative_to(self.root)),
                                           timeout=min(self.execution_timeout, self.budget.remaining_seconds))
            recorded = asdict(result)
            if self.events is not None:
                self.events.emit("execution_output", script=arguments["path"], result=recorded)
            if self.summaries_only:
                return execution_summary(recorded, self.owned_files, self.known_debug_keys())
            preview = dict(recorded)
            for field, limit in (("stdout", 4000), ("stderr", 2000)):
                preview[field] = recorded[field][:limit]
                preview[field + "_truncated"] = len(recorded[field]) > limit
            if result.returncode or result.timed_out:
                return {"is_error": True, **preview}
            return preview
        if name == "audit_candidate":
            self.last_audit_hash = None
            if not (self.root / "solution/pipeline.py").is_file():
                raise DataAccessError("audit_candidate requires a complete solution/pipeline.py and current validation. It cannot audit features alone; run local feature smoke checks instead.")
            with tempfile.TemporaryDirectory(prefix="forecast-audit-") as directory:
                snapshot = Path(directory) / "solution"
                digest = freeze_solution(self.root / "solution", snapshot)
                if digest != self.last_validation_hash:
                    raise DataAccessError("Validate the current candidate before audit_candidate")
                try:
                    report = audit_candidate(self.panel, self.task, snapshot, self.executor,
                                             self.execution_timeout, self.budget.remaining_seconds)
                except AuditFailure as exc:
                    raise DataAccessError(str(exc)) from None
            self.last_audit_hash = digest
            result = {"candidate_hash": digest, **report}
            (self.root / "controller_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            self.events.emit("candidate_audit", **result)
            return rounded_summary(result) if self.summaries_only else result
        if name == "validate_candidate":
            self.last_validation_hash = None
            self.last_validation_summary = None
            if not (self.root / "solution/pipeline.py").is_file():
                raise DataAccessError("No solution/pipeline.py exists. Write the fit/predict pipeline before validate_candidate; feature-only checks use run_python.")
            self.validation_count += 1
            snapshot = self.root.parent / "validation_snapshots" / str(self.validation_count)
            digest = freeze_solution(self.root / "solution", snapshot)
            report = evaluate(self.panel, self.task, snapshot, self.executor, "validation",
                              self.execution_timeout, self.budget.remaining_seconds, require_prob_up=True)
            target = self.root / "validation.json"
            # Validation targets are historical development data, never final targets.
            target.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
            self.last_validation_hash = digest
            self.events.emit("validation", candidate_hash=digest, metrics=report["metrics"])
            result = {"report": "validation.json", "candidate_hash": digest,
                      "metrics": summary_metrics(report["metrics"]), "folds": report["blocks"]}
            self.last_validation_summary = rounded_summary(result) if self.summaries_only else result
            return self.last_validation_summary
        if name == "finish_stage":
            arguments = {"next_role": "", **arguments}
            for artifact in arguments["artifacts"]:
                if not self.path(artifact).is_file():
                    raise ValueError(f"Missing declared artifact: {artifact}")
            if arguments["status"] == "revise" and not arguments["next_role"]:
                raise ValueError("Revision requires next_role")
            if arguments["status"] != "revise" and arguments["next_role"]:
                raise ValueError("next_role is only valid for revisions")
            return {"finished": True, **arguments}
        raise AssertionError("Unreachable tool")
