import json

from forecast_agents import templates
from .types import Message, ModelRequest, ModelResponse, ToolCall, Usage


def _write(path, content):
    return "write_file", {"path": path, "content": content}


def _finish(artifacts):
    return "finish_stage", {"status": "completed", "artifacts": artifacts,
                            "findings": ["Deterministic fixture; not an LLM research conclusion"],
                            "recommendations": [], "next_role": ""}


class MockBackend:
    """Scripted tool calls exercise the same runtime without API credentials."""

    name = "mock"

    def generate(self, request: ModelRequest) -> ModelResponse:
        plans = {
            "diagnosis": [
                _write("diagnose.py", templates.DIAGNOSIS_CODE),
                ("run_python", {"path": "diagnose.py"}),
                _finish(["diagnostics.json"]),
            ],
            "features": [_write("solution/features.py", templates.FEATURES_CODE),
                         _finish(["solution/features.py"])],
            "modeling": [_write("solution/pipeline.py", templates.PIPELINE_CODE),
                         ("validate_candidate", {}), _finish(["solution/pipeline.py"])],
            "auditing": [_write("audit.py", templates.AUDIT_CODE),
                         ("run_python", {"path": "audit.py"}),
                         ("validate_candidate", {}), ("audit_candidate", {}), _finish(["audit.json"])],
        }
        plans["single"] = [step for name in ("diagnosis", "features", "modeling", "auditing")
                           for step in plans[name] if step[0] != "finish_stage"]
        plans["single"].append(_finish(["solution/pipeline.py", "audit.json"]))
        # Propagate failed tool execution rather than masking it as fixture success.
        if request.messages[-1].role == "tool":
            if json.loads(request.messages[-1].text).get("is_error"):
                raise RuntimeError("Mock fixture tool failed; inspect events.jsonl")
        index = (request.turn_index if request.turn_index is not None else
                 sum(m.role == "assistant" for m in request.messages))
        if request.stage not in plans or index >= len(plans[request.stage]):
            raise RuntimeError("Mock plan exhausted")
        name, arguments = plans[request.stage][index]
        message = Message("assistant", tool_calls=(ToolCall(f"{request.stage}-{index}", name, arguments),))
        # Explicit synthetic counts test accounting; they are not tokenizer measurements.
        input_count = max(1, len(json.dumps([m.text for m in request.messages])) // 4)
        output_count = max(1, len(json.dumps(arguments)) // 4)
        if output_count > request.max_output_tokens:
            return ModelResponse(Message("assistant", "Fixture output does not fit"), "length",
                                 Usage(input_count, request.max_output_tokens, source="mock"))
        return ModelResponse(message, "tool_calls", Usage(input_count, output_count, source="mock"))
