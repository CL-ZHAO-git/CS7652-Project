"""Check DeepSeek authentication; optionally test a tiny fictional tool-call cycle.

No dataset files are read. By default only GET /models is requested. --probe
makes two small, billable generation requests with at most 256 output tokens each.
The key is read from DEEPSEEK_API_KEY or key.txt and never printed or saved.
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forecast_agents.llm.backends import OpenAICompatibleBackend
from forecast_agents.llm.credentials import read_key_file
from forecast_agents.llm.types import Message, ModelRequest

BASE_URL = "https://api.deepseek.com"


def load_key(path):
    value = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if value:
        return value
    if not path.is_file():
        raise ValueError("Set DEEPSEEK_API_KEY or provide a local key.txt")
    return read_key_file(path)


def list_models(key):
    request = urllib.request.Request(BASE_URL + "/models",
                                     headers={"Authorization": "Bearer " + key})
    with urllib.request.urlopen(request, timeout=20) as response:
        raw = json.load(response)
    return [row["id"] for row in raw.get("data", []) if isinstance(row.get("id"), str)]


def probe(model, key):
    previous = os.environ.get("DEEPSEEK_API_KEY")
    os.environ["DEEPSEEK_API_KEY"] = key
    try:
        backend = OpenAICompatibleBackend(model, BASE_URL, "DEEPSEEK_API_KEY", "max_tokens",
                                          request_options={"thinking": {"type": "disabled"}})
        tool = {"name": "add_numbers", "description": "Add two integers.", "parameters": {
            "type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"], "additionalProperties": False}}
        messages = [Message("user", "Call add_numbers with a=2 and b=3. After receiving the tool result, reply with only the result digit.")]
        first = backend.generate(ModelRequest(tuple(messages), (tool,), 256, 30))
        if first.stop_reason != "tool_calls" or len(first.message.tool_calls) != 1:
            raise ValueError("Authentication passed, but expected one complete tool call")
        call = first.message.tool_calls[0]
        if call.name != "add_numbers" or call.arguments != {"a": 2, "b": 3}:
            raise ValueError("Model returned unexpected test tool arguments")
        messages += [first.message, Message("tool", '{"result": 5}', tool_call_id=call.id)]
        second = backend.generate(ModelRequest(tuple(messages), (tool,), 256, 30))
        if second.stop_reason != "end" or second.message.text.strip() != "5":
            raise ValueError("Tool result was sent, but the model did not finish with the expected answer")
        return {"tool_call": "passed", "continuation": "passed", "answer": "5",
                "usage": [{"input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens}
                          for r in (first, second)]}
    finally:
        if previous is None:
            os.environ.pop("DEEPSEEK_API_KEY", None)
        else:
            os.environ["DEEPSEEK_API_KEY"] = previous


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path, default=ROOT / "key.txt")
    parser.add_argument("--probe", action="store_true", help="make two small billable tool-call test requests")
    parser.add_argument("--model", help="model ID from the authenticated model list")
    args = parser.parse_args(argv)
    key = ""
    report = {"authentication": "not verified"}
    try:
        key = load_key(args.key_file)
        models = list_models(key)
        report.update(authentication="passed", available_models=models)
        if args.probe:
            model = args.model or next((m for m in ("deepseek-flash", "deepseek-chat") if m in models), None)
            if not model or model not in models:
                raise ValueError("Choose an available model explicitly using --model")
            report.update(model=model, probe=probe(model, key))
    except Exception as exc:
        message = str(exc).replace(key, "[REDACTED]") if key else str(exc)
        report.update(error_type=type(exc).__name__, error=message[:400])
        print(json.dumps(report, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
