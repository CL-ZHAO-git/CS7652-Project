"""HTTP adapters. No implicit retries, provider switching, or tool execution."""

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable

from .types import BackendError, Message, ModelRequest, ModelResponse, ToolCall, Usage

Transport = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]


def post_json(url, headers, payload, timeout):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        # Do not copy server bodies/headers, URLs, or credentials into run logs.
        raise BackendError(f"Model API returned HTTP {exc.code}; usage unknown") from None
    except (OSError, ValueError) as exc:
        raise BackendError(f"Model request failed ({type(exc).__name__}); usage unknown") from None


def _calls(calls):
    if len({call.id for call in calls}) != len(calls):
        raise ValueError("Duplicate tool-call IDs")
    for call in calls:
        if not call.id or not call.name or not isinstance(call.arguments, dict):
            raise ValueError("Invalid tool call")
    return tuple(calls)


def chat_request_options(options):
    """Allow explicit reasoning controls without overriding tools or credentials."""
    if not isinstance(options, dict) or set(options) - {"thinking", "reasoning_effort"}:
        raise ValueError("request_options supports only thinking and reasoning_effort")
    if "thinking" in options:
        thinking = options["thinking"]
        if (not isinstance(thinking, dict) or set(thinking) != {"type"}
                or thinking["type"] not in ("enabled", "disabled")):
            raise ValueError("thinking must be {type: enabled} or {type: disabled}")
    if "reasoning_effort" in options and options["reasoning_effort"] not in (
            "none", "minimal", "low", "medium", "high", "xhigh", "max"):
        raise ValueError("Unsupported reasoning_effort")
    return json.loads(json.dumps(options))


def _tool_choice(request):
    choice = request.tool_choice
    if choice is not None and not isinstance(choice, str):
        raise BackendError("Tool choice must be a string", Usage(0, 0))
    if choice is not None and choice != "required" and choice not in {t["name"] for t in request.tools}:
        raise BackendError("Requested tool choice is unavailable", Usage(0, 0))
    return choice


class OpenAICompatibleBackend:
    name = "openai_compatible"

    def __init__(self, model, base_url, api_key_env, token_parameter="max_completion_tokens",
                 transport: Transport = post_json, request_options=None):
        if token_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("Unsupported token limit parameter")
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key_env, self.token_parameter = api_key_env, token_parameter
        self.transport = transport
        self.request_options = chat_request_options(request_options if request_options is not None else {})

    def generate(self, request: ModelRequest) -> ModelResponse:
        messages = []
        for message in request.messages:
            if message.role == "assistant" and self.name in message.native:
                item = message.native[self.name]
            else:
                item = {"role": message.role, "content": message.text}
                if message.tool_call_id:
                    item["tool_call_id"] = message.tool_call_id
                if message.tool_calls:
                    item["tool_calls"] = [
                        {"id": c.id, "type": "function", "function": {
                            "name": c.name, "arguments": json.dumps(c.arguments)}}
                        for c in message.tool_calls
                    ]
            messages.append(item)
        payload = {
            "model": self.model, "messages": messages,
            self.token_parameter: request.max_output_tokens,
            "tools": [{"type": "function", "function": tool} for tool in request.tools],
        }
        payload.update(self.request_options)
        choice = _tool_choice(request)
        if choice is not None:
            payload["tool_choice"] = ("required" if choice == "required" else
                                      {"type": "function", "function": {"name": choice}})
        key = os.environ.get(self.api_key_env)
        if not key:
            raise BackendError(f"Missing credential environment variable: {self.api_key_env}", Usage(0, 0))
        raw = self.transport(self.base_url + "/chat/completions",
                             {"Authorization": f"Bearer {key}"}, payload, request.timeout_seconds)
        usage_raw = raw.get("usage") or {}
        usage = Usage(usage_raw.get("prompt_tokens"), usage_raw.get("completion_tokens"), raw=usage_raw)
        try:
            choice = raw["choices"][0]
            original = choice["message"]
            calls = _calls([
                ToolCall(c["id"], c["function"]["name"], json.loads(c["function"]["arguments"]))
                for c in original.get("tool_calls", [])
            ])
            text = original.get("content") or ""
            if not isinstance(text, str):
                raise ValueError("Only text and function calls are supported")
            reason = {"stop": "end", "tool_calls": "tool_calls", "length": "length",
                      "content_filter": "refusal"}.get(choice["finish_reason"], "unsupported")
            if original.get("refusal"):
                reason = "refusal"
            return ModelResponse(Message("assistant", text, calls,
                                         native={self.name: original}), reason, usage, raw.get("id"))
        except (KeyError, IndexError, TypeError, ValueError):
            raise BackendError("Malformed or unsupported chat-completion response", usage) from None


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, model, base_url, api_key_env, transport: Transport = post_json):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key_env, self.transport = api_key_env, transport

    def generate(self, request: ModelRequest) -> ModelResponse:
        messages = []
        system = []
        for message in request.messages:
            if message.role == "system":
                system.append(message.text)
                continue
            role = "user" if message.role == "tool" else message.role
            if message.role == "tool":
                blocks = [{"type": "tool_result", "tool_use_id": message.tool_call_id,
                           "content": message.text, "is_error": message.is_error}]
            elif message.role == "assistant" and self.name in message.native:
                blocks = message.native[self.name]["content"]
            else:
                blocks = [{"type": "text", "text": message.text}] if message.text else []
                blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                           for c in message.tool_calls]
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"].extend(blocks)
            else:
                messages.append({"role": role, "content": list(blocks)})
        key = os.environ.get(self.api_key_env)
        if not key:
            raise BackendError(f"Missing credential environment variable: {self.api_key_env}", Usage(0, 0))
        payload = {
            "model": self.model, "system": "\n\n".join(system), "messages": messages,
            "max_tokens": request.max_output_tokens,
            "tools": [{"name": t["name"], "description": t["description"],
                       "input_schema": t["parameters"]} for t in request.tools],
        }
        choice = _tool_choice(request)
        if choice is not None:
            payload["tool_choice"] = {"type": "any"} if choice == "required" else {"type": "tool", "name": choice}
        raw = self.transport(self.base_url + "/messages",
                             {"x-api-key": key, "anthropic-version": "2023-06-01"},
                             payload, request.timeout_seconds)
        u = raw.get("usage") or {}
        # Cache input is reported separately by this API; count it once in total input.
        input_tokens = u.get("input_tokens")
        if input_tokens is not None:
            input_tokens += u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0)
        usage = Usage(input_tokens, u.get("output_tokens"), raw=u)
        try:
            blocks = raw["content"]
            calls = _calls([ToolCall(b["id"], b["name"], b["input"])
                            for b in blocks if b["type"] == "tool_use"])
            text = "\n".join(b["text"] for b in blocks if b["type"] == "text")
            reason = {"tool_use": "tool_calls", "end_turn": "end", "max_tokens": "length",
                      "refusal": "refusal"}.get(raw["stop_reason"], "unsupported")
            return ModelResponse(Message("assistant", text, calls,
                                         native={self.name: {"content": blocks}}),
                                 reason, usage, raw.get("id"))
        except (KeyError, TypeError, ValueError):
            raise BackendError("Malformed or unsupported Anthropic response", usage) from None
