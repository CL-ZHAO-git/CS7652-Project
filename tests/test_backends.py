import unittest
from dataclasses import replace
from unittest.mock import patch

from forecast_agents.llm.backends import AnthropicBackend, OpenAICompatibleBackend
from forecast_agents.llm.types import BackendError, Message, ModelRequest


def request(messages):
    return ModelRequest(tuple(messages), ({"name": "read", "description": "Read data",
                                          "parameters": {"type": "object", "properties": {}}},), 100, 5)


class BackendTests(unittest.TestCase):
    def test_explicit_tool_choice_is_mapped_to_each_provider(self):
        for choice in ("read", "required"):
            captured = []

            def transport(url, headers, payload, timeout):
                captured.append(payload)
                return {"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
                        "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "input_tokens": 1, "output_tokens": 1}}

            with patch.dict("os.environ", {"TEST_MODEL_KEY": "fixture-only"}):
                for cls in (OpenAICompatibleBackend, AnthropicBackend):
                    backend = cls("test", "http://localhost/v1", "TEST_MODEL_KEY", transport=transport)
                    backend.generate(replace(request([Message("user", "Close")]), tool_choice=choice))
            self.assertEqual(captured[0]["tool_choice"], "required" if choice == "required" else
                             {"type": "function", "function": {"name": "read"}})
            self.assertEqual(captured[1]["tool_choice"], {"type": "any"} if choice == "required" else
                             {"type": "tool", "name": "read"})

    def test_unavailable_forced_tool_is_rejected_before_transport(self):
        for cls in (OpenAICompatibleBackend, AnthropicBackend):
            with patch.dict("os.environ", {"TEST_MODEL_KEY": "fixture-only"}):
                def transport(*args):
                    self.fail("Invalid tool choice reached the network transport")
                backend = cls("test", "http://localhost/v1", "TEST_MODEL_KEY", transport=transport)
                with self.assertRaises(BackendError):
                    backend.generate(replace(request([Message("user", "Close")]), tool_choice="unavailable"))

    def test_openai_tool_call_round_trip_preserves_native_state_and_usage(self):
        payloads = []

        def transport(url, headers, payload, timeout):
            payloads.append(payload)
            return {"id": "r1", "choices": [{"finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "reasoning_content": "opaque state",
                "tool_calls": [{"id": "c1", "type": "function", "function": {
                    "name": "read", "arguments": "{}"}}]}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20,
                              "completion_tokens_details": {"reasoning_tokens": 5}}}

        with patch.dict("os.environ", {"TEST_MODEL_KEY": "fixture-only"}):
            backend = OpenAICompatibleBackend("test", "http://localhost/v1", "TEST_MODEL_KEY", transport=transport)
            response = backend.generate(request([Message("user", "hello")]))
            backend.generate(request([Message("user", "hello"), response.message,
                                      Message("tool", "done", tool_call_id="c1")]))
        self.assertEqual(response.usage.total, 30)  # reasoning is a subset, not added twice
        self.assertEqual(response.message.tool_calls[0].arguments, {})
        self.assertEqual(payloads[1]["messages"][1]["reasoning_content"], "opaque state")
        self.assertEqual(payloads[1]["messages"][2]["tool_call_id"], "c1")

    def test_anthropic_preserves_thinking_blocks_and_groups_tool_results(self):
        payloads = []
        native = [{"type": "thinking", "thinking": "opaque", "signature": "signature"},
                  {"type": "tool_use", "id": "a", "name": "read", "input": {}},
                  {"type": "tool_use", "id": "b", "name": "read", "input": {}}]

        def transport(url, headers, payload, timeout):
            payloads.append(payload)
            return {"content": native, "stop_reason": "tool_use", "usage": {
                "input_tokens": 4, "output_tokens": 6, "cache_read_input_tokens": 10}}

        with patch.dict("os.environ", {"TEST_MODEL_KEY": "fixture-only"}):
            backend = AnthropicBackend("test", "http://localhost/v1", "TEST_MODEL_KEY", transport)
            response = backend.generate(request([Message("system", "system"), Message("user", "start")]))
            backend.generate(request([Message("user", "start"), response.message,
                                      Message("tool", "one", tool_call_id="a"),
                                      Message("tool", "two", tool_call_id="b")]))
        self.assertEqual(response.usage.total, 20)
        self.assertEqual(payloads[1]["messages"][1]["content"], native)
        self.assertEqual(len(payloads[1]["messages"][2]["content"]), 2)
        self.assertEqual(payloads[1]["messages"][2]["role"], "user")

    def test_malformed_tool_arguments_keep_reported_usage(self):
        def transport(*args):
            return {"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [
                {"id": "c", "function": {"name": "read", "arguments": "not JSON"}}]}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 7}}
        with patch.dict("os.environ", {"TEST_MODEL_KEY": "fixture-only"}):
            backend = OpenAICompatibleBackend("test", "http://localhost/v1", "TEST_MODEL_KEY", transport=transport)
            with self.assertRaises(BackendError) as caught:
                backend.generate(request([Message("user", "start")]))
        self.assertEqual(caught.exception.usage.total, 10)

    def test_missing_usage_stays_unknown(self):
        with patch.dict("os.environ", {"TEST_MODEL_KEY": "fixture-only"}):
            backend = AnthropicBackend("test", "http://localhost/v1", "TEST_MODEL_KEY",
                                       lambda *a: {"content": [], "stop_reason": "end_turn"})
            response = backend.generate(request([Message("user", "start")]))
        self.assertIsNone(response.usage.total)
