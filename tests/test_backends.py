import unittest
from unittest.mock import patch

from forecast_agents.llm.backends import AnthropicBackend, OpenAICompatibleBackend
from forecast_agents.llm.types import BackendError, Message, ModelRequest


def request(messages):
    return ModelRequest(tuple(messages), ({"name": "read", "description": "Read data",
                                          "parameters": {"type": "object", "properties": {}}},), 100, 5)


class BackendTests(unittest.TestCase):
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
