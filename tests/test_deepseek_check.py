import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from forecast_agents.llm.types import Message, ModelResponse, ToolCall, Usage

spec = importlib.util.spec_from_file_location("check_deepseek", Path(__file__).parents[1] / "scripts/check_deepseek.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
FIXTURE_KEY = "sk-" + "fixture" * 5


class DeepSeekCheckTests(unittest.TestCase):
    def test_key_file_is_read_without_echoing_credentials(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"DEEPSEEK_API_KEY": ""}):
            path = Path(directory) / "key.txt"
            path.write_text('DEEPSEEK_API_KEY="' + FIXTURE_KEY + '"\n')
            self.assertEqual(check.load_key(path), FIXTURE_KEY)

    def test_failed_requests_redact_the_key(self):
        output = io.StringIO()
        with patch.object(check, "load_key", return_value=FIXTURE_KEY), \
                patch.object(check, "list_models", side_effect=RuntimeError("failed " + FIXTURE_KEY)), \
                contextlib.redirect_stdout(output):
            self.assertEqual(check.main([]), 1)
        self.assertNotIn(FIXTURE_KEY, output.getvalue())
        self.assertEqual(json.loads(output.getvalue())["authentication"], "not verified")

    def test_default_check_makes_no_generation_calls(self):
        with patch.object(check, "load_key", return_value=FIXTURE_KEY), \
                patch.object(check, "list_models", return_value=["deepseek-flash"]), \
                patch.object(check, "probe") as probe, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(check.main([]), 0)
            probe.assert_not_called()

    def test_probe_round_trips_tools_and_restores_environment(self):
        responses = [ModelResponse(Message("assistant", tool_calls=(ToolCall("call-1", "add_numbers", {"a": 2, "b": 3}),)),
                                   "tool_calls", Usage(10, 10)),
                     ModelResponse(Message("assistant", "5"), "end", Usage(15, 1))]
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "original"}), \
                patch.object(check.OpenAICompatibleBackend, "generate", side_effect=responses) as generate:
            result = check.probe("deepseek-flash", FIXTURE_KEY)
            self.assertEqual(result["continuation"], "passed")
            self.assertEqual(generate.call_args.args[0].messages[-1].tool_call_id, "call-1")
            self.assertEqual(check.os.environ["DEEPSEEK_API_KEY"], "original")
