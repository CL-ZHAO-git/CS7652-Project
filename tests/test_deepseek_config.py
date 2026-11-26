import contextlib
import io
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from forecast_agents.cli import main
from forecast_agents.config import config_record, load_config, validate_config
from forecast_agents.llm.backends import OpenAICompatibleBackend
from forecast_agents.llm.credentials import key_environment
from forecast_agents.llm.types import Message, ModelRequest

ROOT = Path(__file__).parents[1]
FIXTURE_KEY = "sk-" + "fixture" * 5


class DeepSeekConfigTests(unittest.TestCase):
    def test_dedicated_config_uses_official_endpoint_and_synthetic_data(self):
        config = load_config(ROOT / "configs/deepseek.toml")
        self.assertEqual(config.llm.base_url, "https://api.deepseek.com")
        self.assertEqual(config.llm.model, "deepseek-flash")
        self.assertEqual(config.llm.request_options, {"thinking": {"type": "disabled"}})
        self.assertEqual(config.data.source, "synthetic")
        self.assertEqual(config.execution.mode, "docker")
        self.assertTrue(config.scheduling.force_tool_choice)

    def test_deepseek_forced_tool_choice_rejects_thinking_mode(self):
        config = load_config(ROOT / "configs/deepseek.toml")
        for options in ({}, {"thinking": {"type": "enabled"}}, {"reasoning_effort": "high"}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "non-thinking"):
                validate_config(replace(config, llm=replace(config.llm, request_options=options)))

    def test_explicit_options_and_token_limit_are_sent_to_backend(self):
        captured = []

        def transport(url, headers, payload, timeout):
            captured.append((url, payload))
            return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "5"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1}}

        with patch.dict(os.environ, {"TEST_MODEL_KEY": FIXTURE_KEY}):
            backend = OpenAICompatibleBackend("deepseek-flash", "https://api.deepseek.com", "TEST_MODEL_KEY",
                                              "max_tokens", transport,
                                              request_options={"thinking": {"type": "disabled"}})
            backend.generate(ModelRequest((Message("user", "test"),), (), 256, 10))
        url, payload = captured[0]
        self.assertEqual(url, "https://api.deepseek.com/chat/completions")
        self.assertEqual(payload["max_tokens"], 256)
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertNotIn("max_completion_tokens", payload)

    def test_options_cannot_override_model_tools_or_credentials(self):
        config = load_config(ROOT / "configs/deepseek.toml")
        for options in ({"model": "other"}, {"tools": []}, {"api_key": "forbidden"},
                        {"thinking": {"type": "adaptive"}}, {"reasoning_effort": "unknown"}):
            with self.subTest(option_names=list(options)), self.assertRaises(ValueError):
                validate_config(replace(config, llm=replace(config.llm, request_options=options)))
        with self.assertRaises(ValueError):
            validate_config(replace(config, llm=replace(config.llm, backend="anthropic")))

    def test_key_environment_restores_original_value_even_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "key.txt"
            path.write_text(FIXTURE_KEY)
            with patch.dict(os.environ, {"TEST_MODEL_KEY": "original"}):
                with self.assertRaises(RuntimeError):
                    with key_environment("TEST_MODEL_KEY", path):
                        self.assertEqual(os.environ["TEST_MODEL_KEY"], FIXTURE_KEY)
                        raise RuntimeError("test failure")
                self.assertEqual(os.environ["TEST_MODEL_KEY"], "original")

    def test_cli_loads_key_only_into_controller_not_configuration_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "key.txt"
            path.write_text(FIXTURE_KEY)
            output = io.StringIO()

            def fake_run(config, destination):
                self.assertEqual(os.environ[config.llm.api_key_env], FIXTURE_KEY)
                self.assertNotIn(FIXTURE_KEY, json.dumps(config_record(config)))
                return {"status": "completed", "mode": config.mode, "mock": False}

            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "original"}), \
                    patch("forecast_agents.cli.run_experiment", side_effect=fake_run), \
                    contextlib.redirect_stdout(output):
                status = main(["run", "--config", str(ROOT / "configs/deepseek.toml"),
                               "--key-file", str(path), "--output", str(Path(directory) / "run")])
                self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "original")
            self.assertEqual(status, 0)
            self.assertNotIn(FIXTURE_KEY, output.getvalue())
