import math
import os
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

from forecast_agents.runtime.budget import BudgetLimits
from forecast_agents.tasks.panel import TaskSpec


@dataclass(frozen=True)
class DataConfig:
    source: str = "synthetic"
    path: str = ""
    stocks: int = 4
    sessions: int = 120
    seed: int = 7
    train_sessions: int = 70
    validation_sessions: int = 25
    horizons: tuple[int, ...] = (1, 3, 5)
    # Optional calendar boundaries (YYYY-MM-DD). When set, they replace the session
    # counts; each resolves to the last panel session on or before the date.
    train_end: str = ""
    validation_end: str = ""
    # Walk-forward refits at each "year", "quarter", or "month" end ("none": one fit).
    refit: str = "none"
    validation_stride: int = 1
    test_stride: int = 1


@dataclass(frozen=True)
class LLMConfig:
    backend: str = "mock"
    model: str = "mock-fixture-v1"
    base_url: str = ""
    api_key_env: str = "LLM_API_KEY"
    token_parameter: str = "max_completion_tokens"
    max_output_tokens: int = 4096
    timeout_seconds: float = 60


@dataclass(frozen=True)
class ExecutionConfig:
    mode: str = "local"
    image: str = "python:3.13-slim"
    cpus: float = 1.0
    memory_mb: int = 1024
    timeout_seconds: float = 15
    evaluation_wall_seconds: float = 120


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str = "multi"
    max_revisions: int = 1
    data: DataConfig = field(default_factory=DataConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    budget: BudgetLimits = field(default_factory=BudgetLimits)


def load_config(path: Path) -> ExperimentConfig:
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    allowed = {"mode", "max_revisions", "data", "llm", "execution", "budget"}
    if set(raw) - allowed:
        raise ValueError("Unknown top-level configuration keys")
    sections = {}
    for name, cls in (("data", DataConfig), ("llm", LLMConfig),
                      ("execution", ExecutionConfig), ("budget", BudgetLimits)):
        values = dict(raw.get(name, {}))
        for key, value in values.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                variable = value[2:-1]
                if not os.environ.get(variable):
                    raise ValueError(f"Missing configuration environment variable: {variable}")
                values[key] = os.environ[variable]
        if name == "data" and "horizons" in values:
            values["horizons"] = tuple(values["horizons"])
        if name == "data":
            for key in ("train_end", "validation_end"):
                if type(values.get(key)) is date:  # unquoted TOML date
                    values[key] = values[key].isoformat()
            if {"train_end", "validation_end"} & set(values) and {"train_sessions", "validation_sessions"} & set(values):
                raise ValueError("Use data.train_end/validation_end dates or session counts, not both")
        try:
            sections[name] = cls(**values)
        except TypeError as exc:
            raise ValueError(f"Invalid {name} configuration: {exc}") from None
    config = ExperimentConfig(raw.get("mode", "multi"), raw.get("max_revisions", 1), **sections)
    return validate_config(config)


def validate_config(config: ExperimentConfig) -> ExperimentConfig:
    if config.mode not in {"single", "multi"}:
        raise ValueError("Mode must be single or multi")
    if type(config.max_revisions) is not int or config.max_revisions < 0:
        raise ValueError("max_revisions must be a nonnegative integer")
    if config.llm.backend not in {"mock", "openai_compatible", "anthropic"}:
        raise ValueError("Unsupported LLM backend")
    if config.execution.mode not in {"local", "docker"}:
        raise ValueError("Unsupported executor")
    if config.llm.backend != "mock" and config.execution.mode != "docker":
        raise ValueError("Real LLM-generated code requires Docker; local mode is for trusted mock fixtures")
    if config.llm.backend != "mock" and not config.llm.base_url:
        raise ValueError("Real backends require an explicit base_url")
    if config.llm.backend != "mock":
        url = urlsplit(config.llm.base_url)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("base_url must be an HTTP(S) endpoint without credentials, query, or fragment")
        if not config.llm.model or not config.llm.api_key_env:
            raise ValueError("Model and credential environment-variable name are required")
    if config.llm.token_parameter not in {"max_tokens", "max_completion_tokens"}:
        raise ValueError("Unsupported token parameter")
    if config.data.source not in {"synthetic", "json"}:
        raise ValueError("Data source must be synthetic or json")
    for key in ("stocks", "sessions", "train_sessions", "validation_sessions"):
        if type(getattr(config.data, key)) is not int or getattr(config.data, key) <= 0:
            raise ValueError(f"data.{key} must be a positive integer")
    numeric = (config.llm.max_output_tokens, config.llm.timeout_seconds,
               config.execution.cpus, config.execution.memory_mb,
               config.execution.timeout_seconds, config.execution.evaluation_wall_seconds)
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in numeric):
        raise ValueError("Execution and generation limits must be finite and positive")
    if type(config.llm.max_output_tokens) is not int or type(config.execution.memory_mb) is not int:
        raise ValueError("Token and memory limits must be integers")
    TaskSpec("2000-01-01", "2000-01-02", config.data.horizons, config.data.refit,
             config.data.validation_stride, config.data.test_stride)
    boundaries =(config.data.train_end, config.data.validation_end)
    if any(type(day) is not str for day in boundaries) or bool(boundaries[0]) != bool(boundaries[1]):
        raise ValueError("Set both data.train_end and data.validation_end, or neither")
    if boundaries[0]:
        TaskSpec(*boundaries, config.data.horizons)  # canonical dates, train before validation
    if config.data.source == "json" and not config.data.path:
        raise ValueError("JSON data source requires data.path")
    return config


def config_record(config):
    # Configuration stores an environment-variable NAME, never the credential value.
    return asdict(config)
