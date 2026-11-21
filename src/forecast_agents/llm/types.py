from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Message:
    role: str
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    # Native assistant blocks can contain signatures/state needed for continuation.
    native: dict[str, Any] = field(default_factory=dict)
    is_error: bool = False


@dataclass(frozen=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    source: str = "provider"
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for value in (self.input_tokens, self.output_tokens):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("Token counts must be nonnegative integers or unknown")

    @property
    def total(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class ModelRequest:
    messages: tuple[Message, ...]
    tools: tuple[dict[str, Any], ...]
    max_output_tokens: int
    timeout_seconds: float
    stage: str = ""


@dataclass(frozen=True)
class ModelResponse:
    message: Message
    stop_reason: str
    usage: Usage
    request_id: str | None = None


class BackendError(RuntimeError):
    def __init__(self, message: str, usage: Usage | None = None):
        super().__init__(message)
        self.usage = usage or Usage()


class LLMBackend(Protocol):
    name: str

    def generate(self, request: ModelRequest) -> ModelResponse: ...


def as_json(value: Any) -> dict[str, Any]:
    return asdict(value)
