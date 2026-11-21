import math
import time
from dataclasses import asdict, dataclass
from typing import Callable

from forecast_agents.llm.types import Usage


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class BudgetLimits:
    max_llm_calls: int = 80
    max_total_tokens: int = 100_000
    max_tool_calls: int = 150
    wall_seconds: float = 300

    def __post_init__(self):
        for value in asdict(self).values():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("Budget limits must be positive")
        if any(type(getattr(self, key)) is not int for key in
               ("max_llm_calls", "max_total_tokens", "max_tool_calls")):
            raise ValueError("Call and token limits must be integers")


class Budget:
    """Shared ledger. Token limit is an observed-use stop, not a pre-count guarantee."""

    def __init__(self, limits: BudgetLimits, clock: Callable[[], float] = time.monotonic):
        self.limits, self.clock = limits, clock
        self.started = clock()
        self.llm_calls = self.tool_calls = self.total_tokens = self.unknown_usage_calls = 0

    @property
    def remaining_seconds(self):
        return max(0.0, self.limits.wall_seconds - (self.clock() - self.started))

    def check(self):
        if self.remaining_seconds <= 0:
            raise BudgetExceeded("Wall-time budget exhausted")
        if self.unknown_usage_calls:
            raise BudgetExceeded("Unknown model usage; run stopped rather than charging zero")
        if self.total_tokens >= self.limits.max_total_tokens:
            raise BudgetExceeded("Observed token budget exhausted")

    def begin_call(self):
        self.check()
        if self.llm_calls >= self.limits.max_llm_calls:
            raise BudgetExceeded("LLM-call budget exhausted")
        self.llm_calls += 1

    def charge(self, usage: Usage):
        if usage.total is None:
            self.unknown_usage_calls += 1
        else:
            self.total_tokens += usage.total
        self.check()

    def begin_tool(self):
        self.check()
        if self.tool_calls >= self.limits.max_tool_calls:
            raise BudgetExceeded("Tool-call budget exhausted")
        self.tool_calls += 1

    def snapshot(self):
        return {"limits": asdict(self.limits), "llm_calls": self.llm_calls,
                "tool_calls": self.tool_calls, "total_tokens": self.total_tokens,
                "unknown_usage_calls": self.unknown_usage_calls,
                "elapsed_seconds": self.clock() - self.started}
