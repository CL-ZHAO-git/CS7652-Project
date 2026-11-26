"""Cumulative role allowances drawn from one shared experiment budget."""

import math
from dataclasses import asdict, dataclass

from forecast_agents.runtime.budget import BudgetExceeded, BudgetLimits


@dataclass(frozen=True)
class SchedulingConfig:
    enabled: bool = True
    diagnosis_share: float = 0.15
    features_share: float = 0.15
    modeling_share: float = 0.45
    auditing_share: float = 0.25
    finish_only_fraction: float = 0.8
    finish_call_reserve: int = 2
    finish_output_tokens: int = 1024
    input_token_margin: int = 1024
    force_tool_choice: bool = False
    max_policy_corrections: int = 1

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("scheduling.enabled must be a boolean")
        if type(self.force_tool_choice) is not bool:
            raise ValueError("scheduling.force_tool_choice must be a boolean")
        if type(self.max_policy_corrections) is not int or self.max_policy_corrections < 0:
            raise ValueError("scheduling.max_policy_corrections must be a nonnegative integer")
        shares = [self.share(role) for role in ("diagnosis", "features", "modeling", "auditing")]
        if any(type(s) not in (int, float) or not math.isfinite(s) or not 0 < s < 1 for s in shares):
            raise ValueError("Stage shares must be finite values between zero and one")
        if not math.isclose(sum(shares), 1.0, abs_tol=1e-9, rel_tol=0):
            raise ValueError("Stage shares must sum to one")
        if (type(self.finish_only_fraction) not in (int, float) or
                not math.isfinite(self.finish_only_fraction) or not 0 < self.finish_only_fraction < 1):
            raise ValueError("finish_only_fraction must be between zero and one")
        for name in ("finish_call_reserve", "finish_output_tokens", "input_token_margin"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"scheduling.{name} must be a positive integer")

    def share(self, role):
        return 1.0 if role == "single" else getattr(self, role + "_share")


class StageBudgetExceeded(BudgetExceeded):
    pass


class StageAllowance:
    """Reused across revisions. Wall time counts only while this role is active."""

    def __init__(self, role, shared, config):
        self.role, self.shared, self.config = role, shared, config
        share = config.share(role)
        self.limits = BudgetLimits(max_llm_calls=max(1, int(shared.limits.max_llm_calls * share)),
                                   max_total_tokens=max(1, int(shared.limits.max_total_tokens * share)),
                                   max_tool_calls=max(1, int(shared.limits.max_tool_calls * share)),
                                   wall_seconds=shared.limits.wall_seconds * share)
        self.llm_calls = self.tool_calls = self.total_tokens = 0
        self.elapsed = 0.0
        self.active_since = None
        self.last_input_tokens = self.last_context_chars = None
        self.finalizing = False
        self.policy_corrections = 0

    def start(self):
        self.active_since = self.shared.clock()

    def stop(self):
        self.elapsed = self.elapsed_seconds
        self.active_since = None

    @property
    def elapsed_seconds(self):
        return self.elapsed + (self.shared.clock() - self.active_since if self.active_since is not None else 0)

    @property
    def remaining_seconds(self):
        if not self.config.enabled:
            return self.shared.remaining_seconds
        return min(self.shared.remaining_seconds, max(0.0, self.limits.wall_seconds - self.elapsed_seconds))

    @property
    def remaining_tokens(self):
        shared = self.shared.limits.max_total_tokens - self.shared.total_tokens
        return min(shared, self.limits.max_total_tokens - self.total_tokens) if self.config.enabled else shared

    def check(self):
        self.shared.check()
        if self.config.enabled:
            if self.total_tokens >= self.limits.max_total_tokens:
                raise StageBudgetExceeded(f"Stage {self.role}: observed token allowance exhausted")
            if self.remaining_seconds <= 0:
                raise StageBudgetExceeded(f"Stage {self.role}: wall-time allowance exhausted")

    def finish_only(self):
        if not self.config.enabled:
            return False
        if self.finalizing:
            return True
        cfg = self.config
        call_left = min(self.limits.max_llm_calls - self.llm_calls,
                        self.shared.limits.max_llm_calls - self.shared.llm_calls)
        tool_left = min(self.limits.max_tool_calls - self.tool_calls,
                        self.shared.limits.max_tool_calls - self.shared.tool_calls)
        token_buffer = cfg.finish_call_reserve * ((self.last_input_tokens or 0)
                                                + cfg.input_token_margin + cfg.finish_output_tokens)
        self.finalizing = (self.total_tokens >= self.limits.max_total_tokens * cfg.finish_only_fraction
                or call_left <= cfg.finish_call_reserve or tool_left <= cfg.finish_call_reserve
                or self.remaining_tokens <= token_buffer
                or self.elapsed_seconds >= self.limits.wall_seconds * cfg.finish_only_fraction)
        return self.finalizing

    def estimate_input(self, context_chars, tool_chars, compact_finalization=False):
        # Admission estimate only: providers report exact usage after generation.
        # Anchor to this role's last prompt usage and account for context growth.
        if compact_finalization and self.last_input_tokens is not None and self.last_context_chars:
            # A deliberately rebuilt decision-only prompt excludes the large old
            # context. Scale its empirical density, with an independent floor.
            estimate = max(math.ceil((context_chars + tool_chars) / 3),
                           math.ceil(self.last_input_tokens * context_chars / self.last_context_chars))
        elif self.last_input_tokens is None:
            estimate = math.ceil((context_chars + tool_chars) / 3)
        else:
            estimate = self.last_input_tokens + max(0, math.ceil((context_chars - self.last_context_chars) / 3))
        return estimate + self.config.input_token_margin

    def begin_call(self):
        self.check()
        if self.config.enabled and self.llm_calls >= self.limits.max_llm_calls:
            raise StageBudgetExceeded(f"Stage {self.role}: model-call allowance exhausted")
        self.shared.begin_call()
        self.llm_calls += 1

    def charge(self, usage, context_chars):
        if usage.total is not None:
            self.total_tokens += usage.total
        self.last_input_tokens, self.last_context_chars = usage.input_tokens, context_chars
        self.shared.charge(usage)
        self.check()

    def begin_tool(self):
        self.check()
        if self.config.enabled and self.tool_calls >= self.limits.max_tool_calls:
            raise StageBudgetExceeded(f"Stage {self.role}: tool-call allowance exhausted")
        self.shared.begin_tool()
        self.tool_calls += 1

    def snapshot(self):
        return {"role": self.role, "enforced": self.config.enabled, "limits": asdict(self.limits),
                "llm_calls": self.llm_calls, "tool_calls": self.tool_calls,
                "total_tokens": self.total_tokens, "elapsed_seconds": self.elapsed_seconds,
                "finish_only": self.finalizing, "policy_corrections": self.policy_corrections}
