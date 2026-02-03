"""Token cost tracking service.

Estimates USD costs for LLM API calls based on provider-specific
pricing tables.  Supports per-session and per-experiment budgets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Pricing per 1M tokens (input/output) — approximate as of 2025
_PRICING: dict[str, dict[str, tuple[float, float]]] = {
    "openai": {
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4-turbo": (10.00, 30.00),
        "gpt-3.5-turbo": (0.50, 1.50),
    },
    "anthropic": {
        "claude-opus-4-20250514": (15.00, 75.00),
        "claude-sonnet-4-20250514": (3.00, 15.00),
        "claude-haiku-3-5-20241022": (0.80, 4.00),
        "claude-3-5-sonnet-20241022": (3.00, 15.00),
    },
    "google": {
        "gemini-2.0-flash": (0.10, 0.40),
        "gemini-1.5-pro": (1.25, 5.00),
        "gemini-1.5-flash": (0.075, 0.30),
    },
    "ollama": {},  # Local models — zero cost
}


@dataclass
class CostRecord:
    """Record of a single LLM call's cost."""

    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
    agent_role: str = ""  # "attacker", "analyzer", "defender", "target"


@dataclass
class CostTracker:
    """Tracks cumulative costs across a session."""

    max_cost_usd: float = 10.0
    records: list[CostRecord] = field(default_factory=list)
    _total_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_tokens(self) -> int:
        return sum(r.prompt_tokens + r.completion_tokens for r in self.records)

    @property
    def budget_remaining(self) -> float:
        return max(0.0, self.max_cost_usd - self._total_cost)

    @property
    def budget_exceeded(self) -> bool:
        return self._total_cost >= self.max_cost_usd

    def record(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        agent_role: str = "",
    ) -> CostRecord:
        """Record a new LLM call and update totals.

        Args:
            provider: Provider name ("openai", "anthropic", "google", "ollama").
            model: Model identifier.
            prompt_tokens: Number of input tokens.
            completion_tokens: Number of output tokens.
            agent_role: Which agent made the call.

        Returns:
            The CostRecord with estimated cost.
        """
        cost = estimate_cost(provider, model, prompt_tokens, completion_tokens)
        rec = CostRecord(
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=cost,
            agent_role=agent_role,
        )
        self.records.append(rec)
        self._total_cost += cost
        return rec

    def breakdown_by_agent(self) -> dict[str, float]:
        """Return cost breakdown by agent role."""
        breakdown: dict[str, float] = {}
        for r in self.records:
            role = r.agent_role or "unknown"
            breakdown[role] = breakdown.get(role, 0.0) + r.estimated_cost_usd
        return breakdown

    def preflight_check(
        self,
        provider: str,
        model: str,
        estimated_prompt_tokens: int,
        estimated_completion_tokens: int,
    ) -> PreflightResult:
        """Check if a planned LLM call would stay within budget.

        Call this before making an LLM API call to avoid exceeding limits.

        Args:
            provider: Provider name.
            model: Model identifier.
            estimated_prompt_tokens: Estimated input tokens.
            estimated_completion_tokens: Estimated output tokens.

        Returns:
            PreflightResult indicating whether the call is safe to proceed.
        """
        est_cost = estimate_cost(
            provider, model, estimated_prompt_tokens, estimated_completion_tokens
        )
        projected_total = self._total_cost + est_cost
        within_budget = projected_total <= self.max_cost_usd

        return PreflightResult(
            allowed=within_budget,
            estimated_cost=est_cost,
            projected_total=projected_total,
            budget_remaining=self.budget_remaining,
            budget_max=self.max_cost_usd,
        )

    def summary(self) -> dict[str, Any]:
        """Return a summary dict of all cost data."""
        return {
            "total_cost_usd": self._total_cost,
            "total_tokens": self.total_tokens,
            "budget_max": self.max_cost_usd,
            "budget_remaining": self.budget_remaining,
            "budget_exceeded": self.budget_exceeded,
            "num_calls": len(self.records),
            "by_agent": self.breakdown_by_agent(),
        }


@dataclass(frozen=True)
class PreflightResult:
    """Result of a pre-flight budget check."""

    allowed: bool
    estimated_cost: float
    projected_total: float
    budget_remaining: float
    budget_max: float


def estimate_cost(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Estimate the USD cost for a single LLM call.

    Returns 0.0 for local providers (Ollama) or unknown models.
    """
    if provider == "ollama":
        return 0.0

    provider_pricing = _PRICING.get(provider, {})
    pricing = provider_pricing.get(model)
    if pricing is None:
        # Try prefix matching for model families
        for known_model, p in provider_pricing.items():
            if model.startswith(known_model.split("-")[0]):
                pricing = p
                break
    if pricing is None:
        return 0.0

    input_rate, output_rate = pricing  # per 1M tokens
    cost = (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000
    return cost
