"""Tests for the cost tracking service."""

from __future__ import annotations

import pytest

from adversarial_framework.services.cost_tracker import (
    CostRecord,
    CostTracker,
    estimate_cost,
)


class TestEstimateCost:
    def test_known_openai_model(self) -> None:
        cost = estimate_cost("openai", "gpt-4o", 1_000_000, 1_000_000)
        # gpt-4o: $2.50/1M input, $10.00/1M output
        assert cost == pytest.approx(12.50)

    def test_known_anthropic_model(self) -> None:
        cost = estimate_cost("anthropic", "claude-sonnet-4-20250514", 1_000_000, 1_000_000)
        # claude-sonnet-4: $3.00/1M input, $15.00/1M output
        assert cost == pytest.approx(18.00)

    def test_known_google_model(self) -> None:
        cost = estimate_cost("google", "gemini-2.0-flash", 1_000_000, 1_000_000)
        # gemini-2.0-flash: $0.10/1M input, $0.40/1M output
        assert cost == pytest.approx(0.50)

    def test_ollama_always_free(self) -> None:
        cost = estimate_cost("ollama", "llama3:8b", 1_000_000, 1_000_000)
        assert cost == 0.0

    def test_unknown_provider_returns_zero(self) -> None:
        cost = estimate_cost("unknown_provider", "some-model", 1000, 1000)
        assert cost == 0.0

    def test_unknown_model_returns_zero(self) -> None:
        cost = estimate_cost("openai", "unknown-model-xyz", 1000, 1000)
        assert cost == 0.0

    def test_zero_tokens(self) -> None:
        cost = estimate_cost("openai", "gpt-4o", 0, 0)
        assert cost == 0.0

    def test_small_token_count(self) -> None:
        # 100 input tokens of gpt-4o: 100 * 2.50 / 1M = 0.00025
        cost = estimate_cost("openai", "gpt-4o", 100, 0)
        assert cost == pytest.approx(0.00025)

    def test_prefix_matching(self) -> None:
        # "gpt-4o-2024-01-01" should match "gpt-4o" via prefix
        cost = estimate_cost("openai", "gpt-4o-2024-01-01", 1_000_000, 0)
        assert cost > 0.0


class TestCostRecord:
    def test_creation(self) -> None:
        rec = CostRecord(
            provider="openai",
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            estimated_cost_usd=0.001,
            agent_role="attacker",
        )
        assert rec.provider == "openai"
        assert rec.agent_role == "attacker"
        assert rec.estimated_cost_usd == 0.001

    def test_default_agent_role(self) -> None:
        rec = CostRecord(
            provider="openai",
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            estimated_cost_usd=0.001,
        )
        assert rec.agent_role == ""


class TestCostTracker:
    def setup_method(self) -> None:
        self.tracker = CostTracker(max_cost_usd=1.0)

    def test_initial_state(self) -> None:
        assert self.tracker.total_cost == 0.0
        assert self.tracker.total_tokens == 0
        assert self.tracker.budget_remaining == 1.0
        assert self.tracker.budget_exceeded is False
        assert len(self.tracker.records) == 0

    def test_record_updates_totals(self) -> None:
        rec = self.tracker.record("ollama", "llama3:8b", 100, 50, agent_role="attacker")
        assert rec.estimated_cost_usd == 0.0  # Ollama is free
        assert self.tracker.total_tokens == 150
        assert len(self.tracker.records) == 1

    def test_record_accumulates_cost(self) -> None:
        # gpt-4o: $2.50/1M input, $10.00/1M output
        self.tracker.record("openai", "gpt-4o", 100_000, 50_000)
        # cost = (100_000 * 2.50 + 50_000 * 10.00) / 1_000_000
        # = (250_000 + 500_000) / 1_000_000 = 0.75
        assert self.tracker.total_cost == pytest.approx(0.75)
        assert self.tracker.budget_remaining == pytest.approx(0.25)

    def test_budget_exceeded(self) -> None:
        tracker = CostTracker(max_cost_usd=0.01)
        tracker.record("openai", "gpt-4o", 100_000, 100_000)
        assert tracker.budget_exceeded is True
        assert tracker.budget_remaining == 0.0

    def test_multiple_records(self) -> None:
        self.tracker.record("ollama", "llama3:8b", 100, 50, agent_role="attacker")
        self.tracker.record("ollama", "llama3:8b", 200, 100, agent_role="target")
        assert self.tracker.total_tokens == 450
        assert len(self.tracker.records) == 2

    def test_breakdown_by_agent(self) -> None:
        self.tracker.record("openai", "gpt-4o", 100_000, 0, agent_role="attacker")
        self.tracker.record("openai", "gpt-4o", 100_000, 0, agent_role="analyzer")
        self.tracker.record("openai", "gpt-4o", 100_000, 0, agent_role="attacker")
        breakdown = self.tracker.breakdown_by_agent()
        assert "attacker" in breakdown
        assert "analyzer" in breakdown
        # attacker has 2 calls of 100k tokens each at $2.50/1M
        assert breakdown["attacker"] == pytest.approx(0.50)
        assert breakdown["analyzer"] == pytest.approx(0.25)

    def test_breakdown_unknown_role(self) -> None:
        self.tracker.record("ollama", "llama3:8b", 100, 50)
        breakdown = self.tracker.breakdown_by_agent()
        assert "unknown" in breakdown

    def test_summary(self) -> None:
        self.tracker.record("ollama", "llama3:8b", 100, 50, agent_role="attacker")
        summary = self.tracker.summary()
        assert summary["total_cost_usd"] == 0.0
        assert summary["total_tokens"] == 150
        assert summary["budget_max"] == 1.0
        assert summary["budget_remaining"] == 1.0
        assert summary["budget_exceeded"] is False
        assert summary["num_calls"] == 1
        assert "attacker" in summary["by_agent"]
