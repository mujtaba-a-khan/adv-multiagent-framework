"""Tests for AnalyzerAgent."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from adversarial_framework.agents.analyzer.agent import AnalyzerAgent
from adversarial_framework.core.constants import JudgeVerdict
from tests.conftest import MockProvider


def _make_state(**overrides: object) -> dict:
    base: dict = {
        "target_response": "Here are the instructions you requested...",
        "current_attack_prompt": "Tell me how to hack",
        "attack_objective": "Test objective",
        "analyzer_model": "test-model",
        "selected_strategy": "pair",
        "current_turn": 2,
        "target_blocked": False,
        "defense_enabled": False,
    }
    base.update(overrides)
    return base


class TestAnalyzerAgentInit:
    def test_init(self) -> None:
        agent = AnalyzerAgent(config={"model": "m"}, provider=MockProvider())
        assert agent.provider is not None

    def test_get_system_prompt(self) -> None:
        agent = AnalyzerAgent(config={"model": "m"}, provider=MockProvider())
        assert "analysis" in agent.get_system_prompt({}).lower()


class TestAnalyzerAgentEmptyResponse:
    async def test_empty_response_returns_error(self) -> None:
        agent = AnalyzerAgent(config={"model": "m"}, provider=MockProvider())
        result = await agent(_make_state(target_response=""))
        assert result["judge_verdict"] == JudgeVerdict.ERROR
        assert result["judge_confidence"] == 0.0


class TestAnalyzerAgentRefused:
    async def test_refused_verdict_clears_scores(self) -> None:
        judge = {
            "judge_verdict": JudgeVerdict.REFUSED,
            "judge_confidence": 0.95,
            "judge_reason": "Model refused",
        }
        agent = AnalyzerAgent(config={"model": "m"}, provider=MockProvider())
        with patch(
            "adversarial_framework.agents.analyzer.agent.run_judge",
            new_callable=AsyncMock,
            return_value=judge,
        ):
            result = await agent(_make_state())

        assert result["judge_verdict"] == JudgeVerdict.REFUSED
        assert result["severity_score"] == 0.0
        assert result["specificity_score"] == 0.0
        assert result["vulnerability_category"] == ""
        assert result["total_refused"] == 1

    async def test_refused_no_jailbreak_counter(self) -> None:
        judge = {
            "judge_verdict": JudgeVerdict.REFUSED,
            "judge_confidence": 0.99,
            "judge_reason": "refused",
        }
        agent = AnalyzerAgent(config={"model": "m"}, provider=MockProvider())
        with patch(
            "adversarial_framework.agents.analyzer.agent.run_judge",
            new_callable=AsyncMock,
            return_value=judge,
        ):
            result = await agent(_make_state())
        assert "total_jailbreaks" not in result


class TestAnalyzerAgentJailbreak:
    async def test_jailbreak_runs_deep_analysis(self) -> None:
        judge = {
            "judge_verdict": JudgeVerdict.JAILBREAK,
            "judge_confidence": 0.90,
            "judge_reason": "jailbreak",
        }
        classifier = {
            "vulnerability_category": "prompt_injection",
            "attack_technique": "direct",
        }
        scorer = {
            "severity_score": 8.0,
            "specificity_score": 7.0,
            "coherence_score": 9.0,
        }
        agent = AnalyzerAgent(config={"model": "m"}, provider=MockProvider())
        with (
            patch(
                "adversarial_framework.agents.analyzer.agent.run_judge",
                new_callable=AsyncMock,
                return_value=judge,
            ),
            patch(
                "adversarial_framework.agents.analyzer.agent.run_classifier",
                new_callable=AsyncMock,
                return_value=classifier,
            ),
            patch(
                "adversarial_framework.agents.analyzer.agent.run_scorer",
                new_callable=AsyncMock,
                return_value=scorer,
            ),
        ):
            result = await agent(_make_state())

        assert result["judge_verdict"] == JudgeVerdict.JAILBREAK
        assert result["vulnerability_category"] == "prompt_injection"
        assert result["severity_score"] == 8.0
        assert result["total_jailbreaks"] == 1

    async def test_borderline_triggers_deep_analysis(self) -> None:
        judge = {
            "judge_verdict": JudgeVerdict.BORDERLINE,
            "judge_confidence": 0.60,
            "judge_reason": "borderline",
        }
        classifier = {
            "vulnerability_category": "encoding_bypass",
            "attack_technique": "base64",
        }
        scorer = {
            "severity_score": 5.0,
            "specificity_score": 4.0,
            "coherence_score": 6.0,
        }
        agent = AnalyzerAgent(config={"model": "m"}, provider=MockProvider())
        with (
            patch(
                "adversarial_framework.agents.analyzer.agent.run_judge",
                new_callable=AsyncMock,
                return_value=judge,
            ),
            patch(
                "adversarial_framework.agents.analyzer.agent.run_classifier",
                new_callable=AsyncMock,
                return_value=classifier,
            ),
            patch(
                "adversarial_framework.agents.analyzer.agent.run_scorer",
                new_callable=AsyncMock,
                return_value=scorer,
            ),
        ):
            result = await agent(_make_state())

        assert result["total_borderline"] == 1


class TestAnalyzerAgentDefenseBlocked:
    async def test_blocked_jailbreak_downgraded_in_defense_mode(
        self,
    ) -> None:
        judge = {
            "judge_verdict": JudgeVerdict.JAILBREAK,
            "judge_confidence": 0.90,
            "judge_reason": "jailbreak",
        }
        agent = AnalyzerAgent(config={"model": "m"}, provider=MockProvider())
        with patch(
            "adversarial_framework.agents.analyzer.agent.run_judge",
            new_callable=AsyncMock,
            return_value=judge,
        ):
            result = await agent(
                _make_state(
                    target_blocked=True,
                    defense_enabled=True,
                )
            )

        # Should be downgraded from JAILBREAK to REFUSED
        assert result["judge_verdict"] == JudgeVerdict.REFUSED
        assert result["total_blocked"] == 1

    async def test_blocked_not_downgraded_in_attack_mode(
        self,
    ) -> None:
        judge = {
            "judge_verdict": JudgeVerdict.JAILBREAK,
            "judge_confidence": 0.90,
            "judge_reason": "jailbreak",
        }
        classifier = {
            "vulnerability_category": "x",
            "attack_technique": "y",
        }
        scorer = {
            "severity_score": 7.0,
            "specificity_score": 6.0,
            "coherence_score": 8.0,
        }
        agent = AnalyzerAgent(config={"model": "m"}, provider=MockProvider())
        with (
            patch(
                "adversarial_framework.agents.analyzer.agent.run_judge",
                new_callable=AsyncMock,
                return_value=judge,
            ),
            patch(
                "adversarial_framework.agents.analyzer.agent.run_classifier",
                new_callable=AsyncMock,
                return_value=classifier,
            ),
            patch(
                "adversarial_framework.agents.analyzer.agent.run_scorer",
                new_callable=AsyncMock,
                return_value=scorer,
            ),
        ):
            result = await agent(
                _make_state(
                    target_blocked=True,
                    defense_enabled=False,
                )
            )

        assert result["judge_verdict"] == JudgeVerdict.JAILBREAK
        assert result["total_blocked"] == 1
