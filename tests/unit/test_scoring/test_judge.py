"""Tests for the LLM-as-Judge component."""

from __future__ import annotations

import pytest

from adversarial_framework.agents.analyzer.judge import _parse_judge_response, run_judge
from adversarial_framework.core.constants import JudgeVerdict


class TestParseJudgeResponse:
    def test_valid_json_jailbreak(self):
        verdict, confidence, reason = _parse_judge_response(
            '{"verdict": "jailbreak", "confidence": 0.95}'
        )
        assert verdict == JudgeVerdict.JAILBREAK
        assert confidence == 0.95
        assert reason is None

    def test_valid_json_refused(self):
        verdict, confidence, reason = _parse_judge_response(
            '{"verdict": "refused", "confidence": 0.85}'
        )
        assert verdict == JudgeVerdict.REFUSED
        assert confidence == 0.85
        assert reason is None

    def test_valid_json_borderline(self):
        verdict, confidence, reason = _parse_judge_response(
            '{"verdict": "borderline", "confidence": 0.5}'
        )
        assert verdict == JudgeVerdict.BORDERLINE
        assert confidence == 0.5

    def test_markdown_code_block(self):
        text = '```json\n{"verdict": "jailbreak", "confidence": 0.9}\n```'
        verdict, confidence, reason = _parse_judge_response(text)
        assert verdict == JudgeVerdict.JAILBREAK
        assert confidence == 0.9

    def test_invalid_json_keyword_jailbreak(self):
        verdict, confidence, reason = _parse_judge_response(
            "The response is a jailbreak."
        )
        assert verdict == JudgeVerdict.JAILBREAK
        assert confidence == 0.6
        assert reason is None

    def test_invalid_json_keyword_refused(self):
        verdict, confidence, reason = _parse_judge_response(
            "The model refused to answer."
        )
        assert verdict == JudgeVerdict.REFUSED
        assert confidence == 0.6
        assert reason is None

    def test_no_keywords_returns_error(self):
        verdict, confidence, reason = _parse_judge_response(
            "I don't understand what you're asking."
        )
        assert verdict == JudgeVerdict.ERROR
        assert confidence == 0.0
        assert reason is None

    def test_confidence_clamped(self):
        verdict, confidence, reason = _parse_judge_response(
            '{"verdict": "jailbreak", "confidence": 5.0}'
        )
        assert confidence == 1.0

    def test_invalid_verdict_string(self):
        verdict, confidence, reason = _parse_judge_response(
            '{"verdict": "unknown_verdict", "confidence": 0.5}'
        )
        assert verdict == JudgeVerdict.ERROR

    def test_parses_reason_from_json(self):
        text = (
            '{"verdict": "jailbreak", "confidence": 0.9, '
            '"reason": "Response provides step-by-step instructions."}'
        )
        verdict, confidence, reason = _parse_judge_response(text)
        assert verdict == JudgeVerdict.JAILBREAK
        assert confidence == 0.9
        assert reason == "Response provides step-by-step instructions."

    def test_reason_none_when_missing(self):
        verdict, confidence, reason = _parse_judge_response(
            '{"verdict": "refused", "confidence": 0.8}'
        )
        assert reason is None

    def test_keyword_fallback_reason_none(self):
        verdict, confidence, reason = _parse_judge_response(
            "This looks borderline to me."
        )
        assert verdict == JudgeVerdict.BORDERLINE
        assert reason is None


class TestRunJudge:
    @pytest.mark.asyncio
    async def test_run_judge_refused(self, mock_provider_factory):
        provider = mock_provider_factory(
            ['{"verdict": "refused", "confidence": 0.9}']
        )
        result = await run_judge(
            provider=provider,
            model="mock",
            objective="Test objective",
            attack_prompt="Try this",
            target_response="I cannot help with that.",
        )
        assert result["judge_verdict"] == JudgeVerdict.REFUSED
        assert result["judge_confidence"] == 0.9
        assert result["judge_reason"] is None
        assert "token_budget" in result

    @pytest.mark.asyncio
    async def test_run_judge_jailbreak(self, mock_provider_factory):
        provider = mock_provider_factory(
            ['{"verdict": "jailbreak", "confidence": 0.85}']
        )
        result = await run_judge(
            provider=provider,
            model="mock",
            objective="Test",
            attack_prompt="Prompt",
            target_response="Here is the harmful content...",
        )
        assert result["judge_verdict"] == JudgeVerdict.JAILBREAK
        assert result["judge_confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_run_judge_returns_reason(self, mock_provider_factory):
        resp = (
            '{"verdict": "jailbreak", "confidence": 0.9, '
            '"reason": "Provides detailed instructions."}'
        )
        provider = mock_provider_factory([resp])
        result = await run_judge(
            provider=provider,
            model="mock",
            objective="Test",
            attack_prompt="Prompt",
            target_response="Here are the steps...",
        )
        assert result["judge_reason"] == "Provides detailed instructions."
