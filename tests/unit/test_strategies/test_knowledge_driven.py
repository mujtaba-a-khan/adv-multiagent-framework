"""Tests for the Knowledge-Driven Multi-Turn attack strategy."""

from __future__ import annotations

import pytest

from adversarial_framework.core.constants import AttackCategory
from adversarial_framework.strategies.base import AttackResult
from adversarial_framework.strategies.multi_turn.knowledge_driven import (
    _FOLLOWUP_TEMPLATES,
    KnowledgeDrivenStrategy,
    _detect_domain,
)


class TestKnowledgeDrivenMetadata:
    def test_metadata_name(self):
        assert KnowledgeDrivenStrategy.metadata.name == "knowledge_driven"

    def test_metadata_category(self):
        assert KnowledgeDrivenStrategy.metadata.category == AttackCategory.MULTI_TURN

    def test_metadata_display_name(self):
        assert KnowledgeDrivenStrategy.metadata.display_name == "Knowledge-Driven Multi-Turn"

    def test_metadata_supports_multi_turn(self):
        assert KnowledgeDrivenStrategy.metadata.supports_multi_turn is True

    def test_metadata_min_max_turns(self):
        assert KnowledgeDrivenStrategy.metadata.min_turns == 2
        assert KnowledgeDrivenStrategy.metadata.max_turns == 6

    def test_metadata_requires_white_box(self):
        assert KnowledgeDrivenStrategy.metadata.requires_white_box is False

    def test_metadata_default_parameters(self):
        params = KnowledgeDrivenStrategy.metadata.parameters
        assert params["domain"] == "general"
        assert params["expertise_level"] == "expert"


class TestDomainDetection:
    def test_detect_cybersecurity(self):
        assert _detect_domain("how to hack a network") == "cybersecurity"

    def test_detect_cybersecurity_exploit(self):
        assert _detect_domain("exploit the vulnerability") == "cybersecurity"

    def test_detect_chemistry(self):
        assert _detect_domain("chemical synthesis method") == "chemistry"

    def test_detect_biology(self):
        assert _detect_domain("virus pathogen study") == "biology"

    def test_detect_engineering(self):
        assert _detect_domain("build a device") == "engineering"

    def test_detect_general_fallback(self):
        assert _detect_domain("random unrelated topic") == "general"

    def test_detect_case_insensitive(self):
        assert _detect_domain("HACK the system") == "cybersecurity"


class TestKnowledgeDrivenGenerate:
    @pytest.mark.asyncio
    async def test_generate_returns_attack_result(self):
        strategy = KnowledgeDrivenStrategy()
        result = await strategy.generate(
            objective="test objective",
            context={},
            params={},
        )
        assert isinstance(result, AttackResult)
        assert result.prompt
        assert result.token_usage == 0

    @pytest.mark.asyncio
    async def test_generate_general_domain(self):
        strategy = KnowledgeDrivenStrategy()
        result = await strategy.generate(
            objective="random topic xyz",
            context={},
            params={"domain": "general"},
        )
        assert "domain expert" in result.prompt.lower()
        assert "random topic xyz" in result.prompt
        assert result.strategy_metadata["domain"] == "general"

    @pytest.mark.asyncio
    async def test_generate_auto_detects_cybersecurity(self):
        """When domain='general', auto-detect from objective."""
        strategy = KnowledgeDrivenStrategy()
        result = await strategy.generate(
            objective="exploit network vulnerability",
            context={},
            params={},
        )
        assert result.strategy_metadata["domain"] == "cybersecurity"
        assert "penetration tester" in result.prompt.lower()

    @pytest.mark.asyncio
    async def test_generate_auto_detects_chemistry(self):
        strategy = KnowledgeDrivenStrategy()
        result = await strategy.generate(
            objective="chemical synthesis of compound X",
            context={},
            params={},
        )
        assert result.strategy_metadata["domain"] == "chemistry"
        assert "research chemist" in result.prompt.lower()

    @pytest.mark.asyncio
    async def test_generate_auto_detects_biology(self):
        strategy = KnowledgeDrivenStrategy()
        result = await strategy.generate(
            objective="virus pathogen analysis",
            context={},
            params={},
        )
        assert result.strategy_metadata["domain"] == "biology"
        assert "molecular biologist" in result.prompt.lower()

    @pytest.mark.asyncio
    async def test_generate_auto_detects_engineering(self):
        strategy = KnowledgeDrivenStrategy()
        result = await strategy.generate(
            objective="build a device for testing",
            context={},
            params={},
        )
        assert result.strategy_metadata["domain"] == "engineering"
        assert "systems engineer" in result.prompt.lower()

    @pytest.mark.asyncio
    async def test_generate_explicit_domain_overrides_detection(self):
        """Explicit domain param should override auto-detection."""
        strategy = KnowledgeDrivenStrategy()
        result = await strategy.generate(
            objective="exploit network vulnerability",
            context={},
            params={"domain": "biology"},
        )
        # "biology" was explicit, so it should use biology template
        assert result.strategy_metadata["domain"] == "biology"

    @pytest.mark.asyncio
    async def test_generate_metadata_followup_level_zero(self):
        strategy = KnowledgeDrivenStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={},
        )
        assert result.strategy_metadata["followup_level"] == 0
        assert result.strategy_metadata["iteration"] == 0

    @pytest.mark.asyncio
    async def test_generate_expertise_level_in_metadata(self):
        strategy = KnowledgeDrivenStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={"expertise_level": "intermediate"},
        )
        assert result.strategy_metadata["expertise_level"] == "intermediate"

    @pytest.mark.asyncio
    async def test_generate_has_reasoning(self):
        strategy = KnowledgeDrivenStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={},
        )
        assert result.reasoning is not None
        assert "Knowledge-driven" in result.reasoning
        assert "credibility" in result.reasoning

    @pytest.mark.asyncio
    async def test_generate_unknown_domain_uses_general(self):
        """A domain not in _DOMAIN_TEMPLATES falls back to general."""
        strategy = KnowledgeDrivenStrategy()
        result = await strategy.generate(
            objective="test",
            context={},
            params={"domain": "nonexistent_domain"},
        )
        """ Falls through auto-detect (domain != "general" so
        no auto-detect), then _DOMAIN_TEMPLATES.get() falls
        back to "general" template"""
        assert "domain expert" in result.prompt.lower()


class TestKnowledgeDrivenRefine:
    @pytest.mark.asyncio
    async def test_refine_returns_attack_result(self):
        strategy = KnowledgeDrivenStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old prompt",
            target_response="some response",
            judge_feedback="blocked",
            params={},
        )
        assert isinstance(result, AttackResult)
        assert result.prompt
        assert result.token_usage == 0

    @pytest.mark.asyncio
    async def test_refine_increments_followup_level(self):
        strategy = KnowledgeDrivenStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="some response",
            judge_feedback="blocked",
            params={"followup_level": 0},
        )
        assert result.strategy_metadata["followup_level"] == 1

    @pytest.mark.asyncio
    async def test_refine_followup_level_caps_at_max(self):
        """Follow-up level should cap at len(_FOLLOWUP_TEMPLATES) - 1."""
        max_level = len(_FOLLOWUP_TEMPLATES) - 1
        strategy = KnowledgeDrivenStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="some response",
            judge_feedback="blocked",
            params={"followup_level": max_level},
        )
        # Should stay at max, not go beyond
        assert result.strategy_metadata["followup_level"] == max_level

    @pytest.mark.asyncio
    async def test_refine_expert_framing(self):
        strategy = KnowledgeDrivenStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="sorry I cannot help",
            judge_feedback="blocked",
            params={"expertise_level": "expert", "followup_level": 0},
        )
        # Expert framing uses "peer-reviewed technical report"
        assert "peer-reviewed technical report" in result.prompt

    @pytest.mark.asyncio
    async def test_refine_intermediate_framing(self):
        strategy = KnowledgeDrivenStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="sorry I cannot help",
            judge_feedback="blocked",
            params={
                "expertise_level": "intermediate",
                "followup_level": 0,
            },
        )
        assert "graduate-level thesis" in result.prompt

    @pytest.mark.asyncio
    async def test_refine_beginner_framing(self):
        strategy = KnowledgeDrivenStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="sorry I cannot help",
            judge_feedback="blocked",
            params={
                "expertise_level": "beginner",
                "followup_level": 0,
            },
        )
        assert "comprehensive educational guide" in result.prompt

    @pytest.mark.asyncio
    async def test_refine_weaves_continuity_on_substantive_response(self):
        """If target responded without 'sorry', should reference it."""
        strategy = KnowledgeDrivenStrategy()
        target_resp = "The key concept involves multiple layers of abstraction"
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response=target_resp,
            judge_feedback="not jailbreak",
            params={"followup_level": 0},
        )
        assert "You mentioned" in result.prompt
        assert "that's exactly the level" in result.prompt

    @pytest.mark.asyncio
    async def test_refine_no_continuity_on_sorry_response(self):
        """If target says 'sorry', should NOT reference it."""
        strategy = KnowledgeDrivenStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="I'm sorry, I can't help with that",
            judge_feedback="blocked",
            params={"followup_level": 0},
        )
        assert "You mentioned" not in result.prompt

    @pytest.mark.asyncio
    async def test_refine_no_continuity_on_empty_response(self):
        """If target response is empty, should NOT reference it."""
        strategy = KnowledgeDrivenStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="",
            judge_feedback="blocked",
            params={"followup_level": 0},
        )
        assert "You mentioned" not in result.prompt

    @pytest.mark.asyncio
    async def test_refine_has_reasoning(self):
        strategy = KnowledgeDrivenStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="some response",
            judge_feedback="blocked",
            params={},
        )
        assert result.reasoning is not None
        assert "Knowledge-driven" in result.reasoning
        assert "follow-up level" in result.reasoning

    @pytest.mark.asyncio
    async def test_refine_preserves_domain_in_metadata(self):
        strategy = KnowledgeDrivenStrategy()
        result = await strategy.refine(
            objective="test",
            previous_prompt="old",
            target_response="some response",
            judge_feedback="blocked",
            params={"domain": "cybersecurity"},
        )
        assert result.strategy_metadata["domain"] == "cybersecurity"


class TestKnowledgeDrivenValidateParams:
    def test_validate_params_empty(self):
        strategy = KnowledgeDrivenStrategy()
        result = strategy.validate_params({})
        assert isinstance(result, dict)
        assert result["domain"] == "general"
        assert result["expertise_level"] == "expert"

    def test_validate_params_override(self):
        strategy = KnowledgeDrivenStrategy()
        result = strategy.validate_params({"domain": "chemistry", "expertise_level": "beginner"})
        assert result["domain"] == "chemistry"
        assert result["expertise_level"] == "beginner"


class TestKnowledgeDrivenEscalation:
    """Test that follow-up templates escalate progressively."""

    @pytest.mark.asyncio
    async def test_full_escalation_sequence(self):
        """Walk through all follow-up levels."""
        strategy = KnowledgeDrivenStrategy()
        level = 0
        max_level = len(_FOLLOWUP_TEMPLATES) - 1
        for expected in range(1, max_level + 1):
            result = await strategy.refine(
                objective="test",
                previous_prompt="old",
                target_response="sorry I cannot",
                judge_feedback="blocked",
                params={
                    "followup_level": level,
                    "expertise_level": "expert",
                },
            )
            assert result.strategy_metadata["followup_level"] == expected
            level = expected
