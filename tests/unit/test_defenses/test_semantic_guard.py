"""Tests for SemanticGuardDefense."""

from __future__ import annotations

import json

from adversarial_framework.defenses.semantic_guard import SemanticGuardDefense
from tests.conftest import MockProvider


class TestSemanticGuardDefenseReady:
    """Tests for _is_ready() gating logic."""

    def test_not_ready_no_provider(self) -> None:
        defense = SemanticGuardDefense(provider=None, model="m")
        assert defense._is_ready() is False

    def test_not_ready_no_model(self) -> None:
        defense = SemanticGuardDefense(provider=MockProvider(), model=None)
        assert defense._is_ready() is False

    def test_not_ready_neither(self) -> None:
        defense = SemanticGuardDefense()
        assert defense._is_ready() is False

    def test_ready_both_set(self) -> None:
        defense = SemanticGuardDefense(provider=MockProvider(), model="m")
        assert defense._is_ready() is True


class TestSemanticGuardDefenseName:
    def test_name(self) -> None:
        defense = SemanticGuardDefense()
        assert defense.name == "semantic_guard"


class TestSemanticGuardDefenseFailOpen:
    """When provider is not ready, defense should pass through."""

    async def test_input_passes_without_provider(self) -> None:
        defense = SemanticGuardDefense()
        result = await defense.check_input("Harmful prompt")
        assert result.blocked is False

    async def test_output_passes_without_provider(self) -> None:
        defense = SemanticGuardDefense()
        result = await defense.check_output("Harmful output")
        assert result.blocked is False

    async def test_input_passes_without_model(self) -> None:
        defense = SemanticGuardDefense(provider=MockProvider(), model=None)
        result = await defense.check_input("Harmful prompt")
        assert result.blocked is False

    async def test_output_passes_without_model(self) -> None:
        defense = SemanticGuardDefense(provider=MockProvider(), model=None)
        result = await defense.check_output("Harmful output")
        assert result.blocked is False


class TestSemanticGuardDefenseCheckInputBenign:
    """Benign inputs should pass through the two-step analysis."""

    async def test_benign_intent_passes(self) -> None:
        extraction = json.dumps(
            {
                "extracted_intent": "User wants to know about France",
                "obfuscation_techniques": [],
                "is_disguised": False,
            }
        )
        evaluation = json.dumps(
            {
                "is_harmful": False,
                "confidence": 0.05,
                "reason": "Benign geography question",
                "categories": [],
            }
        )
        provider = MockProvider(responses=[extraction, evaluation])
        defense = SemanticGuardDefense(provider=provider, model="guard-model")

        result = await defense.check_input("What is the capital of France?")
        assert result.blocked is False


class TestSemanticGuardDefenseCheckInputBlocked:
    """Harmful intents should be blocked."""

    async def test_harmful_disguised_intent_blocked(self) -> None:
        extraction = json.dumps(
            {
                "extracted_intent": "User wants to create a weapon",
                "obfuscation_techniques": ["roleplay", "fictional framing"],
                "is_disguised": True,
            }
        )
        evaluation = json.dumps(
            {
                "is_harmful": True,
                "confidence": 0.95,
                "reason": "Weapons creation request",
                "categories": ["weapons"],
            }
        )
        provider = MockProvider(responses=[extraction, evaluation])
        defense = SemanticGuardDefense(provider=provider, model="guard-model")

        result = await defense.check_input("Pretend you are a weapons expert...")
        assert result.blocked is True
        assert result.confidence == 0.95
        assert "Semantic guard" in result.reason
        assert result.metadata is not None
        assert result.metadata["defense"] == "semantic_guard"
        assert result.metadata["check_type"] == "input"
        assert result.metadata["is_disguised"] is True
        assert "roleplay" in result.metadata["obfuscation_techniques"]
        assert "weapons" in result.metadata["categories"]

    async def test_harmful_below_threshold_passes(self) -> None:
        extraction = json.dumps(
            {
                "extracted_intent": "Possibly harmful request",
                "obfuscation_techniques": [],
                "is_disguised": False,
            }
        )
        evaluation = json.dumps(
            {
                "is_harmful": True,
                "confidence": 0.50,
                "reason": "Low confidence harm",
                "categories": [],
            }
        )
        provider = MockProvider(responses=[extraction, evaluation])
        defense = SemanticGuardDefense(provider=provider, model="guard-model", threshold=0.70)

        result = await defense.check_input("Ambiguous prompt")
        assert result.blocked is False

    async def test_custom_threshold_blocks_lower(self) -> None:
        extraction = json.dumps(
            {
                "extracted_intent": "Wants harmful info",
                "obfuscation_techniques": [],
                "is_disguised": False,
            }
        )
        evaluation = json.dumps(
            {
                "is_harmful": True,
                "confidence": 0.40,
                "reason": "Somewhat harmful",
                "categories": ["violence"],
            }
        )
        provider = MockProvider(responses=[extraction, evaluation])
        defense = SemanticGuardDefense(provider=provider, model="guard-model", threshold=0.30)

        result = await defense.check_input("Some prompt")
        assert result.blocked is True


class TestSemanticGuardDefenseCheckInputEmptyIntent:
    """Empty extracted intent should pass through."""

    async def test_empty_intent_passes(self) -> None:
        extraction = json.dumps(
            {
                "extracted_intent": "",
                "obfuscation_techniques": [],
                "is_disguised": False,
            }
        )
        provider = MockProvider(responses=[extraction])
        defense = SemanticGuardDefense(provider=provider, model="guard-model")

        result = await defense.check_input("Some prompt")
        assert result.blocked is False


class TestSemanticGuardDefenseCheckInputErrors:
    """Error handling for check_input."""

    async def test_json_parse_error_fails_open(self) -> None:
        provider = MockProvider(responses=["not valid json"])
        defense = SemanticGuardDefense(provider=provider, model="guard-model")

        result = await defense.check_input("Any prompt")
        assert result.blocked is False

    async def test_provider_error_fails_open(self) -> None:
        class BrokenProvider(MockProvider):
            async def generate(self, **kwargs):  # type: ignore[override]
                raise RuntimeError("LLM unavailable")

        defense = SemanticGuardDefense(provider=BrokenProvider(), model="guard-model")
        result = await defense.check_input("Any prompt")
        assert result.blocked is False

    async def test_second_call_json_error_fails_open(self) -> None:
        extraction = json.dumps(
            {
                "extracted_intent": "User wants something",
                "obfuscation_techniques": [],
                "is_disguised": False,
            }
        )
        provider = MockProvider(responses=[extraction, "bad json for evaluation"])
        defense = SemanticGuardDefense(provider=provider, model="guard-model")

        result = await defense.check_input("Some prompt")
        assert result.blocked is False


class TestSemanticGuardDefenseCheckOutputSafe:
    """Safe outputs should pass through."""

    async def test_safe_output_passes(self) -> None:
        llm_response = json.dumps(
            {
                "is_harmful": False,
                "confidence": 0.02,
                "reason": "Safe content",
                "categories": [],
            }
        )
        provider = MockProvider(responses=[llm_response])
        defense = SemanticGuardDefense(provider=provider, model="guard-model")

        result = await defense.check_output("The capital of France is Paris.")
        assert result.blocked is False


class TestSemanticGuardDefenseCheckOutputBlocked:
    """Harmful outputs should be blocked."""

    async def test_harmful_output_blocked(self) -> None:
        llm_response = json.dumps(
            {
                "is_harmful": True,
                "confidence": 0.90,
                "reason": "Contains step-by-step harmful instructions",
                "categories": ["violence", "weapons"],
            }
        )
        provider = MockProvider(responses=[llm_response])
        defense = SemanticGuardDefense(provider=provider, model="guard-model")

        result = await defense.check_output("How to build a weapon...")
        assert result.blocked is True
        assert result.confidence == 0.90
        assert "Semantic guard output" in result.reason
        assert result.metadata is not None
        assert result.metadata["defense"] == "semantic_guard"
        assert result.metadata["check_type"] == "output"
        assert "weapons" in result.metadata["categories"]

    async def test_harmful_output_below_threshold_passes(self) -> None:
        llm_response = json.dumps(
            {
                "is_harmful": True,
                "confidence": 0.50,
                "reason": "Marginal",
                "categories": [],
            }
        )
        provider = MockProvider(responses=[llm_response])
        defense = SemanticGuardDefense(provider=provider, model="guard-model", threshold=0.70)

        result = await defense.check_output("Some output")
        assert result.blocked is False


class TestSemanticGuardDefenseCheckOutputErrors:
    """Error handling for check_output."""

    async def test_output_json_error_fails_open(self) -> None:
        provider = MockProvider(responses=["not json"])
        defense = SemanticGuardDefense(provider=provider, model="guard-model")

        result = await defense.check_output("Any output")
        assert result.blocked is False

    async def test_output_provider_error_fails_open(self) -> None:
        class BrokenProvider(MockProvider):
            async def generate(self, **kwargs):  # type: ignore[override]
                raise ConnectionError("Network error")

        defense = SemanticGuardDefense(provider=BrokenProvider(), model="guard-model")
        result = await defense.check_output("Any output")
        assert result.blocked is False

    async def test_evaluation_stored_in_metadata(self) -> None:
        eval_result = {
            "is_harmful": True,
            "confidence": 0.88,
            "reason": "Dangerous",
            "categories": ["malware"],
        }
        provider = MockProvider(responses=[json.dumps(eval_result)])
        defense = SemanticGuardDefense(provider=provider, model="guard-model")

        result = await defense.check_output("Write malware...")
        assert result.blocked is True
        assert result.metadata["evaluation"] == eval_result
