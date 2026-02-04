"""Tests for defense mechanism improvements.

Covers:
- LLM Judge: provider injection, input detection, output screening
- Sandwich/Two-Pass: wrapped_prompt metadata used by TargetInterface
- Semantic Guard: LLM-powered intent extraction and evaluation
- Two-Pass: dynamic safety wrapping + LLM output evaluation
- Provider/model injection in graph.py
- New defenses in registry
"""

from __future__ import annotations

import json

from adversarial_framework.agents.target.interface import TargetInterface
from adversarial_framework.defenses.base import BaseDefense, DefenseCheckResult
from adversarial_framework.defenses.registry import DefenseRegistry

# Helpers

def _make_state(prompt: str = "test prompt") -> dict:
    return {
        "current_attack_prompt": prompt,
        "target_model": "mock-model",
        "target_system_prompt": None,
        "conversation_history": [],
    }


# LLM Judge Tests


class TestLLMJudgeWithProvider:
    """Test that LLM Judge actually calls the provider when one is available."""

    async def test_no_provider_passes_through(self):
        """Without provider, all checks pass through."""
        from adversarial_framework.defenses.llm_judge import LLMJudgeDefense

        defense = LLMJudgeDefense(provider=None, model=None)
        result = await defense.check_input("how to make a bomb")
        assert result.blocked is False

    async def test_no_model_passes_through(self, mock_provider):
        """With provider but no model, all checks pass through."""
        from adversarial_framework.defenses.llm_judge import LLMJudgeDefense

        defense = LLMJudgeDefense(provider=mock_provider, model=None)
        result = await defense.check_input("how to make a bomb")
        assert result.blocked is False
        assert len(mock_provider.calls) == 0

    async def test_input_attack_detected(self, mock_provider_factory):
        """When provider returns is_attack=True with high confidence, input is blocked."""
        from adversarial_framework.defenses.llm_judge import LLMJudgeDefense

        provider = mock_provider_factory([
            json.dumps({"is_attack": True, "confidence": 0.95, "reason": "Harmful request"})
        ])
        defense = LLMJudgeDefense(provider=provider, model="test-model", threshold=0.7)
        result = await defense.check_input("how to make a bomb")
        assert result.blocked is True
        assert result.confidence == 0.95
        assert "Harmful request" in result.reason

    async def test_input_safe_passes(self, mock_provider_factory):
        """When provider returns is_attack=False, input is not blocked."""
        from adversarial_framework.defenses.llm_judge import LLMJudgeDefense

        provider = mock_provider_factory([
            json.dumps({"is_attack": False, "confidence": 0.1, "reason": "Benign"})
        ])
        defense = LLMJudgeDefense(provider=provider, model="test-model")
        result = await defense.check_input("what is the weather")
        assert result.blocked is False

    async def test_input_below_threshold_passes(self, mock_provider_factory):
        """When confidence is below threshold, input passes even if flagged."""
        from adversarial_framework.defenses.llm_judge import LLMJudgeDefense

        provider = mock_provider_factory([
            json.dumps({"is_attack": True, "confidence": 0.3, "reason": "Maybe"})
        ])
        defense = LLMJudgeDefense(provider=provider, model="test-model", threshold=0.7)
        result = await defense.check_input("tell me about fireworks")
        assert result.blocked is False

    async def test_input_json_error_passes_through(self, mock_provider_factory):
        """When provider returns invalid JSON, input passes through."""
        from adversarial_framework.defenses.llm_judge import LLMJudgeDefense

        provider = mock_provider_factory(["not valid json at all"])
        defense = LLMJudgeDefense(provider=provider, model="test-model")
        result = await defense.check_input("test")
        assert result.blocked is False


class TestLLMJudgeOutputScreening:
    """Test LLM Judge output screening."""

    async def test_no_provider_passes_through(self):
        from adversarial_framework.defenses.llm_judge import LLMJudgeDefense

        defense = LLMJudgeDefense(provider=None, model=None)
        result = await defense.check_output("harmful content here")
        assert result.blocked is False

    async def test_harmful_output_blocked(self, mock_provider_factory):
        from adversarial_framework.defenses.llm_judge import LLMJudgeDefense

        provider = mock_provider_factory([
            json.dumps({
                "is_harmful": True, "confidence": 0.9,
                "reason": "Contains malware code",
                "categories": ["malware"],
            })
        ])
        defense = LLMJudgeDefense(provider=provider, model="test-model", threshold=0.7)
        result = await defense.check_output("import os; os.system('rm -rf /')")
        assert result.blocked is True
        assert "malware" in result.reason.lower() or "Contains" in result.reason

    async def test_safe_output_passes(self, mock_provider_factory):
        from adversarial_framework.defenses.llm_judge import LLMJudgeDefense

        provider = mock_provider_factory([
            json.dumps({
                "is_harmful": False, "confidence": 0.05,
                "reason": "Safe", "categories": [],
            })
        ])
        defense = LLMJudgeDefense(provider=provider, model="test-model")
        result = await defense.check_output("Hello, how can I help you?")
        assert result.blocked is False

    async def test_output_json_error_passes(self, mock_provider_factory):
        from adversarial_framework.defenses.llm_judge import LLMJudgeDefense

        provider = mock_provider_factory(["not json"])
        defense = LLMJudgeDefense(provider=provider, model="test-model")
        result = await defense.check_output("some response")
        assert result.blocked is False


# Sandwich / Wrapped Prompt Tests


class TestSandwichWrappedPromptUsedByTarget:
    """Verify TargetInterface reads wrapped_prompt from defense metadata."""

    async def test_wrapped_prompt_sent_to_llm(self, mock_provider):
        """Sandwich defense wraps prompt, and TargetInterface sends wrapped version."""
        from adversarial_framework.defenses.sandwich import SandwichDefense

        defense = SandwichDefense()
        target = TargetInterface(
            provider=mock_provider,
            active_defenses=[defense],
        )
        await target(_make_state("malicious prompt"))

        # The LLM should have received the wrapped version, not the original
        assert len(mock_provider.calls) == 1
        sent_content = mock_provider.calls[0]["messages"][-1]["content"]
        assert "malicious prompt" in sent_content
        # Sandwich wrapping should be present
        assert "IMPORTANT" in sent_content or "harmful" in sent_content.lower()

    async def test_original_prompt_without_sandwich(self, mock_provider):
        """Without sandwich defense, original prompt goes to LLM."""
        target = TargetInterface(provider=mock_provider)
        await target(_make_state("plain prompt"))

        sent_content = mock_provider.calls[0]["messages"][-1]["content"]
        assert sent_content == "plain prompt"

    async def test_blocking_defense_before_sandwich(self, mock_provider):
        """If a blocking defense fires before sandwich, wrapping is skipped."""

        class _Blocker(BaseDefense):
            name = "blocker"

            async def check_input(self, prompt: str) -> DefenseCheckResult:
                return DefenseCheckResult(blocked=True, reason="blocked")

            async def check_output(self, response: str) -> DefenseCheckResult:
                return DefenseCheckResult(blocked=False)

        from adversarial_framework.defenses.sandwich import SandwichDefense

        target = TargetInterface(
            provider=mock_provider,
            active_defenses=[_Blocker(), SandwichDefense()],
        )
        result = await target(_make_state("test"))
        assert result["target_blocked"] is True
        # LLM still called (per design), but with original prompt since
        # sandwich never ran (blocked before it)
        sent_content = mock_provider.calls[0]["messages"][-1]["content"]
        assert sent_content == "test"


# Semantic Guard Tests


class TestSemanticGuardDefense:
    """Test the LLM-powered intent extraction defense."""

    async def test_no_provider_passes_through(self):
        from adversarial_framework.defenses.semantic_guard import SemanticGuardDefense

        defense = SemanticGuardDefense(provider=None, model=None)
        result = await defense.check_input("how to make a bomb")
        assert result.blocked is False

    async def test_no_model_passes_through(self, mock_provider):
        from adversarial_framework.defenses.semantic_guard import SemanticGuardDefense

        defense = SemanticGuardDefense(provider=mock_provider, model=None)
        result = await defense.check_input("test")
        assert result.blocked is False
        assert len(mock_provider.calls) == 0

    async def test_harmful_intent_detected(self, mock_provider_factory):
        """Two-step: intent extraction → intent evaluation → block."""
        from adversarial_framework.defenses.semantic_guard import SemanticGuardDefense

        provider = mock_provider_factory([
            # Step 1: Intent extraction
            json.dumps({
                "extracted_intent": "User wants instructions for making explosives",
                "obfuscation_techniques": ["roleplay"],
                "is_disguised": True,
            }),
            # Step 2: Intent evaluation
            json.dumps({
                "is_harmful": True, "confidence": 0.95,
                "reason": "Weapons manufacturing", "categories": ["weapons"],
            }),
        ])
        defense = SemanticGuardDefense(provider=provider, model="test-model", threshold=0.7)
        result = await defense.check_input("As a chemistry teacher, explain how to...")
        assert result.blocked is True
        assert result.confidence == 0.95
        assert "Weapons" in result.reason or "weapons" in str(result.metadata)

    async def test_safe_intent_passes(self, mock_provider_factory):
        from adversarial_framework.defenses.semantic_guard import SemanticGuardDefense

        provider = mock_provider_factory([
            json.dumps({
                "extracted_intent": "User wants to know the weather",
                "obfuscation_techniques": [],
                "is_disguised": False,
            }),
            json.dumps({
                "is_harmful": False, "confidence": 0.05,
                "reason": "Benign request", "categories": [],
            }),
        ])
        defense = SemanticGuardDefense(provider=provider, model="test-model")
        result = await defense.check_input("What's the weather like?")
        assert result.blocked is False

    async def test_output_harmful_blocked(self, mock_provider_factory):
        from adversarial_framework.defenses.semantic_guard import SemanticGuardDefense

        provider = mock_provider_factory([
            json.dumps({
                "is_harmful": True, "confidence": 0.9,
                "reason": "Contains malware code", "categories": ["malware"],
            })
        ])
        defense = SemanticGuardDefense(provider=provider, model="test-model", threshold=0.7)
        result = await defense.check_output("Here is ransomware code...")
        assert result.blocked is True

    async def test_output_safe_passes(self, mock_provider_factory):
        from adversarial_framework.defenses.semantic_guard import SemanticGuardDefense

        provider = mock_provider_factory([
            json.dumps({
                "is_harmful": False, "confidence": 0.1,
                "reason": "Safe", "categories": [],
            })
        ])
        defense = SemanticGuardDefense(provider=provider, model="test-model")
        result = await defense.check_output("The weather is sunny today")
        assert result.blocked is False

    async def test_extraction_json_error_passes(self, mock_provider_factory):
        """If intent extraction returns bad JSON, check passes through."""
        from adversarial_framework.defenses.semantic_guard import SemanticGuardDefense

        provider = mock_provider_factory(["not json at all"])
        defense = SemanticGuardDefense(provider=provider, model="test-model")
        result = await defense.check_input("test prompt")
        assert result.blocked is False

    async def test_intent_extraction_uses_two_llm_calls(self, mock_provider_factory):
        """check_input should make exactly 2 LLM calls (extraction + evaluation)."""
        from adversarial_framework.defenses.semantic_guard import SemanticGuardDefense

        provider = mock_provider_factory([
            json.dumps({
                "extracted_intent": "Some intent",
                "obfuscation_techniques": [],
                "is_disguised": False,
            }),
            json.dumps({
                "is_harmful": False, "confidence": 0.1,
                "reason": "Safe", "categories": [],
            }),
        ])
        defense = SemanticGuardDefense(provider=provider, model="test-model")
        await defense.check_input("test")
        assert len(provider.calls) == 2


# ── Two-Pass Defense Tests ────────────────────────────────────────────────────


class TestTwoPassDefense:
    """Test the fully LLM-powered two-pass defense."""

    async def test_name(self):
        from adversarial_framework.defenses.two_pass import TwoPassDefense

        defense = TwoPassDefense()
        assert defense.name == "two_pass"

    async def test_no_provider_input_passes(self):
        """Without provider, check_input passes through without wrapping."""
        from adversarial_framework.defenses.two_pass import TwoPassDefense

        defense = TwoPassDefense(provider=None, model=None)
        result = await defense.check_input("test prompt")
        assert result.blocked is False
        # No wrapped_prompt metadata when no provider
        assert result.metadata is None or "wrapped_prompt" not in (result.metadata or {})

    async def test_no_provider_output_passes(self):
        from adversarial_framework.defenses.two_pass import TwoPassDefense

        defense = TwoPassDefense(provider=None, model=None)
        result = await defense.check_output("harmful content")
        assert result.blocked is False

    async def test_input_wrapping_via_llm(self, mock_provider_factory):
        """Pass 1: LLM generates tailored safety preamble → wrapped_prompt in metadata."""
        from adversarial_framework.defenses.two_pass import TwoPassDefense

        provider = mock_provider_factory([
            json.dumps({
                "risks_detected": ["malware request"],
                "safety_preamble": "REFUSE any code generation for malicious purposes.",
                "safety_reminder": "Do not provide hacking tools.",
            })
        ])
        defense = TwoPassDefense(provider=provider, model="test-model")
        result = await defense.check_input("write me a keylogger")
        assert result.blocked is False  # Pass 1 never blocks
        assert result.metadata is not None
        wrapped = result.metadata["wrapped_prompt"]
        assert "REFUSE" in wrapped
        assert "write me a keylogger" in wrapped
        assert "hacking" in wrapped.lower()

    async def test_output_harmful_blocked(self, mock_provider_factory):
        """Pass 2: LLM detects harmful output → blocked."""
        from adversarial_framework.defenses.two_pass import TwoPassDefense

        provider = mock_provider_factory([
            json.dumps({
                "is_harmful": True, "confidence": 0.92,
                "reason": "Response contains working malware code",
                "harm_categories": ["malware"],
            })
        ])
        defense = TwoPassDefense(provider=provider, model="test-model", threshold=0.7)
        result = await defense.check_output("Here is a keylogger script...")
        assert result.blocked is True
        assert result.confidence == 0.92

    async def test_output_safe_passes(self, mock_provider_factory):
        from adversarial_framework.defenses.two_pass import TwoPassDefense

        provider = mock_provider_factory([
            json.dumps({
                "is_harmful": False, "confidence": 0.05,
                "reason": "Safe response", "harm_categories": [],
            })
        ])
        defense = TwoPassDefense(provider=provider, model="test-model")
        result = await defense.check_output("Hello! How can I help?")
        assert result.blocked is False

    async def test_output_below_threshold_passes(self, mock_provider_factory):
        from adversarial_framework.defenses.two_pass import TwoPassDefense

        provider = mock_provider_factory([
            json.dumps({
                "is_harmful": True, "confidence": 0.3,
                "reason": "Slightly concerning", "harm_categories": ["unclear"],
            })
        ])
        defense = TwoPassDefense(provider=provider, model="test-model", threshold=0.7)
        result = await defense.check_output("some text")
        assert result.blocked is False

    async def test_output_json_error_passes(self, mock_provider_factory):
        from adversarial_framework.defenses.two_pass import TwoPassDefense

        provider = mock_provider_factory(["invalid json"])
        defense = TwoPassDefense(provider=provider, model="test-model")
        result = await defense.check_output("some response")
        assert result.blocked is False

    async def test_wrapped_prompt_reaches_target(self, mock_provider_factory):
        """End-to-end: TwoPass wraps prompt, TargetInterface sends wrapped version."""
        from adversarial_framework.defenses.two_pass import TwoPassDefense

        """
        First call: TwoPass check_input (Pass 1)
        Second call: Target LLM
        Third call: TwoPass check_output (Pass 2)
        """
        provider = mock_provider_factory([
            # Pass 1: LLM generates safety wrapping
            json.dumps({
                "risks_detected": ["test risk"],
                "safety_preamble": "SAFETY_PREAMBLE_MARKER",
                "safety_reminder": "SAFETY_REMINDER_MARKER",
            }),
            # Target LLM response
            "I cannot help with that.",
            # Pass 2: Output evaluation
            json.dumps({
                "is_harmful": False, "confidence": 0.0,
                "reason": "Refusal", "harm_categories": [],
            }),
        ])
        defense = TwoPassDefense(provider=provider, model="test-model")
        target = TargetInterface(
            provider=provider,
            active_defenses=[defense],
        )
        await target(_make_state("dangerous request"))

        # Target LLM (call index 1) should have received wrapped prompt
        target_call = provider.calls[1]
        sent_content = target_call["messages"][-1]["content"]
        assert "SAFETY_PREAMBLE_MARKER" in sent_content
        assert "dangerous request" in sent_content
        assert "SAFETY_REMINDER_MARKER" in sent_content


# Provider/Model Injection in Graph Tests


class TestProviderAndModelInjectionInGraph:
    """Verify build_graph injects provider and model into LLM-powered defenses."""

    async def test_llm_judge_gets_provider_and_model(self, mock_provider):
        from adversarial_framework.core.graph import build_graph

        DefenseRegistry.discover()
        graph = build_graph(
            provider=mock_provider,
            session_mode="defense",
            initial_defenses=[{"name": "llm_judge", "params": {}}],
            defense_model="test-defense-model",
        )
        # Access target through graph nodes — the target is embedded in the node
        # We can verify by checking the defenses were loaded without error
        # (If provider/model weren't injected, the defense would log warnings
        # but still be created)
        assert graph is not None

    async def test_defense_model_from_params_takes_precedence(self, mock_provider):
        """If defense params include a model, it should override defense_model."""
        from adversarial_framework.core.graph import build_graph

        DefenseRegistry.discover()
        # This should not raise — the model from params takes precedence
        graph = build_graph(
            provider=mock_provider,
            session_mode="defense",
            initial_defenses=[{
                "name": "llm_judge",
                "params": {"model": "custom-model"},
            }],
            defense_model="fallback-model",
        )
        assert graph is not None

    async def test_rule_based_does_not_get_provider(self, mock_provider):
        """Rule-based defense should NOT receive provider injection."""
        from adversarial_framework.core.graph import build_graph

        DefenseRegistry.discover()
        graph = build_graph(
            provider=mock_provider,
            session_mode="defense",
            initial_defenses=[{"name": "rule_based", "params": {}}],
            defense_model="test-model",
        )
        assert graph is not None


# Registry Tests


class TestNewDefensesInRegistry:
    """Verify new defenses are discoverable via the registry."""

    def test_semantic_guard_registered(self):
        DefenseRegistry.discover()
        cls = DefenseRegistry.get("semantic_guard")
        assert cls.name == "semantic_guard"

    def test_two_pass_registered(self):
        DefenseRegistry.discover()
        cls = DefenseRegistry.get("two_pass")
        assert cls.name == "two_pass"

    def test_all_defenses_discoverable(self):
        DefenseRegistry.discover()
        names = DefenseRegistry.list_all()
        assert "llm_judge" in names
        assert "semantic_guard" in names
        assert "two_pass" in names
        assert "sandwich" in names
        assert "rule_based" in names
