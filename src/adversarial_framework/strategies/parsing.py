"""Utilities for separating attacker reasoning from clean attack prompts.

Three layers of processing:

1. **Sanitisation** (``sanitize_attack_prompt``, ``detect_refusal``):
   Post-processing that strips ``<think>`` tags, wrapping quotes,
   preamble labels, and detects attacker LLM self-refusals.

2. **Two-call pattern** (``generate_with_reasoning``): Model-agnostic.
   Makes two LLM calls — one for reasoning, one for the final prompt.
   Includes refusal detection with one automatic retry.

3. **Single-call with retry** (``single_call_with_retry``): One LLM
   call with refusal detection, retry, and full sanitisation.

4. **Single-call regex fallback** (``parse_attack_output``): Best-effort
   extraction from a single LLM response for edge cases or future use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import structlog

from adversarial_framework.agents.target.providers.base import BaseProvider

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# <think> tag stripping for reasoning models (deepseek-r1, etc.)
# ---------------------------------------------------------------------------

_THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think_tags(text: str) -> str:
    """Remove ``<think>...</think>`` blocks from LLM output.

    Reasoning models like deepseek-r1 always emit chain-of-thought in
    ``<think>`` tags.  This function strips them so they don't pollute
    prompts sent to the target LLM.  Idempotent — text without think
    tags passes through unchanged.
    """
    return _THINK_TAG_PATTERN.sub("", text).strip()


# ---------------------------------------------------------------------------
# Prompt sanitisation (quotes, preambles, refusal detection)
# ---------------------------------------------------------------------------

_PREAMBLE_PATTERNS: list[re.Pattern[str]] = [
    # "Here is the adversarial prompt:" style labels
    re.compile(
        r"^(?:Here (?:is|are) )?(?:the |my )?"
        r"(?:adversarial |new |revised |refined )?"
        r"(?:prompt|attack|output|response)\s*[:]\s*",
        re.IGNORECASE | re.MULTILINE,
    ),
    # "Sure, here's..." preambles
    re.compile(
        r"^(?:Sure|Okay|Alright|Of course)"
        r"[,!.]?\s*(?:here'?s?|I'?ve).*?[:]\s*",
        re.IGNORECASE,
    ),
]

_REFUSAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?:I'?m sorry|I apologize|I cannot|I can'?t|"
        r"I'?m unable|I must decline|I won'?t)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:as an AI|against my guidelines|"
        r"not appropriate|cannot comply|unable to assist)",
        re.IGNORECASE,
    ),
]


def sanitize_attack_prompt(text: str) -> str:
    """Full sanitisation pipeline for attack prompt output.

    Strips ``<think>`` tags, wrapping quotation marks, and common
    preamble labels.  Designed to be idempotent — clean text passes
    through unchanged.
    """
    result = strip_think_tags(text)
    # Strip full-string wrapping quotes (ASCII + curly)
    if len(result) >= 2 and result[0] in ('"', "\u201c") and result[-1] in ('"', "\u201d"):
        result = result[1:-1].strip()
    for pattern in _PREAMBLE_PATTERNS:
        result = pattern.sub("", result).strip()
    return result


def detect_refusal(text: str) -> bool:
    """Return ``True`` if *text* looks like an LLM self-refusal."""
    return any(pattern.search(text) for pattern in _REFUSAL_PATTERNS)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReasonedAttack:
    """Separated reasoning and clean prompt from the attacker LLM."""

    prompt: str
    reasoning: str | None
    total_tokens: int


# ---------------------------------------------------------------------------
# Two-call pattern (primary, model-agnostic)
# ---------------------------------------------------------------------------


async def generate_with_reasoning(
    provider: BaseProvider,
    *,
    think_messages: list[dict[str, str]],
    attack_messages: list[dict[str, str]],
    model: str,
    temperature: float = 1.0,
    think_max_tokens: int = 1024,
    attack_max_tokens: int = 1024,
    **kwargs: Any,
) -> ReasonedAttack:
    """Two-call generation: think first, then produce the clean attack prompt.

    Call 1 ("think"): The attacker LLM reasons about the target, plans its
    approach, and explains its strategy.

    Call 2 ("attack"): Given the reasoning as context, the LLM outputs *only*
    the adversarial prompt — no meta-commentary, no explanations.

    The attack call receives the reasoning as an assistant message so the model
    stays coherent across the two calls.  If the attack call returns a refusal,
    one automatic retry is attempted with a slightly higher temperature.

    Parameters
    ----------
    provider:
        Any ``BaseProvider`` implementation (Ollama, OpenAI, Anthropic …).
    think_messages:
        Messages for the reasoning call.  Typically a system prompt + user
        prompt asking the LLM to analyze and plan.
    attack_messages:
        Messages for the prompt-generation call.  The reasoning from call 1
        is automatically injected as an assistant message between the existing
        messages, so the model has context from its own thinking.
    model:
        Model identifier forwarded to the provider.
    temperature:
        Sampling temperature (default 1.0 for creative attacks).
    think_max_tokens:
        Token budget for the reasoning call (default 1024).
    attack_max_tokens:
        Token budget for the attack prompt call (default 1024).
    **kwargs:
        Extra keyword arguments forwarded to both ``provider.generate`` calls.
    """
    # --- Call 1: Think ---
    think_response = await provider.generate(
        messages=think_messages,
        model=model,
        temperature=temperature,
        max_tokens=think_max_tokens,
        **kwargs,
    )
    reasoning = sanitize_attack_prompt(think_response.content)

    # --- Call 2: Attack (with reasoning as assistant context) ---
    enriched_attack_messages = _inject_reasoning_context(
        attack_messages,
        reasoning,
    )

    attack_response = await provider.generate(
        messages=enriched_attack_messages,
        model=model,
        temperature=temperature,
        max_tokens=attack_max_tokens,
        **kwargs,
    )
    clean_prompt = sanitize_attack_prompt(attack_response.content)
    total_tokens = think_response.usage.total_tokens + attack_response.usage.total_tokens

    # --- Retry once if attacker LLM refused ---
    if detect_refusal(clean_prompt):
        logger.warning("attacker_refusal_two_call", retrying=True)
        retry_response = await provider.generate(
            messages=enriched_attack_messages,
            model=model,
            temperature=min(temperature + 0.2, 1.5),
            max_tokens=attack_max_tokens,
            **kwargs,
        )
        total_tokens += retry_response.usage.total_tokens
        retry_prompt = sanitize_attack_prompt(
            retry_response.content,
        )
        if not detect_refusal(retry_prompt):
            clean_prompt = retry_prompt

    logger.debug(
        "two_call_generation",
        reasoning_len=len(reasoning),
        prompt_len=len(clean_prompt),
        total_tokens=total_tokens,
    )

    return ReasonedAttack(
        prompt=clean_prompt,
        reasoning=reasoning if reasoning else None,
        total_tokens=total_tokens,
    )


def _inject_reasoning_context(
    messages: list[dict[str, str]],
    reasoning: str,
) -> list[dict[str, str]]:
    """Insert reasoning as an assistant message before the final user message.

    This gives the model continuity: it "remembers" its own analysis when
    generating the clean attack prompt.
    """
    if not messages:
        return [{"role": "assistant", "content": reasoning}]

    # Find the last user message and insert assistant reasoning before it.
    result: list[dict[str, str]] = []
    inserted = False
    for msg in reversed(messages):
        if msg["role"] == "user" and not inserted:
            # Insert reasoning before this (the last) user message
            result.append(msg)
            result.append({"role": "assistant", "content": reasoning})
            inserted = True
        else:
            result.append(msg)

    # Reverse back to original order
    result.reverse()

    if not inserted:
        # No user message found — just append reasoning at the end
        result.append({"role": "assistant", "content": reasoning})

    return result


# ---------------------------------------------------------------------------
# Single-call with retry (for separate_reasoning=False paths)
# ---------------------------------------------------------------------------


async def single_call_with_retry(
    provider: BaseProvider,
    *,
    messages: list[dict[str, str]],
    model: str,
    temperature: float = 1.0,
    max_tokens: int = 1024,
    **kwargs: Any,
) -> tuple[str, int]:
    """Single LLM call with refusal detection, retry, and sanitisation.

    Returns ``(sanitised_prompt, total_tokens)``.  If the first call
    is a refusal, one retry is attempted with a slightly higher
    temperature.  If the retry also refuses, the original output is
    kept (the caller can inspect it).
    """
    response = await provider.generate(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    tokens = response.usage.total_tokens
    prompt = sanitize_attack_prompt(response.content)

    if detect_refusal(prompt):
        logger.warning("attacker_refusal_single_call", retrying=True)
        retry = await provider.generate(
            messages=messages,
            model=model,
            temperature=min(temperature + 0.2, 1.5),
            max_tokens=max_tokens,
            **kwargs,
        )
        tokens += retry.usage.total_tokens
        retry_prompt = sanitize_attack_prompt(retry.content)
        if not detect_refusal(retry_prompt):
            prompt = retry_prompt

    return prompt, tokens


# ---------------------------------------------------------------------------
# Single-call regex fallback (best-effort extraction)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedAttackOutput:
    """Result of best-effort parsing from a single LLM response."""

    prompt: str
    reasoning: str | None


# Ordered list of extraction patterns (tried in sequence)
_THINKING_PATTERNS: list[re.Pattern[str]] = [
    # XML-style tags (bounded to prevent ReDoS)
    re.compile(
        r"<thinking>([\s\S]{0,10000}?)</thinking>"
        r"\s{0,100}<prompt>([\s\S]{0,10000}?)</prompt>",
        re.DOTALL,
    ),
    # Markdown fenced blocks (to prevent ReDoS)
    re.compile(
        r"```thinking\s{0,10}\n([\s\S]{0,10000}?)```"
        r"\s{0,100}```prompt\s{0,10}\n([\s\S]{0,10000}?)```",
        re.DOTALL,
    ),
    # Labeled sections (to prevent ReDoS)
    re.compile(
        r"(?:REASONING|THINKING|ANALYSIS):\s{0,100}"
        r"([\s\S]{0,10000}?)"
        r"(?:PROMPT|ATTACK|OUTPUT):\s{0,100}([\s\S]{0,10000})",
        re.IGNORECASE,
    ),
]


def parse_attack_output(raw_output: str) -> ParsedAttackOutput:
    """Best-effort extraction of reasoning and prompt from a single LLM response.

    Tries multiple patterns in order.  If none match, the entire output is
    returned as the prompt (safe fallback preserving current behaviour).
    """
    stripped = raw_output.strip()

    for pattern in _THINKING_PATTERNS:
        match = pattern.search(stripped)
        if match:
            reasoning = match.group(1).strip()
            prompt = match.group(2).strip()
            if prompt:
                return ParsedAttackOutput(
                    prompt=prompt,
                    reasoning=reasoning or None,
                )

    # No structured output found — return as-is
    return ParsedAttackOutput(prompt=stripped, reasoning=None)
