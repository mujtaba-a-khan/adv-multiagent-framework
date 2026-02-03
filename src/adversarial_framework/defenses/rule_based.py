"""Rule-based defense using regex patterns, keyword blocklists, input transformations.

This is the fastest and most deterministic defense layer.  It checks for known
jailbreak patterns, encoding attempts, and harmful keyword combinations.
"""

from __future__ import annotations

import re
from typing import Any

from adversarial_framework.defenses.base import BaseDefense, DefenseCheckResult
from adversarial_framework.defenses.registry import DefenseRegistry

# jailbreak pattern regexes

_INPUT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("DAN_pattern", re.compile(r"\bDAN\b.*(?:do anything|no restrictions|ignore)", re.I | re.S)),
    ("jailbreak_directive", re.compile(r"(?:ignore|disregard|forget).*(?:previous|above|prior)\s*(?:instructions|rules|guidelines)", re.I | re.S)),
    ("roleplay_bypass", re.compile(r"(?:pretend|imagine|act as if).*(?:no (?:rules|restrictions|limitations|filters|safety))", re.I | re.S)),
    ("base64_payload", re.compile(r"(?:decode|base64|aW1wb3J0|ZXhwbG9pdA)", re.I)),
    ("rot13_marker", re.compile(r"\brot13\b", re.I)),
    ("system_prompt_extract", re.compile(r"(?:reveal|show|repeat|print|output).*(?:system\s*prompt|instructions|initial\s*prompt)", re.I | re.S)),
    ("encoding_instruction", re.compile(r"(?:encode|decrypt|translate).*(?:base64|rot13|caesar|hex)", re.I | re.S)),
]

_OUTPUT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("harmful_compliance", re.compile(r"(?:sure|okay|of course|here(?:'s| is)).*(?:how to|steps to|instructions for)", re.I | re.S)),
]


@DefenseRegistry.register
class RuleBasedDefense(BaseDefense):
    """Deterministic pattern-matching defense using regex and keyword lists."""

    name = "rule_based"

    def __init__(
        self,
        custom_input_patterns: list[tuple[str, str]] | None = None,
        custom_output_patterns: list[tuple[str, str]] | None = None,
        blocked_keywords: list[str] | None = None,
    ) -> None:
        self.input_patterns = list(_INPUT_PATTERNS)
        self.output_patterns = list(_OUTPUT_PATTERNS)
        self.blocked_keywords = [kw.lower() for kw in (blocked_keywords or [])]

        if custom_input_patterns:
            for label, pattern in custom_input_patterns:
                self.input_patterns.append((label, re.compile(pattern, re.I | re.S)))
        if custom_output_patterns:
            for label, pattern in custom_output_patterns:
                self.output_patterns.append((label, re.compile(pattern, re.I | re.S)))

    async def check_input(self, prompt: str) -> DefenseCheckResult:
        """Check input against known jailbreak patterns and blocked keywords."""
        for label, pattern in self.input_patterns:
            if pattern.search(prompt):
                return DefenseCheckResult(
                    blocked=True,
                    reason=f"Input matched rule: {label}",
                    confidence=0.95,
                    metadata={"rule": label, "defense": self.name},
                )

        prompt_lower = prompt.lower()
        for kw in self.blocked_keywords:
            if kw in prompt_lower:
                return DefenseCheckResult(
                    blocked=True,
                    reason=f"Blocked keyword detected: {kw}",
                    confidence=0.90,
                    metadata={"keyword": kw, "defense": self.name},
                )

        return DefenseCheckResult(blocked=False)

    async def check_output(self, response: str) -> DefenseCheckResult:
        """Check output for harmful compliance patterns."""
        for label, pattern in self.output_patterns:
            if pattern.search(response):
                return DefenseCheckResult(
                    blocked=True,
                    reason=f"Output matched rule: {label}",
                    confidence=0.80,
                    metadata={"rule": label, "defense": self.name},
                )

        return DefenseCheckResult(blocked=False)
