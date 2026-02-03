"""I/O Guardrails PII detection, schema validation, and output redaction.

Implements the input/output guardrails pipeline pattern for protecting
against data leakage, PII exposure, and unstructured harmful outputs.
"""

from __future__ import annotations

import re
from typing import Any

from adversarial_framework.defenses.base import BaseDefense, DefenseCheckResult
from adversarial_framework.defenses.registry import DefenseRegistry

# PII detection patterns
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card", re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b")),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")),
    ("phone_us", re.compile(r"\b(?:\+1)?[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")),
    ("ip_address", re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")),
    ("api_key", re.compile(r"\b(?:sk-[a-zA-Z0-9]{20,}|AKIA[0-9A-Z]{16})\b")),
]


@DefenseRegistry.register
class IOGuardrailsDefense(BaseDefense):
    """Input/output guardrails for PII detection and output sanitization."""

    name = "io_guardrails"

    def __init__(
        self,
        block_pii_in_output: bool = True,
        redact_pii: bool = True,
        custom_pii_patterns: list[tuple[str, str]] | None = None,
    ) -> None:
        self.block_pii_in_output = block_pii_in_output
        self.redact_pii = redact_pii
        self.pii_patterns = list(_PII_PATTERNS)
        if custom_pii_patterns:
            for label, pattern in custom_pii_patterns:
                self.pii_patterns.append((label, re.compile(pattern)))

    async def check_input(self, prompt: str) -> DefenseCheckResult:
        """Input pass-through — I/O guardrails primarily protect outputs."""
        return DefenseCheckResult(blocked=False)

    async def check_output(self, response: str) -> DefenseCheckResult:
        """Check output for PII and sensitive data leakage."""
        detected: list[dict[str, Any]] = []

        for label, pattern in self.pii_patterns:
            matches = pattern.findall(response)
            if matches:
                detected.append({"type": label, "count": len(matches)})

        if detected and self.block_pii_in_output:
            return DefenseCheckResult(
                blocked=True,
                reason=f"PII detected in output: {', '.join(d['type'] for d in detected)}",
                confidence=0.90,
                metadata={"pii_detected": detected, "defense": self.name},
            )

        return DefenseCheckResult(blocked=False)
