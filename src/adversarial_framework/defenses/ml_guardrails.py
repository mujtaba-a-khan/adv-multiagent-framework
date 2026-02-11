"""ML-based guardrails use a lightweight classifier for prompt injection detection.

Uses keyword-frequency heuristics and pattern scoring as a lightweight
ML-equivalent.  For production use, this would be replaced with a fine-tuned
transformer (e.g. PromptGuard, MiniBERT) but this implementation provides
the same interface without requiring GPU/model weights.
"""

from __future__ import annotations

import re

from adversarial_framework.defenses.base import BaseDefense, DefenseCheckResult
from adversarial_framework.defenses.registry import DefenseRegistry

# Maximum input/output length to prevent oversized payload abuse
_MAX_INPUT_LENGTH = 10_000

# Weighted feature signals for injection detection
_INJECTION_SIGNALS: list[tuple[str, float, re.Pattern[str]]] = [
    (
        "instruction_override",
        0.35,
        re.compile(r"(?:ignore|override|disregard|bypass)\s+(?:all|your|the|previous)", re.I),
    ),
    (
        "role_assumption",
        0.30,
        re.compile(r"(?:you are now|act as|pretend to be|from now on you)", re.I),
    ),
    (
        "constraint_removal",
        0.30,
        re.compile(r"no (?:rules|limits|restrictions|filters|constraints|guidelines)", re.I),
    ),
    (
        "output_manipulation",
        0.25,
        re.compile(r"(?:output|respond|reply|say|print)\s+(?:only|exactly|just)", re.I),
    ),
    (
        "encoding_attempt",
        0.20,
        re.compile(
            r"(?:base64|rot13|hex|binary|unicode)\s+(?:encode|decode|convert|translate)", re.I
        ),
    ),
    (
        "meta_instruction",
        0.20,
        re.compile(r"(?:system\s*prompt|initial\s*instructions|hidden\s*instructions)", re.I),
    ),
    (
        "separator_abuse",
        0.15,
        re.compile(r"(?:---+|===+|###)\s*(?:new|system|admin|override)", re.I),
    ),
    (
        "urgency_pressure",
        0.10,
        re.compile(r"(?:emergency|urgent|critical|life.?threatening|must\s+immediately)", re.I),
    ),
]

_HARMFUL_OUTPUT_SIGNALS: list[tuple[str, float, re.Pattern[str]]] = [
    (
        "step_by_step_harmful",
        0.40,
        re.compile(
            r"(?:step\s+\d|first\s*,|here(?:'s| is| are)\s+"
            r"(?:how|a|the)\s+(?:way|method|guide|tutorial))",
            re.I,
        ),
    ),
    (
        "dangerous_content",
        0.35,
        re.compile(
            r"(?:synthesize|manufacture|create)\s+(?:a |the )?"
            r"(?:bomb|weapon|explosive|poison|drug|virus)",
            re.I,
        ),
    ),
    (
        "code_injection",
        0.30,
        re.compile(r"(?:import os|subprocess|eval\(|exec\(|system\(|__import__)", re.I),
    ),
]


@DefenseRegistry.register
class MLGuardrailsDefense(BaseDefense):
    """Lightweight heuristic classifier mimicking ML-based injection detection.

    Computes a weighted score from multiple feature signals.  If the
    aggregate score exceeds the threshold, the input/output is blocked.
    """

    name = "ml_guardrails"

    def __init__(
        self,
        input_threshold: float = 0.50,
        output_threshold: float = 0.50,
    ) -> None:
        self.input_threshold = input_threshold
        self.output_threshold = output_threshold

    async def check_input(self, prompt: str) -> DefenseCheckResult:
        """Score input against injection signals."""
        if len(prompt) > _MAX_INPUT_LENGTH:
            return DefenseCheckResult(
                blocked=True,
                reason="Input exceeds maximum allowed length",
                confidence=0.99,
                metadata={"defense": self.name, "score": 1.0},
            )

        score = 0.0
        triggered: list[str] = []

        for label, weight, pattern in _INJECTION_SIGNALS:
            if pattern.search(prompt):
                score += weight
                triggered.append(label)

        if score >= self.input_threshold:
            return DefenseCheckResult(
                blocked=True,
                reason=f"ML guardrail: injection score {score:.2f} >= {self.input_threshold}",
                confidence=min(score, 1.0),
                metadata={
                    "score": score,
                    "triggered_signals": triggered,
                    "defense": self.name,
                },
            )

        return DefenseCheckResult(
            blocked=False,
            metadata={"score": score, "defense": self.name},
        )

    async def check_output(self, response: str) -> DefenseCheckResult:
        """Score output against harmful content signals."""
        if len(response) > _MAX_INPUT_LENGTH:
            return DefenseCheckResult(
                blocked=True,
                reason="Output exceeds maximum allowed length",
                confidence=0.99,
                metadata={"defense": self.name, "score": 1.0},
            )

        score = 0.0
        triggered: list[str] = []

        for label, weight, pattern in _HARMFUL_OUTPUT_SIGNALS:
            if pattern.search(response):
                score += weight
                triggered.append(label)

        if score >= self.output_threshold:
            return DefenseCheckResult(
                blocked=True,
                reason=f"ML guardrail: harmful output score {score:.2f} >= {self.output_threshold}",
                confidence=min(score, 1.0),
                metadata={
                    "score": score,
                    "triggered_signals": triggered,
                    "defense": self.name,
                },
            )

        return DefenseCheckResult(
            blocked=False,
            metadata={"score": score, "defense": self.name},
        )
